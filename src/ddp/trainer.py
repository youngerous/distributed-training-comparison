import glob
import os
import random
from typing import *

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataset import get_trn_val_loader, get_tst_loader
from utils import AverageMeter, accuracy, reduce_mean


class Trainer:
    def __init__(self, hparams, model, scaler, rank, ngpus_per_node):
        self.hparams = hparams
        self.rank = rank
        self.nprocs = torch.cuda.device_count()
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        self.dset = hparams.dset

        self.model_name = hparams.model
        self.model = model
        self.model = model.to(self.device, non_blocking=True)
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)
        self.scaler = scaler

        hparams.batch_size = int(hparams.batch_size / ngpus_per_node)

        # optimizer, scheduler
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        # metric
        self.criterion = nn.CrossEntropyLoss()

        # dataloader and distributed sampler
        self.train_loader, self.train_sampler, self.val_loader = get_trn_val_loader(
            data_dir=hparams.dpath.strip(),
            batch_size=hparams.batch_size,
            valid_size=0.1,
            num_workers=hparams.workers,
            pin_memory=True,
        )
        self.test_loader = get_tst_loader(
            data_dir=hparams.dpath.strip(),
            batch_size=hparams.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        # model-saving options (only at rank 0)
        if self.rank == 0:
            self.version = 0
            while True:
                self.save_path = os.path.join(
                    hparams.ckpt_path, f"version-{self.version}"
                )
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                    break
                else:
                    self.version += 1
            self.summarywriter = SummaryWriter(self.save_path)
            self.global_step = 0
            self.global_val_loss = 1e5
            self.global_top1_acc = 0
            self.eval_step = hparams.eval_step
            with open(
                os.path.join(self.save_path, "hparams.yaml"), "w", encoding="utf8"
            ) as outfile:
                yaml.dump(
                    hparams, outfile, default_flow_style=False, allow_unicode=True
                )

            # experiment-logging options
            self.best_result = {"version": self.version}

    def configure_optimizers(self):
        # optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )

        # lr scheduler (optional)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_decay_step_size,
            gamma=self.hparams.lr_decay_gamma,
        )
        return optimizer, scheduler

    def save_checkpoint(self, epoch: int, val_acc: float, model: nn.Module) -> None:
        tqdm.write(
            f"Val acc increased ({self.global_top1_acc:.4f} → {val_acc:.4f}). Saving model ..."
        )
        new_path = os.path.join(
            self.save_path, f"best_model_epoch_{epoch}_acc_{val_acc:.4f}.pt"
        )

        for filename in glob.glob(os.path.join(self.save_path, "*.pt")):
            os.remove(filename)  # remove old checkpoint
        torch.save(model.state_dict(), new_path)
        self.global_top1_acc = val_acc

    def fit(self) -> dict:
        for epoch in tqdm(range(self.hparams.epoch), desc="epoch"):
            self.train_sampler.set_epoch(epoch)

            tqdm.write(f"* Learning Rate: {self.optimizer.param_groups[0]['lr']:.5f}")
            result = self._train_epoch(epoch)

            # update checkpoint
            if self.rank == 0 and result["val_acc"] > self.global_top1_acc:
                self.save_checkpoint(epoch, result["val_acc"], self.model)
            self.lr_scheduler.step()

        if self.rank == 0:
            self.summarywriter.close()
        return self.version if self.rank == 0 else None

    def _train_epoch(self, epoch: int) -> dict:
        train_loss = AverageMeter()

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader),
            desc="train_steps",
            total=len(self.train_loader),
        ):
            img, label = map(lambda x: x.to(self.device, non_blocking=True), batch)

            if self.hparams.amp:
                with torch.cuda.amp.autocast():
                    logit = self.model(img)
                    loss = self.criterion(logit, label)
                dist.barrier()
                reduced_loss = reduce_mean(loss, self.nprocs)
                self.scaler.scale(reduced_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logit = self.model(img)
                loss = self.criterion(logit, label)
                dist.barrier()
                reduced_loss = reduce_mean(loss, self.nprocs)
                reduced_loss.backward()
                self.optimizer.step()

            train_loss.update(reduced_loss.item())

            if self.rank == 0:
                self.global_step += 1
                if self.global_step % self.eval_step == 0:
                    tqdm.write(
                        f"[DDP Version {self.version} Epoch {epoch}] global step: {self.global_step}, train loss: {reduced_loss.item():.3f}"
                    )

        train_loss = train_loss.avg

        if self.rank == 0:
            val_loss, val_acc = self.validate(epoch)

            # tensorboard writing
            self.summarywriter.add_scalars(
                "lr", {"lr": self.optimizer.param_groups[0]["lr"]}, epoch
            )
            self.summarywriter.add_scalars(
                "loss/step", {"val": val_loss, "train": train_loss}, self.global_step
            )
            self.summarywriter.add_scalars(
                "loss/epoch", {"val": val_loss, "train": train_loss}, epoch
            )
            self.summarywriter.add_scalars("acc/epoch", {"val": val_acc}, epoch)
            tqdm.write(
                f"** global step: {self.global_step}, val loss: {val_loss:.3f}, val_acc: {val_acc:.2f}%"
            )

            return {"val_loss": val_loss, "val_acc": val_acc}
        return None

    def validate(self, epoch: int) -> Tuple[float]:
        val_loss = AverageMeter()
        top1 = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader),
                desc="valid_steps",
                total=len(self.val_loader),
            ):
                img, label = map(lambda x: x.to(self.device, non_blocking=True), batch)
                pred = self.model(img)

                loss = self.criterion(pred, label)
                val_loss.update(loss.item())

                prec1 = accuracy(pred, label, topk=(1,))[0]
                top1.update(prec1.item())

        return val_loss.avg, top1.avg

    def test(self, state_dict) -> dict:
        test_loss = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.load_state_dict(state_dict)
        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.test_loader),
                desc="tst_steps",
                total=len(self.test_loader),
            ):
                img, label = map(lambda x: x.to(self.device, non_blocking=True), batch)
                pred = self.model(img)

                loss = self.criterion(pred, label)
                test_loss.update(loss.item())

                prec1, prec5 = accuracy(pred, label, topk=(1, 5))
                top1.update(prec1.item())
                top5.update(prec5.item())

        print()
        print(f"** Test Loss: {test_loss.avg:.4f}")
        print(f"** Top-1 Accuracy: {top1.avg:.4f}%")
        print(f"** Top-5 Accuracy: {top5.avg:.4f}%")
        print()

        return {
            "test_loss": test_loss.avg,
            "top_1_acc": top1.avg,
            "top_5_acc": top5.avg,
        }
