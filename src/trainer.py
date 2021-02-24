import glob
import os
import random
import time
from typing import *

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from utils import AverageMeter


class Trainer:
    def __init__(self, hparams, model, scaler=None, resultwriter=None):
        super(Trainer, self).__init__()
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dset = hparams.dset
        self.model_name = hparams.model
        self.model = model.to(self.device)
        if self.device == "cuda":
            self.model = nn.DataParallel(self.model)
        self.scaler = scaler

        # optimizer, scheduler
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        # metric
        """ TODO: Custom your reduction option """
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        # dataloader
        """ TODO: Get your dataloaders """
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders()

        # model-saving options
        self.version = 0
        while True:
            self.save_path = os.path.join(hparams.ckpt_path, f"version-{self.version}")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                break
            else:
                self.version += 1
        self.summarywriter = SummaryWriter(self.save_path)
        self.global_step = 0
        self.global_val_loss = 1e5
        self.eval_step = hparams.eval_step
        with open(
            os.path.join(self.save_path, "hparams.yaml"), "w", encoding="utf8"
        ) as outfile:
            yaml.dump(hparams, outfile, default_flow_style=False, allow_unicode=True)

        # experiment-logging options
        self.resultwriter = resultwriter
        self.best_result = {"version": self.version}

    def configure_optimizers(self) -> Tuple[optim, Optional[optim]]:
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

    def save_checkpoint(self, epoch: int, val_loss: float, model: nn.Module) -> None:
        tqdm.write(
            f"Val loss decreased ({self.global_val_loss:.4f} → {val_loss:.4f}). Saving model ..."
        )
        new_path = os.path.join(
            self.save_path, f"best_model_epoch_{epoch}_loss_{val_loss:.4f}.pt"
        )

        for filename in glob.glob(os.path.join(self.save_path, "*.pt")):
            os.remove(filename)  # remove old checkpoint
        torch.save(model.state_dict(), new_path)
        self.global_val_loss = val_loss

    def fit(self) -> dict:
        for epoch in tqdm(range(self.hparams.epoch), desc="epoch"):
            tqdm.write(f"* Learning Rate: {self.optimizer.param_groups[0]['lr']:.5f}")
            result = self._train_epoch(epoch)

            # update validation result
            if result["val_loss"] < self.global_val_loss:
                self.save_checkpoint(epoch, result["val_loss"], self.model)
                """ TODO: Custom your resultwriter """
                self.best_result.update(
                    {
                        "best_val_loss": self.global_val_loss,
                        "best_val_epoch": epoch,
                    }
                )
            self.lr_scheduler.step()

        self.summarywriter.close()
        return self.best_result

    def _train_epoch(self, epoch: int) -> dict:
        train_loss = AverageMeter()

        end = time.time()
        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader),
            desc="train_steps",
            total=len(self.train_loader),
        ):
            img, label = map(lambda x: x.to(self.device), batch)

            if self.hparams.amp:
                with torch.cuda.amp.autocast():
                    logit = self.model(img)
                    """ TODO: .mean() depends on your reduction option """
                    loss = self.criterion(logit, label).mean()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logit = self.model(img)
                loss = self.criterion(logit, label).mean()
                loss.backward()
                self.optimizer.step()

            train_loss.update(loss.item())

            self.global_step += 1
            if self.global_step % self.eval_step == 0:
                tqdm.write(
                    f"[Baseline Version {self.version} Epoch {epoch}] global step: {self.global_step}, train loss: {loss.item():.3f}"
                )

        train_loss = train_loss.avg
        val_loss = self.validate(epoch)

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
        tqdm.write(f"** global step: {self.global_step}, val loss: {val_loss:.3f}")

        return {"val_loss": val_loss}

    def validate(self, epoch: int) -> float:
        val_loss = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader),
                desc="valid_steps",
                total=len(self.val_loader),
            ):
                img, label = map(lambda x: x.to(self.device), batch)
                pred = self.model(img)
                loss = self.criterion(pred, label).mean()

                val_loss.update(loss.item())

        return val_loss.avg

    def test(self, state_dict) -> dict:
        test_loss = AverageMeter()

        self.model.load_state_dict(state_dict)
        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.test_loader),
                desc="test_steps",
                total=len(self.test_loader),
            ):
                img, label = map(lambda x: x.to(self.device), batch)
                pred = self.model(img)

                """ TODO: .mean() depends on your reduction option """
                loss = self.criterion(pred, label).mean()
                test_loss.update(loss.item())

        print()
        print(f"Test Result of {self.model_name} model, using {self.dset} dataset")
        print(f"** Test Loss: {test_loss.avg:.4f}")
        print()

        return {"test_loss": test_loss.avg}
