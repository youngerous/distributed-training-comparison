import glob
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import load_config
from net import ResNet18
from trainer import Trainer
from utils import fix_seed


def main_worker(rank, ngpus_per_node, hparams):
    print(f"Use GPU {rank} for training")
    fix_seed(hparams.seed)
    hparams.rank = hparams.rank * ngpus_per_node + rank
    dist.init_process_group(
        backend=hparams.dist_backend,
        init_method=hparams.dist_url,
        world_size=hparams.world_size,
        rank=hparams.rank,
    )

    scaler = torch.cuda.amp.GradScaler() if hparams.amp else None
    model = ResNet18()

    # training phase
    trainer = Trainer(hparams, model, scaler, rank, ngpus_per_node)
    version = trainer.fit()

    # testing phase
    if rank == 0 and hparams.contain_test:
        state_dict = torch.load(
            glob.glob(
                os.path.join(hparams.ckpt_path, f"version-{version}/best_model_*.pt")
            )[0]
        )
        trainer.test(state_dict)


if __name__ == "__main__":
    hparams = load_config()

    # 'world_size' means total number of processes to run
    ngpus_per_node = torch.cuda.device_count()
    hparams.world_size = ngpus_per_node * hparams.world_size

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, hparams))
