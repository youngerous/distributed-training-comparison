import glob
import os

import torch

from config import load_config
from net import ResNet18
from trainer import Trainer
from utils import fix_seed


def main(hparams):
    fix_seed(hparams.seed)
    scaler = torch.cuda.amp.GradScaler() if hparams.amp else None
    model = ResNet18()

    # training phase
    trainer = Trainer(hparams, model, scaler)
    version = trainer.fit()

    # testing phase
    if hparams.contain_test:
        state_dict = torch.load(
            glob.glob(
                os.path.join(hparams.ckpt_path, f"version-{version}/best_model_*.pt")
            )[0]
        )
        trainer.test(state_dict)


if __name__ == "__main__":
    hparams = load_config()
    main(hparams)
