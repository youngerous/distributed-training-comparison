import argparse
import glob

import torch
import torch.nn as nn

from config import load_config
from model.net import Model  # TODO: Set your model
from trainer import Trainer
from utils import ResultWriter, fix_seed


def main(hparams):
    fix_seed(hparams.seed)

    resultwriter = ResultWriter(hparams.result_path)
    scaler = torch.cuda.amp.GradScaler() if hparams.amp else None
    model = Model()  #

    # training phase
    trainer = Trainer(hparams, model, scaler, resultwriter)
    best_result = trainer.fit()

    # testing phase
    if hparams.test:
        version = best_result["version"]
        state_dict = torch.load(
            glob.glob(f"checkpoints/version-{version}/best_model_*.pt")[0]
        )
        test_result = trainer.test(state_dict)

    # save result
    best_result.update(test_result)
    resultwriter.update(hparams, **best_result)


if __name__ == "__main__":
    hparams = load_config()
    main(hparams)
