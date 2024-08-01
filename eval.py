# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import argparse
from pathlib import Path

import loguru
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_train import RASTrainingModule
from utils.config import yaml_to_parser


def main(args):

    config_path = args.ckpt_path.parent.parent / "hparams.yaml"
    hparams = yaml_to_parser(config_path)
    hparams = hparams.parse_args([])

    seed_everything(0, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    exp_name, name, save_dir = [config_path.parents[i].name for i in range(3)]
    logger = TensorBoardLogger(save_dir=save_dir, name=name, version=exp_name)

    trainer = Trainer(
        logger=logger,
        enable_progress_bar=True,
        deterministic=True,
        devices=1,
        num_nodes=1,
    )
    # testing
    loguru.logger.info("Begin Testing")
    model = RASTrainingModule.load_from_checkpoint(args.ckpt_path, hparams=hparams, data_path=args.data_path)
    trainer.test(model)
    loguru.logger.info("Testing complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    args, other_options = parser.parse_known_args()
    main(args)
