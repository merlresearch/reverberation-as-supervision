# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import argparse
from pathlib import Path

import loguru
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_train import RASTrainingModule
from utils.config import yaml_to_parser


def main(args):

    hparams = yaml_to_parser(args.config)
    hparams = hparams.parse_args([])
    exp_name = args.config.stem

    seed_everything(hparams.seed, workers=True)

    # some cuda configs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    logger = TensorBoardLogger(save_dir="exp", name="eras", version=exp_name)
    ckpt_dir = Path(logger.log_dir) / "checkpoints"

    model = RASTrainingModule(hparams, args.data_path)

    if (ckpt_dir / "last.ckpt").exists():
        # resume training from the latest checkpoint
        ckpt_path = ckpt_dir / "last.ckpt"
        skip_first_validation_loop = True
        loguru.logger.info(f"Resume training from {str(ckpt_path)}")
    elif getattr(hparams, "pretrained_model_path", None) is not None:
        ckpt_path = None
        skip_first_validation_loop = True
    else:
        print("Train from scratch")
        ckpt_path = None
        skip_first_validation_loop = False

    ckpt_callback = ModelCheckpoint(**hparams.model_checkpoint)
    callbacks = [LearningRateMonitor(logging_interval="epoch"), ckpt_callback]
    if hparams.early_stopping is not None:
        callbacks.append(EarlyStopping(**hparams.early_stopping))

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        deterministic=True,
        devices=-1,
        strategy="ddp",
        **hparams.trainer_conf,
    )

    # validation epoch before training for debugging
    if skip_first_validation_loop:
        loguru.logger.info("Skip validating before train when resuming training")
    else:
        loguru.logger.info("Validating before train")
        trainer.validate(model)

    # training
    loguru.logger.info("Finished initial validation")
    trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    args, other_options = parser.parse_known_args()
    main(args)
