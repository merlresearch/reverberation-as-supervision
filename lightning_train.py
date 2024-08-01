# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import os
import random
from argparse import Namespace

import fast_bss_eval
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from loguru import logger
from pesq import pesq
from pytorch_lightning.utilities import rank_zero_only
from torch import optim

from datasets.dataset_creator import dataset_creator
from loss_functions.ras_loss import RASLoss
from nets.build_model import build_model
from utils.audio_utils import istft_4dim
from utils.collate import collate_seq, collate_seq_eras


class RASTrainingModule(pl.LightningModule):
    def __init__(self, hparams, data_path):
        super().__init__()

        if not isinstance(hparams, Namespace):
            hparams = Namespace(hparams.model_name, **hparams.model_conf)
        self.data_path = data_path

        self.save_hyperparameters(hparams)
        self.model = build_model(hparams.model_name, hparams.model_conf)
        self.loss = RASLoss(**hparams.eras_loss_conf)

        self.current_step = 0  # used for learning-rate warmup

    def load_pretrained_weight(self):
        if self.hparams.pretrained_model_path is not None:
            if torch.cuda.is_available():
                state_dict = torch.load(self.hparams.pretrained_model_path)
            else:
                state_dict = torch.load(self.hparams.pretrained_model_path, map_location=torch.device("cpu"))
            try:
                state_dict = state_dict["state_dict"]
            except KeyError:
                print("No key named state_dict. Directly loading from model.")
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            logger.info("Loaded weights from " + self.hparams.pretrained_model_path)

    def on_batch_end(self):
        # learning rate warmup
        self.warmup_lr()

    @rank_zero_only
    def _symlink_logger(self):
        # Keep track of which log file goes with which tensorboard log folder
        tensorboard_log_dir = self.trainer.logger.log_dir
        logger.info(f"Tensorboard logs: {tensorboard_log_dir}")
        if os.path.exists(self.hparams.log_file):
            _, log_name = os.path.split(self.hparams.log_file)
            new_log_path = os.path.join(tensorboard_log_dir, log_name)

            # when resuming training, symlink already exists
            if not os.path.islink(new_log_path):
                os.symlink(os.path.abspath(self.hparams.log_file), new_log_path)

    def warmup_lr(self):
        # get initial learning rate at step 0
        if self.current_step == 0:
            for param_group in self.optimizers().optimizer.param_groups:
                self.peak_lr = param_group["lr"]

        self.current_step += 1
        if getattr(self.hparams, "warmup_steps", 0) >= self.current_step:
            for param_group in self.optimizers().optimizer.param_groups:
                param_group["lr"] = self.peak_lr * self.current_step / self.hparams.warmup_steps

    def on_train_start(self):
        self._symlink_logger()
        self.load_pretrained_weight()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        input_features, target_dict = batch
        input_features, lens = input_features

        y = self.forward(input_features)  # (batch, frame, freq) -> (batch, frame, freq, num_src)

        loss = self.loss(y, target_dict, device=self.device, training=self.model.training)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        loss_for_logging = {}
        for k, v in loss.items():
            loss_for_logging[f"train/{k}"] = v
        self.log_dict(loss_for_logging, on_step=True, on_epoch=True, sync_dist=True)

        self.on_batch_end()
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        loss_for_logging = {}
        for k, v in loss.items():
            loss_for_logging[f"val/{k}"] = v
        self.log_dict(loss_for_logging, on_epoch=True, sync_dist=True)

        return loss["loss"]

    def test_step(self, batch, batch_idx):
        input_features, target_dict = batch
        input_features, lens = input_features
        sample_rate = self.hparams.dataloading_conf["sr"]

        est = self.forward(input_features)  # (batch, frame, freq) -> (batch, frame, freq, num_src)

        # apply FCP
        est = self.loss.filtering_func(est, input_features)
        est = est[..., self.loss.ref_channel, :]

        # TF-domain -> time-domain by iSTFT
        est = istft_4dim(est, **self.loss.stft_conf)[0].T

        # reference signal
        ref = target_dict["y_srcs"]["reverb"][0][0, ..., self.loss.ref_channel, :].T

        # compute metrics
        m = min(ref.shape[-1], est.shape[-1])
        sisnr, perm = fast_bss_eval.si_sdr(ref[..., :m], est[..., :m], return_perm=True)
        sisnr = sisnr.mean().cpu().numpy()
        perm = perm.cpu().numpy()

        sdr = fast_bss_eval.sdr(ref, est).mean().cpu().numpy()

        ref, est = ref.cpu().numpy(), est.cpu().numpy()
        pesq_score = 0.0
        for i, p in enumerate(perm):
            pesq_score += pesq(sample_rate, ref[i], est[p], mode="nb")
        pesq_score /= i + 1

        result = {
            "test/sisnr": float(sisnr),
            "test/sdr": float(sdr),
            "test/pesq": float(pesq_score),
        }
        self.log_dict(result, on_epoch=True)

        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), **self.hparams.optimizer_conf)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.hparams.scheduler_conf)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    def _init_fn(self, worker_id):
        random.seed(self.hparams.seed + worker_id)
        np.random.seed(self.hparams.seed + worker_id)
        torch.manual_seed(self.hparams.seed + worker_id)

    def _get_data_loader(self, partition):
        shuffle = self.hparams.shuffle if partition == "tr" else None
        if partition == "tr":
            batch_size = self.hparams.batch_size
        elif partition == "cv":
            batch_size = self.hparams.val_batch_size
        else:
            batch_size = 1

        d = dataset_creator(self.hparams, self.data_path, partition)

        if getattr(d, "running_eras", False):
            collate_fn = collate_seq_eras
        else:
            collate_fn = collate_seq

        return data.DataLoader(
            d,
            batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            worker_init_fn=self._init_fn,
        )

    def train_dataloader(self):
        return self._get_data_loader("tr")

    def val_dataloader(self):
        return self._get_data_loader("cv")

    def test_dataloader(self):
        return self._get_data_loader("tt")
