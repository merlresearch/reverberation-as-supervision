# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from .paths.smswsj import get_smswsj_paths
from .paths.whamr import get_whamr_paths
from .stft_dataset import STFTDataset


def dataset_creator(hparams, data_path, partition):
    path_list = datapath_creator(hparams.dataset_name, data_path, partition, hparams.dataset_conf)

    # some setups specific for training stage
    is_training = partition == "tr"

    dataset = STFTDataset(
        path_list,
        is_training,
        hparams.stft_conf,
        **hparams.dataloading_conf,
    )
    return dataset


def datapath_creator(dataset_name, data_path, partition, dataset_conf):
    dataset_conf["partition"] = partition
    if dataset_name == "whamr":
        path_list = get_whamr_paths(data_path, **dataset_conf)
    elif dataset_name == "smswsj":
        path_list = get_smswsj_paths(data_path, **dataset_conf)
    else:
        raise ValueError("Dataset {} not currently supported.".format(dataset_name))
    return path_list
