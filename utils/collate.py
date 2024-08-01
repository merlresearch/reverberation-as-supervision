# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import collections.abc as container_abcs

import torch
import torch.nn.utils.rnn as rnn


def collate_seq(batch):
    elem = batch[0]
    elem_type = type(elem)
    if elem_type.__name__ == "ndarray":
        lengths = torch.tensor([len(b) for b in batch])
        pad_features = rnn.pad_sequence([torch.tensor(b, dtype=torch.float32) for b in batch], batch_first=True)
        return pad_features, lengths

    elif isinstance(elem, torch.Tensor):
        lengths = torch.tensor([len(b) for b in batch])
        pad_features = rnn.pad_sequence([b for b in batch], batch_first=True)
        return pad_features, lengths

    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_seq(samples) for samples in transposed]

    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_seq([d[key] for d in batch]) for key in elem}

    else:
        # for other stuff just return it and do not collate
        return [b for b in batch]


def collate_seq_eras(batch):
    elem = batch[0]
    elem_type = type(elem)
    if elem_type.__name__ == "ndarray":
        raise RuntimeError("Input must be torch Tensor or Dict")

    elif isinstance(elem, torch.Tensor):
        lengths = torch.tensor([len(b) for b in batch])
        pad_features = rnn.pad_sequence([b for b in batch], batch_first=True)
        pad_features = pad_features.movedim(-1, 1)
        pad_features = pad_features.reshape((-1,) + pad_features.shape[2:])
        return pad_features, lengths

    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_seq_eras(samples) for samples in transposed]

    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_seq_eras([d[key] for d in batch]) for key in elem}

    else:
        # for other stuff just return it and do not collate
        return [b for b in batch]
