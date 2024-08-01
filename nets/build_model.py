# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from .tfgridnetv2 import TFGridNetV2


def build_model(model_name, model_conf):
    if model_name == "tfgridnetv2":
        model = TFGridNetV2(**model_conf)
    else:
        raise ValueError("Model type {} not currently supported.".format(model_name))

    return model
