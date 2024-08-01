# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import os
from collections import OrderedDict


def get_whamr_paths(
    data_path,
    task="sep_noisy",
    partition="tt",
    sr=8000,
    use_min=True,
):
    POSSIBLE_TASKS = [
        "sep_clean_reverb",
        "denoise_both_reverb",
        "sep_reverb_all_srcs",
        "sep_noisy_reverb_all_srcs",
    ]

    assert task in POSSIBLE_TASKS, task
    S1_ANECHOIC_DIR = "s1_anechoic"
    S2_ANECHOIC_DIR = "s2_anechoic"
    S1_REVERB_DIR = "s1_reverb"
    S2_REVERB_DIR = "s2_reverb"
    S1_DRY = "s1"
    S2_DRY = "s2"
    BOTH_REVERB_DIR = "mix_both_reverb"
    CLEAN_REVERB_DIR = "mix_clean_reverb"

    if task == "sep_reverb_all_srcs":
        mix_dir = CLEAN_REVERB_DIR
        src_dir_list = [
            S1_ANECHOIC_DIR,
            S2_ANECHOIC_DIR,
            S1_REVERB_DIR,
            S2_REVERB_DIR,
            S1_DRY,
            S2_DRY,
        ]
    elif task == "sep_noisy_reverb_all_srcs":
        mix_dir = BOTH_REVERB_DIR
        src_dir_list = [
            S1_ANECHOIC_DIR,
            S2_ANECHOIC_DIR,
            S1_REVERB_DIR,
            S2_REVERB_DIR,
            S1_DRY,
            S2_DRY,
        ]

    else:
        raise ValueError("WHAMR task {} not available, please choose from {}".format(task, POSSIBLE_TASKS))
    return get_wsj2mix_paths(
        data_path,
        partition=partition,
        sr=sr,
        use_min=use_min,
        mix_dir=mix_dir,
        src_dir_list=src_dir_list,
    )


def get_wsj2mix_paths(
    data_path,
    partition="tt",
    sr=8000,
    use_min=True,
    mix_dir=None,
    src_dir_list=None,
):
    if mix_dir is None:
        mix_dir = "mix"
    if src_dir_list is None:
        src_dir_list = ["s1", "s2"]
    if sr == 8000:
        wav_dir = "wav8k"
    elif sr == 16000:
        wav_dir = "wav16k"
    else:
        raise ValueError("set wsj0-2mix dataset sample rate to either 8kHz or 16kHz")

    if use_min:
        max_or_min_dir = "min"
    else:
        max_or_min_dir = "max"

    root_path = os.path.join(data_path, wav_dir, max_or_min_dir, partition)
    filelist = [f for f in os.listdir(os.path.join(root_path, src_dir_list[0])) if f.endswith(".wav")]
    if partition == "tr":
        assert len(filelist) == 20000, "Expected 20000 files in training set"
    elif partition == "cv":
        assert len(filelist) == 5000, "Expected 5000 files in validation set"
    elif partition == "tt":
        assert len(filelist) == 3000, "Expected 3000 files in testing set"
    path_list = get_path_list(filelist, root_path, mix_dir, src_dir_list)

    return path_list


def get_path_dict(filename, root_dir, mix_dir, src_dir_list):
    id_ = os.path.splitext(filename)[0]
    path_dict = {
        "id": id_,
        "srcs": OrderedDict({s: os.path.join(root_dir, s, filename) for s in src_dir_list}),
        "mix": os.path.join(root_dir, mix_dir, filename),
    }
    return path_dict


def get_path_list(filelist, root_dir, mix_dir, src_dir_list):
    # if type(filelist) == type([]):
    if isinstance(filelist, list):
        path_list = [get_path_dict(f, root_dir, mix_dir, src_dir_list) for f in filelist]
    if isinstance(filelist, dict):
        path_list = {}
        for key in filelist.keys():
            path_list[key] = [get_path_dict(f, root_dir, mix_dir, src_dir_list) for f in filelist[key]]
    return path_list
