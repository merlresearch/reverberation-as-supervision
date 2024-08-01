# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import json

PARTITION_MAP = {
    "tr": {"name": "train_si284", "num_data": 33561},
    "cv": {"name": "cv_dev93", "num_data": 982},
    "tt": {"name": "test_eval92", "num_data": 1332},
}


def get_smswsj_paths(
    data_path,
    partition="tt",
    **kwargs,
):
    # rename partition
    stage = PARTITION_MAP[partition]["name"]

    # load metadata of smswsj
    with open(data_path / "sms_wsj.json") as f:
        mixinfo = json.load(f)
    mixinfo = mixinfo["datasets"][stage]

    pathlist = []
    for key, info in mixinfo.items():
        # information of a sample
        tmp = {
            "id": key,
            "mix": info["audio_path"]["observation"],
            "srcs": {
                "reverb1": info["audio_path"]["speech_image"][0],
                "reverb2": info["audio_path"]["speech_image"][1],
                "anechoic1": info["audio_path"]["speech_reverberation_early"][0],
                "anechoic2": info["audio_path"]["speech_reverberation_early"][1],
                "dry1": info["audio_path"]["speech_source"][0],
                "dry2": info["audio_path"]["speech_source"][1],
            },
            "num_samples": info["num_samples"]["speech_source"],
            "offset": info["offset"],
        }
        # add to pathlist
        pathlist.append(tmp)
    # check number of data
    assert len(pathlist) == PARTITION_MAP[partition]["num_data"]

    if partition == "cv":
        # sort by length for efficient validation
        # decreasing padded-zeros would lead to acccurate validation
        pathlist = sorted(pathlist, key=lambda x: x["num_samples"])

    return pathlist
