<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Enhanced Reverberation as Supervision for Unsupervised Speech Separation

This repository includes source code for training and evaluating the enhanced reverberation as supervision (ERAS), proposed in the following Interspeech 2024 paper:

```
@InProceedings{Saijo2024_eras,
  author    =  {Saijo, Kohei and Wichern, Gordon and Germain, Fran\c{c}ois G. and Pan, Zexu and {Le Roux}, Jonathan},
  title     =  {Enhanced Reverberation as Supervision for Unsupervised Speech Separation},
  booktitle =  {Proc. Annual Conference of International Speech Communication Association (INTERSPEECH)},
  year      =  2024,
  month     =  sep
}
```

## Table of contents

1. [Installation](#installation)
2. [How to run](#how-to-run)
3. [Contributing](#contributing)
4. [Copyright and license](#copyright-and-license)

## Installation

Clone this repo and create the anaconda environment

```sh
git clone https://github.com/merlresearch/reverberation-as-supervision
cd reverberation-as-supervision && conda env create -f environment.yaml
```

## How to run

This repository supports training on two datasets used in the paper, **WHAMR!** and **SMS-WSJ**.
Example training configuration files are under `./configs/*dataset-name*`.

Before starting training, run the following command:

```sh
conda activate ras
```

The main script for training is in `train.py`, which can be run by

```sh
python train.py --config /path/to/config --data_path /path/to/data
```

Here, `/path/to/data` is the directory containing `wav8k` and `wav16k` directories for WHAMR! and that containing `sms_wsj.json` for SMS-WSJ.

As demonstrated in the paper, a best-performing model is obtained by two-stage training.
One can first pre-train a model and then fine-tune it as follows (example commands on WHAMR!).

```sh
# Train a model with ISMS-loss weight of 0.3 for 20 epochs.
python train.py --config ./configs/whamr/eras_whamr_isms0.3_icc0.0.yaml --data_path /path/to/whamr

# Fine-tune the pre-trained model without the ISMS loss and with the ICC loss for 80 epochs.
# Note that the pre-trained model's path has to be specified in the yaml file.
python train.py --config ./configs/whamr/eras_whamr_isms0.0_icc0.1.yaml --data_path /path/to/whamr
```

The checkpoints and tensorboard logs are saved under `exp/eras/*config-name*` directory.
After finishing the training, separation performance can be evaluated using `eval.py`:

```sh
python eval.py --ckpt_path /path/to/.ckpt-file --data_path /path/to/data
```

The evaluation scores are logged in the tensorboard.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## Copyright and license

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

The following file:

- `nets/tfgridnetv2.py`

was adapted from https://github.com/espnet/espnet (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md))

```
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) 2023 ESPnet Developers

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: Apache-2.0
```
