# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import pytest
import torch

from utils.forward_convolutive_prediction import forward_convolutive_prediction as fcp
from utils.forward_convolutive_prediction import stack_past_and_future_taps


@pytest.mark.parametrize("past_tap", [1, 5, 19])
@pytest.mark.parametrize("future_tap", [0, 1])
def test_stack_past_and_future_taps_forward(past_tap, future_tap):
    n_batch, n_frame, n_freq, n_src = 1, 50, 65, 2
    input = torch.randn((n_batch, n_frame, n_freq, n_src), dtype=torch.complex64)

    padded = stack_past_and_future_taps(input, past_tap, future_tap)
    assert padded.shape == (n_batch, n_frame, past_tap + future_tap + 1, n_freq, n_src)


@pytest.mark.parametrize("past_tap", [1, 5, 19])
@pytest.mark.parametrize("future_tap", [0, 1])
@pytest.mark.parametrize("n_chan", [1, 2, 6])
@pytest.mark.parametrize("n_src", [1, 2])
def test_fcp_forward(past_tap, future_tap, n_chan, n_src):
    n_batch, n_frame, n_freq = 1, 50, 65
    est = torch.randn((n_batch, n_frame, n_freq, n_src), dtype=torch.complex64)
    mix = torch.randn((n_batch, n_frame, n_freq, n_chan), dtype=torch.complex64)

    est_filtered = fcp(est, mix, past_tap, future_tap)
    assert est_filtered.shape == (n_batch, n_frame, n_freq, n_chan, n_src)
