# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import pytest
import torch

from loss_functions.complex import complex_l1
from loss_functions.ras_loss import inter_channel_consistency_loss as icc_loss
from loss_functions.ras_loss import intra_source_magnitude_scattering_loss as isms_loss


@pytest.mark.parametrize("permutation_search", [True, False])
@pytest.mark.parametrize("reduction", ["mean_all", "keep_batchdim", None])
def test_complex_l1(permutation_search, reduction):
    n_batch, n_frame, n_freq, n_src = 1, 50, 65, 2
    est = torch.randn((n_batch, n_frame, n_freq, n_src), dtype=torch.complex64)
    ref = torch.randn((n_batch, n_frame, n_freq, n_src), dtype=torch.complex64)

    loss = complex_l1(est, ref, permutation_search=permutation_search, reduction=reduction)

    if reduction is None:
        if permutation_search:
            assert loss.shape == (n_batch,)
        else:
            assert loss.shape == (n_batch, n_src)
    elif reduction == "keep_batchdim":
        assert loss.shape == (n_batch,)
    else:
        assert loss.shape == torch.Size([])


@pytest.mark.parametrize("reduction", ["mean_all", "keep_batchdim", None])
def test_isms_loss(reduction):
    n_batch, n_frame, n_freq, n_chan, n_src = 1, 50, 65, 2, 2
    est = torch.randn((n_batch, n_frame, n_freq, n_chan, n_src), dtype=torch.complex64)
    mix = torch.randn((n_batch, n_frame, n_freq, n_chan), dtype=torch.complex64)

    loss = isms_loss(est, mix, reduction=reduction)

    if reduction is None:
        assert loss.shape == (n_batch, n_chan)
    elif reduction == "keep_batchdim":
        assert loss.shape == (n_batch,)
    else:
        assert loss.shape == torch.Size([])


def test_icc_loss():
    n_batch, n_frame, n_freq, n_chan, n_src = 1, 50, 65, 2, 2
    est_ch1 = torch.randn((n_batch, n_frame, n_freq, n_chan, n_src), dtype=torch.complex64)
    est_ch2 = torch.stack((est_ch1[..., 1, :], est_ch1[..., 0, :]), dim=-2)
    est = torch.cat((est_ch1, est_ch2), dim=0)

    loss = icc_loss(est, complex_l1)

    assert loss.shape == torch.Size([])
    assert loss == 0.0


@pytest.mark.parametrize("n_chan", [1, 3, 6])
def test_icc_loss_invalid_type(n_chan):
    # n_chan != 2 raises the assertion error

    n_batch, n_frame, n_freq, n_src = 2, 50, 65, 2
    est = torch.randn((n_batch, n_frame, n_freq, n_chan, n_src), dtype=torch.complex64)

    with pytest.raises(AssertionError):
        icc_loss(est, complex_l1)
