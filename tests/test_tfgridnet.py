# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import pytest
import torch

from nets.tfgridnetv2 import TFGridNetV2


@pytest.mark.parametrize("fft_size", [128, 256])
@pytest.mark.parametrize("n_srcs", [1, 2])
@pytest.mark.parametrize("n_imics", [1, 3, 6])
@pytest.mark.parametrize("n_layers", [1, 4, 6])
@pytest.mark.parametrize("lstm_hidden_units", [16])
@pytest.mark.parametrize("attn_n_head", [1, 4])
@pytest.mark.parametrize("attn_approx_qk_dim", [32])
@pytest.mark.parametrize("emb_dim", [16])
@pytest.mark.parametrize("emb_ks", [4])
@pytest.mark.parametrize("emb_hs", [1, 4])
@pytest.mark.parametrize("eps", [1.0e-5])
def test_tfgridnetv2_forward_backward(
    fft_size,
    n_srcs,
    n_imics,
    n_layers,
    lstm_hidden_units,
    attn_n_head,
    attn_approx_qk_dim,
    emb_dim,
    emb_ks,
    emb_hs,
    eps,
):

    model = TFGridNetV2(
        fft_size=fft_size,
        n_imics=n_imics,
        n_srcs=n_srcs,
        n_layers=n_layers,
        lstm_hidden_units=lstm_hidden_units,
        attn_n_head=attn_n_head,
        attn_approx_qk_dim=attn_approx_qk_dim,
        emb_dim=emb_dim,
        emb_ks=emb_ks,
        emb_hs=emb_hs,
        eps=eps,
    )
    model.train()

    n_freqs = fft_size // 2 + 1
    n_batch, n_frames = 2, 18
    real = torch.rand(n_batch, n_frames, n_freqs, n_imics)
    imag = torch.rand(n_batch, n_frames, n_freqs, n_imics)
    x = torch.complex(real, imag)

    output = model(x)
    assert output.shape == (n_batch, n_frames, n_freqs, n_srcs)
    sum(output).abs().mean().backward()
