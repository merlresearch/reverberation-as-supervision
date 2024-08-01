# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import torch


def forward_convolutive_prediction(
    est: torch.Tensor,
    ref: torch.Tensor,
    past_taps: int,
    future_taps: int,
    eps: float = 1e-8,
):
    """Forward convolutive prediction (FCP) proposed in [1].

    Parameters
    ----------
    est: torch.Tensor, (n_batch, n_frame, n_freq, n_src)
        Signals to which the FCP is applied.
    ref: torch.Tensor, (n_batch, n_frame, n_freq, n_chan)
        Reference signals of the FCP.
    past_taps: int
        The number of the past taps in the FCP.
    future_taps: int
        The number of the future taps in the FCP.
    eps: float
        Stabilizer for matrix inverse.

    Returns
    ----------
    output: torch.Tensor, (n_batch, n_frame, n_freq, n_chan, n_src)
        Signals after applying the FCP.

    References
    ----------
    [1]: Z.-Q Wang, G. Wichern, and J. Le Roux,
        "Convolutive Prediction for Monaural Speech Dereverberation and Noisy-Reverberant Speaker Separation,"
        IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3476-3490, 2021.
    """

    n_chan = ref.shape[-1]
    # stack past and future frames first
    est_tilde = stack_past_and_future_taps(est, past_taps, future_taps)

    # compute weighting term lambda
    weight = compute_fcp_weight(ref)
    weight = weight[..., None, None]

    # compute FCP filter
    # get auto-covariance matrix
    auto_cov = torch.einsum("...tafn, ...tbfn -> ...tfnab", est_tilde, est_tilde.conj())
    auto_cov = (auto_cov / weight).sum(dim=-5)
    auto_cov = auto_cov.unsqueeze(-4).tile(1, 1, n_chan, 1, 1, 1)

    # get cross-covariance matrix
    cross_cov = torch.einsum("...tafn, ...tfc -> ...tfcna", est_tilde, ref.conj())
    cross_cov = (cross_cov / weight).sum(dim=-5)

    # compute relative RIR: (batch, past+1+future, freq, n_chan, n_src)
    rir = torch.linalg.solve(auto_cov + eps, cross_cov)

    # filter the estimate: (batch, frame, freq, n_chan, n_src)
    output = torch.einsum("...fcna, ...tafn -> ...tfcn", rir.conj(), est_tilde)

    return output


def stack_past_and_future_taps(
    input,
    past_tap,
    future_tap,
):
    """Function to stack the past and future frames of the input.

    Parameters
    ----------
    input: torch.Tensor, (n_batch, n_frame, n_freq, n_src)
        Signals to which the FCP is applied.
    past_taps: int
        Number of past taps in the FCP.
    future_taps: int
        Number of future taps in the FCP.

    Returns
    ----------
    output: torch.Tensor, (n_batch, n_frame, past_tap+1+future_tap, n_freq, n_src)
        A tensor that stacks the past and future frames of the input
    """

    T = input.shape[-3]
    indices = torch.arange(0, past_tap + future_tap + 1).view(1, 1, past_tap + future_tap + 1).to(input.device)
    padded_indices = torch.arange(T).view(1, T, 1).to(input.device) + indices
    output = torch.nn.functional.pad(input, (0, 0, 0, 0, past_tap, future_tap))
    output = output[:, padded_indices]
    return output[:, 0]


def compute_fcp_weight(input, eps=1e-4):
    """Function to calculate the weighting term (lambda) in the FCP [1].

    Parameters
    ----------
    input: torch.Tensor, (n_batch, n_frame, n_freq, n_chan)
        Signal to compute the FCP weight.

    Returns
    ----------
    weight: torch.Tensor, (n_batch, n_frame, n_freq, n_chan)
        FCP weighting term with the same shape as the input.
    """
    power = (abs(input) ** 2).mean(dim=-1)
    max_power = torch.max(power)
    weight = power + eps * max_power
    return weight.unsqueeze(-1)
