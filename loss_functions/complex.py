# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from loss_functions.general import pit_from_pairwise_loss


def complex_l1(est, ref, permutation_search=False, reduction="mean_all", **kwargs):
    """L1 loss on complex stft representations.
    L1 loss on real part, imaginary part, and magnitude are computed.

    Parameters
    ----------
    est: torch.Tensor, (n_batch, n_frame, n_freq, n_src)
        Estimated (separated) signals.
    ref: torch.Tensor, (n_batch, n_frame, n_freq, n_src)
        Reference (ground-truth) signals.
    permutation_search: bool
        Whether to do permutation search between `est` and `ref`.
    reduction: str
        Argument for controlling the shape of returned tensor.
        `keep_batchdim` returns a tensor with (n_batch, ),
        `mean_all` returns a tensor with (),
        and else, just return a tensor as it is.

    Returns
    ----------
    loss: torch.Tensor,
        Loss value.
    """
    # sometimes ref is a tuple of a tensor and a length
    if isinstance(ref, tuple):
        ref = ref[0]
    normalizer = abs(ref).sum(dim=(1, 2))  # (batch, n_src)

    # expand dimension for getting pairwise loss
    if permutation_search:
        est = est.unsqueeze(-1)
        ref = ref.unsqueeze(-2)
        normalizer = normalizer.unsqueeze(-2)

    # loss
    real_l1 = abs(ref.real - est.real).sum(dim=(1, 2))
    imag_l1 = abs(ref.imag - est.imag).sum(dim=(1, 2))
    abs_l1 = abs(abs(ref) - abs(est)).sum(dim=(1, 2))
    loss = real_l1 + imag_l1 + abs_l1

    loss = loss / normalizer
    if permutation_search:
        # compute loss with brute-force optimal permutation search
        loss = pit_from_pairwise_loss(loss)
        loss = loss / ref.shape[-1]

    if reduction is None:
        return loss
    elif reduction == "keep_batchdim":
        if loss.ndim == 1:
            return loss
        else:
            return loss.mean(dim=-1)
    elif reduction == "mean_all":
        return loss.mean()
    else:
        raise RuntimeError(f"Choose proper reduction: {reduction}")
