# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from functools import partial
from typing import Dict

import torch
from fast_bss_eval import si_sdr

from loss_functions.complex import complex_l1
from utils.audio_utils import istft_4dim, stft_3dim
from utils.forward_convolutive_prediction import forward_convolutive_prediction

loss_funcs = {"complex_l1": complex_l1}


class RASLoss(object):
    """
    Reverberation as Supervision (RAS) loss [1, 2].
    This class also supports over-determined conditions (UNSSOR) [3].


    Parameters
    ----------
        loss_func: str
            Name of loss function. Only "complex_l1" is supported now.
        future_taps: int
            The number of future taps in FCP.
        past_taps: int
            The number of past taps in FCP.
        ref_channel_loss_weight: float
            Loss weight on the reference channel (input channel of separation model).
            In ERAS, 0.0 is recommended.
        isms_loss_weight: float
            Loss weight of intra-source magnitude scattering (ISMS) loss.
        icc_loss_weight: float
            Loss weight of inter-channel consistency (ICC) loss.
        ref_channel: int
            Index of reference channel.
        nonref_channel: int
            Index of non-reference channel. Must be different from `ref_channel`.
        unsupervised: bool
            Either of unsupervised or supervised.
        supervised_loss_type: str
            How to compute supervised loss when doing (semi-)supervised learning.
            before_filtering, after_filtering_ref_channel, or after_filtering_nonref_channel.
            Supervised loss is computed when doing unsupervised learning on the validation set
            to monitor the performance.
        stft_conf: dict
            Dictionary containing STFT parameters.

    References
    ----------
    [1]: Rohith Aralikatti, Christoph Boeddeker, Gordon Wichern, Aswin Subramanian, and Jonathan Le Roux,
        "Reverberation as Supervision for Speech Separation," Proc. ICASSP, 2023.

    [2]: Kohei Saijo, Gordon Wichern, François G. Germain, Zexu Pan, and Jonathan Le Roux,
        "Enhanced Reverberation as Supervision for Unsupervised Speech Separation," Proc. Interspeech, 2024.

    [3]: Zhong-Qiu Wang and Shinji Watanabe, "UNSSOR: unsupervised neural speech separation
        by leveraging over-determined training mixtures," Proc. NeurIPS, 2023.
    """

    def __init__(
        self,
        loss_func: str = "complex_l1",
        future_taps: int = 1,
        past_taps: int = 19,
        ref_channel_loss_weight: float = 0.0,
        isms_loss_weight: float = 0.0,
        icc_loss_weight: float = 0.0,
        ref_channel: int = 0,
        nonref_channel: int = 1,
        unsupervised: bool = True,
        supervised_loss_type: str = "after_filtering_ref_channel",
        stft_conf: Dict = None,
    ):
        assert loss_func in loss_funcs, loss_func

        self.loss_func = loss_funcs[loss_func]
        self.ref_channel = ref_channel
        self.nonref_channel = nonref_channel
        self.unsupervised = unsupervised
        self.supervised_loss_type = supervised_loss_type
        self.stft_conf = stft_conf

        # loss weights
        self.ref_channel_loss_weight = ref_channel_loss_weight
        self.isms_loss_weight = isms_loss_weight
        self.icc_loss_weight = icc_loss_weight

        if self.icc_loss_weight > 0:
            self.icc_loss = partial(inter_channel_consistency_loss, loss_func=self.loss_func)

        # define FCP used in the forward path
        self.filtering_func = partial(
            forward_convolutive_prediction,
            past_taps=past_taps,
            future_taps=future_taps,
        )

    def __call__(self, nn_outputs: torch.Tensor, targets: Dict, **kwargs):
        # mixure signal used as supervision in RAS or UNSSOR
        mix = targets["y_mix_stft"][0]

        n_batch, n_channels = mix.shape[0], mix.shape[-1]
        unsup_loss = 0.0

        est_src_filtered = self.filtering_func(nn_outputs, mix)  # (n_batch, n_frames, n_freqs, n_chan, n_src)
        est_mix = est_src_filtered.sum(dim=-1)

        # compute RAS loss
        ras_loss = self.loss_func(
            est_mix,
            mix,
            permutation_search=False,
            reduction=None,
        )  # loss: (n_batch, n_chan)

        assert ras_loss.shape == (
            n_batch,
            n_channels,
        ), f"loss must be (batch x channel) but {ras_loss.shape}"

        # compute ISMS loss
        isms_loss = intra_source_magnitude_scattering_loss(est_src_filtered, mix, reduction=None)
        assert ras_loss.shape == isms_loss.shape, "(loss must be (batch x channel)"
        f"but {ras_loss.shape} and {isms_loss.shape})"
        unsup_loss = ras_loss + self.isms_loss_weight * isms_loss

        # weighting loss on reference channel
        unsup_loss[:, self.ref_channel] *= self.ref_channel_loss_weight
        unsup_loss = unsup_loss.sum(dim=-1)

        # inter-source consistency loss
        training = unsup_loss.requires_grad

        # Currently only one of L or R is loaded during validation
        # and we cannot compute inter-source loss on dev set
        if self.icc_loss_weight > 0 and training:
            icc_loss = self.icc_loss(est_src_filtered)
            unsup_loss += self.icc_loss_weight * icc_loss

        # in supervised case or on validation set, we compute the supervised loss
        if training and self.unsupervised:
            loss = unsup_loss.mean()
            sup_loss = torch.zeros_like(unsup_loss)  # for logging
        else:
            # pick some channels for SI-SDR evaluation
            est_src_ref_channel = est_src_filtered[..., self.ref_channel, :]
            est_src_nonref_channel = est_src_filtered[..., self.nonref_channel, :]

            targets = targets["y_srcs"]

            if self.supervised_loss_type == "before_filtering":
                est = nn_outputs
                tgt = targets["reverb"][0][..., self.ref_channel, :]
            elif self.supervised_loss_type == "after_filtering_ref_channel":
                est = est_src_ref_channel
                tgt = targets["reverb"][0][..., self.ref_channel, :]
            elif self.supervised_loss_type == "after_filtering_nonref_channel":
                est = est_src_nonref_channel
                tgt = targets["reverb"][0][..., self.nonref_channel, :]

            tgt = stft_3dim(tgt, **self.stft_conf)

            sup_loss = self.loss_func(
                est[:, : tgt.shape[1]],
                tgt,
                permutation_search=True,
                reduction="keep_batchdim",
            )

            uloss = unsup_loss.mean()
            sloss = sup_loss.mean()
            loss = uloss if self.unsupervised else sloss

        # metrics for logging and backprop
        metrics = {
            "loss": loss,
            "ras_loss": ras_loss.mean(),
            "isms_loss": isms_loss.mean(),
            "unsup_loss": unsup_loss.mean(),
            "sup_loss": sup_loss.mean(),
        }

        # compute SI-SDR for logging
        if not training:
            # istft to get time-domain signals when using tf-domain loss
            est_src_ref_channel = istft_4dim(est_src_ref_channel, **self.stft_conf).transpose(-1, -2)
            est_src_nonref_channel = istft_4dim(est_src_nonref_channel, **self.stft_conf).transpose(-1, -2)

            ref = targets["reverb"][0].transpose(-1, 1)
            m = min(est_src_ref_channel.shape[-1], ref.shape[-1])
            metrics["sisnr_ref_channel"] = si_sdr(ref[..., self.ref_channel, :m], est_src_ref_channel[..., :m]).mean()
            metrics["sisnr_nonref_channel"] = si_sdr(
                ref[..., self.nonref_channel, :m], est_src_nonref_channel[..., :m]
            ).mean()

        return metrics


def intra_source_magnitude_scattering_loss(est, mix, reduction=None, eps=1e-8):
    """Intra source magnitude scattering loss (ISMS) proposed in [3] (Eq.10).

    Parameters
    ----------
    est: torch.Tensor, (..., n_frame, n_freq, n_chan, n_src)
        Separation estimates AFTER applying FCP
    mix: torch.Tensor, (..., n_frame, n_freq, n_chan)
        Mixture observed at the microphone
    reduction: str
        How to aggregate the loss values.
        Must be chosen from {None, "keep_batchdim", "mean_all"}

    Returns
    ----------
    loss: torch.Tensor
        ISMS loss value. Shape depends on the specified reduction.
    """
    mix_logmag = torch.log(abs(mix) + eps)
    est_logmag = torch.log(abs(est) + eps)

    mix_var = torch.var(mix_logmag, dim=-2).sum(dim=-2)
    est_var = torch.var(est_logmag, dim=-3).mean(dim=-1).sum(dim=-2)

    loss = est_var / mix_var  # (batch, n_chan)

    if reduction is None:
        return loss
    elif reduction == "keep_batchdim":
        return loss.mean(dim=-1)
    elif reduction == "mean_all":
        return loss.mean()
    else:
        raise RuntimeError(f"Choose proper reduction: {reduction}")


def inter_channel_consistency_loss(est: torch.Tensor, loss_func: callable):
    """Inter-channel consistency (ICC) loss proposed in [2].

    In ERAS, each component in the mini-batch `est` is:
        est[0]: separated signals from mix1 at 1st channel and mapped to [1st, 2nd] channels
        est[1]: separated signals from mix1 at 2nd channel and mapped to [2nd, 1st] channels
        est[2]: separated signals from mix2 at 1st channel and mapped to [1st, 2nd] channels
        est[3]: separated signals from mix2 at 2nd channel and mapped to [2nd, 1st] channels
        ...
        est[B-2]: separated signals from mix{B/2} at 1st channel and mapped to [1st, 2nd] channels
        est[B-1]: separated signals from mix{B/2} at 2nd channel and mapped to [2nd, 1st] channels

    Note that mix{1,...,B/2} are different (totally unrelated) mixtures in the same mini-batch.

    The ICC loss uses signals mapped to the reference microphone as the pseudo-targets of
    those mapped to the non-reference microphone, e.g.,
        - est[0][0] is the pseudo-target of est[1][1]
        - est[1][0] is the pseudo-target of est[0][1]


    Parameters
    ----------
    est: torch.Tensor
        Separation outputs after FCP (n_batch, n_frames, n_freqs, n_chan, n_src)
    loss_func: callable
        Loss function

    Returns
    ----------
    loss: torch.Tensor,
        The inter-channel consistency loss value.
    """
    n_batch, n_chan = est.shape[0], est.shape[-2]
    assert n_chan == 2

    # the first channel in `n_chan` dimension is the one mapped to the reference microphone
    # while the second is mapped to the non-reference microphone.
    ref = est[..., 0, :].clone().detach()  # signals mapped to reference microphone
    est = est[..., 1, :]  # signals mapped to non-reference microphone

    # change the batch order of `est` to [1, 0, 3, 2, ...]
    # to align the batch order of ref and est
    p_tmp = torch.arange(n_batch)
    p = p_tmp.clone()
    p[0::2], p[1::2] = p_tmp[1::2], p_tmp[0::2]  # p: [1, 0, 3, 2, ...]
    est = est[p]

    # compute loss
    loss = loss_func(
        est,
        ref,
        permutation_search=True,
        reduction=None,
    )
    return loss.mean()
