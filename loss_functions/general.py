# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import itertools

import torch


def perms(num_sources):
    return list(itertools.permutations(range(num_sources)))


def pit_from_pairwise_loss(pw_dist_mat, reduction="sum", return_perm=False):
    n_batch, n_src, _ = pw_dist_mat.shape
    perm_mat = pw_dist_mat.new_tensor(perms(n_src), dtype=torch.long)  # [n_perm, n_src]
    ind = pw_dist_mat.new_tensor(range(n_src), dtype=torch.long).unsqueeze(0)  # [1, n_src]
    expanded_perm_dist_mat = pw_dist_mat[:, ind, perm_mat]  # [n_batch, n_perm, n_src]
    perm_dist_mat = torch.sum(expanded_perm_dist_mat, dim=2)  # [n_batch, n_perm]
    min_loss_perm, min_inds = torch.min(perm_dist_mat, dim=1)  # [n_batch]

    if return_perm:
        opt_perm, perm_mat = torch.broadcast_tensors(
            min_inds[:, None, None], perm_mat[None]
        )  # [n_batch, n_perm, n_src]
        opt_perm = torch.gather(perm_mat, 1, opt_perm[:, [0]])[:, 0]

    if reduction == "sum":
        if return_perm:
            return min_loss_perm, opt_perm
        else:
            return min_loss_perm
    else:
        # sometimes we want (batch x n_src) loss matrix
        min_inds = min_inds[:, None, None].tile(1, 1, n_src)
        min_loss = torch.gather(expanded_perm_dist_mat, 1, min_inds)
        if return_perm:
            return min_loss[:, 0, :], opt_perm  # (n_batch, n_src)
        else:
            return min_loss
