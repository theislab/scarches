import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def partition(data, partitions, num_partitions):
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        indices = torch.nonzero((partitions == i), as_tuple=False).squeeze(1)
        res += [data[indices]]
    return res


def subsample_conditions(adata, cond_key, subsample):
    mask = np.full(adata.n_obs, False)
    cats = adata.obs[cond_key].unique()
    for cat in cats:
        cat_idx = np.where(adata.obs[cond_key] == cat)[0]
        size = int(len(cat_idx) * subsample)
        mask[np.random.choice(cat_idx, size, replace=False)] = True
    return adata[mask]