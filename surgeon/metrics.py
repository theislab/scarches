import numpy as np
from scipy.stats import itemfreq, entropy
from sklearn.neighbors import NearestNeighbors

from surgeon.utils import remove_sparsity


def entropy_from_indices(indices):
    return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)))


def entropy_batch_mixing(adata, batch_key='batch',
                         n_neighbors=50, n_pools=50, n_samples_per_pool=100, subsample_frac=1.0):
    adata = remove_sparsity(adata)

    n_samples = adata.shape[0]
    keep_idx = np.random.choice(np.arange(n_samples), size=min(n_samples, int(subsample_frac * n_samples)),
                                replace=False)
    adata = adata[keep_idx, :]

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = neighbors.kneighbors(adata.X, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: adata.obs[batch_key].values[i])(indices)

    entropies = np.apply_along_axis(entropy_from_indices, axis=1, arr=batch_indices)

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])

    return score
