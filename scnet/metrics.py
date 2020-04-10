import numpy as np
from scipy.stats import itemfreq, entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from scnet.utils import remove_sparsity


def __entropy_from_indices(indices):
    return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)))


def entropy_batch_mixing(adata, label_key='batch',
                         n_neighbors=50, n_pools=50, n_samples_per_pool=100, subsample_frac=1.0):
    adata = remove_sparsity(adata)

    n_samples = adata.shape[0]
    keep_idx = np.random.choice(np.arange(n_samples), size=min(n_samples, int(subsample_frac * n_samples)),
                                replace=False)
    adata = adata[keep_idx, :]

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = neighbors.kneighbors(adata.X, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: adata.obs[label_key].values[i])(indices)

    entropies = np.apply_along_axis(__entropy_from_indices, axis=1, arr=batch_indices)

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])

    return score


def asw(adata, label_key):
    adata = remove_sparsity(adata)

    labels = adata.obs[label_key].values

    labels_encoded = LabelEncoder().fit_transform(labels)

    return silhouette_score(adata.X, labels_encoded)


def ari(adata, label_key):
    adata = remove_sparsity(adata)

    n_labels = len(adata.obs[label_key].unique().tolist())
    kmeans = KMeans(n_labels, n_init=200)

    labels_pred = kmeans.fit_predict(adata.X)
    labels = adata.obs[label_key].values
    labels_encoded = LabelEncoder().fit_transform(labels)

    return adjusted_rand_score(labels_encoded, labels_pred)


def nmi(adata, label_key):
    adata = remove_sparsity(adata)

    n_labels = len(adata.obs[label_key].unique().tolist())
    kmeans = KMeans(n_labels, n_init=200)

    labels_pred = kmeans.fit_predict(adata.X)
    labels = adata.obs[label_key].values
    labels_encoded = LabelEncoder().fit_transform(labels)

    return normalized_mutual_info_score(labels_encoded, labels_pred)


def knn_purity(adata, label_key, n_neighbors=30):
    adata = remove_sparsity(adata)
    labels = LabelEncoder().fit_transform(adata.obs[label_key].to_numpy())

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = nbrs.kneighbors(adata.X, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: labels[i])(indices)

    # pre cell purity scores
    scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
    res = [
        np.mean(scores[labels == i]) for i in np.unique(labels)
    ]  # per cell-type purity

    return np.mean(res)
