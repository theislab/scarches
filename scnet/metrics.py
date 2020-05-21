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
                         n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    """Computes Entory of Batch mixing metric for ``adata`` given the batch column name.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        n_neighbors: int
            Number of nearest neighbors.
        n_pools: int
            Number of EBM computation which will be averaged.
        n_samples_per_pool: int
            Number of samples to be used in each pool of execution.

        Returns
        -------
        score: float
            EBM score. A float between zero and one.

    """
    adata = remove_sparsity(adata)

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
    """Computes Average Silhouette Width (ASW) metric for ``adata`` given the batch column name.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        Returns
        -------
        score: float
            ASW score. A float between -1 and 1.

    """
    adata = remove_sparsity(adata)

    labels = adata.obs[label_key].values

    labels_encoded = LabelEncoder().fit_transform(labels)

    return silhouette_score(adata.X, labels_encoded)


def ari(adata, label_key):
    """Computes Adjusted Rand Index (ARI) metric for ``adata`` given the batch column name.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        Returns
        -------
        score: float
            ARI score. A float between 0 and 1.

    """
    adata = remove_sparsity(adata)

    n_labels = len(adata.obs[label_key].unique().tolist())
    kmeans = KMeans(n_labels, n_init=200)

    labels_pred = kmeans.fit_predict(adata.X)
    labels = adata.obs[label_key].values
    labels_encoded = LabelEncoder().fit_transform(labels)

    return adjusted_rand_score(labels_encoded, labels_pred)


def nmi(adata, label_key):
    """Computes Normalized Mutual Information (NMI) metric for ``adata`` given the batch column name.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        Returns
        -------
        score: float
            NMI score. A float between 0 and 1.

    """
    adata = remove_sparsity(adata)

    n_labels = len(adata.obs[label_key].unique().tolist())
    kmeans = KMeans(n_labels, n_init=200)

    labels_pred = kmeans.fit_predict(adata.X)
    labels = adata.obs[label_key].values
    labels_encoded = LabelEncoder().fit_transform(labels)

    return normalized_mutual_info_score(labels_encoded, labels_pred)


def knn_purity(adata, label_key, n_neighbors=30):
    """Computes KNN Purity metric for ``adata`` given the batch column name.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        n_neighbors: int
            Number of nearest neighbors.
        Returns
        -------
        score: float
            KNN purity score. A float between 0 and 1.

    """
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
