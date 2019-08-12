import numpy as np
from scipy import sparse
from sklearn import preprocessing


def label_encoder(adata, label_encoder, condition_key='condition'):
    """
        Encode labels of Annotated `adata` matrix using sklearn.preprocessing.LabelEncoder class.
        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        labels: numpy nd-array
            Array of encoded labels
        Example
        --------
        >>> import surgeon
        >>> import scanpy as sc
        >>> train_data = sc.read("./data/train.h5ad")
        >>> train_labels, label_encoder = surgeon.utils.label_encoder(train_data)
    """
    labels = np.zeros(adata.shape[0])
    for condition, label in label_encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), label_encoder


def remove_sparsity(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    return adata


def train_test_split(adata, train_frac=0.85):
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data
