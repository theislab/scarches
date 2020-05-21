import numpy as np
from scipy import sparse


def label_encoder(adata, le=None, condition_key='condition'):
    """
        Encode labels of Annotated `adata` matrix.
        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix.
        le: dict or None
            dictionary of encoded labels. if `None`, will create one.
        condition_key: str
            column name of conditions in `adata.obs` dataframe
        Returns
        -------
        labels: numpy nd-array
            Array of encoded labels
        le: dict
            dictionary with labels and encoded labels as key, value pairs
    """
    labels = np.zeros(adata.shape[0])
    unique_labels = sorted(adata.obs[condition_key].unique().tolist())
    if isinstance(le, dict):
        assert set(unique_labels).issubset(set(le.keys()))
    else:
        le = dict()
        for idx, label in enumerate(unique_labels):
            le[label] = idx

    for label, encoded in le.items():
        labels[adata.obs[condition_key] == label] = encoded

    return labels.reshape(-1, 1), le


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


def create_condition_encoder(conditions, target_conditions):
    if not isinstance(target_conditions, list):
        target_conditions = [target_conditions]

    dictionary = {}
    conditions = [e for e in conditions if e not in target_conditions]
    for idx, condition in enumerate(conditions):
        dictionary[condition] = idx
    return dictionary
