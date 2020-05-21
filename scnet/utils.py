import numpy as np
from scipy import sparse


def label_encoder(adata, le=None, condition_key='condition'):
    """
        Encode labels of Annotated `adata` matrix.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        le: dict or None
            dictionary of encoded labels. if `None`, will create one.
        condition_key: str
            column name of conditions in `adata.obs` data frame.

        Returns
        -------
        labels: :class:`~numpy.ndarray`
            Array of encoded labels
        le: dict
            dictionary with labels and encoded labels as key, value pairs.
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
    """
        If ``adata.X`` is a sparse matrix, this will convert it in to normal matrix.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.

        Returns
        -------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
    """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    return adata


def train_test_split(adata, train_frac=0.85):
    """
        Split ``adata`` into train and test annotated datasets.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        train_frac: float
            Fraction of observations (cells) to be used in training dataset. Has to be a value between 0 and 1.

        Returns
        -------
        train_adata: :class:`~anndata.AnnData`
            Training annotated dataset.
        valid_adata: :class:`~anndata.AnnData`
            Test annotated dataset.
    """
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data


def create_condition_encoder(conditions, target_conditions):
    """
        Creates a ``condition_encoder`` dictionary for the given ``conditions`` excluding ``target_conditions``.

        Parameters
        ----------
        conditions: list
            list of all unqiue conditions.
        target_conditions: list, None
            list of unique condition to be excluded from ``conditions``.

        Returns
        -------
        condition_encoder: dict
            A dictionary with conditions and encoded labels for them as its keys and values respectively.
    """
    if not isinstance(target_conditions, list):
        target_conditions = [target_conditions]

    if target_conditions is None:
        target_conditions = []

    dictionary = {}
    conditions = [e for e in conditions if e not in target_conditions]
    for idx, condition in enumerate(conditions):
        dictionary[condition] = idx
    return dictionary
