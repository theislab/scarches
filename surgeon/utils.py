import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import LabelEncoder


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
    if isinstance(label_encoder, dict):
        for condition, label in label_encoder.items():
            labels[adata.obs[condition_key] == condition] = label
    elif isinstance(label_encoder, LabelEncoder):
        labels = label_encoder.transform(adata.obs[condition_key].values)
    else:
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(adata.obs[condition_key].values)
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


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True,
              n_top_genes=2000):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    adata_count = adata.copy()

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if n_top_genes > 0 and adata.shape[1] > n_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        genes = adata.var['highly_variable']
        adata = adata[:, genes]
        adata_count = adata_count[:, genes]

    if normalize_input:
        sc.pp.scale(adata)

    if sparse.issparse(adata_count.X):
        adata_count.X = adata_count.X.A

    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata_count.copy()
    else:
        adata.raw = adata_count

    return adata


def create_dictionary(conditions, target_conditions):
    if not isinstance(target_conditions, list):
        target_conditions = [target_conditions]

    dictionary = {}
    conditions = [e for e in conditions if e not in target_conditions]
    for idx, condition in enumerate(conditions):
        dictionary[condition] = idx
    return dictionary
