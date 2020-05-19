import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import KNeighborsTransformer
from collections import Counter


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


def normalize(adata, batch_key=None, filter_min_counts=True, size_factors=True, logtrans_input=True,
              target_sum=None, n_top_genes=2000):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    adata_count = adata.copy()

    if size_factors:
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=True, key_added='size_factors')
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if n_top_genes > 0 and adata.shape[1] > n_top_genes:
        if batch_key:
            genes = hvg_batch(adata.copy(), batch_key=batch_key, adataOut=False, target_genes=n_top_genes)
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            genes = adata.var['highly_variable']
        adata = adata[:, genes]
        adata_count = adata_count[:, genes]

    if sparse.issparse(adata_count.X):
        adata_count.X = adata_count.X.A

    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    if size_factors or logtrans_input:
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


def hvg_batch(adata, batch_key=None, target_genes=2000, flavor='cell_ranger', n_bins=20, adataOut=False):
    """

    Method to select HVGs based on mean dispersions of genes that are highly 
    variable genes in all batches. Using a the top target_genes per batch by
    average normalize dispersion. If target genes still hasn't been reached, 
    then HVGs in all but one batches are used to fill up. This is continued 
    until HVGs in a single batch are considered.
    """

    adata_hvg = adata if adataOut else adata.copy()

    n_batches = len(adata_hvg.obs[batch_key].cat.categories)

    # Calculate double target genes per dataset
    sc.pp.highly_variable_genes(adata_hvg,
                                flavor=flavor,
                                n_top_genes=target_genes,
                                n_bins=n_bins,
                                batch_key=batch_key)

    nbatch1_dispersions = adata_hvg.var['dispersions_norm'][adata_hvg.var.highly_variable_nbatches >
                                                            len(adata_hvg.obs[batch_key].cat.categories) - 1]

    nbatch1_dispersions.sort_values(ascending=False, inplace=True)

    if len(nbatch1_dispersions) > target_genes:
        hvg = nbatch1_dispersions.index[:target_genes]

    else:
        enough = False
        print(f'Using {len(nbatch1_dispersions)} HVGs from full intersect set')
        hvg = nbatch1_dispersions.index[:]
        not_n_batches = 1

        while not enough:
            target_genes_diff = target_genes - len(hvg)

            tmp_dispersions = adata_hvg.var['dispersions_norm'][adata_hvg.var.highly_variable_nbatches ==
                                                                (n_batches - not_n_batches)]

            if len(tmp_dispersions) < target_genes_diff:
                print(f'Using {len(tmp_dispersions)} HVGs from n_batch-{not_n_batches} set')
                hvg = hvg.append(tmp_dispersions.index)
                not_n_batches += 1

            else:
                print(f'Using {target_genes_diff} HVGs from n_batch-{not_n_batches} set')
                tmp_dispersions.sort_values(ascending=False, inplace=True)
                hvg = hvg.append(tmp_dispersions.index[:target_genes_diff])
                enough = True

    print(f'Using {len(hvg)} HVGs')

    if not adataOut:
        del adata_hvg
        return hvg.tolist()
    else:
        return adata_hvg[:, hvg].copy()


def weighted_knn(train_adata, valid_adata, label_key, n_neighbors=50, threshold=0.5,
                 pred_unknown=True, return_uncertainty=True):
    print(f'Weighted KNN with n_neighbors = {n_neighbors} and threshold = {threshold} ... ', end='')
    k_neighbors_transformer = KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance',
                                                    algorithm='brute', metric='euclidean',
                                                    n_jobs=-1)
    train_adata = remove_sparsity(train_adata)
    valid_adata = remove_sparsity(valid_adata)

    k_neighbors_transformer.fit(train_adata.X)

    y_train_labels = train_adata.obs[label_key].values
    y_valid_labels = valid_adata.obs[label_key].values

    top_k_distances, top_k_indices = k_neighbors_transformer.kneighbors(X=valid_adata.X)

    stds = np.std(top_k_distances, axis=1)
    stds = (2. / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(top_k_distances_tilda, axis=1, keepdims=True)

    uncertainties = []
    pred_labels = []
    for i in range(len(weights)):
        labels = y_train_labels[top_k_indices[i]]
        most_common_label, _ = Counter(y_train_labels[top_k_indices[i]]).most_common(n=1)[0]
        most_prob = weights[i, y_train_labels[top_k_indices[i]] == most_common_label].sum()
        if pred_unknown:
            if most_prob >= threshold:
                pred_label = most_common_label
            else:
                pred_label = 'Unknown'
        else:
            pred_label = most_common_label

        if pred_label == y_valid_labels[i]:
            uncertainties.append(1 - most_prob)
        else:
            true_prob = weights[i, y_train_labels[top_k_indices[i]] == y_valid_labels[i]].sum()
            uncertainties.append(1 - true_prob)

        pred_labels.append(pred_label)

    pred_labels = np.array(pred_labels).reshape(-1, 1)
    uncertainties = np.array(uncertainties).reshape(-1, 1)

    print('finished!')
    if return_uncertainty:
        return pred_labels, uncertainties
    else:
        return pred_labels


def subsample(adata, study_key, fraction=0.1, specific_cell_types=None, cell_type_key=None):
    studies = adata.obs[study_key].unique().tolist()
    if specific_cell_types and cell_type_key:
        subsampled_adata = adata[adata.obs[cell_type_key].isin(specific_cell_types)]
        other_adata = adata[~adata.obs[cell_type_key].isin(specific_cell_types)]
    else:
        subsampled_adata = None
    for study in studies:
        study_adata = other_adata[other_adata.obs[study_key] == study]
        n_samples = study_adata.shape[0]
        subsample_idx = np.random.choice(n_samples, int(fraction * n_samples), replace=False)
        study_adata_subsampled = study_adata[subsample_idx, :]
        subsampled_adata = study_adata_subsampled if subsampled_adata is None else subsampled_adata.concatenate(
            study_adata_subsampled)
    return subsampled_adata
