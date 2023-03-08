import scanpy as sc
import pandas as pd
import os
import numpy as np
import anndata
from sklearn.neighbors import KNeighborsTransformer

#These function were originally created by Lisa Sikemma

def weighted_knn_trainer(
    train_adata,
    train_adata_emb,
    n_neighbors=50,
    precomputed = False,
    verbose=True,
    ):
    """Trains a weighted KNN classifier on ``train_adata``.
    Parameters
    ----------
    train_adata: :class:`~anndata.AnnData`
        Annotated dataset to be used to train KNN classifier with ``label_key`` as the target variable.
    train_adata_emb: str
        Name of the obsm layer to be used for calculation of neighbors. If set to "X", anndata.X will be
        used
    n_neighbors: int
        Number of nearest neighbors in KNN classifier.
    precomputed: bool
        ``False`` by default.
    verbose: bool
        Whether or not additional processing information should be printed
    """
    print(f"Weighted KNN with n_neighbors = {n_neighbors}")

    metric = "precomputed" if precomputed else "euclidean"

    k_neighbors_transformer = KNeighborsTransformer(
        n_neighbors=n_neighbors,
        mode="distance",
        algorithm="brute",
        metric=metric,
        n_jobs=-1,
    )
    if train_adata_emb == "X":
        train_emb = train_adata.X
    elif train_adata_emb in train_adata.obsm.keys():
        train_emb = train_adata.obsm[train_adata_emb]
    else:
        raise ValueError(
            "train_adata_emb should be set to either 'X' or the name of the obsm layer to be used!"
        )
    if verbose:
        print(f"Fitting the {n_neighbors}-nearest neighbors transformer from the training dataset.")

    k_neighbors_transformer.fit(train_emb)

    if verbose:
        print("Fitting completed.")
    return k_neighbors_transformer    


def weighted_knn_transfer(
    query_adata,
    query_adata_emb,
    ref_adata_obs,
    label_keys,
    knn_model,
    threshold=1,
    pred_unknown=False,
    verbose=True
):
    """Annotates ``query_adata`` cells with an input trained weighted KNN classifier.
    Parameters
    ----------
    query_adata: :class:`~anndata.AnnData`
        Annotated dataset to be used to queryate KNN classifier. Embedding to be used
    query_adata_emb: str
        Name of the obsm layer to be used for label transfer. If set to "X",
        query_adata.X will be used
    ref_adata_obs: :class:`pd.DataFrame`
        obs of ref Anndata
    label_keys: list of str
        Names of the obs columns to be used as target variables (e.g. cell_type) in ``query_adata``.
    knn_model: :class:`~sklearn.neighbors._graph.KNeighborsTransformer`
        knn model trained on reference adata with weighted_knn_trainer function
    threshold: float
        Threshold of uncertainty used to annotating cells as "Unknown". cells with
        uncertainties higher than this value will be annotated as "Unknown".
        Set to 1 to keep all predictions. This enables one to later on play
        with thresholds.
    pred_unknown: bool
        ``False`` by default. Whether to annotate any cell as "unknown" or not.
        If `False`, ``threshold`` will not be used and each cell will be annotated
        with the label which is the most common in its ``n_neighbors`` nearest cells.
    verbose: bool
        Whether or not additional processing information should be printed
    """
    if not type(knn_model) == KNeighborsTransformer:
        raise ValueError(
            "You should use a knn_model of type sklearn.neighbors._graph.KNeighborsTransformer!"
        )


    if query_adata_emb == "X":
        query_emb = query_adata.X
    elif query_adata_emb in query_adata.obsm.keys():
        query_emb = query_adata.obsm[query_adata_emb]
    else:
        raise ValueError(
            "query_adata_emb should be set to either 'X' or the name of the obsm layer to be used!"
        )

    if verbose:
        print(f"Finding the {knn_model.n_neighbors}-neighbors of a point. This may take some time...")
    
    top_k_distances, top_k_indices = knn_model.kneighbors(X=query_emb)

    if verbose:
        print("Neighbors computed.")

    stds = np.std(top_k_distances, axis=1)
    stds = (2.0 / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(
        top_k_distances_tilda, axis=1, keepdims=True
    )

    cols = ref_adata_obs.columns[ref_adata_obs.columns.isin(label_keys)]
    uncertainties = pd.DataFrame(columns=cols, index=query_adata.obs_names)
    pred_labels = pd.DataFrame(columns=cols, index=query_adata.obs_names)

    if verbose:
        print("Label transfer begins...")

    for i in range(len(weights)):
        for j in cols:

            y_train_labels = ref_adata_obs[j].values
            unique_labels = np.unique(y_train_labels[top_k_indices[i]])
            best_label, best_prob = None, 0.0

            for candidate_label in unique_labels:
                candidate_prob = weights[
                    i, y_train_labels[top_k_indices[i]] == candidate_label
                ].sum()
                if best_prob < candidate_prob:
                    best_prob = candidate_prob
                    best_label = candidate_label
                    
            uncertainty = (max(1 - best_prob, 0))
            
            if pred_unknown:
                if uncertainty <= threshold:
                    pred_label = best_label
                else:
                    pred_label = "Unknown"
            else:
                pred_label = best_label
                
            uncertainties.iloc[i][j] = uncertainty

            pred_labels.iloc[i][j] = (pred_label)

    print("Finished!")

    return pred_labels, uncertainties


def knn_label_transfer(
    train_adata,
    train_adata_emb,
    query_adata,
    query_adata_emb,
    ref_adata_obs,
    label_keys,
    n_neighbors=50,
    threshold=1,
    precomputed = False,
    pred_unknown=False,
    verbose=True
):
    """Trains a weighted KNN classifier on ``train_adata`` and annotates ``query_adata`` cells
    with it. Returns the combined embedding with the transferred cell labels

    Parameters
    ----------
    train_adata: :class:`~anndata.AnnData`
        Annotated dataset to be used to train KNN classifier with ``label_key`` as the target variable.
    train_adata_emb: str
        Name of the obsm layer to be used for calculation of neighbors. If set to "X", anndata.X will be
        used
    uery_adata: :class:`~anndata.AnnData`
        Annotated dataset to be used to queryate KNN classifier. Embedding to be used
    query_adata_emb: str
        Name of the obsm layer to be used for label transfer. If set to "X",
        query_adata.X will be used
    ref_adata_obs: :class:`pd.DataFrame`
        obs of ref Anndata
    label_keys: list of str
        Names of the obs columns to be used as target variables (e.g. cell_type) in ``query_adata``.
    n_neighbors: int
        Number of nearest neighbors in KNN classifier.
    threshold: float
        Threshold of uncertainty used to annotating cells as "Unknown". cells with
        uncertainties higher than this value will be annotated as "Unknown".
        Set to 1 to keep all predictions. This enables one to later on play
        with thresholds.
    pred_unknown: bool
        ``False`` by default. Whether to annotate any cell as "unknown" or not.
        If `False`, ``threshold`` will not be used and each cell will be annotated
        with the label which is the most common in its ``n_neighbors`` nearest cells.
    precomputed: bool
        ``False`` by default.
    verbose: bool
        ``True`` by default. Whether additional processing information should be printed.
    """

    knn_transformer = weighted_knn_trainer(
        train_adata,
        train_adata_emb,
        n_neighbors,
        precomputed, 
        verbose)

    labels, uncert = weighted_knn_transfer(
        query_adata,
        query_adata_emb,
        ref_adata_obs,
        label_keys,
        knn_transformer,
        threshold,
        pred_unknown,
        verbose
    )

    combined_emb = train_adata.concatenate(query_adata, index_unique=None)

    labels.rename(columns={f'Level_{lev}':f'Level_{lev}_transferred_label' for lev in range(1,6)},inplace=True)
    uncert.rename(columns={f'Level_{lev}':f'Level_{lev}_transfer_uncert' for lev in range(1,6)},inplace=True)

    print(labels)
    print(uncert)
    
    combined_emb.obs = combined_emb.obs.join(labels)
    combined_emb.obs = combined_emb.obs.join(uncert)

    # combined_emb.obs[uncert] = list(np.array(combined_emb.obs[uncert]))
    for col in labels:
        combined_emb.obs[col] = combined_emb.obs[col].astype('category')
        combined_emb.obs[col].replace('nan',np.nan,inplace=True)

    return combined_emb