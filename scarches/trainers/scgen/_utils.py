import os
from random import shuffle

import anndata
import numpy as np
from scipy import sparse
from sklearn import preprocessing

from scarches.models.scgen._utils import extractor


def data_remover(adata, remain_list, remove_list, cell_type_key, condition_key):
    """
    Removes specific cell type in stimulated condition form `adata`.

    Parameters
    ----------
    adata: `~anndata.AnnData`
        Annotated data matrix
    remain_list: list
        list of cell types which are going to be remained in `adata`.
    remove_list: list
        list of cell types which are going to be removed from `adata`.

    Returns
    -------
    merged_data: list
        returns array of specified cell types in stimulated condition
    """
    source_data = []
    for i in remain_list:
        source_data.append(extractor(adata, i, conditions={"ctrl": "control", "stim": "stimulated"},
                                     cell_type_key=cell_type_key, condition_key=condition_key)[3])
    target_data = []
    for i in remove_list:
        target_data.append(extractor(adata, i, conditions={"ctrl": "control", "stim": "stimulated"},
                                     cell_type_key=cell_type_key, condition_key=condition_key)[1])
    merged_data = training_data_provider(source_data, target_data)
    merged_data.var_names = adata.var_names
    return merged_data



def training_data_provider(train_s, train_t):
    """
    Concatenates two lists containing adata files

    Parameters
    ----------
    train_s: `~anndata.AnnData`
        Annotated data matrix.
    train_t: `~anndata.AnnData`
        Annotated data matrix.

    Returns
    -------
    Concatenated Annotated data matrix.
    """
    train_s_X = []
    train_s_diet = []
    train_s_groups = []
    for i in train_s:
        train_s_X.append(i.X.A)
        train_s_diet.append(i.obs["condition"].tolist())
        train_s_groups.append(i.obs["cell_type"].tolist())
    train_s_X = np.concatenate(train_s_X)
    temp = []
    for i in train_s_diet:
        temp = temp + i
    train_s_diet = temp
    temp = []
    for i in train_s_groups:
        temp = temp + i
    train_s_groups = temp
    train_t_X = []
    train_t_diet = []
    train_t_groups = []
    for i in train_t:
        train_t_X.append(i.X.A)
        train_t_diet.append(i.obs["condition"].tolist())
        train_t_groups.append(i.obs["cell_type"].tolist())
    temp = []
    for i in train_t_diet:
        temp = temp + i
    train_t_diet = temp
    temp = []
    for i in train_t_groups:
        temp = temp + i
    train_t_groups = temp
    train_t_X = np.concatenate(train_t_X)
    train_real = np.concatenate([train_s_X, train_t_X])  # concat all
    train_real = anndata.AnnData(train_real)
    train_real.obs["condition"] = train_s_diet + train_t_diet
    train_real.obs["cell_type"] = train_s_groups + train_t_groups
    return train_real


def shuffle_adata(adata):
    """
    Shuffles the `adata`.

    Parameters
    ----------
    adata: `~anndata.AnnData`
        Annotated data matrix.
    labels: numpy nd-array
        list of encoded labels

    Returns
    -------
    adata: `~anndata.AnnData`
        Shuffled annotated data matrix.
    labels: numpy nd-array
        Array of shuffled labels if `labels` is not None.
    """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    new_adata = adata[ind_list, :]
    return new_adata


def label_encoder(adata):
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
    """
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(adata.obs["condition"].tolist())
    return labels.reshape(-1, 1), le
