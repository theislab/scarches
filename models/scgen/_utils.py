import anndata
import numpy as np
from scipy import sparse
import logging

logger = logging.getLogger(__name__)


def extractor(data, cell_type, conditions, cell_type_key="cell_type", condition_key="condition"):
    """
    Returns a list of `data` files while filtering for a specific `cell_type`.

    Parameters
    ----------
    data: `~anndata.AnnData`
        Annotated data matrix
    cell_type: basestring
        specific cell type to be extracted from `data`.
    conditions: dict
        dictionary of stimulated/control of `data`.

    Returns
    -------
    list of `data` files while filtering for a specific `cell_type`.
    """
    cell_with_both_condition = data[data.obs[cell_type_key] == cell_type]
    condition_1 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["ctrl"])]
    condition_2 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["stim"])]
    training = data[~((data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["stim"]))]
    return [training, condition_1, condition_2, cell_with_both_condition]


def balancer(adata, cell_type_key="cell_type", condition_key="condition"):
    """
    Makes cell type population equal.

    Parameters
    ----------
    adata: `~anndata.AnnData`
        Annotated data matrix.

    Returns
    -------
    balanced_data: `~anndata.AnnData`
        Equal cell type population Annotated data matrix.
    """
    class_names = np.unique(adata.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata.copy()[adata.obs[cell_type_key] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))
    all_data_x = []
    all_data_label = []
    all_data_condition = []
    for cls in class_names:
        temp = adata.copy()[adata.obs[cell_type_key] == cls]
        index = np.random.choice(range(len(temp)), max_number)
        if sparse.issparse(temp.X):
            temp_x = temp.X.A[index]
        else:
            temp_x = temp.X[index]
        all_data_x.append(temp_x)
        temp_ct = np.repeat(cls, max_number)
        all_data_label.append(temp_ct)
        temp_cc = np.repeat(np.unique(temp.obs[condition_key]), max_number)
        all_data_condition.append(temp_cc)
    balanced_data = anndata.AnnData(np.concatenate(all_data_x))
    balanced_data.obs[cell_type_key] = np.concatenate(all_data_label)
    balanced_data.obs[condition_key] = np.concatenate(all_data_label)
    class_names = np.unique(balanced_data.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = len(balanced_data[balanced_data.obs[cell_type_key] == cls])
    return balanced_data
