from random import shuffle
import sys
import anndata
import numpy as np
from scipy import sparse
from sklearn import preprocessing


def print_progress(epoch, logs, n_epochs=10000):
    """Creates Message for '_print_progress_bar'.

       Parameters
       ----------
       epoch: Integer
            Current epoch iteration.
       logs: dict
            Dictionary of all current losses.
       n_epochs: Integer
            Maximum value of epochs.

       Returns
       -------
    """
    message = ""
    for key in logs:
        if "loss" in key and ("epoch_" in key or "val_" in key) and "unweighted" not in key:
            message += f" - {key:s}: {logs[key][-1]:7.10f}"

    _print_progress_bar(epoch + 1, n_epochs, prefix='', suffix=message, decimals=1, length=20)


def _print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """Prints out message with a progress bar.

       Parameters
       ----------
       iteration: Integer
            Current epoch.
       total: Integer
            Maximum value of epochs.
       prefix: String
            String before the progress bar.
       suffix: String
            String after the progress bar.
       decimals: Integer
            Digits after comma for all the losses.
       length: Integer
            Length of the progress bar.
       fill: String
            Symbol for filling the bar.

       Returns
       -------
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_len = int(length * iteration // total)
    bar = fill * filled_len + '-' * (length - filled_len)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


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
    condtion_1 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["ctrl"])]
    condtion_2 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["stim"])]
    training = data[~((data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["stim"]))]
    return [training, condtion_1, condtion_2, cell_with_both_condition]


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
