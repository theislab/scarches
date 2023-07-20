import collections.abc as container_abcs
import numpy as np
import re
import sys
import torch
from ...dataset import MultiConditionAnnotatedDataset

def print_progress(epoch, logs, n_epochs=10000, only_val_losses=True):
    """Creates Message for '_print_progress_bar'.

       Parameters
       ----------
       epoch: Integer
            Current epoch iteration.
       logs: Dict
            Dictionary of all current losses.
       n_epochs: Integer
            Maximum value of epochs.
       only_val_losses: Boolean
            If 'True' only the validation dataset losses are displayed, if 'False' additionally the training dataset
            losses are displayed.

       Returns
       -------
    """
    message = ""
    for key in logs:
        if only_val_losses:
            if "val_" in key and "unweighted" not in key:
                message += f" - {key:s}: {logs[key][-1]:7.2f}"
        else:
            if "unweighted" not in key:
                message += f" - {key:s}: {logs[key][-1]:7.2f}"

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

def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, container_abcs.Mapping):
        output = {key: custom_collate([d[key] for d in batch]) for key in elem}
        return output

def train_test_split(adata, train_frac=0.85, condition_keys=None, cell_type_key=None, labeled_array=None):
    """Splits 'Anndata' object into training and validation data.

       Parameters
       ----------
       adata: `~anndata.AnnData`
            `AnnData` object for training the model.
       train_frac: float
            Train-test split fraction. the model will be trained with train_frac for training
            and 1-train_frac for validation.

       Returns
       -------
       Indices for training and validating the model.
    """
    indices = np.arange(adata.shape[0])

    if train_frac == 1:
        return indices, None

    if cell_type_key is not None:
        labeled_array = np.zeros((len(adata), 1)) if labeled_array is None else labeled_array
        labeled_array = np.ravel(labeled_array)

        labeled_idx = indices[labeled_array == 1]
        unlabeled_idx = indices[labeled_array == 0]

        train_labeled_idx = []
        val_labeled_idx = []
        train_unlabeled_idx = []
        val_unlabeled_idx = []

        if len(labeled_idx) > 0:
            cell_types = adata[labeled_idx].obs[cell_type_key].unique().tolist()
            for cell_type in cell_types:
                ct_idx = labeled_idx[adata[labeled_idx].obs[cell_type_key] == cell_type]
                n_train_samples = int(np.ceil(train_frac * len(ct_idx)))
                np.random.shuffle(ct_idx)
                train_labeled_idx.append(ct_idx[:n_train_samples])
                val_labeled_idx.append(ct_idx[n_train_samples:])
        if len(unlabeled_idx) > 0:
            n_train_samples = int(np.ceil(train_frac * len(unlabeled_idx)))
            train_unlabeled_idx.append(unlabeled_idx[:n_train_samples])
            val_unlabeled_idx.append(unlabeled_idx[n_train_samples:])
        train_idx = train_labeled_idx + train_unlabeled_idx
        val_idx = val_labeled_idx + val_unlabeled_idx

        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)

    elif condition_keys is not None:
        train_idx = []
        val_idx = []
        conditions = adata.obs['conditions_combined'].unique().tolist()
        for condition in conditions:
            cond_idx = indices[adata.obs['conditions_combined'] == condition]
            n_train_samples = int(np.ceil(train_frac * len(cond_idx)))
            np.random.shuffle(cond_idx)
            train_idx.append(cond_idx[:n_train_samples])
            val_idx.append(cond_idx[n_train_samples:])

        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)

    else:
        n_train_samples = int(np.ceil(train_frac * len(indices)))
        np.random.shuffle(indices)
        train_idx = indices[:n_train_samples]
        val_idx = indices[n_train_samples:]

    return train_idx, val_idx

def make_dataset(adata,
                 train_frac=0.9,
                 condition_keys=None,
                 cell_type_keys=None,
                 condition_encoders=None,
                 conditions_combined_encoder=None,
                 cell_type_encoder=None,
                 labeled_indices=None,
                 ):
    """Splits 'adata' into train and validation data and converts them into 'CustomDatasetFromAdata' objects.

       Parameters
       ----------

       Returns
       -------
       Training 'CustomDatasetFromAdata' object, Validation 'CustomDatasetFromAdata' object
    """
    # Preprare data for semisupervised learning
    labeled_array = np.zeros((len(adata), 1))
    if labeled_indices is not None:
        labeled_array[labeled_indices] = 1

    if cell_type_keys is not None:
        finest_level = None
        n_cts = 0
        for cell_type_key in cell_type_keys:
            if len(adata.obs[cell_type_key].unique().tolist()) >= n_cts:
                n_cts = len(adata.obs[cell_type_key].unique().tolist())
                finest_level = cell_type_key

        train_idx, val_idx = train_test_split(adata, train_frac, cell_type_key=finest_level,
                                              labeled_array=labeled_array)

    elif condition_keys is not None:
        train_idx, val_idx = train_test_split(adata, train_frac, condition_keys=condition_keys)
    else:
        train_idx, val_idx = train_test_split(adata, train_frac)

    data_set_train = MultiConditionAnnotatedDataset(
        adata if train_frac == 1 else adata[train_idx],
        condition_keys=condition_keys,
        cell_type_keys=cell_type_keys,
        condition_encoders=condition_encoders,
        conditions_combined_encoder=conditions_combined_encoder,
        cell_type_encoder=cell_type_encoder,
        labeled_array=labeled_array[train_idx]
    )
    if train_frac == 1:
        return data_set_train, None
    else:
        data_set_valid = MultiConditionAnnotatedDataset(
            adata[val_idx],
            condition_keys=condition_keys,
            cell_type_keys=cell_type_keys,
            condition_encoders=condition_encoders,
            conditions_combined_encoder=conditions_combined_encoder,
            cell_type_encoder=cell_type_encoder,
            labeled_array=labeled_array[val_idx]
        )
        return data_set_train, data_set_valid

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def t_dist(x, y, alpha):
    """student t-distribution, as same as used in t-SNE algorithm.
             q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
    Arguments:
        inputs: the variable containing data, shape=(n_samples, n_features)
    Return:
        q: student's t-distribution with degree alpha, or soft labels for each sample. shape=(n_samples, n_clusters)
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    distances = torch.pow(x - y, 2).sum(2) / alpha

    q = 1.0 / (1.0 + distances)
    q = torch.pow(q, (alpha + 1.0) / 2.0)
    q = (q.T / q.sum(1)).T
    return q

def target_distribution(q):
    weight = torch.pow(q, 2) / q.sum(0)
    return (weight.T / weight.sum(1)).T

def kl_loss(p, q):
    return (p * torch.log(p / q)).sum(1).mean()