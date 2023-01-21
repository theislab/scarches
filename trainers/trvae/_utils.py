import sys
import numpy as np
import re
import torch
import collections.abc as container_abcs
from torch.utils.data import DataLoader

from ...dataset import trVAEDataset


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
                message += f" - {key:s}: {logs[key][-1]:7.10f}"
        else:
            if "unweighted" not in key:
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


def train_test_split(adata, train_frac=0.85, condition_key=None, cell_type_key=None):
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
       `AnnData` objects for training and validating the model.
    """
    if train_frac == 1:
        return adata, None
    else:
        indices = np.arange(adata.shape[0])

        if cell_type_key is not None:
            labeled_idx = indices[adata.obs['trvae_labeled'] == 1]
            unlabeled_idx = indices[adata.obs['trvae_labeled'] == 0]
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

        elif condition_key is not None:
            train_idx = []
            val_idx = []
            conditions = adata.obs[condition_key].unique().tolist()
            for condition in conditions:
                cond_idx = indices[adata.obs[condition_key] == condition]
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

        train_data = adata[train_idx, :]
        valid_data = adata[val_idx, :]

        return train_data, valid_data


def make_dataset(adata,
                 train_frac=0.9,
                 condition_key=None,
                 cell_type_keys=None,
                 condition_encoder=None,
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
    size_factors = adata.X.sum(1) #sum the 1000 genes per cell, (1) refers to summation of the colum values (genes) per row (cell).
    if len(size_factors.shape) < 2:
        size_factors = np.expand_dims(size_factors, axis=1)
    adata.obs['trvae_size_factors'] = size_factors

    # Preprare data for semisupervised learning
    labeled_array = np.zeros((len(adata), 1)) #result of this is a matrix (15681,1) that has 15681 zeros (cell) in one column.
    if labeled_indices is not None:
        labeled_array[labeled_indices] = 1
    adata.obs['trvae_labeled'] = labeled_array

    if cell_type_keys is not None:
        finest_level = None
        n_cts = 0
        for cell_type_key in cell_type_keys:
            if len(adata.obs[cell_type_key].unique().tolist()) >= n_cts:
                n_cts = len(adata.obs[cell_type_key].unique().tolist())
                finest_level = cell_type_key

        train_adata, validation_adata = train_test_split(adata, train_frac, cell_type_key=finest_level)

    elif condition_key is not None:
        train_adata, validation_adata = train_test_split(adata, train_frac, condition_key=condition_key)
    else:
        train_adata, validation_adata = train_test_split(adata, train_frac)

    data_set_train = trVAEDataset(
        train_adata,
        condition_key=condition_key,
        cell_type_keys=cell_type_keys,
        condition_encoder=condition_encoder,
        cell_type_encoder=cell_type_encoder,
    )
    if train_frac == 1:
        return data_set_train, None
    else:
        data_set_valid = trVAEDataset(
            validation_adata,
            condition_key=condition_key,
            cell_type_keys=cell_type_keys,
            condition_encoder=condition_encoder,
            cell_type_encoder=cell_type_encoder,
        )
        return data_set_train, data_set_valid


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
