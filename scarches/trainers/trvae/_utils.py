import sys
import numpy as np
import re
import torch
from torch._six import container_abcs
from torch.utils.data import DataLoader, SubsetRandomSampler

from scarches.dataset.trvae import AnnotatedDataset


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
            message += f" - {key:s}: {logs[key][-1]:7.0f}"

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


def train_test_split(adata, train_frac=0.85, condition_key=None):
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
        n_val_samples = int(adata.shape[0] * (1 - train_frac))
        if condition_key is not None:
            conditions = adata.obs[condition_key].unique().tolist()
            n_conditions = len(conditions)
            n_val_samples_per_condition = int(n_val_samples / n_conditions)
            condition_indices_train = []
            condition_indices_val = []
            for i in range(n_conditions):
                idx = indices[adata.obs[condition_key] == conditions[i]]
                np.random.shuffle(idx)
                condition_indices_val.append(idx[:n_val_samples_per_condition])
                condition_indices_train.append(idx[n_val_samples_per_condition:])
            train_idx = np.concatenate(condition_indices_train)
            val_idx = np.concatenate(condition_indices_val)
        else:
            np.random.shuffle(indices)
            val_idx = indices[:n_val_samples]
            train_idx = indices[n_val_samples:]

        train_data = adata[train_idx, :]
        valid_data = adata[val_idx, :]

        return train_data, valid_data


def make_dataset(adata,
                 train_frac=0.9,
                 use_stratified_split=False,
                 condition_key=None,
                 cell_type_key=None,
                 condition_encoder=None,
                 cell_type_encoder=None,
                 ):
    """Splits 'adata' into train and validation data and converts them into 'CustomDatasetFromAdata' objects.

       Parameters
       ----------

       Returns
       -------
       Training 'CustomDatasetFromAdata' object, Validation 'CustomDatasetFromAdata' object
    """
    if use_stratified_split:
        train_adata, validation_adata = train_test_split(adata, train_frac, condition_key=condition_key)
    else:
        train_adata, validation_adata = train_test_split(adata, train_frac)

    data_set_train = AnnotatedDataset(train_adata,
                                      condition_key=condition_key,
                                      cell_type_key=cell_type_key,
                                      condition_encoder=condition_encoder,
                                      cell_type_encoder=cell_type_encoder,
                                      )
    if train_frac == 1:
        return data_set_train, None
    else:
        data_set_valid = AnnotatedDataset(validation_adata,
                                          condition_key=condition_key,
                                          cell_type_key=cell_type_key,
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
        if "celltype" in elem:
            output = dict(total_batch=dict(),
                          labelled_batch=dict())
            for key in elem:
                total_data = [d[key] for d in batch]
                labelled_data = list()
                for d in batch:
                    if d["celltype"] != -1:
                        labelled_data.append(d[key])
                output["total_batch"][key] = custom_collate(total_data)
                output["labelled_batch"][key] = custom_collate(labelled_data)
        else:
            output = dict(total_batch=dict())
            output["total_batch"] = {key: custom_collate([d[key] for d in batch]) for key in elem}
        return output
