import torch
from torch.utils.data import WeightedRandomSampler

from scarches.dataset.trvae import AnnotatedDataset

import re
from torch._six import container_abcs


import scanpy as sc

from scarches.trainers.trvae._utils import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
        if "y" in elem:
            output = dict(total=dict(),
                          labelled=dict())
            for key in elem:
                total_data = [d[key] for d in batch]
                labelled_data = list()
                for d in batch:
                    if d["y"] != -1:
                        labelled_data.append(d[key])
                output["total"][key] = custom_collate(total_data)
                output["labelled"][key] = custom_collate(labelled_data)
        else:
            output = dict(total=dict)
            output["total"] = {key: custom_collate([d[key] for d in batch]) for key in elem}
        return output

def make_dataset(adata,
                 train_frac=0.9,
                 condition_key="study",
                 cell_type_key="cell_type",
                 size_factor_key="size_factors",
                 is_label_key="is_labelled"):
    """Splits 'adata' into train and validation data and converts them into 'CustomDatasetFromAdata' objects.

       Parameters
       ----------

       Returns
       -------
       Training 'CustomDatasetFromAdata' object, Validation 'CustomDatasetFromAdata' object
    """
    train_adata, validation_adata = train_test_split(adata, train_frac)
    data_set_train = AnnotatedDataset(train_adata,
                                      condition_key=condition_key,
                                      cell_type_key=cell_type_key,
                                      size_factors_key=size_factor_key,
                                      )
    data_set_valid = AnnotatedDataset(validation_adata,
                                      condition_key=condition_key,
                                      cell_type_key=cell_type_key,
                                      size_factors_key=size_factor_key,
                                      )
    return data_set_train, data_set_valid

def inference(x=None, c=None, y=None, raw=None, f=None):
    for item in c.unique():
        print("Cell Item: ", item)
        iter = 0
        for i in range(c.size(0)):
            if c[i] == item:
                iter += 1
        print("occurences:", iter)

def forward(total=None, labelled=None):
    for key, batch in total.items():
        batch.to(device)
        print(batch)
    inference(**total)
    inference(**labelled)

def train_scannet(adata, n_epochs=5, batch_size=1024):
    dataset_train, dataset_valid = make_dataset(adata)
    strat_weights = dataset_train.stratifier_weights
    sampler = WeightedRandomSampler(strat_weights, 2048, replacement=True)
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    collate_fn=custom_collate,
                                                    num_workers=0)
    for epoch in range(n_epochs):
        print("Epoch", epoch)
        for iteration, batch_data in enumerate(data_loader_train):
            print("Iteration: ", iteration)
            forward(**batch_data)

target_conditions = ["Pancreas SS2", "Pancreas CelSeq2"]
adata = sc.read("../../pancreas_normalized.h5ad", backup_url=f"https://zenodo.org/record/3930949/files/pancreas_normalized.h5ad?download=1", sparse=True, cache=True)

target_adata = adata[adata.obs["study"].isin(target_conditions)]
train_scannet(target_adata)
