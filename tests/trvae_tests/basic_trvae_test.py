import scanpy as sc
import os
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import numpy as np
import time
import matplotlib.pyplot as plt

n_epochs_vae = 500
early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
batch_key = "study"
cell_type_key = "cell_type"

adata_all = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/pancreas_normalized.h5ad'))
adata = adata_all.raw.to_adata()
adata = remove_sparsity(adata)
adata_conditions = adata.obs[batch_key].tolist()

trvae = sca.models.TRVAE(
    adata=adata,
    condition_key=batch_key,
    conditions=adata_conditions,
    hidden_layer_sizes=[128,128],
)

trvae.train(
    n_epochs=n_epochs_vae,
    alpha_epoch_anneal=200,
    early_stopping_kwargs=early_stopping_kwargs
)