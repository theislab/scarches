import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from _scVI import ADataset, scVI_Trainer
from scvi.models.vae import VAE
import numpy as np
import pandas as pd
import torch
import os
import argparse
from surgeon.utils import remove_sparsity

DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type", "target": ["Pancreas Celseq", "Pancreas CelSeq2"]},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"]},
}

parser = argparse.ArgumentParser(description='scNet')
arguments_group = parser.add_argument_group("Parameters")
arguments_group.add_argument('-d', '--data', type=str, required=True,
                             help='data name')
args = vars(parser.parse_args())

data_dict = DATASETS[args['data']]
data_name = data_dict['name']
batch_key = data_dict['batch_key']
cell_type_key = data_dict['cell_type_key']
target_batches = data_dict['target']

adata = sc.read(f"./data/{data_name}/{data_name}_count.h5ad")
adata = remove_sparsity(adata)

adata_normalized = adata.copy()

sc.pp.normalize_per_cell(adata_normalized)
sc.pp.log1p(adata_normalized)
sc.pp.highly_variable_genes(adata_normalized, n_top_genes=5000)
highly_variable_genes = adata_normalized.var['highly_variable']

adata = adata[:, highly_variable_genes]

le = LabelEncoder()
adata.obs['labels'] = le.fit_transform(adata.obs[cell_type_key])

le = LabelEncoder()
adata.obs['batch_indices'] = le.fit_transform(adata.obs[batch_key])

n_epochs = 300
lr = 1e-3
early_stopping_kwargs = {
    "early_stopping_metric": "elbo",
    "save_best_state_metric": "elbo",
    "patience": 50,
    "threshold": 0,
    "reduce_lr_on_plateau": True,
    "lr_patience": 40,
    "lr_factor": 0.1,
}
use_batches = True
n_samples = adata.shape[0]

for subsample_frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    adata_out_of_sample = adata[adata.obs[batch_key].isin(target_batches)]
    adata_in_sample = adata[~adata.obs[batch_key].isin(target_batches)]

    if subsample_frac < 1.0:
        keep_idx = np.loadtxt(f'./data/subsample/{data_name}_N*{subsample_frac}.csv', dtype='int32')
    else:
        n_samples = adata_out_of_sample.shape[0]
        keep_idx = np.random.choice(n_samples, n_samples, replace=False)

    adata_out_of_sample = adata_out_of_sample[keep_idx, :]
    final_adata = adata_in_sample.concatenate(adata_out_of_sample)

    scvi_dataset = ADataset(final_adata)

    vae = VAE(scvi_dataset.nb_genes, n_batch=scvi_dataset.n_batches * use_batches)

    model = scVI_Trainer(vae, scvi_dataset,
                         train_size=0.85,
                         frequency=5,
                         early_stopping_kwargs=early_stopping_kwargs)

    model.train(f"./results/subsample/{data_name}/scVI_frac={subsample_frac}.csv", n_epochs=n_epochs, lr=lr)

    os.makedirs("./models/scVI/subsample/", exist_ok=True)
    torch.save(model.model.state_dict(), f"./models/scVI/subsample/{data_name}-{subsample_frac}.pt")
