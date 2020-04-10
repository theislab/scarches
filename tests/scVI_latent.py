import sys
sys.path.append("../")

import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from _scVI import ADataset, scVI_Trainer
from scvi.models.vae import VAE
import numpy as np
import pandas as pd
import torch
import os
import argparse
from scnet.utils import remove_sparsity

DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type", "target": ["Pancreas Celseq", "Pancreas CelSeq2"], "HV": True},
    "pbmc": {"name": "pbmc", "batch_key": "study", "cell_type_key": "cell_type", "target": ["Drop-seq", "inDrops"], "HV": False},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"], "HV": True},
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
highly_variable = data_dict['HV']

adata = sc.read(f"./data/{data_name}/{data_name}_count.h5ad")
adata = remove_sparsity(adata)

if highly_variable:
    adata_normalized = adata.copy()

    sc.pp.normalize_per_cell(adata_normalized)
    sc.pp.log1p(adata_normalized)
    sc.pp.highly_variable_genes(adata_normalized, n_top_genes=5000)
    highly_variable_genes = adata_normalized.var['highly_variable']

    adata = adata[:, highly_variable_genes]

adata.obs['cell_types'] = adata.obs[cell_type_key]

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

for i in range(5):
    for subsample_frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        final_adata = None
        for target in target_batches:
            adata_sampled = adata[adata.obs[batch_key] == target, :]
            keep_idx = np.loadtxt(f'./data/subsample/{data_name}/{target}/{subsample_frac}/{i}.csv', dtype='int32')
            adata_sampled = adata_sampled[keep_idx, :]

            if final_adata is None:
                final_adata = adata_sampled
            else:
                final_adata = final_adata.concatenate(adata_sampled)
        
        le = LabelEncoder()
        final_adata.obs['labels'] = le.fit_transform(final_adata.obs["cell_types"])
        scvi_dataset = ADataset(final_adata)
        
        vae = VAE(scvi_dataset.nb_genes, n_batch=scvi_dataset.n_batches * use_batches)

        model = scVI_Trainer(vae, scvi_dataset,
                             train_size=0.85,
                             frequency=5,
                             early_stopping_kwargs=early_stopping_kwargs)
        os.makedirs(f"./results/latent/scVI/{data_name}/", exist_ok=True)
        model.model.load_state_dict(torch.load(f"./models/scVI/subsample/{data_name}-{subsample_frac}-{i}.pt"))
        model.model.eval()
        p = model.create_posterior(model.model, model.gene_dataset, indices=np.arange(len(model.gene_dataset)))
#         latent, _, _ = p.get_latent()
#         np.savetxt(f"./results/latent/scVI/{data_name}/{subsample_frac}-{i}.csv", latent, delimiter=",", )
#         pd.DataFrame(latent, ).to_csv(f"./results/latent/scVI/{data_name}/{subsample_frac}-{i}.csv", index=None)
        p.show_t_sne(color_by='batches and labels', save_name=f"./results/latent/scVI/{data_name}/{subsample_frac}-{i}.png")
        