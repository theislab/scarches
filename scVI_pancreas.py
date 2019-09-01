import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from _scVI import ADataset, scVI_Trainer
from scvi.models.vae import VAE
import numpy as np
import pandas as pd
import torch

adata = sc.read("pancreas/pancreas_count.h5ad")
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
highlyVariables  = adata.var['highly_variable'].tolist()

adata = sc.read("pancreas/pancreas_count.h5ad")
adata = sc.AnnData(adata.X.A[:, highlyVariables], adata.obs)
adata.obs = adata.obs.drop(['batch'], axis=1)

adata.obs.columns = ['batch_indices','cell_types']
adata.obs['labels'] = adata.obs['cell_types']

le = LabelEncoder()
le.fit(adata.obs['labels'])
adata.obs['labels'] = le.transform(adata.obs['labels'])

le = LabelEncoder()
le.fit(adata.obs['batch_indices'])
adata.obs['batch_indices'] = le.transform(adata.obs['batch_indices'])

scviDataset = ADataset(adata)
n_epochs = 300
lr = 1e-3
use_batches = True
n_samples = adata.shape[0]

vae = VAE(scviDataset.nb_genes, n_batch=scviDataset.n_batches * use_batches)
model = scVI_Trainer(vae,scviDataset,train_size=0.85,frequency=5, early_stopping_kwargs = {
            "early_stopping_metric": "elbo",
            "save_best_state_metric": "elbo",
            "patience": 20,
            "threshold": 0,
            "reduce_lr_on_plateau": True,
            "lr_patience": 15,
            "lr_factor": 0.1,
        })
model.train("scVI_N_Panc.csv", n_epochs=n_epochs, lr=lr)

torch.save(model.model.state_dict(), "pancreasModel.pt")


keep_idx = pd.read_csv('pancreas/pancreas_N_0.8.csv', header=None, dtype='int32').values.reshape(-1)
adata08 = adata[keep_idx, :]
scviDataset = ADataset(adata08)
vae = VAE(scviDataset.nb_genes, n_batch=scviDataset.n_labels * use_batches)
model = scVI_Trainer(vae,scviDataset,train_size=0.85,frequency=5, early_stopping_kwargs = {
            "early_stopping_metric": "elbo",
            "save_best_state_metric": "elbo",
            "patience": 20,
            "threshold": 0,
            "reduce_lr_on_plateau": True,
            "lr_patience": 15,
            "lr_factor": 0.1,
        })
model.train("scVI_0.8N_Panc.csv", n_epochs=n_epochs, lr=lr)

torch.save(model.model.state_dict(), "pancreasModel8.pt")



keep_idx = pd.read_csv('pancreas/pancreas_N_0.6.csv', header=None, dtype='int32').values.reshape(-1)
adata06 = adata[keep_idx, :]
scviDataset = ADataset(adata06)
vae = VAE(scviDataset.nb_genes, n_batch=scviDataset.n_labels * use_batches)
model = scVI_Trainer(vae,scviDataset,train_size=0.85,frequency=5, early_stopping_kwargs = {
            "early_stopping_metric": "elbo",
            "save_best_state_metric": "elbo",
            "patience": 20,
            "threshold": 0,
            "reduce_lr_on_plateau": True,
            "lr_patience": 15,
            "lr_factor": 0.1,
        })
model.train("scVI_0.6N_Panc.csv", n_epochs=n_epochs, lr=lr)

torch.save(model.model.state_dict(), "pancreasModel6.pt")


keep_idx = pd.read_csv('pancreas/pancreas_N_0.2.csv', header=None, dtype='int32').values.reshape(-1)
adata04 = adata[keep_idx, :]
scviDataset = ADataset(adata04)
vae = VAE(scviDataset.nb_genes, n_batch=scviDataset.n_labels * use_batches)
model = scVI_Trainer(vae,scviDataset,train_size=0.85,frequency=5, early_stopping_kwargs = {
            "early_stopping_metric": "elbo",
            "save_best_state_metric": "elbo",
            "patience": 20,
            "threshold": 0,
            "reduce_lr_on_plateau": True,
            "lr_patience": 15,
            "lr_factor": 0.1,
        })
model.train("scVI_0.4N_Panc.csv", n_epochs=n_epochs, lr=lr)

torch.save(model.model.state_dict(), "pancreasModel4.pt")


keep_idx = pd.read_csv('pancreas/pancreas_N_0.2.csv', header=None, dtype='int32').values.reshape(-1)
adata02 = adata[keep_idx, :]
scviDataset = ADataset(adata02)
vae = VAE(scviDataset.nb_genes, n_batch=scviDataset.n_labels * use_batches)
model = scVI_Trainer(vae,scviDataset,train_size=0.85,frequency=5, early_stopping_kwargs = {
            "early_stopping_metric": "elbo",
            "save_best_state_metric": "elbo",
            "patience": 20,
            "threshold": 0,
            "reduce_lr_on_plateau": True,
            "lr_patience": 15,
            "lr_factor": 0.1,
        })
model.train("scVI_0.2N_Panc.csv", n_epochs=n_epochs, lr=lr)

torch.save(model.model.state_dict(), "pancreasModel2.pt")
