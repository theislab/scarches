import argparse
import os

import scanpy as sc
from sklearn.preprocessing import LabelEncoder

import surgeon
from _scVI import ADataset, scVI_Trainer
from scvi.models.vae import VAE
import torch

DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type",
                 "target": ["Pancreas SS2", "Pancreas Celseq", "Pancreas CelSeq2"]},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"]},
    "pbmc": {"name": "pbmc", "batch_key": "study", "cell_type_key": "cell_type", "target": ["inDrops", "Drop-seq"]},
}


def train_and_evaluate(data_dict, target_sum=None):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    batch_key = data_dict['batch_key']
    target_conditions = data_dict['target']

    path_to_save = f"./results/convergence/{data_name}/"
    sc.settings.figdir = path_to_save
    os.makedirs(path_to_save, exist_ok=True)

    adata = sc.read(f"./data/{data_name}/{data_name}_count.h5ad")

    adata = surgeon.tl.normalize(adata,
                                 batch_key=batch_key,
                                 target_sum=target_sum,
                                 filter_min_counts=False,
                                 size_factors=True,
                                 logtrans_input=True,
                                 n_top_genes=1000
                                 )

    new_adata = sc.AnnData(X=adata.raw.X)
    new_adata.obs = adata.obs.copy(deep=True)
    new_adata.var = adata.var.copy(deep=True)
    new_adata.var_names = adata.var_names

    adata = new_adata.copy()

    adata_out_of_sample = adata[adata.obs[batch_key].isin(target_conditions)]

    adata_out_of_sample = surgeon.utils.normalize(adata_out_of_sample,
                                                  filter_min_counts=False,
                                                  normalize_input=False,
                                                  size_factors=True,
                                                  logtrans_input=True,
                                                  n_top_genes=5000,
                                                  )
    use_batches = True
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

    adata_out_of_sample.obs['cell_types'] = adata_out_of_sample.obs[cell_type_key].values

    le = LabelEncoder()
    adata_out_of_sample.obs['labels'] = le.fit_transform(adata_out_of_sample.obs[cell_type_key])

    le = LabelEncoder()
    adata_out_of_sample.obs['batch_indices'] = le.fit_transform(adata_out_of_sample.obs[batch_key])

    adata_out_of_sample = ADataset(adata_out_of_sample)

    vae = VAE(adata_out_of_sample.nb_genes, n_batch=adata_out_of_sample.n_batches * use_batches, n_layers = 2)

    model = scVI_Trainer(vae, adata_out_of_sample,
                         train_size=0.85,
                         frequency=5,
                         early_stopping_kwargs=early_stopping_kwargs)

    model.train(os.path.join(path_to_save, f"scVI.csv"), n_epochs=n_epochs, lr=lr)

    os.makedirs("./models/scVI/convergence/", exist_ok=True)
    torch.save(model.model.state_dict(), f"./models/scVI/subsample/{data_name}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scNet')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='data name')
    arguments_group.add_argument('-t', '--target_sum', type=float, required=False, default=1e6,
                                 help='Target sum')
    args = vars(parser.parse_args())

    data_dict = DATASETS[args['data']]
    target_sum = args['target_sum']

    train_and_evaluate(data_dict=data_dict, target_sum=target_sum)
