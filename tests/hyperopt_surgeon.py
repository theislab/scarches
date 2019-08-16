from __future__ import print_function

import argparse
import os

import anndata
import numpy as np
import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from scipy import sparse
from sklearn.metrics import silhouette_score

import surgeon
from surgeon.utils import normalize, train_test_split


def data():
    DATASETS = {
        "PBMC": {'name': 'pbmc', 'need_merge': False,
                 'source_conditions': ['inDrops', '10x Chromium V2 B', '10x Chromium V2 A', 'Smart-seq2', 'CEL-Seq2'],
                 'target_conditions': ['Drop-seq'],
                 'condition_encoder': {'inDrops': 0, '10x Chromium V2 A': 1, '10x Chromium V2 B': 2, 'Smart-seq2': 3,
                                       "CEL-Seq2": 4, '68K': 5, 'Small 3K': 6},
                 'condition': 'Method',
                 'cell_type': 'CellType'},

    }
    data_key = "PBMC"
    data_dict = DATASETS[data_key]
    data_name = data_dict['name']
    condition_key = data_dict['condition']
    cell_type_key = data_dict['cell_type']
    target_conditions = data_dict['target_conditions']
    condition_encoder = data_dict['condition_encoder']

    if data_name.endswith("count"):
        adata = sc.read(f"./data/{data_name}/{data_name}.h5ad")
        adata = normalize(adata,
                          filter_min_counts=False,
                          normalize_input=False,
                          logtrans_input=True,
                          n_top_genes=2000)
        train_adata, valid_adata = train_test_split(adata, 0.80)
    else:
        if os.path.exists(f"./data/{data_name}/train_{data_name}.h5ad"):
            train_adata = sc.read(f"./data/{data_name}/train_{data_name}.h5ad")
            valid_adata = sc.read(f"./data/{data_name}/valid_{data_name}.h5ad")
        else:
            adata = sc.read(f"./data/{data_name}/{data_name}.h5ad")
            train_adata, valid_adata = train_test_split(adata, 0.80)

    net_train_adata_in_sample = train_adata.copy()[~(train_adata.obs[condition_key].isin(target_conditions))]
    net_valid_adata_in_sample = valid_adata.copy()[~(valid_adata.obs[condition_key].isin(target_conditions))]

    net_train_adata_out_of_sample = train_adata.copy()[train_adata.obs[condition_key].isin(target_conditions)]
    net_valid_adata_out_of_sample = valid_adata.copy()[valid_adata.obs[condition_key].isin(target_conditions)]

    n_conditions = len(net_train_adata_in_sample.obs[condition_key].unique().tolist())

    return net_train_adata_in_sample, net_valid_adata_in_sample, net_train_adata_out_of_sample, net_valid_adata_out_of_sample, condition_key, cell_type_key, n_conditions, condition_encoder, data_name, target_conditions


def create_model(net_train_adata_in_sample, net_valid_adata_in_sample,
                 net_train_adata_out_of_sample, net_valid_adata_out_of_sample,
                 condition_key, cell_type_key,
                 n_conditions, condition_encoder, data_name, target_conditions):
    z_dim_choices = {{choice([10, 20, 40, 50, 60, 80, 100])}}

    alpha_choices = {{choice([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])}}
    eta_choices = {{choice([1, 2, 5, 7, 10])}}
    batch_size_choices = {{choice([128, 256, 512, 1024, 1500, 2048])}}
    dropout_rate_choices = {{choice([0.1, 0.2, 0.5, 0.75])}}

    network = surgeon.archs.CVAE(x_dimension=net_train_adata_in_sample.shape[1],
                                 z_dimension=z_dim_choices,
                                 n_conditions=n_conditions,
                                 lr=0.001,
                                 alpha=alpha_choices,
                                 eta=eta_choices,
                                 clip_value=1e6,
                                 loss_fn='mse',
                                 model_path=f"./models/CVAE/hyperopt/{data_name}/",
                                 dropout_rate=dropout_rate_choices,
                                 output_activation='relu')

    network.train(net_train_adata_in_sample,
                  net_valid_adata_in_sample,
                  condition_key=condition_key,
                  le=condition_encoder,
                  n_epochs=10000,
                  batch_size=batch_size_choices,
                  early_stop_limit=100,
                  lr_reducer=50,
                  save=False,
                  verbose=2)

    new_network = surgeon.operate(network,
                                  new_condition=target_conditions[0],
                                  init='Xavier',
                                  freeze=True)

    new_network.train(net_train_adata_out_of_sample,
                      net_valid_adata_out_of_sample,
                      condition_key=condition_key,
                      le=new_network.condition_encoder,
                      n_epochs=10000,
                      batch_size=batch_size_choices,
                      early_stop_limit=100,
                      lr_reducer=50,
                      save=False,
                      verbose=2)
    adata = net_train_adata_in_sample.concatenate(net_valid_adata_in_sample, net_train_adata_out_of_sample, net_valid_adata_out_of_sample)

    encoder_labels, _ = surgeon.utils.label_encoder(adata, label_encoder=new_network.condition_encoder,
                                                    condition_key=condition_key)

    latent_adata = new_network.to_latent(adata, encoder_labels)

    sc.pp.pca(latent_adata, svd_solver="arpack")
    sc.pp.neighbors(latent_adata, n_neighbors=25)

    asw_score = 0
    for cell_type in latent_adata.obs[cell_type_key].unique().tolist():
        cell_type_adata = latent_adata.copy()[latent_adata.obs[cell_type_key] == cell_type]
        nb_conditions = len(cell_type_adata.obs[condition_key].unique().tolist())
        if nb_conditions > 1:
            X_pca = cell_type_adata.obsm["X_pca"]
            conditions_encoded, _ = surgeon.utils.label_encoder(cell_type_adata, new_network.condition_encoder, condition_key)
            asw_score_cell_type = silhouette_score(X_pca, conditions_encoded)
            asw_score += asw_score_cell_type
            print(f"ASW for {cell_type} is {asw_score_cell_type:.6f}")

    print(f"average silhouette_score for C-VAE : {asw_score}")

    objective = asw_score
    print(
        f'alpha = {network.alpha}, eta={network.eta}, z_dim = {network.z_dim}, batch_size = {batch_size_choices}, dropout_rate = {network.dr_rate}')
    return {'loss': objective, 'status': STATUS_OK}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-n', '--max_evals', type=int, required=True,
                                 help='name of dataset you want to train')

    args = vars(parser.parse_args())
    data_key = args['data']

    best_run, best_network = optim.minimize(model=create_model,
                                            data=data,
                                            algo=tpe.suggest,
                                            max_evals=args['max_evals'],
                                            trials=Trials())
    print("All Done!")
    print(best_run)
