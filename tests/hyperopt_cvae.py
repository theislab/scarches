from __future__ import print_function

import argparse

import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe

import surgeon


def data():
    DATASETS = {
        "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type",
                     "target": ["Pancreas Celseq", "Pancreas CelSeq2"]},
        "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"]},
        "pbmc": {"name": "pbmc", "batch_key": "study", "cell_type_key": "cell_type", "target": ["inDrops", "Drop-seq"]},
    }

    data_key = "pancreas"

    data_dict = DATASETS[data_key]

    data_name = data_dict['name']
    batch_key = data_dict['batch_key']
    cell_type_key = data_dict['cell_type_key']
    target_conditions = data_dict['target']

    adata = sc.read(f"./data/{data_name}/{data_name}_count.h5ad")
    adata_out_of_sample = adata[adata.obs[batch_key].isin(target_conditions)]
    adata_for_training = adata[~adata.obs[batch_key].isin(target_conditions)]

    adata_for_training = surgeon.utils.normalize(adata_for_training,
                                                 filter_min_counts=False,
                                                 normalize_input=False,
                                                 size_factors=True,
                                                 logtrans_input=True,
                                                 n_top_genes=2000,
                                                 )

    adata_out_of_sample = surgeon.utils.normalize(adata_out_of_sample,
                                                  filter_min_counts=False,
                                                  normalize_input=False,
                                                  size_factors=True,
                                                  logtrans_input=True,
                                                  n_top_genes=2000,
                                                  )
    train_adata_for_training, valid_adata_for_training = surgeon.utils.train_test_split(adata_for_training, 0.85)
    train_adata_out_of_sample, valid_adata_out_of_sample = surgeon.utils.train_test_split(adata_out_of_sample, 0.85)
    n_conditions = len(train_adata_for_training.obs[batch_key].unique().tolist())
    return adata_out_of_sample, train_adata_for_training, valid_adata_for_training, train_adata_out_of_sample, valid_adata_out_of_sample, batch_key, cell_type_key, target_conditions


def create_model(adata_out_of_sample, train_adata_for_training, valid_adata_for_training, train_adata_out_of_sample, valid_adata_out_of_sample,
                 batch_key, cell_type_key, target_conditions):
    n_conditions = len(train_adata_for_training.obs[batch_key].unique().tolist())

    z_dim_choices = {{choice([5, 10, 15, 20, 25, 30, 40, 50, 75, 100])}}

    alpha_choices = {{choice([0.01, 0.001, 0.0001, 0.00001, 0.000001])}}
    scale_factor_choices = {{choice([0.1, 1.0, 5.0, 10, 50, 100])}}
    clip_value_choices = {{choice([1, 3, 5, 10])}}
    batch_size_choices_before = {{choice([16, 32, 64, 128, 256, 512, 1024])}}
    batch_size_choices_after = {{choice([16, 32, 64, 128, 256, 512, 1024])}}
    dropout_rate_choices = {{choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])}}
    architecture_choices = {{choice([[128], [128, 128], [128, 128, 128]])}}
    network = surgeon.archs.CVAE(x_dimension=train_adata_for_training.shape[1],
                                 z_dimension=z_dim_choices,
                                 n_conditions=n_conditions,
                                 use_batchnorm=True,
                                 lr=0.001,
                                 alpha=alpha_choices,
                                 scale_factor=scale_factor_choices,
                                 clip_value=clip_value_choices,
                                 loss_fn='nb',
                                 architecture=architecture_choices,
                                 model_path=f"./models/CVAE/hyperopt/before/",
                                 dropout_rate=dropout_rate_choices,
                                 )

    conditions = train_adata_for_training.obs[batch_key].unique().tolist()
    condition_encoder = surgeon.utils.create_dictionary(conditions, target_conditions)

    network.train(train_adata_for_training,
                  valid_adata_for_training,
                  condition_key=batch_key,
                  cell_type_key=cell_type_key,
                  le=condition_encoder,
                  n_epochs=10000,
                  batch_size=batch_size_choices_before,
                  early_stop_limit=70,
                  lr_reducer=50,
                  n_per_epoch=0,
                  save=True,
                  retrain=True,
                  verbose=2)

    new_network = surgeon.operate(network,
                                  new_conditions=target_conditions,
                                  init='Xavier',
                                  freeze=True)

    new_network.train(train_adata_out_of_sample,
                      valid_adata_out_of_sample,
                      condition_key=batch_key,
                      cell_type_key=cell_type_key,
                      le=new_network.condition_encoder,
                      n_epochs=10000,
                      batch_size=batch_size_choices_after,
                      early_stop_limit=50,
                      lr_reducer=40,
                      n_per_epoch=0,
                      save=True,
                      verbose=2)

    encoder_labels, _ = surgeon.utils.label_encoder(
        adata_out_of_sample, label_encoder=network.condition_encoder, condition_key=batch_key)

    latent_adata = new_network.to_latent(adata_out_of_sample, encoder_labels)

    ebm = surgeon.metrics.entropy_batch_mixing(latent_adata, label_key=batch_key, n_pools=1)
    asw = surgeon.metrics.asw(latent_adata, label_key=batch_key)
    ari = surgeon.metrics.ari(latent_adata, label_key=cell_type_key)
    nmi = surgeon.metrics.nmi(latent_adata, label_key=cell_type_key)

    objective = ari

    print(f'EBM: {ebm:.4f} - ASW: {asw:.4f} - ARI: {ari:.4f} - NMI: {nmi:.4f}')
    print(
        f'alpha = {new_network.alpha}, scale_factor = {new_network.scale_factor}, z_dim = {new_network.z_dim}, clip_value = {new_network.clip_value}, batch_size_before = {batch_size_choices_before}, batch_size_after = {batch_size_choices_after}, dropout_rate = {new_network.dr_rate}, lr = {new_network.lr}')

    return {'loss': -objective, 'status': STATUS_OK}


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
