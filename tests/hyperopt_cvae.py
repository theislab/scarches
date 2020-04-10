from __future__ import print_function

import argparse

import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from keras import backend as K

import scnet


def data():
    DATASETS = {
        "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type",
                     "target": ["Pancreas SS2", "Pancreas CelSeq2"]},
        "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"]},
        "pbmc": {"name": "pbmc", "batch_key": "study", "cell_type_key": "cell_type", "target": ["inDrops", "Drop-seq"]},
        "mouse_brain": {"name": "mouse_brain", "batch_key": "study", "cell_type_key": "cell_type",
                        "target": ["Tabula_muris", "Zeisel"]},
    }

    data_key = "mouse_brain"

    data_dict = DATASETS[data_key]

    data_name = data_dict['name']
    batch_key = data_dict['batch_key']
    cell_type_key = data_dict['cell_type_key']
    target_conditions = data_dict['target']

    adata = sc.read(f"./data/{data_name}/{data_name}_normalized_hvg.h5ad")
    adata_out_of_sample = adata[adata.obs[batch_key].isin(target_conditions)]
    adata_for_training = adata[~adata.obs[batch_key].isin(target_conditions)]

    train_adata_for_training, valid_adata_for_training = scnet.utils.train_test_split(adata_for_training, 0.80)
    train_adata_out_of_sample, valid_adata_out_of_sample = scnet.utils.train_test_split(adata_out_of_sample, 0.80)
    n_conditions = len(train_adata_for_training.obs[batch_key].unique().tolist())
    return adata_out_of_sample, train_adata_for_training, valid_adata_for_training, train_adata_out_of_sample, valid_adata_out_of_sample, batch_key, cell_type_key, target_conditions


def create_model(adata_out_of_sample, train_adata_for_training, valid_adata_for_training, train_adata_out_of_sample,
                 valid_adata_out_of_sample,
                 batch_key, cell_type_key, target_conditions):
    n_conditions = len(train_adata_for_training.obs[batch_key].unique().tolist())

    z_dim_choices = {{choice([10, 20, 40, 50, 70, 100])}}

    alpha_choices = {{choice([0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])}}
    beta_choices = {{choice([100, 500, 1000])}}
    # eta_choices = {{choice([0.1, 1.0, 5.0, 10, 50, 100, 1000])}}
    clip_value_choices = {{choice([3, 100, 500, 1e3, 1e4, 1e5, 1e6])}}
    batch_size_choices_before = {{choice([64, 128, 256, 512, 1024])}}
    batch_size_choices_after = {{choice([64, 128, 256, 512, 1024])}}
    dropout_rate_choices = {{choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])}}
    # architecture_choices = {{choice([[128, 32], [512, 128, 32], [512, 256, 128, 32], [1024, 512, 256, 128, 32]])}}
    use_batchnorm_choices = {{choice([True, False])}}

    network = scnet.archs.CVAE(x_dimension=train_adata_for_training.shape[1],
                               z_dimension=z_dim_choices,
                               n_conditions=n_conditions,
                               use_batchnorm=use_batchnorm_choices,
                               lr=0.001,
                               alpha=alpha_choices,
                               beta=beta_choices,
                               eta=1.0,
                               clip_value=clip_value_choices,
                               loss_fn='mse',
                               architecture=[128, 64, 20],
                               dropout_rate=dropout_rate_choices,
                               output_activation='relu',
                               )

    conditions = train_adata_for_training.obs[batch_key].unique().tolist()
    condition_encoder = scnet.utils.create_dictionary(conditions, target_conditions)

    network.train(train_adata_for_training,
                  valid_adata_for_training,
                  condition_key=batch_key,
                  cell_type_key=cell_type_key,
                  le=condition_encoder,
                  n_epochs=1000,
                  batch_size=batch_size_choices_before,
                  early_stop_limit=20,
                  lr_reducer=10,
                  n_per_epoch=0,
                  save=False,
                  retrain=True,
                  verbose=2)

    new_network = scnet.operate(network,
                                new_conditions=target_conditions,
                                init='Xavier',
                                freeze=True,
                                freeze_expression_input=True,
                                remove_dropout=True)
    
    new_network.train(train_adata_out_of_sample,
                      valid_adata_out_of_sample,
                      condition_key=batch_key,
                      cell_type_key=cell_type_key,
                      le=new_network.condition_encoder,
                      n_epochs=1000,
                      batch_size=batch_size_choices_after,
                      early_stop_limit=20,
                      lr_reducer=10,
                      n_per_epoch=0,
                      save=False,
                      retrain=True,
                      verbose=2)

    encoder_labels, _ = scnet.utils.label_encoder(
        adata_out_of_sample, label_encoder=new_network.condition_encoder, condition_key=batch_key)

    latent_adata = new_network.to_mmd_layer(adata_out_of_sample, encoder_labels, encoder_labels)

    ebm = scnet.metrics.entropy_batch_mixing(latent_adata, label_key=batch_key, n_neighbors=15, n_pools=1)
    knn = scnet.metrics.knn_purity(latent_adata, label_key=cell_type_key, n_neighbors=15)
    
    if knn >= 0.8 and ebm >= 0.3:
        objective = 1.25 * knn + ebm
    else:
        objective = 0.0
    
    K.clear_session()

    print(f'EBM: {ebm:.4f} - KNN: {knn:.4f}')
    print(
        f'alpha = {new_network.alpha}, beta = {new_network.beta}, eta = {new_network.eta}, arch = {new_network.architecture}, z_dim = {new_network.z_dim}, clip_value = {new_network.clip_value}, batch_size_after = {batch_size_choices_after}, dropout_rate = {new_network.dr_rate}, lr = {new_network.lr}')

    return {'loss': -objective, 'status': STATUS_OK}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-n', '--max_evals', type=int, required=True,
                                 help='name of dataset you want to train')

    args = vars(parser.parse_args())

    best_run, best_network = optim.minimize(model=create_model,
                                            data=data,
                                            algo=tpe.suggest,
                                            max_evals=args['max_evals'],
                                            trials=Trials())
    print("All Done!")
    print(best_run)
