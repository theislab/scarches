import argparse
import os

import scanpy as sc

import surgeon

DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type",
                 "target": ["Pancreas SS2", "Pancreas Celseq", "Pancreas CelSeq2"]},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"]},
    "pbmc": {"name": "pbmc", "batch_key": "study", "cell_type_key": "cell_type", "target": ["inDrops", "Drop-seq"]},
}


def train_and_evaluate(data_dict, freeze=True, count_adata=True):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    batch_key = data_dict['batch_key']
    target_conditions = data_dict['target']

    path_to_save = f"./results/convergence/{data_name}/"
    sc.settings.figdir = os.path.abspath(path_to_save)
    os.makedirs(path_to_save, exist_ok=True)

    adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")

    if count_adata:
        loss_fn = "nb"
    else:
        loss_fn = "mse"

    adata_out_of_sample = adata[adata.obs[batch_key].isin(target_conditions)]
    adata_for_training = adata[~adata.obs[batch_key].isin(target_conditions)]

    if count_adata:
        clip_value = 3.0
    else:
        clip_value = 1e6

    train_adata, valid_adata = surgeon.utils.train_test_split(adata_for_training, 0.80)
    n_conditions = len(train_adata.obs[batch_key].unique().tolist())

    network = surgeon.archs.CVAE(x_dimension=train_adata.shape[1],
                                 z_dimension=10,
                                 architecture=[128],
                                 n_conditions=n_conditions,
                                 lr=0.001,
                                 alpha=0.001,
                                 scale_factor=1.0,
                                 clip_value=clip_value,
                                 loss_fn=loss_fn,
                                 model_path=f"./models/CVAE/Convergence/before-{data_name}-{loss_fn}/",
                                 dropout_rate=0.0,
                                 output_activation='relu')

    conditions = adata_for_training.obs[batch_key].unique().tolist()
    condition_encoder = surgeon.utils.create_dictionary(conditions, target_conditions)

    network.train(train_adata,
                  valid_adata,
                  condition_key=batch_key,
                  cell_type_key=cell_type_key,
                  le=condition_encoder,
                  n_epochs=10000,
                  batch_size=32,
                  early_stop_limit=50,
                  lr_reducer=40,
                  n_per_epoch=0,
                  save=True,
                  retrain=True,
                  verbose=2)

    encoder_labels, _ = surgeon.utils.label_encoder(adata_for_training, label_encoder=network.condition_encoder,
                                                    condition_key=batch_key)

    latent_adata = network.to_latent(adata_for_training, encoder_labels)

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[batch_key, cell_type_key], wspace=0.7,
               save="_latent_training.pdf")

    new_network = surgeon.operate(network,
                                  new_conditions=target_conditions,
                                  init='Xavier',
                                  freeze=freeze)

    new_network.model_path = f"./models/CVAE/Convergence/after-{data_name}-{loss_fn}-{freeze}/"

    train_adata, valid_adata = surgeon.utils.train_test_split(adata_out_of_sample, 0.85)

    filename = path_to_save + "scores_scNet"
    filename += "Freezed" if freeze else "UnFreezed"
    filename += "_count.log" if count_adata else "_normalized.log"

    if freeze:
        new_network.train(train_adata,
                          valid_adata,
                          condition_key=batch_key,
                          cell_type_key=cell_type_key,
                          le=new_network.condition_encoder,
                          n_epochs=300,
                          batch_size=32,
                          early_stop_limit=50,
                          lr_reducer=40,
                          n_per_epoch=5,
                          score_filename=filename,
                          save=True,
                          verbose=2)
    else:
        new_network.train(train_adata,
                          valid_adata,
                          condition_key=batch_key,
                          cell_type_key=cell_type_key,
                          le=new_network.condition_encoder,
                          n_epochs=300,
                          n_epochs_warmup=400,
                          batch_size=32,
                          early_stop_limit=50,
                          lr_reducer=40,
                          n_per_epoch=5,
                          score_filename=filename,
                          save=True,
                          verbose=2)

    encoder_labels, _ = surgeon.utils.label_encoder(adata_out_of_sample, label_encoder=new_network.condition_encoder,
                                                    condition_key=batch_key)

    latent_adata = new_network.to_latent(adata_out_of_sample, encoder_labels)

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[batch_key, cell_type_key], wspace=0.7, frameon=False,
               save="_latent_out_of_sample.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scNet')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='data name')
    arguments_group.add_argument('-f', '--freeze', type=int, default=1, required=True,
                                 help='freeze')
    arguments_group.add_argument('-c', '--count', type=int, default=0, required=False,
                                 help='latent space dimension')
    args = vars(parser.parse_args())

    freeze = True if args['freeze'] > 0 else False
    count_adata = True if args['count'] > 0 else False
    target_sum = args['target_sum']

    data_dict = DATASETS[args['data']]

    train_and_evaluate(data_dict=data_dict, freeze=freeze, count_adata=count_adata)
