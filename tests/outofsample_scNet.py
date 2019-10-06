import argparse
import os

import scanpy as sc

import surgeon

DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type",
                 "target": ["Pancreas CelSeq2", "Pancreas SS2", ], 'cell_type': "Pancreas Alpha"},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype",
            "source": ["Batch1", "Batch2", 'Batch3', 'Batch4', 'Batch5', 'Batch6', 'Batch7']},
    "pbmc": {"name": "pbmc", "batch_key": "study", "cell_type_key": "cell_type", "source": ["inDrops", "Drop-seq"]},
}


def train(data_dict, freeze_level=0, loss_fn='nb'):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    batch_key = data_dict['batch_key']
    target_conditions = data_dict['target']
    target_cell_type = data_dict['cell_type']

    if loss_fn == 'nb':
        clip_value = 3.0
    else:
        clip_value = 1e6

    path_to_save = f"./results/outofsample/{data_name}-{loss_fn}-freeze_level={freeze_level}/"
    sc.settings.figdir = path_to_save
    os.makedirs(path_to_save, exist_ok=True)

    adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")

    adata.obs['study'] = adata.obs[batch_key].values
    batch_key = 'study'

    adata_out_of_sample = adata[adata.obs[batch_key].isin(target_conditions)]
    adata_for_training = adata[~adata.obs[batch_key].isin(target_conditions)]

    adata_for_training = adata_for_training[adata_for_training.obs[cell_type_key] != target_cell_type]

    train_adata, valid_adata = surgeon.utils.train_test_split(adata_for_training, 0.80)
    n_conditions = len(train_adata.obs[batch_key].unique().tolist())

    architecture = [128]
    z_dim = 10
    network = surgeon.archs.CVAE(x_dimension=train_adata.shape[1],
                                 z_dimension=z_dim,
                                 architecture=architecture,
                                 n_conditions=n_conditions,
                                 lr=0.001,
                                 alpha=0.00001,
                                 use_batchnorm=True,
                                 eta=1.0,
                                 scale_factor=1.0,
                                 clip_value=clip_value,
                                 loss_fn=loss_fn,
                                 model_path=f"./models/CVAE/outofsample/before-{data_name}-{loss_fn}-{architecture}-{z_dim}/",
                                 dropout_rate=0.1,
                                 output_activation='relu')

    conditions = adata_for_training.obs[batch_key].unique().tolist()
    condition_encoder = surgeon.utils.create_dictionary(conditions, [])

    network.train(train_adata,
                  valid_adata,
                  condition_key=batch_key,
                  cell_type_key=cell_type_key,
                  le=condition_encoder,
                  n_epochs=10000,
                  batch_size=128,
                  early_stop_limit=100,
                  lr_reducer=80,
                  n_per_epoch=0,
                  save=True,
                  retrain=False,
                  verbose=2)

    encoder_labels, _ = surgeon.utils.label_encoder(adata_for_training, label_encoder=network.condition_encoder,
                                                    condition_key=batch_key)

    latent_adata = network.to_latent(adata_for_training, encoder_labels)

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[batch_key, cell_type_key], wspace=0.7, frameon=False,
               save="_latent_first.pdf")

    if freeze_level == 0:
        freeze = False
        freeze_expression_input = False
    elif freeze_level == 1:
        freeze = True
        freeze_expression_input = False
    elif freeze_level == 2:
        freeze = True
        freeze_expression_input = True
    else:
        raise Exception("Invalid freeze_level value")

    new_network = surgeon.operate(network,
                                  new_conditions=target_conditions,
                                  init='Xavier',
                                  freeze=freeze,
                                  freeze_expression_input=freeze_expression_input,
                                  remove_dropout=True)

    new_network.model_path = f"./models/CVAE/outofsample/after-{data_name}-{loss_fn}-{architecture}-{z_dim}/"
    train_adata, valid_adata = surgeon.utils.train_test_split(adata_out_of_sample, 0.80)

    new_network.train(train_adata,
                      valid_adata,
                      condition_key=batch_key,
                      cell_type_key=cell_type_key,
                      le=new_network.condition_encoder,
                      n_epochs=10000,
                      batch_size=128,
                      n_epochs_warmup=0,
                      early_stop_limit=50,
                      lr_reducer=40,
                      n_per_epoch=0,
                      save=True,
                      retrain=True,
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
    arguments_group.add_argument('-f', '--freeze_level', type=int, default=1, required=True,
                                 help='freeze')
    arguments_group.add_argument('-c', '--count_adata', type=int, default=1, required=True,
                                 help='freeze')
    args = vars(parser.parse_args())

    freeze_level = args['freeze_level']
    loss_fn = 'nb' if args['count_adata'] > 0 else 'mse'
    data_dict = DATASETS[args['data']]

    train(data_dict=data_dict, freeze_level=freeze_level, loss_fn=loss_fn)
