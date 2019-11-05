import argparse
import os

import scanpy as sc

import surgeon

DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type",
                 "target": ["Pancreas SS2", "Pancreas CelSeq2"]},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"]},
    "pbmc": {"name": "pbmc_subset", "batch_key": "study", "cell_type_key": "cell_type", "target": ["inDrops", "Drop-seq"]},
}


def train_and_evaluate(data_dict, freeze_level=0, loss_fn='nb'):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    batch_key = data_dict['batch_key']
    target_conditions = data_dict['target']

    path_to_save = f"./results/convergence/{data_name}/"
    sc.settings.figdir = os.path.abspath(path_to_save)
    os.makedirs(path_to_save, exist_ok=True)

    adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")

    adata_out_of_sample = adata[adata.obs[batch_key].isin(target_conditions)]
    adata_for_training = adata[~adata.obs[batch_key].isin(target_conditions)]

    if loss_fn == 'nb':
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
                                 beta=1.0,
                                 eta=1.0,
                                 scale_factor=1.0,
                                 clip_value=10000,
                                 loss_fn=loss_fn,
                                 model_path=f"./models/CVAE/Convergence/MMD/before-{data_name}-{loss_fn}/",
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
                  batch_size=512,
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
    sc.pl.umap(latent_adata, color=[batch_key, cell_type_key], wspace=0.7,
               save="_latent_training.pdf")

    if freeze_level == 0:
        freeze = False
        freeze_expression_input = False
    elif freeze_level == 1:
        freeze = True
        freeze_expression_input = False
    elif freeze_level == 2:
        freeze = True
        freeze_expression_input = True


    new_network = surgeon.operate(network,
                                  new_conditions=target_conditions,
                                  init='Xavier',
                                  freeze=freeze,
                                  freeze_expression_input=freeze_expression_input,
                                  remove_dropout=True)

    new_network.model_path = f"./models/CVAE/Convergence/MMD/after-{data_name}-{loss_fn}-{freeze}/"

    train_adata, valid_adata = surgeon.utils.train_test_split(adata_out_of_sample, 0.85)

    filename = path_to_save + f"scores_scNet_freeze_level={freeze_level}"
    filename += "_count.log" if loss_fn == 'nb' else "_normalized.log"

    new_network.train(train_adata,
                    valid_adata,
                    condition_key=batch_key,
                    cell_type_key=cell_type_key,
                    le=new_network.condition_encoder,
                    n_epochs=10000,
                    n_epochs_warmup=500 if not freeze else 0,
                    batch_size=512,
                    early_stop_limit=100,
                    lr_reducer=80,
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
    arguments_group.add_argument('-f', '--freeze_level', type=int, default=1, required=True,
                                 help='freeze_level')
    arguments_group.add_argument('-c', '--count', type=int, default=0, required=False,
                                 help='latent space dimension')
    args = vars(parser.parse_args())

    freeze_level = args['freeze_level']
    loss_fn = 'nb' if args['count'] > 0 else 'mse'

    data_dict = DATASETS[args['data']]

    train_and_evaluate(data_dict=data_dict, freeze_level=freeze_level, loss_fn=loss_fn)
