import argparse
import os

import scanpy as sc
from scanpy.plotting import palettes

import scnet

DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type",
                 "target": ["Pancreas SS2", "Pancreas CelSeq2"]},
}


def train_and_evaluate(data_dict, freeze_level=0, loss_fn='nb'):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    batch_key = data_dict['batch_key']
    target_conditions = data_dict['target']

    path_to_save = f"./results/iterative_surgery/{data_name}-{loss_fn}-freeze_level={freeze_level}/"
    sc.settings.figdir = path_to_save
    os.makedirs(path_to_save, exist_ok=True)

    adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")

    if loss_fn == 'nb':
        clip_value = 3.0
    else:
        clip_value = 1000

    if freeze_level == 0:
        freeze = False
        freeze_expression = False
    elif freeze_level == 1:
        freeze = True
        freeze_expression = False
    elif freeze_level == 2:
        freeze = True
        freeze_expression = True
    else:
        raise Exception("Invalid freeze level")

    adata.obs['study'] = adata.obs[batch_key].values
    
    n_batches = len(adata.obs[batch_key].unique().tolist())
    n_cell_types = len(adata.obs[cell_type_key].unique().tolist())

    batch_key = 'study'

    batch_colors = palettes.vega_10[n_batches:2 * n_batches] 
    cell_type_colors = palettes.godsnot_64[:n_cell_types]

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=[batch_key], title="", wspace=0.7, frameon=False, palette=batch_colors,
               save="_latent_orig_condition.pdf")
    sc.pl.umap(adata, color=[cell_type_key], title="", wspace=0.7, frameon=False, palette=cell_type_colors,
               save="_latent_orig_celltype.pdf")

    adata_for_training = adata[~adata.obs[batch_key].isin(target_conditions)]

    train_adata, valid_adata = scnet.utils.train_test_split(adata_for_training, 0.80)
    n_conditions = len(train_adata.obs[batch_key].unique().tolist())

    architecture = [128, 64, 32]
    z_dim = 10
    network = scnet.archs.CVAE(x_dimension=train_adata.shape[1],
                               z_dimension=z_dim,
                               architecture=architecture,
                               n_conditions=n_conditions,
                               lr=0.001,
                               alpha=0.001,
                               beta=100.0,
                               use_batchnorm=False,
                               eta=1.0,
                               clip_value=clip_value,
                               loss_fn=loss_fn,
                               model_path=f"./models/CVAE/iterative_surgery/MMD/before-{data_name}-{loss_fn}-{architecture}-{z_dim}/",
                               dropout_rate=0.1,
                               output_activation='relu')

    conditions = adata_for_training.obs[batch_key].unique().tolist()
    condition_encoder = scnet.utils.create_dictionary(conditions, [])

    network.train(train_adata,
                  valid_adata,
                  condition_key=batch_key,
                  cell_type_key=cell_type_key,
                  le=condition_encoder,
                  n_epochs=10000,
                  batch_size=1024,
                  early_stop_limit=50,
                  lr_reducer=40,
                  n_per_epoch=0,
                  save=True,
                  retrain=False,
                  verbose=2)

    encoder_labels, _ = scnet.utils.label_encoder(adata_for_training, label_encoder=network.condition_encoder,
                                                  condition_key=batch_key)

    latent_adata = network.to_latent(adata_for_training, encoder_labels)
    latent_adata.uns[f'{batch_key}_colors'] = adata_for_training.uns[f'{batch_key}_colors']
    latent_adata.uns[f'{cell_type_key}_colors'] = adata_for_training.uns[f'{cell_type_key}_colors']

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[batch_key], title="", wspace=0.7, frameon=False,
               save="_latent_first_condition.pdf")
    sc.pl.umap(latent_adata, color=[cell_type_key], title="", wspace=0.7, frameon=False,
               save="_latent_first_celltype.pdf")


    network.beta = 0.0
    network.eta = 0.1

    new_network = network
    adata_vis = adata_for_training
    for idx, new_batch in enumerate(target_conditions):
        print(f"Operating surgery for {new_batch}")
        batch_adata = adata[adata.obs[batch_key] == new_batch]

        new_network = scnet.operate(new_network,
                                    new_conditions=[new_batch],
                                    init='Xavier',
                                    freeze=freeze,
                                    freeze_expression_input=freeze_expression)

        new_network.model_path = f"./models/CVAE/iterative_surgery/MMD/after-({idx}:{new_batch})-{data_name}-{loss_fn}-{freeze}/"

        train_adata, valid_adata = scnet.utils.train_test_split(batch_adata, 0.80)

        new_network.train(train_adata,
                          valid_adata,
                          condition_key=batch_key,
                          cell_type_key=cell_type_key,
                          le=new_network.condition_encoder,
                          n_epochs=10000,
                          n_epochs_warmup=300 if not freeze else 0,
                          batch_size=512,
                          early_stop_limit=100,
                          lr_reducer=80,
                          n_per_epoch=0,
                          save=True,
                          retrain=False,
                          verbose=2)
        if not isinstance(adata_vis.uns[f'{batch_key}_colors'], list):
            prev_batch_colors = adata_vis.uns[f'{batch_key}_colors'].tolist()
            prev_cell_type_colors = adata_vis.uns[f'{cell_type_key}_colors'].tolist()
        else:
            prev_batch_colors = adata_vis.uns[f'{batch_key}_colors']
            prev_cell_type_colors = adata_vis.uns[f'{cell_type_key}_colors']
        adata_vis = adata_vis.concatenate(batch_adata)
        adata_vis.uns[f'{batch_key}_colors'] = prev_batch_colors + batch_adata.uns[f'{batch_key}_colors'].tolist()
        adata_vis.uns[f'{cell_type_key}_colors'] = prev_cell_type_colors + batch_adata.uns[f'{cell_type_key}_colors'].tolist()
        
        encoder_labels, _ = scnet.utils.label_encoder(adata_vis, label_encoder=new_network.condition_encoder,
                                                      condition_key=batch_key)

        latent_adata = new_network.to_latent(adata_vis, encoder_labels)
        latent_adata.uns[f'{batch_key}_colors'] = adata_vis.uns[f'{batch_key}_colors']
        latent_adata.uns[f'{cell_type_key}_colors'] = adata_vis.uns[f'{cell_type_key}_colors']

        sc.pp.neighbors(latent_adata)
        sc.tl.umap(latent_adata)
        sc.pl.umap(latent_adata, color=[batch_key], title="", wspace=0.7, frameon=False,
                   save=f"_latent_({idx}:{new_batch})_condition.pdf")
        sc.pl.umap(latent_adata, color=[cell_type_key], title="", wspace=0.7, frameon=False,
                   save=f"_latent_({idx}:{new_batch})_celltype.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scNet')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='data name')
    arguments_group.add_argument('-f', '--freeze_level', type=int, default=1, required=True,
                                 help='freeze')
    arguments_group.add_argument('-c', '--count', type=int, default=0, required=False,
                                 help='latent space dimension')
    args = vars(parser.parse_args())

    freeze_level = args['freeze_level']
    loss_fn = 'nb' if args['count'] > 0 else 'mse'
    data_dict = DATASETS[args['data']]

    train_and_evaluate(data_dict=data_dict, freeze_level=freeze_level, loss_fn=loss_fn)
