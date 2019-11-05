import os
import argparse
import surgeon
import numpy as np
import scanpy as sc

DATASETS = {
    "pbmc": {"name": "pbmc_perturb", "cell_type_key": "cell_type", "dataset_key": "study", "condition_key": "condition", 
             "target_condition": "stimulated", "target_cell_type": "NK", "target_dataset": "Kang"
    },
}

def train_scNet(data_dict, freeze_level, loss_fn):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    dataset_key = data_dict['dataset_key']
    condition_key = data_dict['condition_key']
    target_condition = data_dict['target_condition']
    target_dataset = data_dict['target_dataset']
    target_cell_type = data_dict['target_cell_type']

    if loss_fn == 'nb':
        clip_value = 3.0
    else:
        clip_value = 1e6

    target_dataset = [target_dataset]

    path_to_save = f"./results/perturbation/{data_name}-{loss_fn}-freeze_level={freeze_level}/"
    sc.settings.figdir = path_to_save
    os.makedirs(path_to_save, exist_ok=True)

    adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")

    adata.obs['study'] = adata.obs[dataset_key].values
    dataset_key = 'study'

    adata_out_of_sample = adata[adata.obs[dataset_key].isin(target_dataset)]
    adata_for_training = adata[~adata.obs[dataset_key].isin(target_dataset)]

    train_adata, valid_adata = surgeon.utils.train_test_split(adata_for_training, 0.80)
    n_datasets = len(train_adata.obs[dataset_key].unique().tolist())
    n_conditions = len(train_adata.obs[condition_key].unique().tolist())

    architecture = [128]
    z_dim = 10
    network = surgeon.archs.CVAEFair(x_dimension=train_adata.shape[1],
                                     z_dimension=z_dim,
                                     architecture=architecture,
                                     n_conditions=n_datasets,
                                     n_cell_types=n_conditions,
                                     lr=0.001,
                                     alpha=0.00001,
                                     beta=0.0,
                                     use_batchnorm=True,
                                     eta=1.0,
                                     scale_factor=1.0,
                                     clip_value=clip_value,
                                     loss_fn=loss_fn,
                                     model_path=f"./models/CVAE/perturbation/before-{data_name}-{loss_fn}-{architecture}-{z_dim}/",
                                     dropout_rate=0.0,
                                     output_activation='relu')

    datasets = adata_for_training.obs[dataset_key].unique().tolist()
    conditions = adata_for_training.obs[condition_key].unique().tolist()
    dataset_encoder = surgeon.utils.create_dictionary(datasets, [])
    condition_encoder = surgeon.utils.create_dictionary(conditions, [])

    network.train(train_adata,
                  valid_adata,
                  condition_key=dataset_key,
                  cell_type_key=condition_key,
                  condition_encoder=dataset_encoder,
                  cell_type_encoder=condition_encoder,
                  n_epochs=10000,
                  batch_size=512,
                  early_stop_limit=100,
                  lr_reducer=80,
                  n_per_epoch=0,
                  save=True,
                  retrain=True,
                  verbose=2)

    encoder_conditions, _ = surgeon.utils.label_encoder(adata_for_training, label_encoder=network.condition_encoder,
                                                    condition_key=dataset_key)
    encoder_cell_types, _ = surgeon.utils.label_encoder(adata_for_training, label_encoder=network.cell_type_encoder,
                                                    condition_key=condition_key)

    latent_adata = network.to_latent(adata_for_training, encoder_conditions, encoder_cell_types)

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[dataset_key, cell_type_key, condition_key], wspace=0.7, frameon=False,
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
        raise Exception("Invalid freeze level.")


    new_network = surgeon.operate_fair(network, 
                                       new_conditions=target_dataset,
                                       new_cell_types=target_condition, 
                                       freeze=freeze, 
                                       freeze_expression_input=freeze_expression_input,
                                       remove_dropout=True,
                                       )

    new_network.model_path = f"./models/CVAE/perturbation/after-{data_name}-{loss_fn}-{architecture}-{z_dim}-{freeze}/"
    train_adata, valid_adata = surgeon.utils.train_test_split(adata_out_of_sample, 0.80)

    net_train_adata = train_adata[~((train_adata.obs[cell_type_key] == target_cell_type) & (train_adata.obs[condition_key] == target_condition))]
    net_valid_adata = valid_adata[~((valid_adata.obs[cell_type_key] == target_cell_type) & (valid_adata.obs[condition_key] == target_condition))]
    
    new_network.train(net_train_adata,
                      net_valid_adata,
                      condition_key=dataset_key,
                      cell_type_key=condition_key,
                      condition_encoder=new_network.condition_encoder,
                      cell_type_encoder=new_network.cell_type_encoder,
                      n_epochs=10000,
                      batch_size=512,
                      n_epochs_warmup=0,
                      early_stop_limit=100,
                      lr_reducer=80,
                      n_per_epoch=0,
                      save=True,
                      retrain=True,
                      verbose=2)

    cell_type_adata = adata_out_of_sample[adata_out_of_sample.obs[cell_type_key] == target_cell_type]

    source_adata = cell_type_adata[cell_type_adata.obs[condition_key] != target_condition]
    target_adata = cell_type_adata[cell_type_adata.obs[condition_key] == target_condition]
    
    encoder_conditions, _ = surgeon.utils.label_encoder(source_adata, label_encoder=new_network.condition_encoder,
                                                    condition_key=dataset_key)
    encoder_cell_types, _ = surgeon.utils.label_encoder(source_adata, label_encoder=new_network.cell_type_encoder,
                                                    condition_key=condition_key)
    
    decoder_conditions = encoder_conditions
    decoder_cell_types = np.zeros_like(encoder_conditions) + new_network.cell_type_encoder[target_condition]

    reconstructed_adata = new_network.predict(source_adata, encoder_conditions, decoder_cell_types, decoder_conditions, decoder_cell_types)

    reconstructed_adata.write_h5ad(f"./results/perturbation/reconstructed-{freeze_level}-{loss_fn}.h5ad")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scNet')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-f', '--freeze_level', type=int, default=1, required=True,
                                 help='if 1 will freeze the network after surgery')
    arguments_group.add_argument('-c', '--count', type=int, default=1, required=False,
                                 help='if 1 will use count adata')
    args = vars(parser.parse_args())

    freeze_level = args['freeze_level']
    loss_fn = 'nb' if args['count'] > 0 else 'mse'
    data_dict = DATASETS["pbmc"]

    train_scNet(data_dict=data_dict, freeze_level=freeze_level, loss_fn=loss_fn)
