import os
import argparse
import scnet
import scanpy as sc
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

DATASETS = {
    "panorama": {"name": "panorama", "batch_key": "study", "cell_type_key": "cell_type", 
                 "target": ["Pancreas SS2", "PBMC 68K", "Macrophage Mixed", "Jurkat"]},
}


def train_scNet(data_dict, freeze_level, loss_fn):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    batch_key = data_dict['batch_key']
    target_conditions = data_dict['target']

    if loss_fn == 'nb':
        clip_value = 3.0
    else:
        clip_value = 1000

    path_to_save = f"./results/classification/{data_name}/freeze_level={freeze_level}/"
    sc.settings.figdir = path_to_save
    os.makedirs(path_to_save, exist_ok=True)

    adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")

    adata.obs['study'] = adata.obs[batch_key].values
    batch_key = 'study'

    adata_out_of_sample = adata[adata.obs[batch_key].isin(target_conditions)]
    adata_for_training = adata[~adata.obs[batch_key].isin(target_conditions)]

    sc.pp.neighbors(adata_out_of_sample)
    sc.tl.umap(adata_out_of_sample)
    sc.pl.umap(adata_out_of_sample, color=batch_key, frameon=False, title="",
               save="_original_condition.pdf")
    sc.pl.umap(adata_out_of_sample, color=cell_type_key, frameon=False, title="", palette=sc.pl.palettes.godsnot_64,
               save="_original_celltype.pdf")


    train_adata, valid_adata = scnet.utils.train_test_split(adata_for_training, 0.80)
    n_conditions = len(train_adata.obs[batch_key].unique().tolist())

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
        raise Exception("Invalid Freeze Level")

    architecture = [128]
    z_dim = 10
    network = scnet.archs.CVAE(x_dimension=train_adata.shape[1],
                               z_dimension=z_dim,
                               architecture=architecture,
                               n_conditions=n_conditions,
                               lr=0.001,
                               alpha=0.001,
                               beta=1.0,
                               use_batchnorm=True,
                               eta=1.0,
                               scale_factor=1.0,
                               clip_value=clip_value,
                               loss_fn=loss_fn,
                               model_path=f"./models/CVAE/classification/MMD/before-{data_name}-{loss_fn}-{architecture}-{z_dim}/",
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
                  batch_size=512,
                  early_stop_limit=100,
                  lr_reducer=80,
                  n_per_epoch=0,
                  save=True,
                  retrain=False,
                  verbose=2)

    encoder_labels, _ = scnet.utils.label_encoder(adata_for_training, label_encoder=network.condition_encoder,
                                                  condition_key=batch_key)

    latent_adata = network.to_latent(adata_for_training, encoder_labels)

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[batch_key], wspace=0.7, frameon=False, title="",
               save="_latent_training_condition.pdf")
    sc.pl.umap(latent_adata, color=[cell_type_key], wspace=0.7, frameon=False, title="", palette=sc.pl.palettes.godsnot_64,
               save="_latent_training_celltype.pdf")

    new_network = scnet.operate(network,
                                target_conditions,
                                freeze=freeze,
                                freeze_expression_input=freeze_expression_input,
                                remove_dropout=True,
                                )

    new_network.model_path = f"./models/CVAE/classification/MMD/after-{data_name}-{loss_fn}-{architecture}-{z_dim}-{freeze}/"
    train_adata, valid_adata = scnet.utils.train_test_split(adata_out_of_sample, 0.80)

    new_network.train(train_adata,
                      valid_adata,
                      condition_key=batch_key,
                      cell_type_key=cell_type_key,
                      le=new_network.condition_encoder,
                      n_epochs=10000,
                      batch_size=512,
                      n_epochs_warmup=0,
                      early_stop_limit=50,
                      lr_reducer=40,
                      n_per_epoch=0,
                      save=True,
                      retrain=True,
                      verbose=2)

    encoder_labels, _ = scnet.utils.label_encoder(
        adata, label_encoder=network.condition_encoder, condition_key=batch_key)

    latent_adata = new_network.to_latent(adata_out_of_sample, encoder_labels)
    
    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[batch_key], wspace=0.7, frameon=False, title="",
               save="_latent_beforeTF_condition.pdf")
    sc.pl.umap(latent_adata, color=[cell_type_key], wspace=0.7, frameon=False, title="", palette=sc.pl.palettes.godsnot_64,
               save="_latent_beforeTF_celltype.pdf")

    classifier_network = scnet.archs.NNClassifier(x_dimension=adata.shape[1],
                                                  z_dimension=network.z_dim,
                                                  cvae_network=new_network,
                                                  n_labels=len(adata.obs[cell_type_key].unique().tolist()),
                                                  use_batchnorm=True,
                                                  model_path=f"./models/classification/MMD/classifier-{data_name}-{architecture}-{z_dim}/",
                                                  dropout_rate=0.1,
                                                  )

    train_adata, valid_adata = scnet.utils.train_test_split(adata_out_of_sample, 0.80)
    classifier_network.train(train_adata,
                             valid_adata,
                             cell_type_key,
                             n_epochs=10000,
                             batch_size=128,
                             early_stop_limit=50,
                             lr_reducer=40,
                             save=True,
                             verbose=2)

    predictions = classifier_network.predict(adata_out_of_sample)
    adata_out_of_sample.obs['predicted_TF'] = predictions

    classifier_network = scnet.archs.NNClassifier(x_dimension=adata.shape[1],
                                                  z_dimension=new_network.z_dim,
                                                  cvae_network=None,
                                                  n_labels=len(adata.obs[cell_type_key].unique().tolist()),
                                                  use_batchnorm=True,
                                                  model_path=f"./models/classification/after-{data_name}-{architecture}-{z_dim}/",
                                                  dropout_rate=0.1,
                                                  )

    train_adata, valid_adata = scnet.utils.train_test_split(adata_out_of_sample, 0.80)
    classifier_network.train(train_adata,
                             valid_adata,
                             cell_type_key,
                             n_epochs=10000,
                             batch_size=1024,
                             early_stop_limit=50,
                             lr_reducer=40,
                             save=True,
                             verbose=2)

    predictions = classifier_network.predict(adata_out_of_sample)
    adata_out_of_sample.obs['predicted_scratch'] = predictions

    knn_classifier = KNeighborsClassifier(n_neighbors=len(adata_out_of_sample.obs[cell_type_key].unique().tolist()))
    knn_classifier.fit(latent_adata.X, adata_out_of_sample.obs[cell_type_key].values)

    knn_predictions = knn_classifier.predict(latent_adata.X)
    adata_out_of_sample.obs['predicted_KNN'] = knn_predictions
    
    os.makedirs("./results/classification/", exist_ok=True)
    adata_out_of_sample.write_h5ad(f"./results/classification/predictions-{freeze_level}-{loss_fn}.h5ad")



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
    data_dict = DATASETS["panorama"]

    train_scNet(data_dict=data_dict, freeze_level=freeze_level, loss_fn=loss_fn)
