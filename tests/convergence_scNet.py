import argparse
import os

import scanpy as sc

import surgeon


def train_and_evaluate(data_name, freeze=True, count_adata=True):
    path_to_save = f"./results/convergence/{data_name}/"
    sc.settings.figdir = path_to_save
    if data_name == "toy":
        condition_key = "batch"
        cell_type_key = "celltype"
        target_conditions = ["Batch8", "Batch9"]
    else:
        condition_key = "study"
        cell_type_key = "cell_type"
        target_conditions = ["Pancreas Celseq", "Pancreas CelSeq2"]

    os.makedirs(path_to_save, exist_ok=True)

    if count_adata:
        adata = sc.read(f"./data/{data_name}/{data_name}_count.h5ad")
        loss_fn = "nb"
    else:
        adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")
        loss_fn = "mse"

    adata_out_of_sample = adata[adata.obs[condition_key].isin(target_conditions)]
    adata_for_training = adata[~adata.obs[condition_key].isin(target_conditions)]

    if count_adata:
        adata_for_training = surgeon.utils.normalize(adata_for_training,
                                                     filter_min_counts=False,
                                                     normalize_input=False,
                                                     size_factors=True,
                                                     logtrans_input=True,
                                                     n_top_genes=5000,
                                                     )

        adata_out_of_sample = surgeon.utils.normalize(adata_out_of_sample,
                                                      filter_min_counts=False,
                                                      normalize_input=False,
                                                      size_factors=True,
                                                      logtrans_input=True,
                                                      n_top_genes=5000,
                                                      )
        clip_value = 5.0
    else:
        clip_value = 1e6

    scores = []

    train_adata, valid_adata = surgeon.utils.train_test_split(adata_for_training, 0.85)
    n_conditions = len(train_adata.obs[condition_key].unique().tolist())

    network = surgeon.archs.CVAE(x_dimension=train_adata.shape[1],
                                 z_dimension=20,
                                 n_conditions=n_conditions,
                                 lr=0.001,
                                 alpha=0.001,
                                 eta=1.0,
                                 clip_value=clip_value,
                                 loss_fn=loss_fn,
                                 model_path=f"./models/CVAE/Convergence/before-{data_name}-{loss_fn}/",
                                 dropout_rate=0.2,
                                 output_activation='relu')

    conditions = adata_for_training.obs[condition_key].unique().tolist()
    condition_encoder = surgeon.utils.create_dictionary(conditions, target_conditions)

    network.train(train_adata,
                  valid_adata,
                  condition_key=condition_key,
                  cell_type_key=cell_type_key,
                  le=condition_encoder,
                  n_epochs=10000,
                  batch_size=32,
                  early_stop_limit=50,
                  lr_reducer=40,
                  n_per_epoch=0,
                  save=True,
                  retrain=False,
                  verbose=2)

    encoder_labels, _ = surgeon.utils.label_encoder(adata_for_training, label_encoder=network.condition_encoder,
                                                    condition_key=condition_key)

    latent_adata = network.to_latent(adata_for_training, encoder_labels)

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[condition_key, cell_type_key], wspace=0.7,
               save="_latent_training.pdf")

    new_network = surgeon.operate(network,
                                  new_conditions=target_conditions,
                                  init='Xavier',
                                  freeze=freeze)

    new_network.model_path = f"./models/CVAE/Convergence/after-{data_name}-{loss_fn}/"

    train_adata, valid_adata = surgeon.utils.train_test_split(adata_out_of_sample, 0.85)

    filename = path_to_save + "scores_scNet"
    filename += "Freezed" if freeze else "UnFreezed"
    filename += "_count.log" if count_adata else "_normalized.log"

    new_network.train(train_adata,
                      valid_adata,
                      condition_key=condition_key,
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

    encoder_labels, _ = surgeon.utils.label_encoder(adata_out_of_sample, label_encoder=new_network.condition_encoder,
                                                    condition_key=condition_key)

    latent_adata = new_network.to_latent(adata_out_of_sample, encoder_labels)

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=[condition_key, cell_type_key], wspace=0.7,
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

    data_name = args['data']
    freeze = True if args['freeze'] > 0 else False
    count_adata = True if args['count'] > 0 else False

    train_and_evaluate(data_name=data_name, freeze=freeze, count_adata=count_adata)
