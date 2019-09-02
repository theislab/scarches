import argparse
import os

import numpy as np
import scanpy as sc

import surgeon

DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type",
                 "target": ["Pancreas Celseq", "Pancreas CelSeq2"]},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"]},
}


def train_and_evaluate(data_dict, freeze=True, count_adata=True):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    condition_key = data_dict['batch_key']
    target_conditions = data_dict['target']

    path_to_save = f"./results/subsample/{data_name}/"
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

    for i in range(5):
        scores = []
        for subsample_frac in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]:
            adata_out_of_sample_subsampled, raw_out_of_sample = None, None
            for condition in target_conditions:
                condition_adata = adata_out_of_sample[adata_out_of_sample.obs[condition_key] == condition]
                keep_idx = np.loadtxt(f'./data/subsample/{data_name}/{condition}/{subsample_frac}/{i}.csv',
                                      dtype='int32')
                condition_adata_subsampled = condition_adata[keep_idx, :]
                adata_out_of_sample_subsampled = condition_adata_subsampled if adata_out_of_sample_subsampled is None \
                    else adata_out_of_sample_subsampled.concatenate(condition_adata_subsampled)
                raw_out_of_sample = sc.AnnData(condition_adata_subsampled.raw.X) if raw_out_of_sample is None else raw_out_of_sample.concatenate(sc.AnnData(condition_adata_subsampled.raw.X))
            adata_out_of_sample_subsampled.raw = raw_out_of_sample

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
                                         model_path=f"./models/CVAE/Subsample/before-{data_name}-{loss_fn}/",
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
                          early_stop_limit=20,
                          lr_reducer=15,
                          n_per_epoch=0,
                          save=True,
                          retrain=False,
                          verbose=2)

            new_network = surgeon.operate(network,
                                          new_conditions=target_conditions,
                                          init='Xavier',
                                          freeze=freeze)

            new_network.model_path = f"./models/CVAE/subsample/after-{data_name}-{loss_fn}-{subsample_frac}-{freeze}/"
            train_adata, valid_adata = surgeon.utils.train_test_split(adata_out_of_sample_subsampled, 0.85)

            new_network.train(train_adata,
                              valid_adata,
                              condition_key=condition_key,
                              cell_type_key=cell_type_key,
                              le=new_network.condition_encoder,
                              n_epochs=300,
                              batch_size=32,
                              early_stop_limit=40,
                              lr_reducer=35,
                              n_per_epoch=0,
                              save=True,
                              verbose=2)

            encoder_labels, _ = surgeon.utils.label_encoder(
                adata_out_of_sample_subsampled, label_encoder=network.condition_encoder, condition_key=condition_key)

            latent_adata = new_network.to_latent(adata_out_of_sample_subsampled, encoder_labels)

            ebm = surgeon.metrics.entropy_batch_mixing(latent_adata, label_key=condition_key, n_pools=1)
            asw = surgeon.metrics.asw(latent_adata, label_key=condition_key)
            ari = surgeon.metrics.ari(latent_adata, label_key=cell_type_key)
            nmi = surgeon.metrics.nmi(latent_adata, label_key=cell_type_key)

            scores.append([subsample_frac, ebm, asw, ari, nmi])
            print([subsample_frac, ebm, asw, ari, nmi])

        scores = np.array(scores)

        filename = "scores_scNet"
        filename += "Freezed" if freeze else "UnFreezed"
        filename += "_count" if count_adata else "_normalized"
        filename += f"_{i}.log"

        np.savetxt(os.path.join(path_to_save, filename), X=scores, delimiter=",")


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
    data_dict = DATASETS[data_name]

    train_and_evaluate(data_dict=data_dict, freeze=freeze, count_adata=count_adata)
