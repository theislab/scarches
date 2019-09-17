import argparse
import os

import numpy as np
import scanpy as sc

import surgeon

DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type",
                 "target": ["Pancreas SS2", "Pancreas CelSeq2"]},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"]},
    "pbmc": {"name": "pbmc_subset", "batch_key": "study", "cell_type_key": "cell_type",
             "target": ["inDrops", "Drop-seq"]},
    "mouse_brain": {"name": "mouse_brain_subset", "batch_key": "study", "cell_type_key": "cell_type",
                    "target": ["Rosenberg", "Zeisel"]},
}


def train_and_evaluate(data_dict, loss_fn="mse"):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    condition_key = data_dict['batch_key']
    target_conditions = data_dict['target']

    adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")

    path_to_save = f"./results/subsample/{data_name}/{loss_fn}/"
    os.makedirs(path_to_save, exist_ok=True)

    if loss_fn == "mse":
        clip_value = 1e6
    else:
        clip_value = 3.0



    for i in range(5):
        scores = []
        for subsample_frac in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]:
            final_adata, raw_out_of_sample = None, None
            for condition in target_conditions:
                condition_adata = adata[adata.obs[condition_key] == condition]
                keep_idx = np.loadtxt(f'./data/subsample/{data_name}/{condition}/{subsample_frac}/{i}.csv',
                                      dtype='int32')
                condition_adata_subsampled = condition_adata[keep_idx, :]
                final_adata = condition_adata_subsampled if final_adata is None \
                    else final_adata.concatenate(condition_adata_subsampled)
                raw_out_of_sample = sc.AnnData(
                    condition_adata_subsampled.raw.X) if raw_out_of_sample is None else raw_out_of_sample.concatenate(
                    sc.AnnData(condition_adata_subsampled.raw.X))
            final_adata.raw = raw_out_of_sample
            
            
            
            train_adata, valid_adata = surgeon.utils.train_test_split(final_adata, 0.80)
            n_conditions = len(train_adata.obs[condition_key].unique().tolist())

            z_dim = 10
            architecture = [128]

            network = surgeon.archs.CVAE(x_dimension=train_adata.shape[1],
                                         z_dimension=z_dim,
                                         architecture=architecture,
                                         use_batchnorm=True,
                                         n_conditions=n_conditions,
                                         lr=0.001,
                                         alpha=0.00001,
                                         scale_factor=1.0,
                                         clip_value=clip_value,
                                         loss_fn=loss_fn,
                                         model_path=f"./models/CVAE/subsample/{data_name}/{loss_fn}/{i}-{subsample_frac}-{architecture}-{z_dim}/",
                                         dropout_rate=0.2,
                                         output_activation='relu')

            conditions = final_adata.obs[condition_key].unique().tolist()
            condition_encoder = surgeon.utils.create_dictionary(conditions, target_conditions)

            network.train(train_adata,
                          valid_adata,
                          condition_key=condition_key,
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

            encoder_labels, _ = surgeon.utils.label_encoder(final_adata, label_encoder=network.condition_encoder, condition_key=condition_key)
            latent_adata = network.to_latent(final_adata, encoder_labels)


            ebm = surgeon.metrics.entropy_batch_mixing(latent_adata, label_key=condition_key, n_pools=1)
            asw = surgeon.metrics.asw(latent_adata, label_key=condition_key)
            ari = surgeon.metrics.ari(latent_adata, label_key=cell_type_key)
            nmi = surgeon.metrics.nmi(latent_adata, label_key=cell_type_key)
            knn_purity = surgeon.metrics.knn_purity(final_adata, label_key=cell_type_key)

            scores.append([subsample_frac, ebm, asw, ari, nmi, knn_purity])
            print([subsample_frac, ebm, asw, ari, nmi])

        scores = np.array(scores)

        filename = "scores_scNet"
        filename += loss_fn
        filename += f"_{i}.log"

        np.savetxt(os.path.join(path_to_save, filename), X=scores, delimiter=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scNet')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='data name')
    arguments_group.add_argument('-l', '--loss', type=int, default=0, required=False,
                                 help='if 1 will use nb else will use mse')
    args = vars(parser.parse_args())

    data_name = args['data']
    loss_fn = "nb" if args['loss'] > 0 else "mse"
    data_dict = DATASETS[data_name]

    train_and_evaluate(data_dict=data_dict, loss_fn=loss_fn)
