import argparse
import os

import numpy as np
import scanpy as sc

from keras import backend as K
import surgeon

DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type",
                 "target": ["Pancreas SS2", "Pancreas CelSeq2"]},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"]},
    "pbmc": {"name": "pbmc", "batch_key": "study", "cell_type_key": "cell_type",
             "target": ["inDrops", "Drop-seq"]},
    "mouse_brain": {"name": "mouse_brain", "batch_key": "study", "cell_type_key": "cell_type",
                    "target": ["Tabula_muris", "Zeisel"]},
}


def train_and_evaluate(data_dict, freeze_level=0, loss_fn='nb'):
    data_name = data_dict['name']
    cell_type_key = data_dict['cell_type_key']
    condition_key = data_dict['batch_key']
    target_conditions = data_dict['target']

    path_to_save = f"./results/subsample/{data_name}/"
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

    adata_out_of_sample = adata[adata.obs[condition_key].isin(target_conditions)]
    adata_for_training = adata[~adata.obs[condition_key].isin(target_conditions)]

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

            train_adata, valid_adata = surgeon.utils.train_test_split(adata_for_training, 0.80)
            n_conditions = len(train_adata.obs[condition_key].unique().tolist())

            z_dim = 10
            architecture = [128, 64, 32]

            network = surgeon.archs.CVAE(x_dimension=train_adata.shape[1],
                                         z_dimension=z_dim,
                                         architecture=architecture,
                                         use_batchnorm=False,
                                         n_conditions=n_conditions,
                                         lr=0.001,
                                         alpha=0.00005,
                                         beta=1000.0,
                                         eta=1.0,
                                         clip_value=clip_value,
                                         loss_fn=loss_fn,
                                         model_path=f"./models/CVAE/{data_name}/before/",
                                         dropout_rate=0.05,
                                         output_activation='relu')

            conditions = adata_for_training.obs[condition_key].unique().tolist()
            condition_encoder = surgeon.utils.create_dictionary(conditions, target_conditions)

            # network.restore_model()
            network.train(train_adata,
                          valid_adata,
                          condition_key=condition_key,
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

            new_network = surgeon.operate(network,
                                          new_conditions=target_conditions,
                                          init='Xavier',
                                          freeze=freeze,
                                          freeze_expression_input=freeze_expression,
                                          new_training_kwargs={"beta": 10000, "eta": 0.1})

            new_network.model_path = f"./models/CVAE/{data_name}/after-{subsample_frac}-{freeze_level}/"
            train_adata, valid_adata = surgeon.utils.train_test_split(adata_out_of_sample_subsampled, 0.80)

            new_network.train(train_adata,
                              valid_adata,
                              condition_key=condition_key,
                              cell_type_key=cell_type_key,
                              le=new_network.condition_encoder,
                              n_epochs=10000,
                              batch_size=1024,
                              n_epochs_warmup=0,
                              early_stop_limit=50,
                              lr_reducer=40,
                              n_per_epoch=-1,
                              score_filename=os.path.join(path_to_save, f"scores_scNet_freeze_level={freeze_level}_normalized_{i}.log"),
                              save=True,
                              retrain=True,
                              verbose=2)

            encoder_labels, _ = surgeon.utils.label_encoder(
                adata_out_of_sample_subsampled, label_encoder=network.condition_encoder, condition_key=condition_key)

            latent_adata = new_network.to_mmd_layer(adata_out_of_sample_subsampled, encoder_labels, encoder_labels)

            # asw = surgeon.metrics.asw(latent_adata, label_key=condition_key)
            # ari = surgeon.metrics.ari(latent_adata, label_key=cell_type_key)
            # nmi = surgeon.metrics.nmi(latent_adata, label_key=cell_type_key)
            # knn_15 = surgeon.metrics.knn_purity(latent_adata, label_key=cell_type_key, n_neighbors=15)
            # knn_25 = surgeon.metrics.knn_purity(latent_adata, label_key=cell_type_key, n_neighbors=25)
            # knn_50 = surgeon.metrics.knn_purity(latent_adata, label_key=cell_type_key, n_neighbors=50)
            # knn_100 = surgeon.metrics.knn_purity(latent_adata, label_key=cell_type_key, n_neighbors=100)
            # knn_200 = surgeon.metrics.knn_purity(latent_adata, label_key=cell_type_key, n_neighbors=200)
            # knn_300 = surgeon.metrics.knn_purity(latent_adata, label_key=cell_type_key, n_neighbors=300)
            # ebm_15 = surgeon.metrics.entropy_batch_mixing(latent_adata, label_key=condition_key, n_pools=1,
            #                                               n_neighbors=15)
            # ebm_25 = surgeon.metrics.entropy_batch_mixing(latent_adata, label_key=condition_key, n_pools=1,
            #                                               n_neighbors=25)
            # ebm_50 = surgeon.metrics.entropy_batch_mixing(latent_adata, label_key=condition_key, n_pools=1,
            #                                               n_neighbors=50)
            # ebm_100 = surgeon.metrics.entropy_batch_mixing(latent_adata, label_key=condition_key, n_pools=1,
            #                                                n_neighbors=100)
            # ebm_200 = surgeon.metrics.entropy_batch_mixing(latent_adata, label_key=condition_key, n_pools=1,
            #                                                n_neighbors=200)
            # ebm_300 = surgeon.metrics.entropy_batch_mixing(latent_adata, label_key=condition_key, n_pools=1,
            #                                                n_neighbors=300)

            # scores.append(
            #     [subsample_frac, asw, ari, nmi, knn_15, knn_25, knn_50, knn_100, knn_200, knn_300, ebm_15, ebm_25,
            #      ebm_50, ebm_100, ebm_200, ebm_300])
            # print([subsample_frac, asw, ari, nmi, knn_15, knn_25, knn_50, knn_100, knn_200, knn_300, ebm_15, ebm_25,
            #      ebm_50, ebm_100, ebm_200, ebm_300])

            K.clear_session()

        # scores = np.array(scores)

        # filename = "scores_scNet"
        # filename += f"_freeze_level={freeze_level}"
        # filename += "_count" if loss_fn == 'nb' else "_normalized"
        # filename += f"_{i}_mmd.log"

        # np.savetxt(os.path.join(path_to_save, filename), X=scores, delimiter=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scNet')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='data name')
    arguments_group.add_argument('-f', '--freeze_level', type=int, default=1, required=True,
                                 help='if 1 will freeze the network after surgery')
    arguments_group.add_argument('-c', '--count', type=int, default=1, required=False,
                                 help='if 1 will use count adata')
    args = vars(parser.parse_args())

    data_name = args['data']
    freeze_level = args['freeze_level']
    loss_fn = 'nb' if args['count'] > 0 else 'mse'
    data_dict = DATASETS[data_name]

    if freeze_level < 0:
        for i in range(3):
            train_and_evaluate(data_dict=data_dict, freeze_level=i, loss_fn=loss_fn)
    else:
        train_and_evaluate(data_dict=data_dict, freeze_level=freeze_level, loss_fn=loss_fn)
