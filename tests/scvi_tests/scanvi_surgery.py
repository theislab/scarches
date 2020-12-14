import scanpy as sc
import numpy as np
import json
import os
import torch
import scvi as scv
import scarches as sca
from scarches.plotting import SCVI_EVAL
from scarches.dataset.trvae.data_handling import remove_sparsity

data = 'pancreas'

# ------------------------------------------Parameters that can be adjusted------------------------------------------
use_scarches_version = True
n_query_labels = 0
n_epochs_surgery = 200
freeze = True
freeze_expression = True
show_plots = False

# ---------------------------------------------------DATA PREPROCESSING-------------------------------------------------
n_epochs_vae = 500
n_epochs_scanvi = 200
early_stopping_kwargs = {
    "early_stopping_metric": "elbo",
    "save_best_state_metric": "elbo",
    "patience": 10,
    "threshold": 0,
    "reduce_lr_on_plateau": True,
    "lr_patience": 8,
    "lr_factor": 0.1,
}
early_stopping_kwargs_scanvi = {
    "early_stopping_metric": "accuracy",
    "save_best_state_metric": "accuracy",
    "on": "full_dataset",
    "patience": 10,
    "threshold": 0.001,
    "reduce_lr_on_plateau": True,
    "lr_patience": 8,
    "lr_factor": 0.1,
}

arches_params = {
    "use_layer_norm": "both",
    "use_batch_norm": "none",
    "encode_covariates": use_scarches_version,
    "dropout_rate": 0.2,
    "n_layers": 2
}

condition_key = 'study'
cell_type_key = 'cell_type'
conditions_dict = dict()
cell_type_dict = dict()
results = dict()

if use_scarches_version:
    version = 'scarches'
else:
    version = 'base'
if freeze_expression:
    surgery_version = 'surgery_only_cond'
elif freeze:
    surgery_version = 'surgery_only_first'
else:
    surgery_version = 'surgery_full_retrain'

if n_query_labels == 0:
    query_label = 'unlabelled_query'
else:
    query_label = f'labelled_query_{n_query_labels}'
epoch_label = f'epochs_{n_epochs_surgery}'
save_path = os.path.expanduser(f'~/Documents/benchmark_results/scanvi_umaps/{data}/'
                               f'{version}/{query_label}/{surgery_version}_{epoch_label}/')
if not os.path.exists(save_path):
    os.makedirs(save_path)

if data == 'pancreas':
    target_batches = ["Pancreas SS2", "Pancreas CelSeq2"]
    batch_key = "study"
    cell_type_key = "cell_type"
    adata = sc.read(os.path.expanduser(f'~/Documents/datasets/pancreas_normalized.h5ad'))
elif data == 'mouse_brain':
    target_batches = ["Tabula_muris", "Zeisel"]
    batch_key = "study"
    cell_type_key = "cell_type"
    adata = sc.read(os.path.expanduser(f'~/Documents/datasets/mouse_brain_normalized.h5ad'))

adata.X = adata.raw.X
adata = remove_sparsity(adata)
celltypes = adata.obs[cell_type_key].unique().tolist()
conditions = adata.obs[condition_key].unique().tolist()

# Save Source Adata
source_adata = adata[~adata.obs[condition_key].isin(target_batches)].copy()
scv.data.setup_anndata(source_adata, batch_key=condition_key, labels_key=cell_type_key)
source_stats = source_adata.uns["_scvi"]["summary_stats"]
print('SOURCE ADATA...\n', source_adata)
print('SOURCE STATS...\n', source_stats)

# Save Target Adata
target_adata = adata[adata.obs[condition_key].isin(target_batches)].copy()
scv.data.setup_anndata(target_adata, batch_key=condition_key, labels_key=cell_type_key)
target_stats = target_adata.uns["_scvi"]["summary_stats"]
print('\nTARGET ADATA...\n', target_adata)
print('TARGET STATS...\n', target_stats)

# Train scVI Model on Reference Data
print("\nSCVI MODEL ARCH...")
scvi = sca.models.SCVI(source_adata, use_cuda=True, **arches_params)

trainer = sca.trainers.scVITrainer(
    scvi,
    train_size=0.9,
    use_cuda=True,
    frequency=1,
    silent=False,
    early_stopping_kwargs=early_stopping_kwargs
)

trainer.train(n_epochs_vae)

# Train scANVI Model on Query Data
print('\n', 20 * '###')
print("STARTING WITH SCANVI...")
scanvi = sca.models.SCANVI(
    source_adata,
    unlabeled_category='Unknown',
    classifier_parameters={'dropout_rate': 0.2, 'n_hidden': 10, 'n_layers': 1},
    pretrained_model=scvi,
    use_cuda=True,
    **arches_params
)

trainer_scanvi = sca.trainers.scANVITrainer(
    scanvi,
    n_labelled_samples_per_class=source_stats["n_cells"],
    train_size=0.9,
    use_cuda=True,
    frequency=1,
    silent=False,
    early_stopping_kwargs=early_stopping_kwargs_scanvi
)
trainer_scanvi.train(n_epochs=n_epochs_scanvi)
scanvi_eval = SCVI_EVAL(
    scanvi,
    source_adata,
    trainer_scanvi,
    cell_type_key=cell_type_key,
    batch_key=condition_key)

results["reference_ebm"] = scanvi_eval.get_ebm()
results["reference_knn"] = scanvi_eval.get_knn_purity()
results["reference_asw_b"], results["scanvi_asw_c"] = scanvi_eval.get_asw()
results["reference_nmi"] = scanvi_eval.get_nmi()
results["reference_acc"] = scanvi_eval.get_classification_accuracy()
results["reference_f1"] = scanvi_eval.get_f1_score()
scanvi_eval.plot_history(show=show_plots, save=True, dir_path=f'{save_path}reference_history')
scanvi_eval.plot_latent(show=show_plots, save=True, dir_path=f'{save_path}reference_latent')
scanvi_eval.post_adata.write_h5ad(filename=f'{save_path}reference_latent_data.h5ad')
torch.save(scanvi.state_dict(), f'{save_path}reference_model')

new_scanvi, new_trainer, target_gene_adata = sca.scvi_operate(
    scanvi,
    target_adata,
    n_epochs=n_epochs_surgery,
    early_stopping=False,
    labels_per_class=n_query_labels,
    freeze=freeze,
    freeze_expression=freeze_expression,
)
surgery_eval = SCVI_EVAL(
    new_scanvi,
    target_gene_adata,
    new_trainer,
    cell_type_key=cell_type_key,
    batch_key=condition_key
)
results["query_ebm"] = surgery_eval.get_ebm()
results["query_knn"] = surgery_eval.get_knn_purity()
results["query_asw_b"], results["surgery_asw_c"] = surgery_eval.get_asw()
results["query_nmi"] = surgery_eval.get_nmi()
results["query_acc"] = surgery_eval.get_classification_accuracy()
results["query_f1"] = surgery_eval.get_f1_score()
surgery_eval.plot_history(show=show_plots, save=True, dir_path=f'{save_path}surgery_history')
surgery_eval.plot_latent(show=show_plots, save=True, dir_path=f'{save_path}query_latent')
surgery_eval.post_adata.write_h5ad(filename=f'{save_path}query_latent_data.h5ad')
torch.save(new_scanvi.state_dict(), f'{save_path}surgery_model')

scv.data.setup_anndata(adata, batch_key=condition_key, labels_key=cell_type_key)

# Recreate Right Batch Encoding
source_conditions = source_adata.obs[condition_key].unique().tolist()
for condition in source_conditions:
    condition_label = source_adata[source_adata.obs[condition_key] == condition].obs['_scvi_batch'].unique().tolist()[0]
    conditions_dict[condition] = condition_label
target_conditions = target_gene_adata.obs[condition_key].unique().tolist()
for condition in target_conditions:
    condition_label = target_gene_adata[
        target_gene_adata.obs[condition_key] == condition].obs['_scvi_batch'].unique().tolist()[0]
    conditions_dict[condition] = condition_label
adata_conditions = adata.obs[condition_key].copy()
new_conditions = np.zeros_like(adata_conditions)
for idx in range(len(adata_conditions)):
    new_conditions[idx] = conditions_dict[adata_conditions[idx]]
adata.obs['_scvi_batch'] = new_conditions

full_eval = SCVI_EVAL(
    new_scanvi,
    adata,
    new_trainer,
    cell_type_key=cell_type_key,
    batch_key=condition_key
)
results["full_ebm"] = full_eval.get_ebm()
results["full_knn"] = full_eval.get_knn_purity()
results["full_asw_b"], results["surgery_asw_c"] = full_eval.get_asw()
results["full_nmi"] = full_eval.get_nmi()
results["full_acc"] = full_eval.get_classification_accuracy()
results["full_f1"] = full_eval.get_f1_score()
full_eval.plot_latent(show=show_plots, save=True, dir_path=f'{save_path}full_latent')
full_eval.post_adata.write_h5ad(filename=f'{save_path}full_latent_data.h5ad')

with open(f'{save_path}metrics.txt', 'w') as filehandle:
    json.dump(results, filehandle)
