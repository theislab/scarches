import scanpy as sc
import os
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import numpy as np
import time
import matplotlib.pyplot as plt


n_epochs_surgery = 300
leave_out_cell_types = ['Pancreas Alpha']
target_batches = ["Pancreas SS2", "Pancreas CelSeq2"]
batch_key = "study"
cell_type_key = "cell_type"
n_epochs_vae = 500
early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
dir_path = os.path.expanduser(f'~/Documents/benchmarking_results/figure_1/trvae_mse/')

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
control_path = f'{dir_path}controlling/'
if not os.path.exists(control_path):
    os.makedirs(control_path)

adata_all = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/pancreas_normalized.h5ad'))
adata = adata_all.raw.to_adata()
adata = remove_sparsity(adata)
query = np.array([s in target_batches for s in adata.obs[batch_key]])
query_1 = np.array([s in [target_batches[0]] for s in adata.obs[batch_key]])
query_2 = np.array([s in [target_batches[1]] for s in adata.obs[batch_key]])
adata_ref_full = adata[~query].copy()
adata_ref = adata_ref_full[~adata_ref_full.obs[cell_type_key].isin(leave_out_cell_types)].copy()
adata_query_1 = adata[query_1].copy()
adata_query_2 = adata[query_2].copy()

trvae = sca.models.TRVAE(
    adata=adata_ref,
    condition_key=batch_key,
    hidden_layer_sizes=[128,128],
    recon_loss='nb',
)
ref_time = time.time()
trvae.train(
    n_epochs=n_epochs_vae,
    alpha_epoch_anneal=200,
    early_stopping_kwargs=early_stopping_kwargs
)
ref_time = time.time() - ref_time
'''
adata_latent = sc.AnnData(trvae.get_y())
adata_latent.obs['celltype'] = adata_ref.obs[cell_type_key].tolist()
adata_latent.obs['batch'] = adata_ref.obs[batch_key].tolist()
sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
adata_latent.write_h5ad(filename=f'{dir_path}reference_data_y.h5ad')

plt.figure()
sc.pl.umap(adata_latent,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(f'{control_path}umap_reference_y.png', bbox_inches='tight')
'''

adata_latent = sc.AnnData(trvae.get_latent())
adata_latent.obs['celltype'] = adata_ref.obs[cell_type_key].tolist()
adata_latent.obs['batch'] = adata_ref.obs[batch_key].tolist()
sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
adata_latent.write_h5ad(filename=f'{dir_path}reference_data_z.h5ad')

plt.figure()
sc.pl.umap(adata_latent,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(f'{control_path}umap_reference_z.png', bbox_inches='tight')

ref_path = f'{dir_path}ref_model/'
if not os.path.exists(ref_path):
    os.makedirs(ref_path)
trvae.save(ref_path, overwrite=True)

new_trvae = sca.models.TRVAE.load_query_data(adata=adata_query_1, reference_model=ref_path)
query_1_time = time.time()
new_trvae.train(
    n_epochs=n_epochs_surgery,
    alpha_epoch_anneal=200,
    early_stopping_kwargs=early_stopping_kwargs
)
query_1_time = time.time() - query_1_time
'''
adata_q1_y = sc.AnnData(trvae.get_y())
adata_q1_y.obs['celltype'] = adata_ref.obs[cell_type_key].tolist()
adata_q1_y.obs['batch'] = adata_ref.obs[batch_key].tolist()
sc.pp.neighbors(adata_q1_y, n_neighbors=8)
sc.tl.leiden(adata_q1_y)
sc.tl.umap(adata_q1_y)
adata_q1_y.write_h5ad(filename=f'{dir_path}query_data_y.h5ad')

plt.figure()
sc.pl.umap(adata_q1_y,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(f'{control_path}umap_query_data_y.png', bbox_inches='tight')
'''
adata_q1_z = sc.AnnData(new_trvae.get_latent())
adata_q1_z.obs['celltype'] = adata_query_1.obs[cell_type_key].tolist()
adata_q1_z.obs['batch'] = adata_query_1.obs[batch_key].tolist()
sc.pp.neighbors(adata_q1_z, n_neighbors=8)
sc.tl.leiden(adata_q1_z)
sc.tl.umap(adata_q1_z)
adata_q1_z.write_h5ad(filename=f'{dir_path}query_data_z.h5ad')

plt.figure()
sc.pl.umap(adata_q1_z,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(f'{control_path}umap_query_data_z.png', bbox_inches='tight')

adata_f1 = adata_ref.concatenate(adata_query_1)
'''
full_latent_y = sc.AnnData(new_trvae.get_y(adata_f1.X, adata_f1.obs[batch_key]))
full_latent_y.obs['celltype'] = adata_f1.obs[cell_type_key].tolist()
full_latent_y.obs['batch'] = adata_f1.obs[batch_key].tolist()

sc.pp.neighbors(full_latent_y)
sc.tl.leiden(full_latent_y)
sc.tl.umap(full_latent_y)
full_latent_y.write_h5ad(filename=f'{dir_path}full_1_data_y.h5ad')
plt.figure()
sc.pl.umap(full_latent_y,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(f'{control_path}umap_full_data_y.png', bbox_inches='tight')
'''

full_latent_z = sc.AnnData(new_trvae.get_latent(adata_f1.X, adata_f1.obs[batch_key]))
full_latent_z.obs['celltype'] = adata_f1.obs[cell_type_key].tolist()
full_latent_z.obs['batch'] = adata_f1.obs[batch_key].tolist()

sc.pp.neighbors(full_latent_z)
sc.tl.leiden(full_latent_z)
sc.tl.umap(full_latent_z)
full_latent_z.write_h5ad(filename=f'{dir_path}full_1_data_z.h5ad')
plt.figure()
sc.pl.umap(full_latent_z,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(f'{control_path}umap_full_data_z.png', bbox_inches='tight')