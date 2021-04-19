import scanpy as sc
import os
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt

n_epochs_vae = 500
early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
batch_key = "study"
cell_type_key = "cell_type"

adata_all = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/pancreas_normalized.h5ad'))
adata = adata_all.raw.to_adata()
adata = remove_sparsity(adata)
adata_conditions = adata.obs[batch_key].tolist()

trvae = sca.models.TRVAE(
    adata=adata,
    condition_key=batch_key,
    hidden_layer_sizes=[128,128],
    use_mmd=False
)

trvae.train(
    n_epochs=n_epochs_vae,
    alpha_epoch_anneal=200,
    early_stopping_kwargs=early_stopping_kwargs
)

adata_latent = sc.AnnData(trvae.get_latent())
adata_latent.obs['celltype'] = adata.obs[cell_type_key].tolist()
adata_latent.obs['batch'] = adata.obs[batch_key].tolist()

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)

sc.pl.umap(adata_latent,
           color=['batch', 'celltype'],
           frameon=False,
           wspace=0.6,
           )
