import gdown
import scanpy as sc
import scarches as sca
import matplotlib.pyplot as plt
import os


hvg = 2000
query_condition = ["Freytag"]
batch_key = "batch"
cell_type_key = "final_annotation"
n_epochs_vae = 500

ref_path = os.path.expanduser(f'~/Documents/trvae_surg_with_unseen_genes/hvg_{hvg}/')
ref_model_path = f'{ref_path}ref_model/'
if not os.path.exists(ref_path):
    os.makedirs(ref_path)
query_path = os.path.expanduser(f'~/Documents/trvae_surg_with_unseen_genes/hvg_{hvg}/surg/')
query_model_path = f'{query_path}surgery_model/'
if not os.path.exists(query_path):
    os.makedirs(query_path)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

url = 'https://drive.google.com/uc?id=1Vh6RpYkusbGIZQC8GMFe3OKVDk5PWEpC'
output = 'pbmc.h5ad'
gdown.download(url, output, quiet=False)

# Show properties of Adata
adata = sc.read('pbmc.h5ad') 
adata.X = adata.layers["counts"].copy()
adata = adata[~adata.obs.study.isin(["Villani"])].copy()
studies = adata.obs.study.unique().tolist()
for study in studies:
    print(study,
          len(adata[adata.obs.study.isin([study])]),
          len(adata[adata.obs.study.isin([study])].obs.final_annotation.unique().tolist())
          )
print(adata)

# Divide into reference and query
source_adata = adata[~adata.obs.study.isin(query_condition)].copy()
target_adata = adata[adata.obs.study.isin(query_condition)].copy()

# Filter for HVG in reference
source_adata.raw = source_adata.copy()
sc.pp.normalize_total(source_adata)
sc.pp.log1p(source_adata)
sc.pp.highly_variable_genes(
    source_adata,
    n_top_genes=hvg,
    batch_key="batch",
    subset=True)
source_adata.X = source_adata.raw[:, source_adata.var_names].X

# Filter for HVG in query
target_adata.raw = target_adata.copy()
sc.pp.normalize_total(target_adata)
sc.pp.log1p(target_adata)
sc.pp.highly_variable_genes(
    target_adata,
    n_top_genes=hvg,
    batch_key="batch",
    subset=True)
target_adata.X = target_adata.raw[:, target_adata.var_names].X

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

trvae = sca.models.TRVAE(
    adata=source_adata,
    condition_key=batch_key,
    hidden_layer_sizes=[128,128],
    use_mmd=False
)

trvae.train(
    n_epochs=n_epochs_vae,
    alpha_epoch_anneal=200,
)
reference_latent = sc.AnnData(trvae.get_latent())
reference_latent.obs['celltype'] = source_adata.obs[cell_type_key].tolist()
reference_latent.obs['batch'] = source_adata.obs[batch_key].tolist()

sc.pp.neighbors(reference_latent, n_neighbors=8)
sc.tl.leiden(reference_latent)
sc.tl.umap(reference_latent)

plt.figure()
sc.pl.umap(
    reference_latent,
    color=['batch', 'celltype'],
    frameon=False,
    wspace=0.6,
    show=False
)
plt.savefig(f'{ref_path}umap_reference.png', bbox_inches='tight')
trvae.save(ref_model_path, overwrite=True)
reference_latent.write_h5ad(filename=f'{ref_path}latent_source_adata.h5ad')


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Train query model
new_trvae = sca.models.TRVAE.load_query_data(adata=target_adata, reference_model=ref_model_path)
new_trvae.train(
    n_epochs=500,
    alpha_epoch_anneal=200,
)
query_latent = sc.AnnData(new_trvae.get_latent())
query_latent.obs['celltype'] = target_adata.obs[cell_type_key].tolist()
query_latent.obs['batch'] = target_adata.obs[batch_key].tolist()

sc.pp.neighbors(query_latent, n_neighbors=8)
sc.tl.leiden(query_latent)
sc.tl.umap(query_latent)

plt.figure()
sc.pl.umap(
    query_latent,
    color=['batch', 'celltype'],
    frameon=False,
    wspace=0.6,
    show=False
)
plt.savefig(f'{query_path}umap_query.png', bbox_inches='tight')
new_trvae.save(query_model_path, overwrite=True)
query_latent.write_h5ad(filename=f'{query_path}latent_target_adata.h5ad')


# Show Full Representation
adata_full = source_adata.concatenate(new_trvae.adata, batch_key="ref_query")
full_latent = sc.AnnData(new_trvae.get_latent(adata_full.X, adata_full.obs[batch_key]))
full_latent.obs['celltype'] = adata_full.obs[cell_type_key].tolist()
full_latent.obs['batch'] = adata_full.obs[batch_key].tolist()

sc.pp.neighbors(full_latent)
sc.tl.leiden(full_latent)
sc.tl.umap(full_latent)
plt.figure()
sc.pl.umap(
    full_latent,
    color=["batch", "celltype"],
    frameon=False,
    wspace=0.6,
    show=False
)
plt.savefig(f'{query_path}umap_full.png', bbox_inches='tight')
full_latent.write_h5ad(filename=f'{query_path}latent_full_adata.h5ad')
