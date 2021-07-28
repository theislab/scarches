from scarches.trainers.trvae._utils import train_test_split
import os
import scanpy as sc
import numpy as np

adata = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_pancreas_shrinked.h5ad'))
indices = np.arange(adata.shape[0])
reference = ["inDrop1", "inDrop2", "inDrop3", "inDrop4", "fluidigmc1", "smartseq2", "smarter"]
query = ["celseq", "celseq2"]
cell_type_key = "cell_type"
condition_key = "study"

labeled_indices = indices[adata.obs.study.isin(reference)].tolist()
# labeled_indices = np.arange(adata.shape[0])

labeled_adata = adata[adata.obs.study.isin(reference)].copy()
unlabeled_adata = adata[adata.obs.study.isin(query)].copy()

# Preprare data for semisupervised learning
labeled_array = np.zeros((len(adata), 1))
if labeled_indices is not None:
    labeled_array[labeled_indices] = 1
adata.obs['trvae_labeled'] = labeled_array


cts = labeled_adata.obs[cell_type_key].unique().tolist()
conds = adata.obs[condition_key].unique().tolist()


ct_dict = dict()
cond_dict = dict()

labeled_indices = np.arange(labeled_adata.shape[0])
for ct in cts:
    ct_idx = labeled_indices[labeled_adata.obs[cell_type_key] == ct]
    ct_dict[ct] = len(ct_idx)
for cond in conds:
    cond_idx = indices[adata.obs[condition_key] == cond]
    cond_dict[cond] = len(cond_idx)

print(cond_dict)
print(ct_dict)

train_ad, val_ad = train_test_split(adata, train_frac=0.9, condition_key=None, cell_type_key=cell_type_key)

train_indices = np.arange(train_ad.shape[0])
labeled_train_indices = np.arange(train_ad[train_ad.obs['trvae_labeled'] == 1].shape[0])
val_indices = np.arange(val_ad.shape[0])

for ct in cts:
    ct_idx = labeled_train_indices[train_ad[train_ad.obs['trvae_labeled'] == 1].obs[cell_type_key] == ct]
    print(ct, len(ct_idx)/ct_dict[ct])

print("\n\n")
for cond in conds:
    cond_idx = train_indices[train_ad.obs[condition_key] == cond]
    print(cond, len(cond_idx)/cond_dict[cond])