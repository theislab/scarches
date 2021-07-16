from scarches.dataset import trVAEDataset
import os
import scanpy as sc
import numpy as np

adata = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/benchmark_pancreas_shrinked.h5ad'))
cell_type_key = "cell_type"
condition_key = "study"
cell_types_ = adata.obs[cell_type_key].unique().tolist()
conditions = adata.obs[condition_key].unique().tolist()
cell_type_encoder = {k: v for k, v in zip(cell_types_, range(len(cell_types_)))}
condition_encoder = {k: v for k, v in zip(conditions, range(len(conditions)))}
size_factors = adata.X.sum(1)
if len(size_factors.shape) < 2:
    size_factors = np.expand_dims(size_factors, axis=1)
adata.obs['trvae_size_factors'] = size_factors
labeled_array = np.ones((len(adata), 1))
adata.obs['trvae_labeled'] = labeled_array

condition_counts = adata.obs[condition_key].value_counts().to_frame()
condition_counts.columns = ['count']
condition_counts['weight'] = 1 / condition_counts['count']
condition_counts['weight'] = condition_counts['weight'] / condition_counts['weight'].sum()
condition_weights = condition_counts['weight'].to_dict()

dataset = trVAEDataset(
    adata,
    condition_key,
    condition_encoder,
    [cell_type_key],
    cell_type_encoder,
    condition_weights,
    condition_key
)
print(dataset)
