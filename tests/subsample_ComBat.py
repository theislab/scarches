import sys
sys.path.append("../")

import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import entropy, itemfreq
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score
import os
import argparse
from sklearn.preprocessing import LabelEncoder
from scnet.utils import remove_sparsity
import numpy as np
import csv



def clustering_scores(labels, newX, batch_ind):
    n_labels = labels.nunique()
    labels_pred = KMeans(n_labels, n_init=200).fit_predict(newX)
    asw_score = silhouette_score(newX, batch_ind)
    nmi_score = NMI(labels, labels_pred)
    ari_score = ARI(labels, labels_pred)
        
    return asw_score, nmi_score, ari_score   


def entropy_batch_mixing(latent, labels, n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    
    def entropy_from_indices(indices):
        return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)))

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = neighbors.kneighbors(latent, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: labels[i])(indices)

    entropies = np.apply_along_axis(entropy_from_indices, axis=1, arr=batch_indices)

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])    
    
    return score



DATASETS = {
    "pancreas": {"name": "pancreas", "batch_key": "study", "cell_type_key": "cell_type", "target": ["Pancreas Celseq", "Pancreas CelSeq2"], "HV": True},
    "pbmc": {"name": "pbmc", "batch_key": "study", "cell_type_key": "cell_type", "target": ["Drop-seq", "inDrops"], "HV": False},
    "toy": {"name": "toy", "batch_key": "batch", "cell_type_key": "celltype", "target": ["Batch8", "Batch9"], "HV": True},
}

parser = argparse.ArgumentParser(description='scNet')
arguments_group = parser.add_argument_group("Parameters")
arguments_group.add_argument('-d', '--data', type=str, required=True,
                             help='data name')
args = vars(parser.parse_args())

data_dict = DATASETS[args['data']]
data_name = data_dict['name']
batch_key = data_dict['batch_key']
cell_type_key = data_dict['cell_type_key']
target_batches = data_dict['target']
highly_variable = data_dict['HV']

adata = sc.read(f"./data/{data_name}/{data_name}_count.h5ad")
adata = remove_sparsity(adata)

adata.obs['cell_types'] = adata.obs[cell_type_key]

le = LabelEncoder()
adata.obs['labels'] = le.fit_transform(adata.obs[cell_type_key])

le = LabelEncoder()
adata.obs['batch_indices'] = le.fit_transform(adata.obs[batch_key])



for i in range(5):
    for subsample_frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        final_adata = None
        for target in target_batches:
            adata_sampled = adata[adata.obs[batch_key] == target, :]
            keep_idx = np.loadtxt(f'./data/subsample/{data_name}/{target}/{subsample_frac}/{i}.csv', dtype='int32')
            adata_sampled = adata_sampled[keep_idx, :]

            if final_adata is None:
                final_adata = adata_sampled
            else:
                final_adata = final_adata.concatenate(adata_sampled)
        
        le = LabelEncoder()
        final_adata.obs['labels'] = le.fit_transform(final_adata.obs["cell_types"])
        
        os.makedirs(f"./results/ComBat/{data_name}/", exist_ok=True)
        row = ["ASW", "NMI" , "ARI" , "EBM"]
        with open(f"./results/ComBat/{data_name}/{subsample_frac}-{i}.csv", 'w+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        
        sc.pp.combat(final_adata, key = "batch_indices")
        
        asw_score, nmi_score, ari_score = clustering_scores(final_adata.obs['labels'], final_adata.X, final_adata.obs['batch_indices'])
        ebm_score = entropy_batch_mixing(final_adata.X, final_adata.obs['batch_indices'])
        
        row = [asw_score, nmi_score, ari_score, ebm_score]
        with open(f"./results/ComBat/{data_name}/{subsample_frac}-{i}.csv", 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
