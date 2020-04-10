from sklearn.metrics import silhouette_score
from scvi.inference import UnsupervisedTrainer, Trainer
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.dataset.anndataset import extract_data_from_anndata
import anndata
import numpy as np
import csv
import time
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import entropy, itemfreq
from sklearn.neighbors import NearestNeighbors
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score


class ADataset(GeneExpressionDataset):

    def __init__(self, ad: anndata.AnnData):
        super().__init__()
        (
            X,
            batch_indices,
            labels,
            gene_names,
            cell_types,
            self.obs,
            self.obsm,
            self.var,
            self.varm,
            self.uns,
        ) = extract_data_from_anndata(ad)
        self.populate_from_data(
            X=X,
            labels=labels,
            batch_indices=batch_indices,
            gene_names=gene_names,
            cell_types=cell_types,
        )
        self.filter_cells_by_count()


class scVI_Trainer(UnsupervisedTrainer):
    def __init__(self, model, gene_dataset, train_size=0.8, test_size=None, n_epochs_kl_warmup=400, **kwargs):
        super(UnsupervisedTrainer, self).__init__(model, gene_dataset, **kwargs)
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.train_set, self.test_set, self.validation_set = self.train_test_validation(
            model, gene_dataset, train_size, test_size
        )
        self.train_set.to_monitor = ["elbo"]
        self.test_set.to_monitor = ["elbo"]
        self.validation_set.to_monitor = ["elbo"]

        self.start_time = 0
        self.elapsed_time = 0
        self.writing_time = 0

        self.file_name = None
        self.ks = [15, 25, 50, 100, 200, 300]

        self.n_iter_kl_warmup = None
        self.n_samples = 1.0
        normalize_loss = None
        self.normalize_loss = (
            not (
                    hasattr(self.model, "reconstruction_loss")
                    and self.model.reconstruction_loss == "autozinb"
            )
            if normalize_loss is None
            else normalize_loss
        )

    def on_epoch_end(self):
        epoch = self.epoch
        if self.file_name is not None and self.frequency and (
                epoch == 0 or epoch == self.n_epochs - 1 or (epoch % self.frequency == 0)):
            begin = time.time()

            p = self.create_posterior(self.model, self.gene_dataset, indices=np.arange(len(self.gene_dataset)))
            clus = clustering_scores(p)
            if clus is not None:
                asw_score, nmi_score, ari_score = clus

                latent, batch_ind, labels = p.get_latent()
                ebm_scores, knn_scores = [], []

                for k in self.ks:
                    ebm_score = entropy_batch_mixing(latent, batch_ind, n_neighbors=k)
                    ebm_scores.append(ebm_score)

                    knn_score = knn_purity(latent, labels, n_neighbors=k)
                    knn_scores.append(knn_score)

                end = time.time()
                self.scores_time += end - begin
                self.elapsed_time = (time.time() - self.start_time) - (
                            self.compute_metrics_time + self.scores_time + self.writing_time)

                begin = time.time()

                row = [epoch, self.elapsed_time, asw_score, nmi_score, ari_score] + ebm_scores + knn_scores
                with open(self.file_name, 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()

                end = time.time()
                self.writing_time += (end - begin)

        return super().on_epoch_end()

    def train(self, file_name, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        self.file_name = file_name
        if self.file_name is not None:
            row = ["Epoch", "Elapsed Time", "ASW", "NMI", "ARI"] + [f"EBM_{k}" for k in self.ks] + [f'KNN_{k}' for k in
                                                                                                    self.ks]
            with open(self.file_name, 'w+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
        self.scores_time = 0
        self.start_time = time.time()
        super().train(n_epochs, lr, eps, params)


@torch.no_grad()
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


@torch.no_grad()
def knn_purity(latent, labels, n_neighbors=30):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = nbrs.kneighbors(latent, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: labels[i])(indices)

    # pre cell purity scores
    scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
    res = [
        np.mean(scores[labels == i]) for i in np.unique(labels)
    ]  # per cell-type purity

    return np.mean(res)


@torch.no_grad()
def clustering_scores(pos, prediction_algorithm="knn"):
    if pos.gene_dataset.n_labels > 1:
        latent, batch_ind, labels = pos.get_latent()
        if prediction_algorithm == "knn":
            labels_pred = KMeans(
                pos.gene_dataset.n_labels, n_init=200
            ).fit_predict(
                latent
            )  # n_jobs>1 ?
        elif prediction_algorithm == "gmm":
            gmm = GMM(pos.gene_dataset.n_labels)
            gmm.fit(latent)
            labels_pred = gmm.predict(latent)

        asw_score = silhouette_score(latent, batch_ind)
        nmi_score = NMI(labels, labels_pred)
        ari_score = ARI(labels, labels_pred)

        return asw_score, nmi_score, ari_score
