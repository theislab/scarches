import os
import time
from typing import TypeVar

import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.models import Model
from keras.utils import to_categorical
from scipy.stats import entropy, itemfreq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

list_or_str = TypeVar('list_or_str', list, str)


class ScoreCallback(Callback):
    def __init__(self,
                 filename: str,
                 data: np.ndarray,
                 batch_labels: np.ndarray,
                 celltype_labels: np.ndarray,
                 encoder_model: Model,
                 n_per_epoch: int = 5,
                 n_batch_labels: int = 0,
                 n_celltype_labels: int = 0,
                 clustering_scores: list_or_str = 'all'
                 ):
        super(ScoreCallback, self).__init__()
        self.X = data

        self.batch_labels = np.reshape(batch_labels, (-1,))
        self.batch_labels_onehot = to_categorical(self.batch_labels)

        self.celltype_labels = np.reshape(celltype_labels, (-1,))
        self.celltype_labels_onehot = to_categorical(self.celltype_labels)

        self.filename = filename
        self.encoder_model = encoder_model
        self.n_per_epoch = n_per_epoch

        self.n_batch_labels = n_batch_labels
        self.n_celltype_labels = n_celltype_labels

        self.clustering_scores = clustering_scores
        self.score_computers = {"asw": self.asw,
                                "ari": self.ari,
                                "nmi": self.nmi,
                                "ebm": self.entropy_of_batch_mixing}

        self.kmeans_batch = KMeans(self.n_batch_labels, n_init=200)
        self.kmeans_celltype = KMeans(self.n_celltype_labels, n_init=200)

    def on_train_begin(self, logs=None):
        self.scores = []
        self.epochs = []
        self.times = []

    def on_train_end(self, logs=None):
        scores_df = pd.DataFrame({"epoch": self.epochs, "time": self.times})

        self.scores_np = np.array(self.scores)
        if self.clustering_scores == 'all':
            self.clustering_scores = ['ASW', 'ARI', 'NMI', 'EBM']
        for i, clustering_score in enumerate(self.clustering_scores):
            computed_scores = self.scores_np[:, i]
            scores_df[clustering_score] = computed_scores
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        scores_df.to_csv(self.filename, index=False)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.n_per_epoch == 0:
            latent_X = self.encoder_model.predict([self.X, self.batch_labels_onehot])[2]

            self.epochs.append(epoch)
            last_time_record = self.times[-1] if len(self.times) > 0 else 0.0
            print(f"Epoch {epoch}: ", end="\t")
            if self.clustering_scores == 'all':
                asw, asw_time = self.asw(latent_X)
                ari, ari_time = self.ari(latent_X)
                nmi, nmi_time = self.nmi(latent_X)
                ebm, ebm_time = self.entropy_of_batch_mixing(latent_X)
                self.scores.append([asw, ari, nmi, ebm])
                print(f"ASW: {asw:.4f} - ARI: {ari:.4f} - NMI: {nmi:.4f} - EBM: {ebm:.4f}")
                computation_times = asw_time + ari_time + nmi_time + ebm_time
            else:
                scores = []
                computation_times = 0
                for clustering_score in self.clustering_scores:
                    score, computation_time = self.score_computers[clustering_score](latent_X)
                    scores.append(score)
                    computation_times += computation_time
                    print(f"{clustering_score}: {scores[-1]:.4f}", end=" - ")
                print()
                self.scores.append(scores)

            self.times.append(time.time() - self.epoch_time_start + last_time_record - computation_times)

    def asw(self, latent):
        start_time = time.time()
        score = silhouette_score(latent, self.batch_labels)
        end_time = time.time()
        return score, end_time - start_time

    def ari(self, latent):
        start_time = time.time()
        labels_pred = self.kmeans_celltype.fit_predict(latent)
        score = adjusted_rand_score(self.celltype_labels, labels_pred)
        end_time = time.time()
        return score, end_time - start_time

    def nmi(self, latent):
        start_time = time.time()
        labels_pred = self.kmeans_celltype.fit_predict(latent)
        score = normalized_mutual_info_score(self.celltype_labels, labels_pred)
        end_time = time.time()
        return score, end_time - start_time

    def entropy_of_batch_mixing(self, latent,
                                n_neighbors=50, n_pools=50, n_samples_per_pool=100, subsample_frac=1.0):
        start_time = time.time()

        def entropy_from_indices(indices):
            return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)))

        n_samples = latent.shape[0]
        keep_idx = np.random.choice(np.arange(n_samples), size=min(n_samples, int(subsample_frac * n_samples)),
                                    replace=False)
        latent = latent[keep_idx, :]

        neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
        indices = neighbors.kneighbors(latent, return_distance=False)[:, 1:]
        batch_indices = np.vectorize(lambda i: self.batch_labels[i])(indices)

        entropies = np.apply_along_axis(entropy_from_indices, axis=1, arr=batch_indices)

        # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
        if n_pools == 1:
            score = np.mean(entropies)
        else:
            score = np.mean([
                np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
                for _ in range(n_pools)
            ])
        end_time = time.time()
        return score, end_time - start_time
