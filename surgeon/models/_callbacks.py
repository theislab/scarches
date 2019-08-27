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
                 labels: np.ndarray,
                 encoder_model: Model,
                 n_per_epoch: int = 5,
                 n_labels: int = 0,
                 clustering_scores: list_or_str = 'all'
                 ):
        super(ScoreCallback, self).__init__()
        self.X = data
        self.labels = np.reshape(labels, (-1,))
        self.labels_onehot = to_categorical(self.labels)
        self.filename = filename
        self.encoder_model = encoder_model
        self.n_per_epoch = n_per_epoch
        self.n_labels = n_labels
        self.clustering_scores = clustering_scores
        self.score_computers = {"asw": self.asw,
                                "ari": self.ari,
                                "nmi": self.nmi,
                                "ebm": self.entropy_of_batch_mixing}

        self.kmeans = KMeans(n_labels, n_init=200)

    def on_train_begin(self, logs=None):
        self.scores = []
        self.epochs = []
        self.times = []

    def on_train_end(self, logs=None):
        scores_df = pd.DataFrame({"epoch": self.epochs, "time": self.times})

        self.scores_np = np.array(self.scores)
        for i, clustering_score in enumerate(self.clustering_scores):
            computed_scores = self.scores_np[:, i]
            scores_df[clustering_score] = computed_scores

        scores_df.to_csv(self.filename, index=False)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.n_per_epoch == 0:
            latent_X = self.encoder_model.predict([self.X, self.labels_onehot])[2]

            self.epochs.append(epoch)
            last_time_record = self.times[-1] if len(self.times) > 0 else 0.0
            self.times.append(time.time() - self.epoch_time_start + last_time_record)
            print(f"Epoch {epoch}: ", end="\t")
            if self.clustering_scores == 'all':
                asw = self.asw(latent_X)
                ari = self.ari(latent_X)
                nmi = self.nmi(latent_X)
                ebm = self.entropy_of_batch_mixing(latent_X)
                self.scores.append([asw, ari, nmi, ebm])
                print(f"ASW = {asw:.4f} - ARI = {ari:.4f} - NMI = {nmi:.4f} - EBM = {ebm:.4f}")
            else:
                scores = []
                for clustering_score in self.clustering_scores:
                    scores += [self.score_computers[clustering_score](latent_X)]
                    print(f"{clustering_score} = {scores[-1]:.4f}", end=" - ")
                print()
                self.scores.append(scores)

            print(f"Epoch {epoch}: {self.scores[-1]}")

    def asw(self, latent):
        return silhouette_score(latent, self.labels)

    def ari(self, latent):
        labels_pred = self.kmeans.fit_predict(latent)
        return adjusted_rand_score(self.labels, labels_pred)

    def nmi(self, latent):
        labels_pred = self.kmeans.fit_predict(latent)
        return normalized_mutual_info_score(self.labels, labels_pred)

    def entropy_of_batch_mixing(self, latent,
                                n_neighbors=50, n_pools=50, n_samples_per_pool=100, subsample_frac=1.0):

        def entropy_from_indices(indices):
            return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)))

        n_samples = latent.shape[0]
        keep_idx = np.random.choice(np.arange(n_samples), size=min(n_samples, int(subsample_frac * n_samples)),
                                    replace=False)
        latent = latent[keep_idx, :]

        neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
        indices = neighbors.kneighbors(latent, return_distance=False)[:, 1:]
        batch_indices = np.vectorize(lambda i: self.labels[i])(indices)

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
