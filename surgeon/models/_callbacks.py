import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.models import Model
from keras.utils import to_categorical
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


class ScoreCallback(Callback):
    def __init__(self,
                 filename: str,
                 data: np.ndarray,
                 labels: np.ndarray,
                 encoder_model: Model,
                 n_per_epoch: int = 5,
                 n_labels: int = 0,
                 ):
        super(ScoreCallback, self).__init__()
        self.X = data
        self.labels = labels
        self.filename = filename
        self.encoder_model = encoder_model
        self.n_per_epoch = n_per_epoch
        self.n_labels = n_labels
        self.kmeans = KMeans(n_labels, n_init=200)

    def on_train_begin(self, logs=None):
        self.scores = []
        self.epochs = []

    def on_train_end(self, logs=None):
        scores_df = pd.DataFrame({"epoch": self.epochs, "score": self.scores})
        scores_df.to_csv(self.filename)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.n_per_epoch == 0:
            labels_onehot = to_categorical(self.labels)
            latent_X = self.encoder_model.predict([self.X, labels_onehot])[2]

            self.epochs.append(epoch)
            self.scores.append(self.asw(latent_X))

            print(f"Epoch {epoch}: ASW = {self.scores[-1]:.4f}")

    def asw(self, latent):
        return silhouette_score(latent, self.labels)

    def ari(self, latent):
        labels_pred = self.kmeans.fit_predict(latent)
        return adjusted_rand_score(self.labels, labels_pred)

    def nmi(self, latent):
        labels_pred = self.kmeans.fit_predict(latent)
        return normalized_mutual_info_score(self.labels, labels_pred)
