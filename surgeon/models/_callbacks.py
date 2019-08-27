import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.models import Model
from sklearn.metrics import silhouette_score


class ScoreCallback(Callback):
    def __init__(self,
                 filename: str,
                 data: np.ndarray,
                 labels: np.ndarray,
                 encoder_model: Model,
                 n_per_epoch: int = 5,
                 ):
        super(ScoreCallback, self).__init__()
        self.X = data
        self.labels = labels
        self.filename = filename
        self.encoder_model = encoder_model
        self.n_per_epoch = n_per_epoch

    def on_train_begin(self, logs=None):
        self.scores = []
        self.epochs = []

    def on_train_end(self, logs=None):
        scores_df = pd.DataFrame({"epoch": self.epochs, "score": self.scores})
        scores_df.to_csv(self.filename)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.n_per_epoch == 0:
            latent_X = self.encoder_model.predict([self.X, self.labels])[2]

            self.epochs.append(epoch)
            self.scores.append(self.asw(latent_X))

            print(f"Epoch {epoch}: ASW = {self.scores[-1]:.4f}")

    def asw(self, latent):
        return silhouette_score(latent, self.labels)
