import logging
import os

import anndata
import keras
import numpy as np
from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, Dropout, Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import to_categorical

from surgeon.models._losses import LOSSES
from surgeon.models._utils import sample_z
from surgeon.utils import label_encoder, remove_sparsity

log = logging.getLogger(__file__)


class NNClassifier:
    def __init__(self, x_dimension, cvae_network, n_labels, z_dimension=100, **kwargs):
        self.cvae = cvae_network
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.n_labels = n_labels

        self.lr = kwargs.get("learning_rate", 0.001)
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_path = kwargs.get("model_path", "./models/NNClassifier/")
        self.clip_value = kwargs.get('clip_value', 1e6)
        self.use_batchnorm = kwargs.get("use_batchnorm", False)
        self.lambda_l1 = kwargs.get("lambda_l1", 0.0)
        self.lambda_l2 = kwargs.get("lambda_l2", 0.0)
        if self.cvae is not None:
            self.architecture = self.cvae.architecture
        else:
            self.architecture = kwargs.get("architecture", [128])

        self.x = Input(shape=(self.x_dim,), name="data")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.label_encoder = None
        self.aux_models = {}

        self.network_kwargs = {
            "x_dimension": self.x_dim,
            "z_dimension": self.z_dim,
            "dropout_rate": self.dr_rate,
            "architecture": self.architecture,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "use_batchnorm": self.use_batchnorm
        }

        self.training_kwargs = {
            "learning_rate": self.lr,
            "clip_value": self.clip_value,
            "model_path": self.model_path,
        }

        self.init_w = keras.initializers.glorot_normal()
        self.regularizer = keras.regularizers.l1_l2(self.lambda_l1, self.lambda_l2)

        self._create_networks()
        self.compile_models()

        print_summary = kwargs.get("print_summary", True)
        if print_summary:
            self.get_summary_of_networks()

    def _network(self, name="encoder"):
        """
            Constructs the encoder sub-network of C-VAE. This function implements the
            encoder part of Variational Auto-encoder. It will transform primary
            data in the `n_vars` dimension-space to the `z_dimension` latent space.
            # Parameters
                No parameters are needed.
            # Returns
                mean: Tensor
                    A dense layer consists of means of gaussian distributions of latent space dimensions.
                log_var: Tensor
                    A dense layer consists of log transformed variances of gaussian distributions of latent space dimensions.
        """
        for idx, n_neuron in enumerate(self.architecture):
            if idx == 0:
                h = Dense(n_neuron, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer,
                          use_bias=False, name="first_layer")(self.x)
            else:
                h = Dense(n_neuron, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer,
                          use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization(axis=1, trainable=True)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)

        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])

        probs = Dense(self.n_labels, activation='softmax', kernel_initializer=self.init_w,
                      kernel_regularizer=self.regularizer)(z)
        model = Model(inputs=self.x, outputs=probs, name=name)
        return model

    def _create_networks(self):
        """
            Constructs the whole C-VAE network. It is step-by-step constructing the C-VAE
            network. First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of C-VAE.
            # Parameters
                No parameters are needed.
            # Returns
                Nothing will be returned.
        """
        self.classifier_model = self._network(name="classifier")

        if self.cvae is not None:
            for idx, layer in enumerate(self.cvae.encoder_model.layers[2:]):
                if layer.name == "first_layer":
                    weights = layer.get_weights()[0]
                    self.classifier_model.layers[idx + 1].set_weights([weights])
                else:
                    weights = layer.get_weights()
                    self.classifier_model.layers[idx + 1].set_weights(weights)

    def _calculate_loss(self):
        loss = LOSSES['cce']
        return loss

    def compile_models(self):
        """
            Defines the loss function of C-VAE network after constructing the whole
            network. This will define the KL Divergence and Reconstruction loss for
            C-VAE and also defines the Optimization algorithm for network. The C-VAE Loss
            will be weighted sum of reconstruction loss and KL Divergence loss.
            # Parameters
                No parameters are needed.
            # Returns
                Nothing will be returned.
        """
        optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value)
        loss = self._calculate_loss()

        self.classifier_model.compile(optimizer=optimizer,
                                      loss=loss,
                                      metrics=['acc'],
                                      )

    def get_summary_of_networks(self):
        self.classifier_model.summary()

    def to_latent(self, adata):
        """
            Map `data` in to the latent space. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.
            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
            # Returns
                latent: numpy nd-array
                    returns array containing latent space encoding of 'data'
        """
        adata = remove_sparsity(adata)
        model = Model(self.classifier_model.inputs, self.classifier_model.layers[-2].output)

        latent = model.predict(adata.X)

        adata_latent = anndata.AnnData(X=latent)
        adata_latent.obs = adata.obs.copy(deep=True)

        return adata_latent

    def predict(self, adata):
        """
            Predicts the cell type provided by the user in stimulated condition.
            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix whether in primary space.
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
            # Returns
                stim_pred: numpy nd-array
                    `numpy nd-array` of predicted cells in primary space.
            # Example
            ```python
            ```
        """
        adata = remove_sparsity(adata)

        probs = self.classifier_model.predict(adata.X)
        predictions = self.label_encoder.inverse_transform(np.argmax(probs, axis=1))

        predictions = self.label_encoder.inverse_transform(predictions)

        return predictions

    def restore_model(self):
        """
            restores model weights from `model_to_use`.
            # Parameters
                No parameters are needed.
            # Returns
                Nothing will be returned.
            # Example
            ```python
            ```
        """
        self.classifier_model = load_model(os.path.join(self.model_path, 'classifier.h5'), compile=False)
        self.compile_models()

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        self.classifier_model.save(os.path.join(self.model_path, "classifier.h5"), overwrite=True)
        log.info(f"Model saved in file: {self.model_path}. Training finished")

    def train(self, train_adata, valid_adata, cell_type_key,
              n_epochs=25, batch_size=32, early_stop_limit=20,
              lr_reducer=10, save=True, verbose=2):
        """
            Trains the network `n_epochs` times with given `train_data`
            and validates the model using validation_data if it was given
            in the constructor function. This function is using `early stopping`
            technique to prevent overfitting.
            # Parameters
                n_epochs: int
                    number of epochs to iterate and optimize network weights
                early_stop_limit: int
                    number of consecutive epochs in which network loss is not going lower.
                    After this limit, the network will stop training.
                threshold: float
                    Threshold for difference between consecutive validation loss values
                    if the difference is upper than this `threshold`, this epoch will not
                    considered as an epoch in early stopping.
                full_training: bool
                    if `True`: Network will be trained with all batches of data in each epoch.
                    if `False`: Network will be trained with a random batch of data in each epoch.
                initial_run: bool
                    if `True`: The network will initiate training and log some useful initial messages.
                    if `False`: Network will resume the training using `restore_model` function in order
                        to restore last model which has been trained with some training dataset.
            # Returns
                Nothing will be returned
            # Example
            ```python
            ```
        """
        train_adata = remove_sparsity(train_adata)
        valid_adata = remove_sparsity(valid_adata)

        train_classes_encoded, self.label_encoder = label_encoder(train_adata, condition_key=cell_type_key,
                                                                  label_encoder=None)
        valid_classes_encoded, _ = label_encoder(valid_adata, condition_key=cell_type_key, label_encoder=None)

        train_classes_onehot = to_categorical(train_classes_encoded, num_classes=self.n_labels)
        valid_classes_onehot = to_categorical(valid_classes_encoded, num_classes=self.n_labels)

        x_train = train_adata.X
        y_train = train_classes_onehot

        x_valid = valid_adata.X
        y_valid = valid_classes_onehot

        callbacks = [
            History(),
        ]

        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor='val_loss'))

        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=lr_reducer))

        self.classifier_model.fit(x=x_train,
                                  y=y_train,
                                  validation_data=(x_valid, y_valid),
                                  epochs=n_epochs,
                                  batch_size=batch_size,
                                  verbose=verbose,
                                  callbacks=callbacks,
                                  )
        if save:
            self.save_model()
