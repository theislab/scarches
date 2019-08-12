import logging
import os

import anndata
import keras
from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, Lambda, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from surgeon.models._losses import kl_recon
from surgeon.models._utils import sample_z
from surgeon.utils import label_encoder, remove_sparsity

log = logging.getLogger(__file__)


class CVAE:
    """
        C-VAE vector Network class. This class contains the implementation of Conditional
        Variational Auto-encoder network.
        # Parameters
            kwargs:
                key: `dropout_rate`: float
                        dropout rate
                key: `learning_rate`: float
                    learning rate of optimization algorithm
                key: `model_path`: basestring
                    path to save the model after training
                key: `alpha`: float
                    alpha coefficient for loss.
                key: `beta`: float
                    beta coefficient for loss.
            x_dimension: integer
                number of gene expression space dimensions.
            z_dimension: integer
                number of latent space dimensions.
    """

    def __init__(self, x_dimension, n_conditions, z_dimension=100, **kwargs):
        self.x_dim = x_dimension
        self.z_dim = z_dimension

        self.n_conditions = n_conditions

        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.001)
        self.eta = kwargs.get("eta", 1.0)
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_path = kwargs.get("model_path", "./models/trVAE/")

        self.x = Input(shape=(self.x_dim,), name="data")
        self.encoder_labels = Input(shape=(self.n_conditions,), name="encoder_labels")
        self.decoder_labels = Input(shape=(self.n_conditions,), name="decoder_labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.condition_encoder = None

        self.network_kwargs = {
            "x_dimension": self.x_dim,
            "z_dimension": self.z_dim,
            "n_conditions": self.n_conditions,
            "dropout_rate": self.dr_rate,
        }

        self.training_kwargs = {
            "learning_rate": self.lr,
            "alpha": self.alpha,
            "eta": self.eta,
            "model_path": self.model_path,
        }

        self.init_w = keras.initializers.glorot_normal()
        self._create_networks()
        self.compile_models()

        get_summary = kwargs.get("summary", True)
        if get_summary:
            self.encoder_model.summary()
            self.decoder_model.summary()
            self.cvae_model.summary()

    def _encoder(self, x, y, name="encoder"):
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
        xy = concatenate([x, y], axis=1)
        h = Dense(512, kernel_initializer=self.init_w, use_bias=False, name='first_layer')(xy)
        h = BatchNormalization()(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(256, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization()(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])
        model = Model(inputs=[x, y], outputs=[mean, log_var, z], name=name)
        return mean, log_var, model

    def _decoder(self, z, y, name="decoder"):
        """
            Constructs the decoder sub-network of C-VAE. This function implements the
            decoder part of Variational Auto-encoder. It will transform constructed
            latent space to the previous space of data with n_dimensions = n_vars.
            # Parameters
                No parameters are needed.
            # Returns
                h: Tensor
                    A Tensor for last dense layer with the shape of [n_vars, ] to reconstruct data.
        """
        zy = concatenate([z, y], axis=1)
        h = Dense(256, kernel_initializer=self.init_w, use_bias=False, name='first_layer')(zy)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(self.x_dim, kernel_initializer=self.init_w, use_bias=True)(h)
        h = Activation('relu', name="reconstruction_output")(h)
        model = Model(inputs=[z, y], outputs=h, name=name)
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

        inputs = [self.x, self.encoder_labels, self.decoder_labels]

        self.mu, self.log_var, self.encoder_model = self._encoder(*inputs[:2], name="encoder")
        self.decoder_model = self._decoder(self.z,
                                           self.decoder_labels,
                                           name="decoder")

        decoder_outputs = self.decoder_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])
        reconstruction_output = Lambda(lambda x: x, name="kl_reconstruction")(decoder_outputs)

        self.cvae_model = Model(inputs=inputs,
                                outputs=reconstruction_output,
                                name="cvae")

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
        self.optimizer = keras.optimizers.Adam(lr=self.lr)
        self.classifier_optimizer = keras.optimizers.Adam(lr=self.lr)
        self.cvae_model.compile(optimizer=self.optimizer,
                                loss=kl_recon(self.mu, self.log_var, self.alpha, self.eta),
                                metrics=[kl_recon(self.mu, self.log_var, self.alpha, self.eta)],
                                )

    def get_summary_of_networks(self):
        self.encoder_model.summary()
        self.decoder_model.summary()
        self.cvae_model.summary()

    def to_latent(self, adata, encoder_labels):
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

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)

        latent = self.encoder_model.predict([adata.X, encoder_labels])[2]

        adata_latent = anndata.AnnData(X=latent)
        adata_latent.obs = adata.obs

        return adata_latent

    def predict(self, adata, encoder_labels, decoder_labels):
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
            import scanpy as sc
            import scgen
            train_data = sc.read("train_kang.h5ad")
            validation_data = sc.read("./data/validation.h5ad")
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.train(n_epochs=20)
            prediction = network.predict('CD4T', obs_key={"cell_type": ["CD8T", "NK"]})
            ```
        """
        adata = remove_sparsity(adata)

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        x_hat = self.cvae_model.predict([adata.X, encoder_labels, decoder_labels])

        adata_pred = anndata.AnnData(X=x_hat)
        adata_pred.obs = adata.obs
        adata_pred.var_names = adata.var_names

        return adata_pred

    def restore_model(self):
        """
            restores model weights from `model_to_use`.
            # Parameters
                No parameters are needed.
            # Returns
                Nothing will be returned.
            # Example
            ```python
            import scanpy as sc
            import scgen
            train_data = sc.read("./data/train_kang.h5ad")
            validation_data = sc.read("./data/valiation.h5ad")
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.restore_model()
            ```
        """
        self.cvae_model = load_model(os.path.join(self.model_path, 'mmd_cvae.h5'), compile=False)
        self.encoder_model = load_model(os.path.join(self.model_path, 'encoder.h5'), compile=False)
        self.decoder_model = load_model(os.path.join(self.model_path, 'decoder.h5'), compile=False)
        self.compile_models()

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        self.cvae_model.save(os.path.join(self.model_path, "mmd_cvae.h5"), overwrite=True)
        self.encoder_model.save(os.path.join(self.model_path, "encoder.h5"), overwrite=True)
        self.decoder_model.save(os.path.join(self.model_path, "decoder.h5"), overwrite=True)
        log.info(f"Model saved in file: {self.model_path}. Training finished")

    def train(self, train_adata, valid_adata, condition_key, le,
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
            import scanpy as sc
            import scgen
            train_data = sc.read(train_katrain_kang.h5ad           >>> validation_data = sc.read(valid_kang.h5ad)
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.train(n_epochs=20)
            ```
        """
        train_adata = remove_sparsity(train_adata)
        valid_adata = remove_sparsity(valid_adata)

        train_conditions_encoded, new_le = label_encoder(train_adata, label_encoder=le,
                                                         condition_key=condition_key)
        valid_conditions_encoded, _ = label_encoder(valid_adata, label_encoder=le, condition_key=condition_key)

        if self.condition_encoder is None:
            self.condition_encoder = new_le

        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        x_train = [train_adata.X, train_conditions_onehot, train_conditions_onehot]
        y_train = train_adata.X

        x_valid = [valid_adata.X, valid_conditions_onehot, valid_conditions_onehot]
        y_valid = valid_adata.X

        callbacks = [
            History(),
        ]

        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor='val_loss'))

        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=lr_reducer))

        self.cvae_model.fit(x=x_train,
                            y=y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=n_epochs,
                            batch_size=batch_size,
                            verbose=verbose,
                            callbacks=callbacks,
                            )
        if save:
            self.save_model()
