import logging
import os

import anndata
import keras
import numpy as np
from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, Lambda, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from scipy import sparse

from surgeon.models._activations import ACTIVATIONS
from surgeon.models._callbacks import ScoreCallback
from surgeon.models._layers import LAYERS
from surgeon.models._losses import LOSSES
from surgeon.models._utils import sample_z
from surgeon.utils import label_encoder, remove_sparsity

log = logging.getLogger(__file__)


class CVAEFair:
    """
        C-VAE Fair vector Network class. This class contains the implementation of Conditional
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

    def __init__(self, x_dimension, n_datasets, n_conditions, z_dimension=100, **kwargs):
        self.x_dim = x_dimension
        self.z_dim = z_dimension

        self.n_datasets = n_datasets
        self.n_conditions = n_conditions

        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.001)
        self.beta = kwargs.get("beta", 1.0)
        self.eta = kwargs.get("eta", 1.0)
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_path = kwargs.get("model_path", "./models/trVAE/")
        self.loss_fn = kwargs.get("loss_fn", 'mse')
        self.ridge = kwargs.get('ridge', 0.1)
        self.scale_factor = kwargs.get("scale_factor", 1.0)
        self.clip_value = kwargs.get('clip_value', 3.0)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.lambda_l2 = kwargs.get('lambda_l2', 1e-6)
        self.output_activation = kwargs.get("output_activation", 'relu')
        self.use_batchnorm = kwargs.get("use_batchnorm", False)
        self.architecture = kwargs.get("architecture", [128])
        self.freeze_expression_input = kwargs.get("freeze_expression_input", False)

        self.x = Input(shape=(self.x_dim,), name="data")
        self.size_factor = Input(shape=(1,), name='size_factor')
        self.encoder_conditions = Input(shape=(self.n_datasets,), name="encoder_datasets")
        self.encoder_cell_types = Input(shape=(self.n_conditions,), name="encoder_conditions")
        self.decoder_conditions = Input(shape=(self.n_datasets,), name="decoder_datasets")
        self.decoder_cell_types = Input(shape=(self.n_conditions,), name="decoder_conditions")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.dataset_encoder = None
        self.condition_encoder = None
        self.aux_models = {}

        self.network_kwargs = {
            "x_dimension": self.x_dim,
            "z_dimension": self.z_dim,
            "n_datasets": self.n_datasets,
            "n_conditions": self.n_conditions,
            "dropout_rate": self.dr_rate,
            "loss_fn": self.loss_fn,
            "output_activation": self.output_activation,
            "architecture": self.architecture,
            "use_batchnorm": self.use_batchnorm,
            "freeze_expression_input": self.freeze_expression_input,
        }

        self.training_kwargs = {
            "learning_rate": self.lr,
            "alpha": self.alpha,
            "eta": self.eta,
            "ridge": self.ridge,
            "scale_factor": self.scale_factor,
            "clip_value": self.clip_value,
            "model_path": self.model_path,
        }

        self.init_w = keras.initializers.glorot_normal()
        self.regularizer = keras.regularizers.l2(self.lambda_l2)
        self._create_networks()
        self.compile_models()

        print_summary = kwargs.get("print_summary", True)
        if print_summary:
            self.encoder_model.summary()
            self.decoder_model.summary()
            self.cvae_model.summary()

    def _encoder(self, name="encoder"):
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
                h = LAYERS['FirstLayer'](n_neuron, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer,
                                         use_bias=False, name="first_layer", freeze=self.freeze_expression_input)(
                    [self.x, self.encoder_conditions, self.encoder_cell_types])
            else:
                h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False,
                          kernel_regularizer=self.regularizer)(h)
            if self.use_batchnorm:
                h = BatchNormalization(axis=1, trainable=True)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)

        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])
        model = Model(inputs=[self.x, self.encoder_conditions, self.encoder_cell_types], outputs=[mean, log_var, z],
                      name=name)
        return mean, log_var, model

    def _output_decoder(self, h):
        if self.loss_fn == 'nb':
            h_mean = Dense(self.x_dim, activation=None, kernel_initializer=self.init_w,
                           kernel_regularizer=self.regularizer, use_bias=True)(h)
            h_mean = ACTIVATIONS['mean_activation'](h_mean)

            h_disp = Dense(self.x_dim, activation=None, kernel_initializer=self.init_w,
                           kernel_regularizer=self.regularizer, use_bias=True)(h)
            h_disp = ACTIVATIONS['disp_activation'](h_disp)

            h_mean = LAYERS['ColWiseMultLayer']()([h_mean, self.size_factor])

            model_outputs = LAYERS['SliceLayer'](0, name='kl_nb')([h_mean, h_disp])

            model_inputs = [self.z, self.decoder_conditions, self.decoder_cell_types, self.size_factor]
            model_outputs = [model_outputs]

            self.aux_models['disp'] = Model(
                inputs=[self.z, self.decoder_conditions, self.decoder_cell_types, self.size_factor],
                output=h_disp)
        elif self.loss_fn == 'zinb':
            h_pi = Dense(self.x_dim, activation=ACTIVATIONS['sigmoid'], kernel_initializer=self.init_w, use_bias=True,
                         name='decoder_pi')(h)
            h_mean = Dense(self.x_dim, activation=None, kernel_initializer=self.init_w,
                           kernel_regularizer=self.regularizer, use_bias=True)(h)
            h_mean = ACTIVATIONS['mean_activation'](h_mean)

            h_disp = Dense(self.x_dim, activation=None, kernel_initializer=self.init_w,
                           kernel_regularizer=self.regularizer, use_bias=True)(h)
            h_disp = ACTIVATIONS['disp_activation'](h_disp)

            mean_output = LAYERS['ColWiseMultLayer']()([h_mean, self.size_factor])

            model_outputs = LAYERS['SliceLayer'](0, name='kl_zinb')(
                [mean_output, h_disp, h_pi])

            model_inputs = [self.z, self.decoder_conditions, self.decoder_cell_types, self.size_factor]
            model_outputs = [model_outputs]

            self.aux_models['disp'] = Model(
                inputs=[self.z, self.decoder_conditions, self.decoder_cell_types, self.size_factor],
                output=h_disp)

            self.aux_models['pi'] = Model(
                inputs=[self.z, self.decoder_conditions, self.decoder_cell_types, self.size_factor],
                output=h_pi)

        else:
            h = Dense(self.x_dim, activation=None, kernel_regularizer=self.regularizer,
                      kernel_initializer=self.init_w,
                      use_bias=True)(h)
            h = ACTIVATIONS[self.output_activation](h)
            model_inputs = [self.z, self.decoder_conditions, self.decoder_cell_types]
            model_outputs = [h]

        return model_inputs, model_outputs

    def _decoder(self, name="decoder"):
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

        for idx, n_neuron in enumerate(self.architecture[::-1]):
            if idx == 0:
                h = LAYERS['FirstLayer'](n_neuron, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer,
                                         use_bias=False, name="first_layer", freeze=self.freeze_expression_input)(
                    [self.z, self.decoder_conditions, self.decoder_cell_types])
            else:
                h = Dense(n_neuron, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer,
                          use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization(axis=1, trainable=True)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            if idx == 0:
                h_mmd = h

        model_inputs, model_outputs = self._output_decoder(h)
        model = Model(inputs=model_inputs, outputs=model_outputs, name=name)
        mmd_model = Model(inputs=model_inputs, outputs=h_mmd, name="mmd_decoder")
        return model, mmd_model

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

        self.mu, self.log_var, self.encoder_model = self._encoder(name="encoder")
        self.decoder_model, self.decoder_mmd_model = self._decoder(name="decoder")

        if self.loss_fn in ['nb', 'zinb']:
            inputs = [self.x, self.encoder_conditions, self.encoder_cell_types, self.decoder_conditions,
                      self.decoder_cell_types, self.size_factor]
            decoder_inputs = [self.encoder_model(inputs[:3])[2], self.decoder_conditions, self.decoder_cell_types,
                              self.size_factor]
            self.disp_output = self.aux_models['disp'](decoder_inputs)
            if self.loss_fn == 'zinb':
                self.pi_output = self.aux_models['pi'](decoder_inputs)
        else:
            inputs = [self.x, self.encoder_conditions, self.encoder_cell_types, self.decoder_conditions,
                      self.decoder_cell_types]
            decoder_inputs = [self.encoder_model(inputs[:3])[2], self.decoder_conditions, self.decoder_cell_types]

        decoder_outputs = self.decoder_model(decoder_inputs)
        decoder_mmd_outputs = self.decoder_mmd_model(decoder_inputs)

        reconstruction_output = Lambda(lambda x: x, name="reconstruction")(decoder_outputs)
        mmd_output = Lambda(lambda x: x, name="mmd")(decoder_mmd_outputs)

        self.cvae_model = Model(inputs=inputs,
                                outputs=[reconstruction_output, mmd_output],
                                name="cvae")
        self.custom_objects = {'mean_activation': ACTIVATIONS['mean_activation'],
                               'disp_activation': ACTIVATIONS['disp_activation'],
                               'SliceLayer': LAYERS['SliceLayer'],
                               'ColwiseMultLayer': LAYERS['ColWiseMultLayer'],
                               'FirstLayer': LAYERS['FirstLayer']}

        get_custom_objects().update(self.custom_objects)

    def _calculate_loss(self):
        if self.loss_fn == 'nb':
            loss = LOSSES[self.loss_fn](self.disp_output, self.mu, self.log_var, self.scale_factor, self.alpha,
                                        self.eta)
            mmd_loss = LOSSES['mmd'](self.n_datasets, self.beta)
        elif self.loss_fn == 'zinb':
            loss = LOSSES[self.loss_fn](self.pi_output, self.disp_output, self.mu, self.log_var, self.ridge, self.alpha,
                                        self.eta)
            mmd_loss = LOSSES['mmd'](self.n_datasets, self.beta)
        else:
            loss = LOSSES[self.loss_fn](self.mu, self.log_var, self.alpha, self.eta)
            mmd_loss = LOSSES['mmd'](self.n_datasets, self.beta)
        return loss, mmd_loss

    def freeze_condition_irrelevant_parts(self, trainable):
        for encoder_layer in self.cvae_model.get_layer("encoder").layers:
            if encoder_layer.name != 'first_layer':
                encoder_layer.trainable = trainable

        for decoder_layer in self.cvae_model.get_layer("decoder").layers:
            if decoder_layer.name != 'first_layer':
                decoder_layer.trainable = trainable

        self.compile_models()

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
        optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value, epsilon=self.epsilon)
        loss, mmd_loss = self._calculate_loss()

        self.cvae_model.compile(optimizer=optimizer,
                                loss=[loss, mmd_loss],
                                metrics={self.cvae_model.outputs[0].name: loss,
                                         self.cvae_model.outputs[1].name: mmd_loss},
                                )

    def get_summary_of_networks(self):
        self.encoder_model.summary()
        self.decoder_model.summary()
        self.cvae_model.summary()

    def to_latent(self, adata, encoder_datasets, encoder_conditions):
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

        encoder_datasets = to_categorical(encoder_datasets, num_classes=self.n_datasets)
        encoder_conditions = to_categorical(encoder_conditions, num_classes=self.n_conditions)

        latent = self.encoder_model.predict([adata.X, encoder_datasets, encoder_conditions])[2]
        latent = np.nan_to_num(latent)

        adata_latent = anndata.AnnData(X=latent)
        adata_latent.obs = adata.obs

        return adata_latent

    def to_mmd_layer(self, adata, encoder_datasets, encoder_conditions, decoder_datasets, decoder_conditions):
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

        encoder_datasets = to_categorical(encoder_datasets, num_classes=self.n_datasets)
        decoder_datasets = to_categorical(decoder_datasets, num_classes=self.n_datasets)

        encoder_conditions = to_categorical(encoder_conditions, num_classes=self.n_conditions)
        decoder_conditions = to_categorical(decoder_conditions, num_classes=self.n_conditions)

        latent = self.cvae_model.predict(
            [adata.X, encoder_datasets, encoder_conditions, decoder_datasets, decoder_conditions])[1]
        latent = np.nan_to_num(latent)

        adata_latent = anndata.AnnData(X=latent)
        adata_latent.obs = adata.obs

        return adata_latent

    def predict(self, adata, encoder_datasets, encoder_conditions, decoder_datasets, decoder_conditions):
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

        encoder_datasets = to_categorical(encoder_datasets, num_classes=self.n_datasets)
        encoder_conditions = to_categorical(encoder_conditions, num_classes=self.n_conditions)

        decoder_datasets = to_categorical(decoder_datasets, num_classes=self.n_datasets)
        decoder_conditions = to_categorical(decoder_conditions, num_classes=self.n_conditions)

        if self.loss_fn in ['nb', 'zinb']:
            x_hat = self.cvae_model.predict(
                [adata.X, encoder_datasets, encoder_conditions, decoder_datasets, decoder_conditions,
                 adata.obs['size_factors'].values])[0]
        else:
            x_hat = self.cvae_model.predict(
                [adata.X, encoder_datasets, encoder_conditions, decoder_datasets, decoder_conditions])[0]

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
        self.cvae_model = load_model(os.path.join(self.model_path, 'cvae.h5'), compile=False,
                                     custom_objects=self.custom_objects)
        self.encoder_model = self.cvae_model.get_layer("encoder")

        self.decoder_model = self.cvae_model.get_layer("decoder")

        self.compile_models()
        print("Model has been successfully restored!")

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        self.cvae_model.save(os.path.join(self.model_path, "cvae.h5"), overwrite=True)
        log.info(f"Model saved in file: {self.model_path}. Training finished")

    def train(self, train_adata, valid_adata, dataset_key, condition_key='condition',
              mmd_calculation="datasets",
              dataset_encoder=None, condition_encoder=None,
              n_epochs=25, batch_size=32, early_stop_limit=20,
              lr_reducer=10, save=True, verbose=2, retrain=True):
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

        if self.loss_fn in ['nb', 'zinb']:
            if train_adata.raw is not None and sparse.issparse(train_adata.raw.X):
                train_adata.raw.X = train_adata.raw.X.A
            if valid_adata.raw is not None and sparse.issparse(valid_adata.raw.X):
                valid_adata.raw.X = valid_adata.raw.X.A

        train_datasets_encoded, new_le_condition = label_encoder(train_adata, label_encoder=dataset_encoder,
                                                                   condition_key=dataset_key)
        train_conditions_encoded, new_le_cell_type = label_encoder(train_adata, label_encoder=condition_encoder,
                                                                   condition_key=condition_key)

        valid_datasets_encoded, _ = label_encoder(valid_adata, label_encoder=dataset_encoder,
                                                    condition_key=dataset_key)
        valid_conditions_encoded, _ = label_encoder(valid_adata, label_encoder=condition_encoder,
                                                    condition_key=condition_key)

        if self.dataset_encoder is None:
            self.dataset_encoder = new_le_condition
            self.condition_encoder = new_le_cell_type

        if not retrain and os.path.exists(self.model_path):
            self.restore_model()
            return

        train_datasets_onehot = to_categorical(train_datasets_encoded, num_classes=self.n_datasets)
        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)

        valid_datasets_onehot = to_categorical(valid_datasets_encoded, num_classes=self.n_datasets)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        if mmd_calculation == "datasets":
            train_mmd_target, valid_mmd_target = train_datasets_encoded, valid_datasets_encoded
        else:
            train_mmd_target, valid_mmd_target = train_conditions_encoded, valid_conditions_encoded

        if self.loss_fn in ['nb', 'zinb']:
            x_train = [train_adata.X, train_datasets_onehot, train_conditions_onehot,
                       train_datasets_onehot, train_conditions_onehot,
                       train_adata.obs['size_factors'].values]
            y_train = [train_adata.raw.X, train_mmd_target]

            x_valid = [valid_adata.X, valid_datasets_onehot, valid_conditions_onehot,
                       valid_datasets_onehot, valid_conditions_onehot,
                       valid_adata.obs['size_factors'].values]
            y_valid = [valid_adata.raw.X, valid_mmd_target]
        else:
            x_train = [train_adata.X, train_datasets_onehot, train_conditions_onehot,
                       train_datasets_onehot, train_conditions_onehot]
            y_train = [train_adata.X, train_mmd_target]

            x_valid = [valid_adata.X, valid_datasets_onehot, valid_conditions_onehot,
                       valid_datasets_onehot, valid_conditions_onehot]
            y_valid = [valid_adata.X, valid_mmd_target]

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
