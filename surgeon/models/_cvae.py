import logging
import os

import anndata
import keras
import numpy as np
from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau, LambdaCallback
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
from surgeon.models._utils import sample_z, print_message, print_progress
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
        self.beta = kwargs.get('beta', 0.0)
        self.eta = kwargs.get("eta", 1.0)
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_path = kwargs.get("model_path", "./models/trVAE/")
        self.loss_fn = kwargs.get("loss_fn", 'mse')
        self.ridge = kwargs.get('ridge', 0.1)
        self.scale_factor = kwargs.get("scale_factor", 1.0)
        self.clip_value = kwargs.get('clip_value', 3.0)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.output_activation = kwargs.get("output_activation", 'relu')
        self.use_batchnorm = kwargs.get("use_batchnorm", False)
        self.architecture = kwargs.get("architecture", [128])
        self.freeze_expression_input = kwargs.get("freeze_expression_input", False)
        self.n_mmd_conditions = kwargs.get("n_mmd_conditions", n_conditions)
        self.mmd_computation_method = kwargs.get("mmd_computation_method", "general")

        self.x = Input(shape=(self.x_dim,), name="data")
        self.size_factor = Input(shape=(1,), name='size_factor')
        self.encoder_labels = Input(shape=(self.n_conditions,), name="encoder_labels")
        self.decoder_labels = Input(shape=(self.n_conditions,), name="decoder_labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.condition_encoder = None
        self.aux_models = {}

        self.network_kwargs = {
            "x_dimension": self.x_dim,
            "z_dimension": self.z_dim,
            "n_conditions": self.n_conditions,
            "n_mmd_conditions": self.n_mmd_conditions,
            "dropout_rate": self.dr_rate,
            "loss_fn": self.loss_fn,
            "output_activation": self.output_activation,
            "architecture": self.architecture,
            "use_batchnorm": self.use_batchnorm,
            "freeze_expression_input": self.freeze_expression_input,
            "mmd_computation_method": self.mmd_computation_method,
        }

        self.training_kwargs = {
            "learning_rate": self.lr,
            "alpha": self.alpha,
            "beta": self.beta,
            "eta": self.eta,
            "ridge": self.ridge,
            "scale_factor": self.scale_factor,
            "clip_value": self.clip_value,
            "model_path": self.model_path,
        }

        self.init_w = keras.initializers.glorot_normal()
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
                    A dense layer consists f means of gaussian distributions of latent space dimensions.
                log_var: Tensor
                    A dense layer consists of log transformed variances of gaussian distributions of latent space dimensions.
        """

        for idx, n_neuron in enumerate(self.architecture):
            if idx == 0:
                h = LAYERS['FirstLayer'](n_neuron, kernel_initializer=self.init_w,
                                         use_bias=False, name="first_layer", freeze=self.freeze_expression_input)(
                    [self.x, self.encoder_labels])
            else:
                h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)

        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])
        model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
        return mean, log_var, model

    def _output_decoder(self, h):
        if self.loss_fn == 'nb':
            h_mean = Dense(self.x_dim, activation=None, kernel_initializer=self.init_w, use_bias=True)(h)
            h_mean = ACTIVATIONS['mean_activation'](h_mean)

            h_disp = Dense(self.x_dim, activation=None, kernel_initializer=self.init_w, use_bias=True)(h)
            h_disp = ACTIVATIONS['disp_activation'](h_disp)

            h_mean = LAYERS['ColWiseMultLayer']()([h_mean, self.size_factor])

            model_outputs = LAYERS['SliceLayer'](0, name='kl_nb')([h_mean, h_disp])

            model_inputs = [self.z, self.decoder_labels, self.size_factor]
            model_outputs = [model_outputs]

            self.aux_models['disp'] = Model(inputs=[self.z, self.decoder_labels, self.size_factor],
                                            output=h_disp)
        elif self.loss_fn == 'zinb':
            h_pi = Dense(self.x_dim, activation=ACTIVATIONS['sigmoid'], kernel_initializer=self.init_w, use_bias=True,
                         name='decoder_pi')(h)
            h_mean = Dense(self.x_dim, activation=None, kernel_initializer=self.init_w,
                           use_bias=True)(h)
            h_mean = ACTIVATIONS['mean_activation'](h_mean)

            h_disp = Dense(self.x_dim, activation=None, kernel_initializer=self.init_w,
                           use_bias=True)(h)
            h_disp = ACTIVATIONS['disp_activation'](h_disp)

            mean_output = LAYERS['ColWiseMultLayer']()([h_mean, self.size_factor])

            model_outputs = LAYERS['SliceLayer'](0, name='kl_zinb')(
                [mean_output, h_disp, h_pi])

            model_inputs = [self.z, self.decoder_labels, self.size_factor]
            model_outputs = [model_outputs]

            self.aux_models['disp'] = Model(inputs=[self.z, self.decoder_labels, self.size_factor],
                                            output=h_disp)

            self.aux_models['pi'] = Model(inputs=[self.z, self.decoder_labels, self.size_factor],
                                          output=h_pi)

        else:
            h = Dense(self.x_dim, activation=None, 
                      kernel_initializer=self.init_w,
                      use_bias=True)(h)
            h = ACTIVATIONS[self.output_activation](h)
            model_inputs = [self.z, self.decoder_labels]
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
                h = LAYERS['FirstLayer'](n_neuron, kernel_initializer=self.init_w,
                                         use_bias=False, name="first_layer", freeze=self.freeze_expression_input)(
                    [self.z, self.decoder_labels])
            else:
                h = Dense(n_neuron, kernel_initializer=self.init_w, 
                          use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            if idx == 0:
                h_mmd = h
            h = Dropout(self.dr_rate)(h)
        # h_mmd = self.z
        model_inputs, model_outputs = self._output_decoder(h)
        model = Model(inputs=model_inputs, outputs=model_outputs, name=name)
        mmd_model = Model(inputs=model_inputs, outputs=h_mmd, name='mmd_decoder')
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
            inputs = [self.x, self.encoder_labels, self.decoder_labels, self.size_factor]
            encoder_outputs = self.encoder_model(inputs[:2])[2]
            decoder_inputs = [encoder_outputs, self.decoder_labels, self.size_factor]
            self.disp_output = self.aux_models['disp'](decoder_inputs)
            if self.loss_fn == 'zinb':
                self.pi_output = self.aux_models['pi'](decoder_inputs)
        else:
            inputs = [self.x, self.encoder_labels, self.decoder_labels]
            encoder_outputs = self.encoder_model(inputs[:2])[2]
            decoder_inputs = [encoder_outputs, self.decoder_labels]
        
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
            mmd_loss = LOSSES['mmd'](self.n_mmd_conditions, self.beta)
            kl_loss = LOSSES['kl'](self.mu, self.log_var)
            recon_loss = LOSSES['nb_wo_kl']
            
        elif self.loss_fn == 'zinb':
            loss = LOSSES[self.loss_fn](self.pi_output, self.disp_output, self.mu, self.log_var, self.ridge, self.alpha,
                                        self.eta)
            mmd_loss = LOSSES['mmd'](self.n_mmd_conditions, self.beta)
            kl_loss = LOSSES['kl'](self.mu, self.log_var)
            recon_loss = LOSSES['zinb_wo_kl']

        else:
            loss = LOSSES[self.loss_fn](self.mu, self.log_var, self.alpha, self.eta)
            mmd_loss = LOSSES['mmd'](self.n_mmd_conditions, self.beta)
            kl_loss = LOSSES['kl'](self.mu, self.log_var)
            recon_loss = LOSSES['recon']

        return loss, mmd_loss, kl_loss, recon_loss

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
        loss, mmd_loss, kl_loss, recon_loss = self._calculate_loss()

        # self.cvae_model.compile(optimizer=optimizer,
        #                         loss=[loss, mmd_loss],
        #                         metrics={'reconstruction': [recon_loss, kl_loss],
        #                                  'mmd': [mmd_loss]}
        #                         )
        self.cvae_model.compile(optimizer=optimizer,
                                loss=[loss, mmd_loss],
                                metrics={self.cvae_model.outputs[0].name: loss,
                                         self.cvae_model.outputs[0].name: mmd_loss}
                                )

    def get_summary_of_networks(self):
        self.encoder_model.summary()
        self.decoder_model.summary()
        self.cvae_model.summary()

    def to_mmd_layer(self, adata, encoder_labels, decoder_labels):
        adata = remove_sparsity(adata)

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)
        if self.loss_fn == 'nb':
            cvae_inputs = [adata.X, encoder_labels, decoder_labels, adata.obs['size_factors'].values]
        else:
            cvae_inputs = [adata.X, encoder_labels, decoder_labels]
        mmd = self.cvae_model.predict(cvae_inputs)[1]
        mmd = np.nan_to_num(mmd, nan=0.0, posinf=0.0, neginf=0.0)

        adata_mmd = anndata.AnnData(X=mmd)
        adata_mmd.obs = adata.obs.copy(deep=True)

        return adata_mmd

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

        encoder_inputs = [adata.X, encoder_labels]

        latent = self.encoder_model.predict(encoder_inputs)[2]
        latent = np.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0)

        adata_latent = anndata.AnnData(X=latent)
        adata_latent.obs = adata.obs.copy(deep=True)

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

        x_hat = self.cvae_model.predict([adata.X, encoder_labels, decoder_labels])[0]

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
        # self.cvae_model = load_model(os.path.join(self.model_path, 'cvae.h5'), compile=False,
                                    #  custom_objects=self.custom_objects)
        self.cvae_model.load_weights(os.path.join(self.model_path, 'cvae.h5'))
        self.decoder_mmd_model = self.cvae_model.get_layer("mmd_decoder")
        self.encoder_model = self.cvae_model.get_layer("encoder")

        self.decoder_model = self.cvae_model.get_layer("decoder")

        self.compile_models()
        print("Model has been successfully restored!")

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        self.cvae_model.save_weights(os.path.join(self.model_path, "cvae.h5"), overwrite=True)
        log.info(f"Model saved in file: {self.model_path}. Training finished")

    def train(self, train_adata, valid_adata, condition_key, cell_type_key='cell_type', le=None,
              n_epochs=25, batch_size=32, early_stop_limit=20, n_per_epoch=5, n_epochs_warmup=0,
              score_filename="./scores.log", lr_reducer=10, save=True, verbose=2, retrain=True):
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
                train_adata.raw = anndata.AnnData(X=train_adata.raw.X.A)
            if valid_adata.raw is not None and sparse.issparse(valid_adata.raw.X):
                valid_adata.raw = anndata.AnnData(X=valid_adata.raw.X.A)

        train_conditions_encoded, new_le = label_encoder(train_adata, label_encoder=le,
                                                         condition_key=condition_key)

        valid_conditions_encoded, _ = label_encoder(valid_adata, label_encoder=le, condition_key=condition_key)

        if self.condition_encoder is None:
            self.condition_encoder = new_le

        if not retrain and os.path.exists(os.path.join(self.model_path, "cvae.h5")):
            self.restore_model()
            return

        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        if self.loss_fn in ['nb', 'zinb']:
            x_train = [train_adata.X, train_conditions_onehot, train_conditions_onehot,
                       train_adata.obs['size_factors'].values]
            y_train = [train_adata.raw.X, train_conditions_encoded]

            x_valid = [valid_adata.X, valid_conditions_onehot, valid_conditions_onehot,
                       valid_adata.obs['size_factors'].values]
            y_valid = [valid_adata.raw.X, valid_conditions_encoded]
        else:
            x_train = [train_adata.X, train_conditions_onehot, train_conditions_onehot]
            y_train = [train_adata.X, train_conditions_encoded]

            x_valid = [valid_adata.X, valid_conditions_onehot, valid_conditions_onehot]
            y_valid = [valid_adata.X, valid_conditions_encoded]

        callbacks = [
            History(),
        ]

        if verbose > 2:
            callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: print_progress(epoch, logs, n_epochs, verbose)))
            fit_verbose = 0
        else:
            fit_verbose = verbose

        if n_per_epoch > 0 or n_per_epoch == -1:
            adata = train_adata.concatenate(valid_adata)

            train_celltypes_encoded, _ = label_encoder(train_adata, label_encoder=None, condition_key=cell_type_key)
            valid_celltypes_encoded, _ = label_encoder(valid_adata, label_encoder=None, condition_key=cell_type_key)
            batch_labels = np.concatenate([train_conditions_encoded, valid_conditions_encoded], axis=0)
            celltype_labels = np.concatenate([train_celltypes_encoded, valid_celltypes_encoded], axis=0)

            callbacks.append(ScoreCallback(score_filename, adata.X, batch_labels, celltype_labels, self.cvae_model,
                                           n_per_epoch=n_per_epoch, n_batch_labels=self.n_conditions,
                                           n_celltype_labels=len(np.unique(celltype_labels))))

        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor='val_loss'))

        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=lr_reducer))

        self.cvae_model.fit(x=x_train,
                            y=y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=n_epochs,
                            batch_size=batch_size,
                            verbose=fit_verbose,
                            callbacks=callbacks,
                            )
        if save:
            self.save_model()
