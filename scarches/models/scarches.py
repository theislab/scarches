import os

import anndata
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, History, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Lambda, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_custom_objects
from scipy import sparse

from scarches.models import CVAE
from scarches.models._activations import ACTIVATIONS
from scarches.models._callbacks import ScoreCallback
from scarches.models._data_generator import make_dataset
from scarches.models._layers import LAYERS
from scarches.models._losses import LOSSES
from scarches.models._utils import print_progress
from scarches.utils import label_encoder, remove_sparsity, train_test_split


class scArches(CVAE):
    """scArches class. This class contains the implementation of scArches network.

        Parameters
        ----------
        x_dimension: int
            number of gene expression space dimensions.
        conditions: list
            list of unique conditions (i.e. batch ids) in the data used for one-hot encoding.
        z_dimension: int
            number of latent space dimensions.
        task_name: str
            name of the task.

        kwargs:
            `learning_rate`: float
                scArches's optimizer's step size (learning rate).
            `alpha`: float
                KL divergence coefficient in the loss function.
            `beta`: float
                MMD loss coefficient in the loss function.
            `eta`: float
                Reconstruction coefficient in the loss function.
            `dropout_rate`: float
                dropout rate for Dropout layers in scArches' architecture.
            `model_path`: str
                path to save model config and its weights.
            `clip_value`: float
                Optimizer's clip value used for clipping the computed gradients.
            `output_activation`: str
                Output activation of scArches which Depends on the range of data. For positive value you can use "relu" or "linear" but if your data
                have negative value set to "linear" which is default.
            `use_batchnorm`: bool
                Whether use batch normalization in scArches or not.
            `architecture`: list
                Architecture of scArches. Must be a list of integers.
            `gene_names`: list
                names of genes fed as scArches' input. Must be a list of strings.
    """

    def __new__(cls, *args, **kwargs):
        loss_fn = kwargs.get("loss_fn", "nb")
        if loss_fn in ['nb', 'zinb']:
            if loss_fn == 'nb':
                from .scarchesnb import scArchesNB
                return scArchesNB(*args, **kwargs)
            elif loss_fn == 'zinb':
                from .scarcheszinb import scArchesZINB
                return scArchesZINB(*args, **kwargs)
        elif kwargs.get('beta', 0.0) == 0:
            from .cvae import CVAE
            if kwargs.get("beta", None) is not None:
                kwargs.pop("beta")
            return CVAE(*args, **kwargs)
        else:
            return super(scArches, cls).__new__(cls)

    def __init__(self, x_dimension, conditions, task_name="unknown", z_dimension=10, **kwargs):
        self.beta = kwargs.pop('beta', 0.0)
        self.mmd_computation_method = kwargs.pop("mmd_computation_method", "general")

        if kwargs.get("loss_fn", "mse") in ['nb', 'zinb']:
            kwargs['loss_fn'] = 'mse'

        kwargs.update({"model_name": "cvae", "class_name": "scArches"})

        super().__init__(x_dimension, conditions, task_name, z_dimension, **kwargs)

        self.network_kwargs.update({
            "mmd_computation_method": self.mmd_computation_method,
        })

        self.training_kwargs.update({
            "beta": self.beta,
        })

    def update_kwargs(self):
        super().update_kwargs()
        self.network_kwargs.update({
            "mmd_computation_method": self.mmd_computation_method,
        })

        self.training_kwargs.update({
            "beta": self.beta,
        })

    @classmethod
    def from_config(cls, config_path, new_params=None, compile=True, construct=True):
        """create class object from exsiting class' config file.

        Parameters
        ----------
        config_path: str
            Path to scArches' config json file.
        new_params: dict, optional
            Python dict of parameters which you wanted to assign new values to them.
        compile: bool
            ``True`` by default. if ``True``, will compile scArches' model after creating an instance.
        construct: bool
            ``True`` by default. if ``True``, will construct scArches' model after creating an instance.
        """
        import json
        with open(config_path, 'rb') as f:
            class_config = json.load(f)

        class_config['construct_model'] = construct
        class_config['compile_model'] = compile

        if new_params:
            class_config.update(new_params)

        return cls(**class_config)

    def _decoder(self, name="decoder"):
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
        model_inputs, model_outputs = self._output_decoder(h)
        self.decoder = Model(inputs=model_inputs, outputs=model_outputs, name=name)
        self.mmd_decoder = Model(inputs=model_inputs, outputs=h_mmd, name='mmd_decoder')

    def call(self, x, training=None, mask=None):
        if isinstance(x, list):
            expression, encoder_labels, decoder_labels = x
        else:
            expression = x['expression']
            encoder_labels = x['encoder_label']
            decoder_labels = x['decoder_label']

        z_mean, z_log_var, z = self.encoder([expression, encoder_labels])

        x_hat = self.decoder([z, decoder_labels])
        mmd_output = self.mmd_decoder([z, decoder_labels])
        return x_hat, mmd_output, z_mean, z_log_var

    def calc_losses(self, y_true, y_pred, mmd_true, mmd_pred, z_mean, z_log_var):
        """
            Defines the loss function of class' network after constructing the whole
            network.
        """
        recon_loss = LOSSES[f'{self.loss_fn}_recon'](y_true, y_pred)
        mmd_loss = LOSSES['mmd'](self.n_conditions)(mmd_true, mmd_pred)
        kl_loss = LOSSES['kl'](z_mean, z_log_var)(y_true, y_pred)

        loss = self.eta * recon_loss + self.alpha * kl_loss + self.beta * mmd_loss

        return loss, self.eta * recon_loss, kl_loss, self.beta * mmd_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, recon_loss, kl_loss, mmd_loss = self.forward_with_loss(data)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": loss,
            f'{self.loss_fn}_loss': recon_loss,
            "kl_loss": kl_loss,
            "mmd_loss": mmd_loss
        }

    def forward_with_loss(self, data):
        x, y = data

        y_true, mmd_true = y['reconstruction'], y['mmd']

        y_pred, mmd_output, z_mean, z_log_var = self.call(x)
        loss, recon_loss, kl_loss, mmd_loss = self.calc_losses(y_true, y_pred, mmd_true, mmd_output, z_mean, z_log_var)

        return loss, recon_loss, kl_loss, mmd_loss

    def test_step(self, data):
        loss, recon_loss, kl_loss, mmd_loss = self.forward_with_loss(data)

        return {
            'loss': loss,
            f'{self.loss_fn}_loss': recon_loss,
            'kl_loss': kl_loss,
            'mmd_loss': mmd_loss,
        }

    def to_mmd_layer(self, adata, batch_key):
        """
            Map ``adata`` in to the MMD space. This function will feed data
            in ``mmd_model`` of scArches and compute the MMD space coordinates
            for each sample in data.

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated data matrix to be mapped to MMD latent space.
                Please note that ``adata.X`` has to be in shape [n_obs, x_dimension]
            encoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as scArches' encoder condition array.
            decoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as scArches' decoder condition array.

            Returns
            -------
            adata_mmd: :class:`~anndata.AnnData`
                returns Annotated data containing MMD latent space encoding of ``adata``
        """
        adata = remove_sparsity(adata)

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        decoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        cvae_inputs = [adata.X, encoder_labels, decoder_labels]

        mmd = self(cvae_inputs)[1].numpy()
        mmd = np.nan_to_num(mmd, nan=0.0, posinf=0.0, neginf=0.0)

        adata_mmd = anndata.AnnData(X=mmd)
        adata_mmd.obs = adata.obs.copy(deep=True)

        return adata_mmd

    def get_latent(self, adata, batch_key, return_z=True):
        """ Transforms `adata` in latent space of scArches and returns the latent
        coordinates in the annotated (adata) format.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset matrix in Primary space.
        batch_key: str
            Name of the column containing the study (batch) names for each sample.
        return_z: bool
            ``False`` by defaul. if ``True``, the output of bottleneck layer of network will be computed.

        Returns
        -------
        adata_pred: `~anndata.AnnData`
            Annotated data of transformed ``adata`` into latent space.
        """
        if set(self.gene_names).issubset(set(adata.var_names)):
            adata = adata[:, self.gene_names]
        else:
            raise Exception("set of gene names in train adata are inconsistent with scArches' gene_names")

        if self.beta == 0:
            return_z = True

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)

        if return_z or self.beta == 0:
            return self.get_z_latent(adata, encoder_labels)
        else:
            return self.to_mmd_layer(adata, batch_key)

    def predict(self, adata, encoder_labels, decoder_labels):
        """Feeds ``adata`` to scArches and produces the reconstructed data.

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated data matrix whether in primary space.
            encoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as scArches' encoder condition array.
            decoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as scArches' decoder condition array.

            Returns
            -------
            adata_pred: `~anndata.AnnData`
                Annotated data of predicted cells in primary space.
        """
        adata = remove_sparsity(adata)

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        x_hat = self.cvae_model.predict([adata.X, encoder_labels, decoder_labels])[0]

        adata_pred = anndata.AnnData(X=x_hat)
        adata_pred.obs = adata.obs
        adata_pred.var_names = adata.var_names

        return adata_pred

    def _fit_dataset(self, adata,
                     condition_key, train_size=0.9,
                     n_epochs=300, batch_size=32, steps_per_epoch=50,
                     early_stop_limit=10, lr_reducer=7,
                     save=True, retrain=True, verbose=3):
        train_adata, valid_adata = train_test_split(adata, train_size)

        if self.gene_names is None:
            self.gene_names = train_adata.var_names.tolist()
        else:
            if set(self.gene_names).issubset(set(train_adata.var_names)):
                train_adata = train_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in train adata are inconsistent with class' gene_names")

            if set(self.gene_names).issubset(set(valid_adata.var_names)):
                valid_adata = valid_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in valid adata are inconsistent with class' gene_names")

        if not retrain and os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.restore_model_weights()
            return

        callbacks = [
            History(),
        ]

        if verbose > 2:
            callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: print_progress(epoch, logs, n_epochs)))
            fit_verbose = 0
        else:
            fit_verbose = verbose

        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor='val_loss'))

        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=lr_reducer))

        train_dataset, self.condition_encoder = make_dataset(train_adata, condition_key, self.condition_encoder,
                                                             batch_size, n_epochs, True,
                                                             self.loss_fn, self.n_conditions, use_mmd=True)
        valid_dataset, _ = make_dataset(valid_adata, condition_key, self.condition_encoder, valid_adata.shape[0],
                                        n_epochs, False,
                                        self.loss_fn, self.n_conditions, use_mmd=True)

        self.log_history = self.fit(train_dataset,
                                    validation_data=valid_dataset,
                                    epochs=n_epochs,
                                    batch_size=batch_size,
                                    verbose=fit_verbose,
                                    callbacks=callbacks,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_steps=1,
                                    )
        if save:
            self.update_kwargs()
            self.save(make_dir=True)
