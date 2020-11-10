import os
import numpy as np

from scarches.models import CVAE
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import get_custom_objects

import tensorflow as tf

from scarches.models._activations import ACTIVATIONS
from scarches.models._callbacks import ScoreCallback
from scarches.models._data_generator import make_dataset
from scarches.models._layers import LAYERS
from scarches.models._losses import LOSSES

from scipy import sparse
from tensorflow.keras.callbacks import History, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.utils import to_categorical

from scarches.models._utils import print_progress
from scarches.utils import train_test_split, label_encoder


class scArchesZINB(CVAE):
    """
        scArches network with ZINB for loss function class. This class contains the implementation of scNet network.

        Parameters
        ----------
        x_dimension: int
            number of gene expression space dimensions.
        n_conditions: list
            list of unique conditions (i.e. batch ids) in the data used for one-hot encoding.
        z_dimension: int
            number of latent space dimensions.
        task_name: str
            name of the task.

        kwargs:
            `learning_rate`: float
                scNet's optimizer's step size (learning rate).
            `alpha`: float
                KL divergence coefficient in the loss function.
            `eta`: float
                Reconstruction coefficient in the loss function.
            `dropout_rate`: float
                dropout rate for Dropout layers in scNet's architecture.
            `model_path`: str
                path to save model config and its weights.
            `clip_value`: float
                Optimizer's clip value used for clipping the computed gradients.
            `output_activation`: str
                Output activation of scNet which Depends on the range of data.
            `use_batchnorm`: bool
                Whether use batch normalization in scNet or not.
            `architecture`: list
                Architecture of scNet. Must be a list of integers.
            `gene_names`: list
                names of genes fed as scNet's input. Must be a list of strings.
    """

    def __init__(self, x_dimension, conditions, task_name="unknown", z_dimension=10, **kwargs):
        kwargs.update({'loss_fn': 'zinb',
                       "model_name": "cvae_zinb", "class_name": "scArchesZINB"})

        self.size_factor_key = kwargs.pop("size_factor_key", 'size_factors')
        self.ridge = kwargs.pop("ridge", 0.1)
        self.size_factor = Input(shape=(1,), name='size_factor')

        super().__init__(x_dimension, conditions, task_name, z_dimension, **kwargs)

        self.network_kwargs.update({
            "size_factor_key": self.size_factor_key,
        })

        self.training_kwargs.update({
            "ridge": self.ridge,
        })

    def update_kwargs(self):
        super().update_kwargs()
        self.network_kwargs.update({
            "size_factor_key": self.size_factor_key,
        })

        self.training_kwargs.update({
            "ridge": self.ridge,
        })

    def _output_decoder(self, h):
        h_pi = Dense(self.x_dim, activation=ACTIVATIONS['sigmoid'], kernel_initializer=self.init_w, use_bias=True,
                     name='decoder_pi')(h)
        h_mean = Dense(self.x_dim, activation=None, kernel_initializer=self.init_w,
                       use_bias=True)(h)
        h_mean = ACTIVATIONS['mean_activation'](h_mean)

        h_disp = Dense(self.x_dim, activation=None, kernel_initializer=self.init_w,
                       use_bias=True)(h)
        h_disp = ACTIVATIONS['disp_activation'](h_disp)

        mean_output = LAYERS['ColWiseMultLayer']()([h_mean, self.size_factor])

        model_outputs = [mean_output, h_disp, h_pi]

        model_inputs = [self.z, self.decoder_labels, self.size_factor]

        return model_inputs, model_outputs

    @classmethod
    def from_config(cls, config_path, new_params=None, compile=True, construct=True):
        """create CVAE_ZINB object from exsiting CVAE_ZINB's config file.

        Parameters
        ----------
        config_path: str
            Path to class' config json file.
        new_params: dict, optional
            Python dict of parameters which you wanted to assign new values to them.
        compile: bool
            ``True`` by default. if ``True``, will compile class' model after creating an instance.
        construct: bool
            ``True`` by default. if ``True``, will construct class' model after creating an instance.
        """
        import json
        with open(config_path, 'rb') as f:
            class_config = json.load(f)

        class_config['construct_model'] = construct
        class_config['compile_model'] = compile

        if new_params:
            class_config.update(new_params)

        return cls(**class_config)

    def call(self, x, training=None, mask=None):
        if isinstance(x, list):
            expression, encoder_labels, decoder_labels, size_factors = x
        else:
            expression = x['expression']
            encoder_labels = x['encoder_label']
            decoder_labels = x['decoder_label']
            size_factors = x['size_factor']

        z_mean, z_log_var, z = self.encoder([expression, encoder_labels])

        x_hat, disp, pi = self.decoder([z, decoder_labels, size_factors])
        return x_hat, z_mean, z_log_var, disp, pi

    def construct_network(self):
        """
            Constructs the whole class' network. It is step-by-step constructing the scNet
            network. First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of scNet.
        """
        self._encoder("encoder")
        self._decoder("decoder")

        self.custom_objects = {'mean_activation': ACTIVATIONS['mean_activation'],
                               'disp_activation': ACTIVATIONS['disp_activation'],
                               'ColwiseMultLayer': LAYERS['ColWiseMultLayer'],
                               'FirstLayer': LAYERS['FirstLayer']}

        # Building the model via calling it with a random input
        input_arr = [tf.random.uniform((1, self.x_dim)), tf.ones((1, self.n_conditions)),
                     tf.ones((1, self.n_conditions)), tf.ones(1, dtype=tf.float32)]
        self(input_arr)

        get_custom_objects().update(self.custom_objects)
        print(f"{self.class_name}'s network has been successfully constructed!")

    def calc_losses(self, y_true, y_pred, z_mean, z_log_var, disp=None, pi=None):
        """
            Defines the loss function of class' network after constructing the whole
            network.
        """
        loss = LOSSES[self.loss_fn](pi, disp, z_mean, z_log_var, self.ridge,
                                    self.alpha, self.eta)(y_true, y_pred)
        recon_loss = LOSSES[f'{self.loss_fn}_wo_kl'](pi, disp, self.ridge, self.eta)(y_true, y_pred)
        kl_loss = LOSSES['kl'](z_mean, z_log_var)(y_true, y_pred)

        return loss, self.eta * recon_loss, self.alpha * kl_loss

    def forward_with_loss(self, data):
        x, y = data

        y = y['reconstruction']
        y_pred, z_mean, z_log_var, disp, pi = self.call(x)
        loss, recon_loss, kl_loss = self.calc_losses(y, y_pred, z_mean, z_log_var, disp, pi)

        return loss, recon_loss, kl_loss

    def _fit_dataset(self, adata,
                     condition_key, train_size=0.9,
                     n_epochs=100, batch_size=32, steps_per_epoch=50,
                     early_stop_limit=10, lr_reducer=8,
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
                                                             self.loss_fn, self.n_conditions, self.size_factor_key)
        valid_dataset, _ = make_dataset(valid_adata, condition_key, self.condition_encoder, valid_adata.shape[0],
                                        n_epochs, False,
                                        self.loss_fn, self.n_conditions, self.size_factor_key)

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
