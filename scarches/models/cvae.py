import os

import anndata
import keras
import numpy as np
import tensorflow as tf
from scipy import sparse
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, History, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, LeakyReLU
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed

from scarches.models._activations import ACTIVATIONS
from scarches.models._callbacks import ScoreCallback
from scarches.models._data_generator import make_dataset
from scarches.models._layers import LAYERS
from scarches.models._losses import LOSSES
from scarches.models._utils import print_progress
from scarches.utils import label_encoder, remove_sparsity, create_condition_encoder, train_test_split


class CVAE(Model):
    """CVAE class. This class contains the implementation of Conditional Variational Autoencoder network.

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
                CVAE's optimizer's step size (learning rate).
            `alpha`: float
                KL divergence coefficient in the loss function.
            `eta`: float
                Reconstruction coefficient in the loss function.
            `dropout_rate`: float
                dropout rate for Dropout layers in CVAE's architecture.
            `model_path`: str
                path to save model config and its weights.
            `clip_value`: float
                Optimizer's clip value used for clipping the computed gradients.
            `output_activation`: str
                Output activation of CVAE which Depends on the range of data.
            `use_batchnorm`: bool
                Whether use batch normalization in CVAE or not.
            `architecture`: list
                Architecture of CVAE. Must be a list of integers.
            `gene_names`: list
                names of genes fed as CVAE's input. Must be a list of strings.

    """

    def __init__(self, x_dimension: int, conditions: list, task_name: str = "unknown", z_dimension: int = 10, *args,
                 **kwargs):

        tf.config.run_functions_eagerly(True)
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.task_name = task_name

        self.conditions = list(sorted(conditions))
        self.n_conditions = len(self.conditions)

        self.lr = kwargs.pop("lr", 0.001)
        self.alpha = kwargs.pop("alpha", 0.0001)
        self.eta = kwargs.pop("eta", 1.0)
        self.dr_rate = kwargs.pop("dropout_rate", 0.1)
        self.model_base_path = kwargs.pop("model_path", "./models/CVAE/")
        self.model_path = os.path.join(self.model_base_path, self.task_name)
        self.loss_fn = kwargs.pop("loss_fn", 'mse')
        self.clip_value = kwargs.pop('clip_value', 3.0)
        self.epsilon = kwargs.pop('epsilon', 0.01)
        self.output_activation = kwargs.pop("output_activation", 'linear')
        self.use_batchnorm = kwargs.pop("use_batchnorm", True)
        self.architecture = kwargs.pop("architecture", [128, 128])
        self.device = kwargs.pop("device", None)
        self.gene_names = kwargs.pop("gene_names", None)
        self.model_name = kwargs.pop("model_name", "cvae")
        self.class_name = kwargs.pop("class_name", 'CVAE')
        self.freeze_expression_input = kwargs.pop("freeze_expression_input", False)
        self.condition_encoder = kwargs.pop("condition_encoder", None)
        self.seed = kwargs.pop('seed', 2020)
        set_seed(self.seed)

        construct_model = kwargs.pop("construct_model", True)
        compile_model = kwargs.pop("compile_model", True)
        print_summary = kwargs.pop("print_summary", False)

        super().__init__(*args, **kwargs)

        if self.device is None:
            self.device = 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu'

        if self.device == 'gpu' and len(tf.config.list_physical_devices('GPU')) == 0:
            print("WARNING: You have set the variable `device` to \'GPU\' but your system does not have any GPUs.")
            self.device = 'cpu'

        print(f"Start running on {self.device}...")

        self.x = Input(shape=(self.x_dim,), name="expression")
        self.encoder_labels = Input(shape=(self.n_conditions,), name="encoder_label")
        self.decoder_labels = Input(shape=(self.n_conditions,), name="decoder_label")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.network_kwargs = {
            "x_dimension": self.x_dim,
            "z_dimension": self.z_dim,
            "conditions": self.conditions,
            "dropout_rate": self.dr_rate,
            "loss_fn": self.loss_fn,
            "output_activation": self.output_activation,
            "architecture": self.architecture,
            "use_batchnorm": self.use_batchnorm,
            "freeze_expression_input": self.freeze_expression_input,
            "gene_names": self.gene_names,
            "condition_encoder": self.condition_encoder,
            "train_device": self.device,
            "model_path": self.model_base_path,
            "seed": self.seed,
        }

        self.training_kwargs = {
            "lr": self.lr,
            "alpha": self.alpha,
            "eta": self.eta,
            "clip_value": self.clip_value,
        }

        self.init_w = keras.initializers.glorot_normal()

        if construct_model:
            self.construct_network()

        if construct_model and compile_model:
            self.compile_models()

        if print_summary:
            self.encoder_model.summary()
            self.decoder_model.summary()
            self.summary()

    def update_kwargs(self):
        self.network_kwargs = {
            "x_dimension": self.x_dim,
            "z_dimension": self.z_dim,
            "conditions": self.conditions,
            "dropout_rate": self.dr_rate,
            "loss_fn": self.loss_fn,
            "output_activation": self.output_activation,
            "architecture": self.architecture,
            "use_batchnorm": self.use_batchnorm,
            "freeze_expression_input": self.freeze_expression_input,
            "gene_names": self.gene_names,
            "condition_encoder": self.condition_encoder,
            "model_path": self.model_base_path,
            "device": self.device,
            "seed": self.seed,
        }

        self.training_kwargs = {
            "lr": self.lr,
            "alpha": self.alpha,
            "eta": self.eta,
            "clip_value": self.clip_value,
        }

    @classmethod
    def from_config(cls, config_path, new_params=None, compile=True, construct=True):
        """create CVAE object from exsiting CVAE's config file.

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

    def _encoder(self, name="encoder"):
        """
           Constructs the decoder sub-network of CVAE. This function implements the
           decoder part of CVAE. It will transform primary space input to
           latent space to with n_dimensions = z_dimension.
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
        z = LAYERS['Sampling']()([mean, log_var])
        self.encoder = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
        return mean, log_var, z

    def _decoder(self, name="decoder"):
        """
            Constructs the decoder sub-network of CVAE. This function implements the
            decoder part of scNet. It will transform constructed
            latent space to the previous space of data with n_dimensions = x_dimension.
        """
        for idx, n_neuron in enumerate(self.architecture[::-1]):
            if idx == 0:
                h = LAYERS['FirstLayer'](n_neuron, kernel_initializer=self.init_w,
                                         use_bias=False, name="first_layer", freeze=self.freeze_expression_input)(
                    [self.z, self.decoder_labels])
            else:
                h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
        model_inputs, model_outputs = self._output_decoder(h)
        self.decoder = Model(inputs=model_inputs, outputs=model_outputs, name=name)

    def _output_decoder(self, h):
        h = Dense(self.x_dim, activation=None,
                  kernel_initializer=self.init_w,
                  use_bias=True)(h)
        h = ACTIVATIONS[self.output_activation](h)
        model_inputs = [self.z, self.decoder_labels]
        model_outputs = [h]

        return model_inputs, model_outputs

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
                     tf.ones((1, self.n_conditions))]
        self(input_arr)

        get_custom_objects().update(self.custom_objects)
        print(f"{self.class_name}'s network has been successfully constructed!")

    def compile_models(self):
        """
            Compiles scNet network with the defined loss functions and
            Adam optimizer with its pre-defined hyper-parameters.
        """
        self.optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value, epsilon=self.epsilon)
        self.compile(optimizer=self.optimizer)

        print(f"{self.class_name}'s network has been successfully compiled!")

    def call(self, x, training=None, mask=None):
        if isinstance(x, list):
            expression, encoder_labels, decoder_labels = x
        else:
            expression = x['expression']
            encoder_labels = x['encoder_label']
            decoder_labels = x['decoder_label']

        z_mean, z_log_var, z = self.encoder([expression, encoder_labels])

        x_hat = self.decoder([z, decoder_labels])
        return x_hat, z_mean, z_log_var

    def calc_losses(self, y_true, y_pred, z_mean, z_log_var, disp=None, pi=None):
        """
            Defines the loss function of class' network after constructing the whole
            network.
        """
        recon_loss = LOSSES[f'{self.loss_fn}_recon'](y_true, y_pred)
        kl_loss = LOSSES['kl'](z_mean, z_log_var)(y_true, y_pred)
        loss = self.eta * recon_loss + self.alpha * kl_loss

        return loss, self.eta*recon_loss, self.alpha*kl_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, recon_loss, kl_loss = self.forward_with_loss(data)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": loss,
            f'{self.loss_fn}_loss': recon_loss,
            "kl_loss": kl_loss
        }

    def forward_with_loss(self, data):
        x, y = data
        y = y['reconstruction']
        y_pred, z_mean, z_log_var = self.call(x)
        loss, recon_loss, kl_loss = self.calc_losses(y, y_pred, z_mean, z_log_var)

        return loss, recon_loss, kl_loss

    def test_step(self, data):
        loss, recon_loss, kl_loss = self.forward_with_loss(data)

        return {
            'loss': loss,
            f'{self.loss_fn}_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def get_summary_of_networks(self):
        """Prints summary of scNet sub-networks.
        """
        self.encoder_model.summary()
        self.decoder_model.summary()
        self.summary()

    def to_mmd_layer(self, adata, encoder_labels, decoder_labels):
        """
            CVAE has no MMD Layer to project input on it.

            Raises
            ------
            Exception
        """
        raise NotImplementedError("There are no MMD layer in CVAE")

    def get_latent(self, adata, batch_key, return_mean=False):
        """ Transforms `adata` in latent space of CVAE and returns the latent
        coordinates in the annotated (adata) format.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset matrix in Primary space.
        batch_key: str
            key for the observation that has batch labels in adata.obs.

        return_mean: bool
            if False, z will be sampled. Set to `True` if want a fix z value every time you call
             get_latent.

        Returns
        -------
        latent_adata: :class:`~anndata.AnnData`
            Annotated dataset matrix in Latent space.



        """
        if set(self.gene_names).issubset(set(adata.var_names)):
            adata = adata[:, self.gene_names]
        else:
            raise Exception("set of gene names in train adata are inconsistent with scNet's gene_names")

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)

        return self.get_z_latent(adata, encoder_labels, return_mean)

    def get_z_latent(self, adata, encoder_labels, return_mean=False):
        """
            Map ``adata`` in to the latent space. This function will feed data
            in encoder part of scNet and compute the latent space coordinates
            for each sample in data.

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated data matrix to be mapped to latent space.
                Please note that `adata.X` has to be in shape [n_obs, x_dimension]
            encoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as class' condition array.

            return_mean: bool
                if False, z will be sampled. Set to `True` if want a fix z value every time you call
                get_latent.


            Returns
            -------
            adata_latent: :class:`~anndata.AnnData`
                returns Annotated data containing latent space encoding of ``adata``
        """
        adata = remove_sparsity(adata)

        encoder_inputs = [adata.X, encoder_labels]
        if return_mean:
            latent = self.encoder.predict(encoder_inputs)[0]
        else:
            latent = self.encoder.predict(encoder_inputs)[2]

        latent = np.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0)

        adata_latent = anndata.AnnData(X=latent)
        adata_latent.obs = adata.obs.copy(deep=True)

        return adata_latent

    def predict(self, adata, encoder_labels, decoder_labels):
        """Feeds ``adata`` to scNet and produces the reconstructed data.

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated data matrix whether in primary space.
            encoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as class' encoder condition array.
            decoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as class' decoder condition array.

            Returns
            -------
            adata_pred: `~anndata.AnnData`
                Annotated data of predicted cells in primary space.
        """
        adata = remove_sparsity(adata)

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        inputs = [adata.X, encoder_labels, decoder_labels]

        x_hat = super().predict(inputs)

        adata_pred = anndata.AnnData(X=x_hat)
        adata_pred.obs = adata.obs
        adata_pred.var_names = adata.var_names

        return adata_pred

    def restore_model_weights(self, compile=True):
        """
            restores model weights from ``model_path``.

            Parameters
            ----------
            compile: bool
                if ``True`` will compile model after restoring its weights.

            Returns
            -------
            ``True`` if the model has been successfully restored.
            ``False`` if ``model_path`` is invalid or the model weights couldn't be found in the specified ``model_path``.
        """
        if os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.load_weights(os.path.join(self.model_path, f'{self.model_name}.h5'))

            self.encoder = self.get_layer("encoder")
            self.decoder = self.get_layer("decoder")

            if compile:
                self.compile_models()
            print(f"{self.model_name}'s weights has been successfully restored!")
            return True
        return False

    def restore_class_config(self, compile_and_consturct=True):
        """
            restores class' config from ``model_path``.

            Parameters
            ----------
            compile_and_consturct: bool
                if ``True`` will construct and compile model from scratch.

            Returns
            -------
            ``True`` if the scNet config has been successfully restored.
            ``False`` if `model_path` is invalid or the class' config couldn't be found in the specified ``model_path``.
        """
        import json
        if os.path.exists(os.path.join(self.model_path, f"{self.class_name}.json")):
            with open(os.path.join(self.model_path, f"{self.class_name}.json"), 'rb') as f:
                scArches_config = json.load(f)

            # Update network_kwargs and training_kwargs dictionaries
            for key, value in scArches_config.items():
                if key in self.network_kwargs.keys():
                    self.network_kwargs[key] = value
                elif key in self.training_kwargs.keys():
                    self.training_kwargs[key] = value

            # Update class attributes
            for key, value in scArches_config.items():
                setattr(self, key, value)
                if key == 'model_path':
                    self.model_base_path = self.model_path
                    self.model_path = os.path.join(self.model_base_path, self.task_name)

            if compile_and_consturct:
                self.construct_network()
                self.compile_models()

            print(f"{self.class_name}'s config has been successfully restored!")
            return True
        else:
            return False

    def save(self, make_dir=True):
        """
            Saves all model weights, configs, and hyperparameters in the ``model_path``.

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
            ``True`` if the model has been successfully saved.
            ``False`` if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            self.save_model_weights(make_dir)
            self.save_class_config(make_dir)
            print(f"\n{self.class_name} has been successfully saved in {self.model_path}.")
            return True
        else:
            return False

    def save_model_weights(self, make_dir=True):
        """
            Saves model weights in the ``model_path``.

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
            ``True`` if the model has been successfully saved.
            ``False`` if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            self.save_weights(os.path.join(self.model_path, f"{self.model_name}.h5"),
                              overwrite=True)
            return True
        else:
            return False

    def save_class_config(self, make_dir=True):
        """
            Saves class' config in the ``model_path``.

            Parameters
            ----------
            make_dir: bool
                Whether makes ``model_path`` directory if it does not exists.

            Returns
            -------
                ``True`` if the model has been successfully saved.
                ``False`' if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        import json

        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            config = {"x_dimension": self.x_dim,
                      "z_dimension": self.z_dim,
                      "n_conditions": self.n_conditions,
                      "task_name": self.task_name,
                      "condition_encoder": self.condition_encoder,
                      "gene_names": self.gene_names}
            all_configs = dict(list(self.network_kwargs.items()) +
                               list(self.training_kwargs.items()) +
                               list(config.items()))
            with open(os.path.join(self.model_path, f"{self.class_name}.json"), 'w') as f:
                json.dump(all_configs, f)

            return True
        else:
            return False

    def set_condition_encoder(self, condition_encoder=None, conditions=None):
        """
            Sets condition encoder of scNet

            Parameters
            ----------
            condition_encoder: dict
                dictionary with conditions as key and integers as value
            conditions: list
                list of unique conditions exist in annotated data for training

            Returns
            -------
            ``True`` if the model has been successfully saved.
            ``False`' if ``model_path`` is an invalid path and ``make_dir`` is set to ``False``.
        """
        if condition_encoder:
            self.condition_encoder = condition_encoder
        elif not condition_encoder and conditions:
            self.condition_encoder = create_condition_encoder(conditions, [])
        else:
            raise Exception("Either condition_encoder or conditions have to be passed.")

    def train(self, adata,
              condition_key, train_size=0.8,
              n_epochs=300, batch_size=32, steps_per_epoch=100,
              early_stop_limit=15, lr_reducer=10,
              save=True, retrain=True, verbose=3):

        """
            Trains the network with ``n_epochs`` times given ``adata``.
            This function is using ``early stopping`` and ``learning rate reduce on plateau``
            techniques to prevent over-fitting.
            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated dataset used to train & evaluate scNet.
            condition_key: str
                column name for conditions in the `obs` matrix of `train_adata` and `valid_adata`.
            train_size: float
                fraction of samples in `adata` used to train scNet.
            n_epochs: int
                number of epochs.
            steps_per_epoch: int
                Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
            batch_size: int
                number of samples in the mini-batches used to optimize scNet.
            early_stop_limit: int
                patience of EarlyStopping
            lr_reducer: int
                patience of LearningRateReduceOnPlateau.
            save: bool
                Whether to save scNet after the training or not.
            verbose: int
                Verbose level
            retrain: bool
                ``True`` by default. if ``True`` scNet will be trained regardless of existance of pre-trained scNet in ``model_path``. if ``False`` scNet will not be trained if pre-trained scNet exists in ``model_path``.

        """
        return self._fit_dataset(adata, condition_key, train_size, n_epochs, batch_size, steps_per_epoch,
                                 early_stop_limit,
                                 lr_reducer, save, retrain, verbose)

    def _fit_dataset(self, adata,
                     condition_key, train_size=0.8,
                     n_epochs=100, batch_size=128, steps_per_epoch=100,
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
            self.restore_class_config(compile_and_consturct=False)
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
                                                             batch_size, n_epochs, steps_per_epoch=steps_per_epoch,
                                                             is_training=True, loss_fn=self.loss_fn,
                                                             n_conditions=self.n_conditions)
        valid_dataset, _ = make_dataset(valid_adata, condition_key, self.condition_encoder, valid_adata.shape[0],
                                        n_epochs, steps_per_epoch=1,
                                        is_training=False, loss_fn=self.loss_fn, n_conditions=self.n_conditions)

        self.log_history = self.fit(train_dataset,
                                    validation_data=valid_dataset,
                                    epochs=n_epochs,
                                    batch_size=batch_size,
                                    verbose=fit_verbose,
                                    callbacks=callbacks,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_steps=1,
                                    use_multiprocessing=True,
                                    workers=8,
                                    )

        if save:
            self.update_kwargs()
            self.save(make_dir=True)

    def plot_training_history(self):
        from matplotlib import pyplot as plt
        import numpy as np

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(20, 10))
        fig.suptitle('Training History')

        min_loss = min(self.log_history.history['val_loss'] + self.log_history.history['loss'])
        max_loss = max(self.log_history.history['val_loss'] + self.log_history.history['loss'])
        n_epochs = len(self.log_history.history['loss'])
        ax1.plot(self.log_history.history['loss'], label='Train')
        ax1.plot(self.log_history.history['val_loss'], label='Validation')
        ax1.set_yticks(np.arange(min_loss, max_loss, (max_loss - min_loss) / 10))
        ax1.set_xticks(np.arange(0, n_epochs + 1, int(n_epochs) // 10))
        ax1.legend()
        ax1.set_title('Total Loss')

        min_loss = min(
            self.log_history.history[f'val_{self.loss_fn}_loss'] + self.log_history.history[f'{self.loss_fn}_loss'])
        max_loss = max(
            self.log_history.history[f'val_{self.loss_fn}_loss'] + self.log_history.history[f'{self.loss_fn}_loss'])
        ax2.plot(self.log_history.history[f'{self.loss_fn}_loss'], label='Train')
        ax2.plot(self.log_history.history[f'val_{self.loss_fn}_loss'], label='Validation')
        ax2.set_yticks(np.arange(min_loss, max_loss, (max_loss - min_loss) / 10))
        ax2.set_xticks(np.arange(0, n_epochs + 1, int(n_epochs) // 10))
        ax2.legend()
        ax2.set_title('Reconstruction Loss')

        min_loss = min(self.log_history.history['val_kl_loss'] + self.log_history.history['kl_loss'])
        max_loss = max(self.log_history.history['val_kl_loss'] + self.log_history.history['kl_loss'])
        ax3.plot(self.log_history.history['kl_loss'], label='Train')
        ax3.plot(self.log_history.history['val_kl_loss'], label='Validation')
        ax3.set_yticks(np.arange(min_loss, max_loss, (max_loss - min_loss) / 10))
        ax3.set_xticks(np.arange(0, n_epochs + 1, int(n_epochs) // 10))
        ax3.legend()
        ax3.set_title('KL Loss')

        plt.show()
