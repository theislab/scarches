import os
import random

import anndata
import keras
import numpy as np
from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau, LambdaCallback
from keras.layers import Dense, BatchNormalization, Dropout, Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, model_from_json
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from scipy import sparse

from scarches.models._activations import ACTIVATIONS
from scarches.models._callbacks import ScoreCallback
from scarches.models._layers import LAYERS
from scarches.models._losses import LOSSES
from scarches.models._utils import sample_z, print_progress
from scarches.utils import label_encoder, remove_sparsity, create_condition_encoder, train_test_split


class CVAE(object):
    """CVAE class. This class contains the implementation of Conditional Variational Autoencoder network.

        Parameters
        ----------
        x_dimension: int
            number of gene expression space dimensions.
        n_conditions: int
            number of conditions used for one-hot encoding.
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

    def __init__(self, x_dimension, conditions, task_name="unknown", z_dimension=10, **kwargs):
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.task_name = task_name

        self.conditions = sorted(conditions)
        self.n_conditions = len(self.conditions)

        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.0001)
        self.eta = kwargs.get("eta", 1.0)
        self.dr_rate = kwargs.get("dropout_rate", 0.05)
        self.model_path = os.path.join(kwargs.get("model_path", "./models/CVAE/"), self.task_name)
        self.loss_fn = kwargs.get("loss_fn", 'mse')
        self.ridge = kwargs.get('ridge', 0.1)
        self.scale_factor = kwargs.get("scale_factor", 1.0)
        self.clip_value = kwargs.get('clip_value', 3.0)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.output_activation = kwargs.get("output_activation", 'relu')
        self.use_batchnorm = kwargs.get("use_batchnorm", True)
        self.architecture = kwargs.get("architecture", [128, 32])
        self.size_factor_key = kwargs.get("size_factor_key", 'size_factors')
        self.train_device = kwargs.get("train_device", "cpu")
        self.gene_names = kwargs.get("gene_names", None)
        self.model_name = kwargs.get("model_name", "cvae")
        self.class_name = kwargs.get("class_name", 'CVAE')

        self.freeze_expression_input = kwargs.get("freeze_expression_input", False)

        self.x = Input(shape=(self.x_dim,), name="data")
        self.size_factor = Input(shape=(1,), name='size_factor')
        self.encoder_labels = Input(shape=(self.n_conditions,), name="encoder_labels")
        self.decoder_labels = Input(shape=(self.n_conditions,), name="decoder_labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.condition_encoder = kwargs.get("condition_encoder", None)
        self.aux_models = {}

        self.network_kwargs = {
            "x_dimension": self.x_dim,
            "z_dimension": self.z_dim,
            "conditions": self.conditions,
            "dropout_rate": self.dr_rate,
            "loss_fn": self.loss_fn,
            "output_activation": self.output_activation,
            "size_factor_key": self.size_factor_key,
            "architecture": self.architecture,
            "use_batchnorm": self.use_batchnorm,
            "freeze_expression_input": self.freeze_expression_input,
            "gene_names": self.gene_names,
            "condition_encoder": self.condition_encoder,
            "train_device": self.train_device,
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

        if kwargs.get("construct_model", True):
            self.construct_network()

        if kwargs.get("construct_model", True) and kwargs.get("compile_model", True):
            self.compile_models()

        print_summary = kwargs.get("print_summary", False)
        if print_summary:
            self.encoder_model.summary()
            self.decoder_model.summary()
            self.cvae_model.summary()

    def update_kwargs(self):
        self.network_kwargs = {
            "x_dimension": self.x_dim,
            "z_dimension": self.z_dim,
            "conditions": self.conditions,
            "dropout_rate": self.dr_rate,
            "loss_fn": self.loss_fn,
            "output_activation": self.output_activation,
            "size_factor_key": self.size_factor_key,
            "architecture": self.architecture,
            "use_batchnorm": self.use_batchnorm,
            "freeze_expression_input": self.freeze_expression_input,
            "gene_names": self.gene_names,
            "condition_encoder": self.condition_encoder,
            "train_device": self.train_device,
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
        z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])
        model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
        return mean, log_var, model

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
                h = Dense(n_neuron, kernel_initializer=self.init_w,
                          use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
        model_inputs, model_outputs = self._output_decoder(h)
        model = Model(inputs=model_inputs, outputs=model_outputs, name=name)
        return model

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

    def construct_network(self):
        """
            Constructs the whole class' network. It is step-by-step constructing the scNet
            network. First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of scNet.
        """

        self.mu, self.log_var, self.encoder_model = self._encoder(name="encoder")
        self.decoder_model = self._decoder(name="decoder")

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

        reconstruction_output = Lambda(lambda x: x, name=self.loss_fn)(decoder_outputs)

        self.cvae_model = Model(inputs=inputs,
                                outputs=reconstruction_output,
                                name="cvae")

        self.custom_objects = {'mean_activation': ACTIVATIONS['mean_activation'],
                               'disp_activation': ACTIVATIONS['disp_activation'],
                               'SliceLayer': LAYERS['SliceLayer'],
                               'ColwiseMultLayer': LAYERS['ColWiseMultLayer'],
                               'FirstLayer': LAYERS['FirstLayer']}

        get_custom_objects().update(self.custom_objects)
        print(f"{self.class_name}'s network has been successfully constructed!")

    def _calculate_loss(self):
        """
            Defines the loss function of class' network after constructing the whole
            network.
        """
        if self.loss_fn == 'nb':
            loss = LOSSES[self.loss_fn](self.disp_output, self.mu, self.log_var, self.scale_factor, self.alpha,
                                        self.eta)
            kl_loss = LOSSES['kl'](self.mu, self.log_var)
            recon_loss = LOSSES['nb_wo_kl'](self.disp_output, self.scale_factor, self.eta)

        elif self.loss_fn == 'zinb':
            loss = LOSSES[self.loss_fn](self.pi_output, self.disp_output, self.mu, self.log_var, self.ridge, self.alpha,
                                        self.eta)
            kl_loss = LOSSES['kl'](self.mu, self.log_var)
            recon_loss = LOSSES['zinb_wo_kl'](self.pi_output, self.disp_output, self.ridge, self.eta)

        else:
            loss = LOSSES[self.loss_fn](self.mu, self.log_var, self.alpha, self.eta)
            kl_loss = LOSSES['kl'](self.mu, self.log_var)
            recon_loss = LOSSES[f'{self.loss_fn}_recon']

        return loss, kl_loss, recon_loss

    def compile_models(self):
        """
            Compiles scNet network with the defined loss functions and
            Adam optimizer with its pre-defined hyper-parameters.
        """
        optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value, epsilon=self.epsilon)
        loss, kl_loss, recon_loss = self._calculate_loss()

        self.cvae_model.compile(optimizer=optimizer,
                                loss=loss,
                                metrics=[recon_loss, kl_loss],
                                )

        print(f"{self.class_name}'s network has been successfully compiled!")

    def get_summary_of_networks(self):
        """Prints summary of scNet sub-networks.
        """
        self.encoder_model.summary()
        self.decoder_model.summary()
        self.cvae_model.summary()

    def to_mmd_layer(self, adata, encoder_labels, decoder_labels):
        """
            CVAE has no MMD Layer to project input on it.

            Raises
            ------
            Exception
        """
        raise NotImplementedError("There are no MMD layer in CVAE")

    def get_latent(self, adata, batch_key):
        """ Transforms `adata` in latent space of CVAE and returns the latent
        coordinates in the annotated (adata) format.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset matrix in Primary space.




        """
        if set(self.gene_names).issubset(set(adata.var_names)):
            adata = adata[:, self.gene_names]
        else:
            raise Exception("set of gene names in train adata are inconsistent with scNet's gene_names")

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)

        return self.get_z_latent(adata, encoder_labels)

    def get_z_latent(self, adata, encoder_labels):
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

            Returns
            -------
            adata_latent: :class:`~anndata.AnnData`
                returns Annotated data containing latent space encoding of ``adata``
        """
        adata = remove_sparsity(adata)

        encoder_inputs = [adata.X, encoder_labels]

        latent = self.encoder_model.predict(encoder_inputs)[2]
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
        if self.loss_fn in ['nb', 'zinb']:
            inputs = [adata.X, encoder_labels, decoder_labels, self.adata.obs[self.size_factor_key]]
        else:
            inputs = [adata.X, encoder_labels, decoder_labels]

        x_hat = self.cvae_model.predict(inputs)

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
            self.cvae_model.load_weights(os.path.join(self.model_path, f'{self.model_name}.h5'))

            self.encoder_model = self.cvae_model.get_layer("encoder")
            self.decoder_model = self.cvae_model.get_layer("decoder")

            if compile:
                self.compile_models()
            print(f"{self.model_name}'s weights has been successfully restored!")
            return True
        return False

    def restore_model_config(self, compile=True):
        """
            restores model config from ``model_path``.

            Parameters
            ----------
            compile: bool
                if ``True`` will compile model after restoring its config.

            Returns
            -------
            ``True`` if the model config has been successfully restored.
            ``False`` if `model_path` is invalid or the model config couldn't be found in the specified ``model_path``.
        """
        if os.path.exists(os.path.join(self.model_path, f"{self.model_name}.json")):
            json_file = open(os.path.join(self.model_path, f"{self.model_name}.json"), 'rb')
            loaded_model_json = json_file.read()
            self.cvae_model = model_from_json(loaded_model_json)
            self.encoder_model = self.cvae_model.get_layer("encoder")
            self.decoder_model = self.cvae_model.get_layer("decoder")

            if compile:
                self.compile_models()

            print(f"{self.model_name}'s network's config has been successfully restored!")
            return True
        else:
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
                scNet_config = json.load(f)

            # Update network_kwargs and training_kwargs dictionaries
            for key, value in scNet_config.items():
                if key in self.network_kwargs.keys():
                    self.network_kwargs[key] = value
                elif key in self.training_kwargs.keys():
                    self.training_kwargs[key] = value

            # Update class attributes
            for key, value in scNet_config.items():
                setattr(self, key, value)

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
            self.save_model_config(make_dir)
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
            self.cvae_model.save_weights(os.path.join(self.model_path, f"{self.model_name}.h5"),
                                         overwrite=True)
            return True
        else:
            return False

    def save_model_config(self, make_dir=True):
        """
            Saves model's config in the ``model_path``.

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
            model_json = self.cvae_model.to_json()
            with open(os.path.join(self.model_path, f"{self.model_name}.json"), 'w') as file:
                file.write(model_json)
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
              condition_key, train_size=0.8, cell_type_key='cell_type',
              n_epochs=25, batch_size=32,
              early_stop_limit=20, lr_reducer=10,
              n_per_epoch=0, score_filename=None,
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

        if self.train_device == 'gpu':
            return self._fit(adata, condition_key, train_size, cell_type_key, n_epochs, batch_size, early_stop_limit,
                             lr_reducer, n_per_epoch, score_filename, save, retrain, verbose)
        else:
            return self._train_on_batch(adata, condition_key, train_size, cell_type_key, n_epochs, batch_size,
                                        early_stop_limit, lr_reducer, n_per_epoch, score_filename, save, retrain,
                                        verbose)

    def _fit(self, adata,
             condition_key, train_size=0.8, cell_type_key='cell_type',
             n_epochs=25, batch_size=32,
             early_stop_limit=20, lr_reducer=10,
             n_per_epoch=0, score_filename=None,
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

        if self.loss_fn in ['nb', 'zinb']:
            train_raw_expr = train_adata.raw.X.A if sparse.issparse(train_adata.raw.X) else train_adata.raw.X
            valid_raw_expr = valid_adata.raw.X.A if sparse.issparse(valid_adata.raw.X) else valid_adata.raw.X

        train_expr = train_adata.X.A if sparse.issparse(train_adata.X) else train_adata.X
        valid_expr = valid_adata.X.A if sparse.issparse(valid_adata.X) else valid_adata.X

        train_conditions_encoded, self.condition_encoder = label_encoder(train_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        valid_conditions_encoded, self.condition_encoder = label_encoder(valid_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

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

        if (n_per_epoch > 0 or n_per_epoch == -1) and not score_filename:
            adata = train_adata.concatenate(valid_adata)

            train_celltypes_encoded, _ = label_encoder(train_adata, le=None, condition_key=cell_type_key)
            valid_celltypes_encoded, _ = label_encoder(valid_adata, le=None, condition_key=cell_type_key)
            celltype_labels = np.concatenate([train_celltypes_encoded, valid_celltypes_encoded], axis=0)

            callbacks.append(ScoreCallback(score_filename, adata, condition_key, cell_type_key, self.cvae_model,
                                           n_per_epoch=n_per_epoch, n_batch_labels=self.n_conditions,
                                           n_celltype_labels=len(np.unique(celltype_labels))))

        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor='val_loss'))

        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=lr_reducer))

        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        x_train = [train_expr, train_conditions_onehot, train_conditions_onehot]
        x_valid = [valid_expr, valid_conditions_onehot, valid_conditions_onehot]

        if self.loss_fn in ['nb', 'zinb']:
            x_train.append(train_adata.obs[self.size_factor_key].values)
            y_train = train_raw_expr

            x_valid.append(valid_adata.obs[self.size_factor_key].values)
            y_valid = valid_raw_expr
        else:
            y_train = train_expr
            y_valid = valid_expr

        self.cvae_model.fit(x=x_train,
                            y=y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=n_epochs,
                            batch_size=batch_size,
                            verbose=fit_verbose,
                            callbacks=callbacks,
                            )
        if save:
            self.update_kwargs()
            self.save(make_dir=True)

    def _train_on_batch(self, adata,
                        condition_key, train_size=0.8, cell_type_key='cell_type',
                        n_epochs=25, batch_size=32,
                        early_stop_limit=20, lr_reducer=10,
                        n_per_epoch=0, score_filename=None,
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

        train_conditions_encoded, self.condition_encoder = label_encoder(train_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        valid_conditions_encoded, self.condition_encoder = label_encoder(valid_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

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

        if (n_per_epoch > 0 or n_per_epoch == -1) and not score_filename:
            adata = train_adata.concatenate(valid_adata)

            train_celltypes_encoded, _ = label_encoder(train_adata, le=None, condition_key=cell_type_key)
            valid_celltypes_encoded, _ = label_encoder(valid_adata, le=None, condition_key=cell_type_key)
            celltype_labels = np.concatenate([train_celltypes_encoded, valid_celltypes_encoded], axis=0)

            callbacks.append(ScoreCallback(score_filename, adata, condition_key, cell_type_key, self.cvae_model,
                                           n_per_epoch=n_per_epoch, n_batch_labels=self.n_conditions,
                                           n_celltype_labels=len(np.unique(celltype_labels))))

        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor='val_loss'))

        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=lr_reducer))

        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        valid_expr = valid_adata.X.A if sparse.issparse(valid_adata.X) else valid_adata.X
        x_valid = [valid_expr, valid_conditions_onehot, valid_conditions_onehot]

        if self.loss_fn in ['nb', 'zinb']:
            x_valid.append(valid_adata.obs[self.size_factor_key].values)
            y_valid = valid_adata.raw.X.A if sparse.issparse(valid_adata.raw.X) else valid_adata.raw.X
        else:
            y_valid = valid_expr

        for i in range(n_epochs):
            train_loss = train_recon_loss = train_kl_loss = 0.0
            for j in range(min(1000, train_adata.shape[0] // batch_size)):
                batch_indices = np.random.choice(train_adata.shape[0], batch_size)

                batch_expr = train_adata.X[batch_indices, :]

                batch_expr = batch_expr.A if sparse.issparse(batch_expr) else batch_expr

                x_train = [batch_expr, train_conditions_onehot[batch_indices], train_conditions_onehot[batch_indices]]

                if self.loss_fn in ['nb', 'zinb']:
                    x_train.append(train_adata.obs[self.size_factor_key].values[batch_indices])
                    y_train = train_adata.raw.X[batch_indices].A if sparse.issparse(
                        train_adata.raw.X[batch_indices]) else train_adata.raw.X[batch_indices]
                else:
                    y_train = batch_expr

                batch_loss, batch_recon_loss, batch_kl_loss = self.cvae_model.train_on_batch(x_train, y_train)

                train_loss += batch_loss / batch_size
                train_recon_loss += batch_recon_loss / batch_size
                train_kl_loss += batch_kl_loss / batch_size

            valid_loss, valid_recon_loss, valid_kl_loss = self.cvae_model.evaluate(x_valid, y_valid, verbose=0)

            logs = {"loss": train_loss, "recon_loss": train_recon_loss, "kl_loss": train_kl_loss,
                    "val_loss": valid_loss, "val_recon_loss": valid_recon_loss, "val_kl_loss": valid_kl_loss}
            print_progress(i, logs, n_epochs)

        if save:
            self.update_kwargs()
            self.save(make_dir=True)
