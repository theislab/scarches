import os

import anndata
import keras
import numpy as np
from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau, LambdaCallback
from keras.layers import Dense, BatchNormalization, Dropout, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects

from scnet.models import CVAE
from scnet.models._activations import ACTIVATIONS
from scnet.models._callbacks import ScoreCallback
from scnet.models._layers import LAYERS
from scnet.models._losses import LOSSES
from scnet.models._utils import print_progress
from scnet.utils import label_encoder, remove_sparsity, train_test_split


class scNet(CVAE):
    """scNet class. This class contains the implementation of scNet network.

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
                scNet's optimizer's step size (learning rate).
            `alpha`: float
                KL divergence coefficient in the loss function.
            `beta`: float
                MMD loss coefficient in the loss function.
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

    def __new__(cls, *args, **kwargs):
        loss_fn = kwargs.get("loss_fn", "mse")
        if loss_fn in ['nb', 'zinb']:
            kwargs.pop('beta')
            kwargs.pop('n_mmd_conditions')
            kwargs.pop('mmd_computation_method')

            if loss_fn == 'nb':
                from .scnetnb import scNetNB
                return scNetNB(*args, **kwargs)
            elif loss_fn == 'zinb':
                from .scnetzinb import scNetZINB
                return scNetZINB(*args, **kwargs)
        else:
            return super(scNet, cls).__new__(cls)

    def __init__(self, x_dimension, n_conditions, task_name="unknown", z_dimension=100, **kwargs):
        self.beta = kwargs.pop('beta', 20.0)
        self.n_mmd_conditions = kwargs.pop("n_mmd_conditions", n_conditions)
        self.mmd_computation_method = kwargs.pop("mmd_computation_method", "general")

        if kwargs.get("loss_fn", "mse") in ['nb', 'zinb']:
            kwargs['loss_fn'] = 'mse'

        kwargs.update({"model_name": "cvae", "class_name": "scNet"})

        super().__init__(x_dimension, n_conditions, task_name, z_dimension, **kwargs)

        self.network_kwargs.update({
            "n_mmd_conditions": self.n_mmd_conditions,
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
            Path to scNet's config json file.
        new_params: dict, optional
            Python dict of parameters which you wanted to assign new values to them.
        compile: bool
            ``True`` by default. if ``True``, will compile scNet's model after creating an instance.
        construct: bool
            ``True`` by default. if ``True``, will construct scNet's model after creating an instance.
        """
        import json
        with open(config_path, 'rb') as f:
            class_config = json.load(f)

        class_config['construct_model'] = construct
        class_config['compile_model'] = compile

        if new_params:
            class_config.update(new_params)

        return cls(**class_config)

    def _output_decoder(self, h):
        h = Dense(self.x_dim, activation=None,
                  kernel_initializer=self.init_w,
                  use_bias=True)(h)
        h = ACTIVATIONS[self.output_activation](h)
        model_inputs = [self.z, self.decoder_labels]
        model_outputs = [h]

        return model_inputs, model_outputs

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
        model = Model(inputs=model_inputs, outputs=model_outputs, name=name)
        mmd_model = Model(inputs=model_inputs, outputs=h_mmd, name='mmd_decoder')
        return model, mmd_model

    def construct_network(self):
        """
            Constructs the whole scNet's network. It is step-by-step constructing the scNet network.
            First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of scNet.
        """
        self.mu, self.log_var, self.encoder_model = self._encoder(name="encoder")
        self.decoder_model, self.decoder_mmd_model = self._decoder(name="decoder")

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
        print(f"{self.class_name}'s network has been successfully constructed!")

    def _calculate_loss(self):
        """
            Defines the loss function of scNet's network after constructing the whole
            network.
        """
        loss = LOSSES[self.loss_fn](self.mu, self.log_var, self.alpha, self.eta)
        mmd_loss = LOSSES['mmd'](self.n_mmd_conditions, self.beta)
        kl_loss = LOSSES['kl'](self.mu, self.log_var)
        recon_loss = LOSSES[f'{self.loss_fn}_recon']

        return loss, mmd_loss, kl_loss, recon_loss

    def compile_models(self):
        """
            Compiles scNet network with the defined loss functions and
            Adam optimizer with its pre-defined hyper-parameters.
        """
        optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value, epsilon=self.epsilon)
        loss, mmd_loss, kl_loss, recon_loss = self._calculate_loss()

        self.cvae_model.compile(optimizer=optimizer,
                                loss=[loss, mmd_loss],
                                metrics={self.cvae_model.outputs[0].name: loss,
                                         self.cvae_model.outputs[1].name: mmd_loss}
                                )

        print("scNet's network has been successfully compiled!")

    def to_mmd_layer(self, adata, batch_key):
        """
            Map ``adata`` in to the MMD space. This function will feed data
            in ``mmd_model`` of scNet and compute the MMD space coordinates
            for each sample in data.

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated data matrix to be mapped to MMD latent space.
                Please note that ``adata.X`` has to be in shape [n_obs, x_dimension]
            encoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as scNet's encoder condition array.
            decoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as scNet's decoder condition array.

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

        mmd = self.cvae_model.predict(cvae_inputs)[1]
        mmd = np.nan_to_num(mmd, nan=0.0, posinf=0.0, neginf=0.0)

        adata_mmd = anndata.AnnData(X=mmd)
        adata_mmd.obs = adata.obs.copy(deep=True)

        return adata_mmd

    def get_latent(self, adata, batch_key, return_z=False):
        """ Transforms `adata` in latent space of scNet and returns the latent
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
            raise Exception("set of gene names in train adata are inconsistent with scNet's gene_names")

        if self.beta == 0:
            return_z = True

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)

        if return_z or self.beta == 0:
            return self.get_z_latent(adata, encoder_labels)
        else:
            return self.to_mmd_layer(adata, batch_key)

    def predict(self, adata, encoder_labels, decoder_labels):
        """Feeds ``adata`` to scNet and produces the reconstructed data.

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated data matrix whether in primary space.
            encoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as scNet's encoder condition array.
            decoder_labels: :class:`~numpy.ndarray`
                :class:`~numpy.ndarray` of labels to be fed as scNet's decoder condition array.

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

    def train(self, adata, condition_key,
              cell_type_key='cell_type',
              train_size=0.8,
              n_epochs=25, batch_size=32,
              early_stop_limit=20, lr_reducer=10,
              n_per_epoch=0, score_filename=None,
              save=True, retrain=True, verbose=3):
        """
            Trains scNet with ``n_epochs`` times given ``train_adata``
            and validates the model using ``valid_adata``
            This function is using ``early stopping`` and ``learning rate reduce on plateau``
            techniques to prevent over-fitting.

            Parameters
            ----------
            adata: :class:`~anndata.AnnData`
                Annotated dataset used to train & evaluate scNet.
            condition_key: str
                column name for conditions in the `obs` matrix of `train_adata` and `valid_adata`.
            train_size: float
                fraction of samples used to train scNet.
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
        train_adata, valid_adata = train_test_split(adata, train_size)

        train_adata = remove_sparsity(train_adata)
        valid_adata = remove_sparsity(valid_adata)

        if self.gene_names is None:
            self.gene_names = train_adata.var_names.tolist()
        else:
            if set(self.gene_names).issubset(set(train_adata.var_names)):
                train_adata = train_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in train adata are inconsistent with scNet's gene_names")

            if set(self.gene_names).issubset(set(valid_adata.var_names)):
                valid_adata = valid_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in valid adata are inconsistent with scNet's gene_names")

        train_conditions_encoded, self.condition_encoder = label_encoder(train_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        valid_conditions_encoded, self.condition_encoder = label_encoder(valid_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        if not retrain and self.restore_model_weights():
            return

        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        x_train = [train_adata.X, train_conditions_onehot, train_conditions_onehot]
        y_train = [train_adata.X, train_conditions_encoded]

        x_valid = [valid_adata.X, valid_conditions_onehot, valid_conditions_onehot]
        y_valid = [valid_adata.X, valid_conditions_encoded]

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

        self.cvae_model.fit(x=x_train,
                            y=y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=n_epochs,
                            batch_size=batch_size,
                            verbose=fit_verbose,
                            callbacks=callbacks,
                            )
        if save:
            self.save(make_dir=True)
