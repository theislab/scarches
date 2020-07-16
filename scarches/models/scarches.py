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
from scipy import sparse

from scarches.models import CVAE
from scarches.models._activations import ACTIVATIONS
from scarches.models._callbacks import ScoreCallback
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
        conditions: int
            number of conditions used for one-hot encoding.
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
            Constructs the whole scArches' network. It is step-by-step constructing the scArches network.
            First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of scArches.
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
        print(f"{self.class_name}' network has been successfully constructed!")

    def _calculate_loss(self):
        """
            Defines the loss function of scArches' network after constructing the whole
            network.
        """
        loss = LOSSES[self.loss_fn](self.mu, self.log_var, self.alpha, self.eta)
        mmd_loss = LOSSES['mmd'](self.n_conditions, self.beta)
        kl_loss = LOSSES['kl'](self.mu, self.log_var)
        recon_loss = LOSSES[f'{self.loss_fn}_recon']

        return loss, mmd_loss, kl_loss, recon_loss

    def compile_models(self):
        """
            Compiles scArches network with the defined loss functions and
            Adam optimizer with its pre-defined hyper-parameters.
        """
        optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value, epsilon=self.epsilon)
        loss, mmd_loss, kl_loss, recon_loss = self._calculate_loss()

        self.cvae_model.compile(optimizer=optimizer,
                                loss=[loss, mmd_loss],
                                metrics={self.cvae_model.outputs[0].name: loss,
                                         self.cvae_model.outputs[1].name: mmd_loss}
                                )

        print("scArches' network has been successfully compiled!")

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

        mmd = self.cvae_model.predict(cvae_inputs)[1]
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

    def _fit(self, adata,
             condition_key, train_size=0.8, cell_type_key='cell_type',
             n_epochs=300, batch_size=512,
             early_stop_limit=10, lr_reducer=7,
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

        y_train = [train_expr, train_conditions_encoded]
        y_valid = [valid_expr, valid_conditions_encoded]

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
                        n_epochs=300, batch_size=512,
                        early_stop_limit=10, lr_reducer=7,
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

        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        if sparse.issparse(train_adata.X):
            is_sparse = True
        else:
            is_sparse = False

        train_expr = train_adata.X
        valid_expr = valid_adata.X.A if is_sparse else valid_adata.X
        x_valid = [valid_expr, valid_conditions_onehot, valid_conditions_onehot]

        if self.loss_fn in ['nb', 'zinb']:
            x_valid.append(valid_adata.obs[self.size_factor_key].values)
            y_valid = [valid_adata.raw.X.A if sparse.issparse(valid_adata.raw.X) else valid_adata.raw.X,
                       valid_conditions_encoded]
        else:
            y_valid = [valid_expr, valid_conditions_encoded]

        es_patience, best_val_loss = 0, 1e10
        for i in range(n_epochs):
            train_loss = train_recon_loss = train_mmd_loss = 0.0
            for j in range(min(500, train_adata.shape[0] // batch_size)):
                batch_indices = np.random.choice(train_adata.shape[0], batch_size)

                batch_expr = train_expr[batch_indices, :].A if is_sparse else train_expr[batch_indices, :]

                x_train = [batch_expr, train_conditions_onehot[batch_indices], train_conditions_onehot[batch_indices]]

                if self.loss_fn in ['nb', 'zinb']:
                    x_train.append(train_adata.obs[self.size_factor_key].values[batch_indices])
                    y_train = [train_adata.raw.X[batch_indices].A if sparse.issparse(
                        train_adata.raw.X[batch_indices]) else train_adata.raw.X[batch_indices],
                               train_conditions_encoded[batch_indices]]
                else:
                    y_train = [batch_expr, train_conditions_encoded[batch_indices]]

                batch_loss, batch_recon_loss, batch_kl_loss = self.cvae_model.train_on_batch(x_train, y_train)

                train_loss += batch_loss / batch_size
                train_recon_loss += batch_recon_loss / batch_size
                train_mmd_loss += batch_kl_loss / batch_size

            valid_loss, valid_recon_loss, valid_mmd_loss = self.cvae_model.evaluate(x_valid, y_valid, verbose=0)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                es_patience = 0
            else:
                es_patience += 1
                if es_patience == early_stop_limit:
                    print("Training stopped with Early Stopping")
                    break

            logs = {"loss": train_loss, "recon_loss": train_recon_loss, "mmd_loss": train_mmd_loss,
                    "val_loss": valid_loss, "val_recon_loss": valid_recon_loss, "val_mmd_loss": valid_mmd_loss}
            print_progress(i, logs, n_epochs)

        if save:
            self.update_kwargs()
            self.save(make_dir=True)
