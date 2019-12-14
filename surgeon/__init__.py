import numpy as np
from typing import TypeVar

from keras import backend as K

from . import metrics
from . import models as archs
from . import plotting as pl
from . import utils as tl
from . import kipoi as kp
from . import data_loader as dl

list_str = TypeVar('list_str', str, list)


def operate(network: archs.CVAE,
            new_conditions: list_str,
            init: str = 'Xavier',
            freeze: bool = True,
            freeze_expression_input: bool = False,
            remove_dropout: bool = True,
            print_summary: bool = True,
            new_training_kwargs: dict = None,
            ) -> archs.CVAE:
    if isinstance(new_conditions, str):
        new_conditions = [new_conditions]

    n_new_conditions = len(new_conditions)

    network_kwargs = network.network_kwargs
    training_kwargs = network.training_kwargs

    network_kwargs['n_conditions'] += n_new_conditions
    network_kwargs['n_mmd_conditions'] += n_new_conditions
    network_kwargs['freeze_expression_input'] = freeze_expression_input

    if remove_dropout:
        network_kwargs['dropout_rate'] = 0.0

    for key in new_training_kwargs.keys():
        training_kwargs[key] = new_training_kwargs[key]

    # Instantiate new model with old parameters except `n_conditions`
    new_network = archs.CVAE(**network_kwargs, **training_kwargs,
                             print_summary=False)

    # Get Previous Model's weights
    used_bias_encoder = network.cvae_model.get_layer("encoder").get_layer("first_layer").use_bias
    used_bias_decoder = network.cvae_model.get_layer("decoder").get_layer("first_layer").use_bias

    prev_weights = {}
    for w in network.cvae_model.get_layer("encoder").get_layer("first_layer").weights:
        if "condition_kernel" in w.name:
            prev_weights['c'] = K.batch_get_value(w)
        elif "expression_kernel" in w.name:
            prev_weights['i'] = K.batch_get_value(w)
        else:
            prev_weights['b'] = K.batch_get_value(w)

    if used_bias_encoder:
        prev_input_weights_encoder, prev_condition_weights_encoder, prev_biases_encoder = \
            prev_weights['i'], prev_weights['c'], prev_weights['b']

    else:
        prev_input_weights_encoder, prev_condition_weights_encoder, prev_biases_encoder = \
            prev_weights['i'], prev_weights['c'], None

    prev_weights = {}
    for w in network.cvae_model.get_layer("decoder").get_layer("first_layer").weights:
        if "condition_kernel" in w.name:
            prev_weights['c'] = K.batch_get_value(w)
        elif "expression_kernel" in w.name:
            prev_weights['i'] = K.batch_get_value(w)
        else:
            prev_weights['b'] = K.batch_get_value(w)

    if used_bias_decoder:
        prev_latent_weights_decoder, prev_condition_weights_decoder, prev_biases_decoder = \
            prev_weights['i'], prev_weights['c'], prev_weights['b']
    else:
        prev_latent_weights_decoder, prev_condition_weights_decoder, prev_biases_decoder = \
            prev_weights['i'], prev_weights['c'], None

    # Modify the weights of 1st encoder & decoder layers
    if init == 'ones':
        to_be_added_weights_encoder_condition = np.ones(
            shape=(n_new_conditions, prev_condition_weights_encoder.shape[1]))
        to_be_added_weights_decoder_condition = np.ones(
            shape=(n_new_conditions, prev_condition_weights_decoder.shape[1]))
    elif init == "zeros":
        to_be_added_weights_encoder_condition = np.zeros(
            shape=(n_new_conditions, prev_condition_weights_encoder.shape[1]))
        to_be_added_weights_decoder_condition = np.zeros(
            shape=(n_new_conditions, prev_condition_weights_decoder.shape[1]))
    elif init == "Xavier":
        to_be_added_weights_encoder_condition = np.random.randn(n_new_conditions,
                                                                prev_condition_weights_encoder.shape[1]) * np.sqrt(
            2 / (prev_condition_weights_encoder.shape[0] + 1 + prev_condition_weights_encoder.shape[1]))
        to_be_added_weights_decoder_condition = np.random.randn(n_new_conditions,
                                                                prev_condition_weights_decoder.shape[1]) * np.sqrt(
            2 / (prev_condition_weights_decoder.shape[0] + 1 + prev_condition_weights_decoder.shape[1]))
    else:
        raise Exception("Invalid initialization for new weights")

    new_condition_weights_encoder = np.concatenate(
        [prev_condition_weights_encoder, to_be_added_weights_encoder_condition], axis=0)
    new_condition_weights_decoder = np.concatenate(
        [prev_condition_weights_decoder, to_be_added_weights_decoder_condition], axis=0)

    # Set new model's weights
    for w in new_network.cvae_model.get_layer("encoder").get_layer("first_layer").weights:
        if "condition_kernel" in w.name:
            K.set_value(w, new_condition_weights_encoder)
        elif "expression_kernel" in w.name:
            K.set_value(w, prev_input_weights_encoder)
        else:
            K.set_value(w, prev_biases_encoder)

    for w in new_network.cvae_model.get_layer("decoder").get_layer("first_layer").weights:
        if "condition_kernel" in w.name:
            K.set_value(w, new_condition_weights_decoder)
        elif "expression_kernel" in w.name:
            K.set_value(w, prev_latent_weights_decoder)
        else:
            K.set_value(w, prev_biases_decoder)

    # set weights of other parts of model
    for idx, encoder_layer in enumerate(new_network.encoder_model.layers):
        if encoder_layer.name != 'first_layer' and encoder_layer.get_weights() != []:
            encoder_layer.set_weights(network.encoder_model.layers[idx].get_weights())

    for idx, decoder_layer in enumerate(new_network.decoder_model.layers):
        if decoder_layer.name != 'first_layer' and decoder_layer.get_weights():
            decoder_layer.set_weights(network.decoder_model.layers[idx].get_weights())

    # Freeze old parts of cloned network
    if freeze:
        for encoder_layer in new_network.encoder_model.layers:
            if encoder_layer.name != 'first_layer':
                encoder_layer.trainable = False

        for decoder_layer in new_network.decoder_model.layers:
            if decoder_layer.name != 'first_layer':
                decoder_layer.trainable = False

        new_network.compile_models()

    # Print summary of new network
    if print_summary:
        new_network.get_summary_of_networks()

    # Add new condition to new network condition encoder
    new_network.condition_encoder = network.condition_encoder
    for idx, new_condition in enumerate(new_conditions):
        new_network.condition_encoder[new_condition] = network.n_conditions + idx

    return new_network


def operate_fair(network: archs.CVAEFair,
                 new_datasets: list_str,
                 new_conditions: list_str,
                 init: str = 'Xavier',
                 freeze: bool = True,
                 freeze_expression_input: bool = False,
                 remove_dropout: bool = True,
                 print_summary: bool = True,
                 new_training_kwargs: dict = None,
                 ) -> archs.CVAEFair:
    if isinstance(new_datasets, str):
        new_datasets = [new_datasets]
    if isinstance(new_conditions, str):
        new_conditions = [new_conditions]

    n_new_datasets = len(new_datasets)
    n_new_conditions = len(new_conditions)

    network_kwargs = network.network_kwargs
    training_kwargs = network.training_kwargs

    network_kwargs['n_datasets'] += n_new_datasets
    network_kwargs['n_conditions'] += n_new_conditions
    network_kwargs['freeze_expression_input'] = freeze_expression_input

    if new_training_kwargs:
        for key in new_training_kwargs.keys():
            training_kwargs[key] = new_training_kwargs[key]

    if remove_dropout:
        network_kwargs['dropout_rate'] = 0.0

    # Instantiate new model with old parameters except `n_conditions`
    new_network = archs.CVAEFair(**network_kwargs, **training_kwargs,
                                 print_summary=False)

    # Get Previous Model's weights
    used_bias_encoder = network.cvae_model.get_layer("encoder").get_layer("first_layer").use_bias
    used_bias_decoder = network.cvae_model.get_layer("decoder").get_layer("first_layer").use_bias

    prev_weights = {}
    for w in network.cvae_model.get_layer("encoder").get_layer("first_layer").weights:
        if "condition_kernel" in w.name:
            prev_weights['c'] = K.batch_get_value(w)
        elif "expression_kernel" in w.name:
            prev_weights['i'] = K.batch_get_value(w)
        elif "cell_type_kernel" in w.name:
            prev_weights['t'] = K.batch_get_value(w)
        else:
            prev_weights['b'] = K.batch_get_value(w)

    if used_bias_encoder:
        prev_input_weights_encoder, prev_condition_weights_encoder, prev_cell_type_weights_encoder, prev_biases_encoder = \
            prev_weights['i'], prev_weights['c'], prev_weights['t'], prev_weights['b']

    else:
        prev_input_weights_encoder, prev_condition_weights_encoder, prev_cell_type_weights_encoder, prev_biases_encoder = \
            prev_weights['i'], prev_weights['c'], prev_weights['t'], None

    prev_weights = {}
    for w in network.cvae_model.get_layer("decoder").get_layer("first_layer").weights:
        if "condition_kernel" in w.name:
            prev_weights['c'] = K.batch_get_value(w)
        elif "expression_kernel" in w.name:
            prev_weights['i'] = K.batch_get_value(w)
        elif "cell_type_kernel" in w.name:
            prev_weights['t'] = K.batch_get_value(w)
        else:
            prev_weights['b'] = K.batch_get_value(w)

    if used_bias_decoder:
        prev_latent_weights_decoder, prev_condition_weights_decoder, prev_cell_type_weights_decoder, prev_biases_decoder = \
            prev_weights['i'], prev_weights['c'], prev_weights['t'], prev_weights['b']
    else:
        prev_latent_weights_decoder, prev_condition_weights_decoder, prev_cell_type_weights_decoder, prev_biases_decoder = \
            prev_weights['i'], prev_weights['c'], prev_weights['t'], None

    # Modify the weights of 1st encoder & decoder layers
    if init == 'ones':
        to_be_added_weights_encoder_condition = np.ones(
            shape=(n_new_datasets, prev_condition_weights_encoder.shape[1]))
        to_be_added_weights_encoder_cell_type = np.ones(
            shape=(n_new_conditions, prev_cell_type_weights_encoder.shape[1]))
        to_be_added_weights_decoder_condition = np.ones(
            shape=(n_new_datasets, prev_condition_weights_decoder.shape[1]))
        to_be_added_weights_decoder_cell_type = np.ones(
            shape=(n_new_conditions, prev_cell_type_weights_decoder.shape[1]))
    elif init == "zeros":
        to_be_added_weights_encoder_condition = np.zeros(
            shape=(n_new_datasets, prev_condition_weights_encoder.shape[1]))
        to_be_added_weights_encoder_cell_type = np.zeros(
            shape=(n_new_conditions, prev_cell_type_weights_encoder.shape[1]))
        to_be_added_weights_decoder_condition = np.zeros(
            shape=(n_new_datasets, prev_condition_weights_decoder.shape[1]))
        to_be_added_weights_decoder_cell_type = np.zeros(
            shape=(n_new_conditions, prev_cell_type_weights_decoder.shape[1]))
    elif init == "Xavier":
        to_be_added_weights_encoder_condition = np.random.randn(n_new_datasets,
                                                                prev_condition_weights_encoder.shape[1]) * np.sqrt(
            2 / (prev_condition_weights_encoder.shape[0] + 1 + prev_condition_weights_encoder.shape[1]))
        to_be_added_weights_encoder_cell_type = np.random.randn(n_new_conditions,
                                                                prev_cell_type_weights_encoder.shape[1]) * np.sqrt(
            2 / (prev_condition_weights_encoder.shape[0] + 1 + prev_condition_weights_encoder.shape[1]))
        to_be_added_weights_decoder_condition = np.random.randn(n_new_datasets,
                                                                prev_condition_weights_decoder.shape[1]) * np.sqrt(
            2 / (prev_condition_weights_decoder.shape[0] + 1 + prev_condition_weights_decoder.shape[1]))
        to_be_added_weights_decoder_cell_type = np.random.randn(n_new_conditions,
                                                                prev_cell_type_weights_decoder.shape[1]) * np.sqrt(
            2 / (prev_condition_weights_decoder.shape[0] + 1 + prev_condition_weights_decoder.shape[1]))
    else:
        raise Exception("Invalid initialization for new weights")

    new_condition_weights_encoder = np.concatenate(
        [prev_condition_weights_encoder, to_be_added_weights_encoder_condition], axis=0)
    new_cell_type_weights_encoder = np.concatenate(
        [prev_cell_type_weights_encoder, to_be_added_weights_encoder_cell_type], axis=0)
    new_condition_weights_decoder = np.concatenate(
        [prev_condition_weights_decoder, to_be_added_weights_decoder_condition], axis=0)
    new_cell_type_weights_decoder = np.concatenate(
        [prev_cell_type_weights_decoder, to_be_added_weights_decoder_cell_type], axis=0)

    # Set new model's weights
    for w in new_network.cvae_model.get_layer("encoder").get_layer("first_layer").weights:
        if "condition_kernel" in w.name:
            K.set_value(w, new_condition_weights_encoder)
        elif "expression_kernel" in w.name:
            K.set_value(w, prev_input_weights_encoder)
        elif "cell_type_kernel" in w.name:
            K.set_value(w, new_cell_type_weights_encoder)
        else:
            K.set_value(w, prev_biases_encoder)

    for w in new_network.cvae_model.get_layer("decoder").get_layer("first_layer").weights:
        if "condition_kernel" in w.name:
            K.set_value(w, new_condition_weights_decoder)
        elif "expression_kernel" in w.name:
            K.set_value(w, prev_latent_weights_decoder)
        elif "cell_type_kernel" in w.name:
            K.set_value(w, new_cell_type_weights_decoder)
        else:
            K.set_value(w, prev_biases_decoder)

    # set weights of other parts of model
    for idx, encoder_layer in enumerate(new_network.encoder_model.layers):
        if encoder_layer.name != 'first_layer' and encoder_layer.get_weights() != []:
            encoder_layer.set_weights(network.encoder_model.layers[idx].get_weights())

    for idx, decoder_layer in enumerate(new_network.decoder_model.layers):
        if decoder_layer.name != 'first_layer' and decoder_layer.get_weights():
            decoder_layer.set_weights(network.decoder_model.layers[idx].get_weights())

    # Freeze old parts of cloned network
    if freeze:
        for encoder_layer in new_network.encoder_model.layers:
            if encoder_layer.name != 'first_layer':
                encoder_layer.trainable = False

        for decoder_layer in new_network.decoder_model.layers:
            if decoder_layer.name != 'first_layer':
                decoder_layer.trainable = False

        new_network.compile_models()

    # Print summary of new network
    if print_summary:
        new_network.get_summary_of_networks()

    # Add new condition to new network condition encoder
    new_network.dataset_encoder = network.dataset_encoder
    new_network.condition_encoder = network.condition_encoder
    for idx, new_condition in enumerate(new_datasets):
        new_network.dataset_encoder[new_condition] = network.n_datasets + idx
    for idx, new_cell_type in enumerate(new_conditions):
        new_network.condition_encoder[new_cell_type] = network.n_conditions + idx

    return new_network
