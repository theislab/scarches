from . import models as archs
from . import plotting as pl
from . import utils as tl


def operate(network: archs.CVAE,
            new_condition: str,
            init: str = 'Xavier',
            freeze: bool = True,
            print_summary: bool = True) -> archs.CVAE:
    import numpy as np
    network_kwargs = network.network_kwargs
    training_kwargs = network.training_kwargs

    network_kwargs['n_conditions'] += 1

    # Instantiate new model with old parameters except `n_conditions`
    new_network = archs.CVAE(**network_kwargs, **training_kwargs, summary=False)

    # Get Previous Model's weights
    used_bias_encoder = network.cvae_model.get_layer("encoder").get_layer("first_layer").use_bias
    used_bias_decoder = network.cvae_model.get_layer("decoder").get_layer("first_layer").use_bias

    if used_bias_encoder:
        prev_weights_encoder, prev_biases_encoder = network.cvae_model.get_layer("encoder").get_layer(
            "first_layer").get_weights()
    else:
        prev_weights_encoder, prev_biases_encoder = network.cvae_model.get_layer("encoder").get_layer(
            "first_layer").get_weights()[0], None

    if used_bias_decoder:
        prev_weights_decoder, prev_biases_decoder = network.cvae_model.get_layer("decoder").get_layer(
            "first_layer").get_weights()
    else:
        prev_weights_decoder, prev_biases_decoder = network.cvae_model.get_layer("decoder").get_layer(
            "first_layer").get_weights()[0], None

    # Modify the weights of 1st encoder & decoder layers
    if init == 'ones':
        to_be_added_weights_encoder = np.ones(shape=(1, prev_weights_encoder.shape[1]))
        to_be_added_weights_decoder = np.ones(shape=(1, prev_weights_decoder.shape[1]))
    if init == "zeros":
        to_be_added_weights_encoder = np.zeros(shape=(1, prev_weights_encoder.shape[1]))
        to_be_added_weights_decoder = np.zeros(shape=(1, prev_weights_decoder.shape[1]))

    if init == "Xavier":
        to_be_added_weights_encoder = np.random.randn(1, prev_weights_encoder.shape[1]) * np.sqrt(
            2 / (prev_weights_encoder.shape[0] + 1 + prev_weights_encoder.shape[1]))
        to_be_added_weights_decoder = np.random.randn(1, prev_weights_decoder.shape[1]) * np.sqrt(
            2 / (prev_weights_decoder.shape[0] + 1 + prev_weights_decoder.shape[1]))

    new_weights_encoder = np.concatenate([prev_weights_encoder, to_be_added_weights_encoder], axis=0)
    new_weights_decoder = np.concatenate([prev_weights_decoder, to_be_added_weights_decoder], axis=0)

    # Set new model's weights
    if used_bias_encoder:
        new_network.cvae_model.get_layer("encoder").get_layer("first_layer").set_weights(
            [new_weights_encoder, prev_biases_encoder])
    else:
        new_network.cvae_model.get_layer("encoder").get_layer("first_layer").set_weights(
            [new_weights_encoder])

    if used_bias_decoder:
        new_network.cvae_model.get_layer("decoder").get_layer("first_layer").set_weights(
            [new_weights_decoder, prev_biases_decoder])
    else:
        new_network.cvae_model.get_layer("decoder").get_layer("first_layer").set_weights(
            [new_weights_decoder])

    # set weights of other parts of model
    for idx, encoder_layer in enumerate(new_network.encoder_model.layers):
        if encoder_layer.name != 'first_layer' and encoder_layer.get_weights():
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
    new_network.condition_encoder[new_condition] = new_network.n_conditions - 1

    return new_network