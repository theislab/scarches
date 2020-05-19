import os
import warnings

from scnet.zenodo.file import download_file
from scnet.zenodo.zip import unzip_model_directory

warnings.filterwarnings('ignore')

import numpy as np
from typing import TypeVar, Optional, Union

from keras import backend as K

from . import metrics
from . import models as archs
from . import plotting as pl
from . import utils as tl
from . import kipoi as kp
from . import data_loader as dl

list_str = TypeVar('list_str', str, list)


def operate(network: archs.scNet,
            new_conditions: list_str,
            init: str = 'Xavier',
            freeze: bool = True,
            freeze_expression_input: bool = False,
            remove_dropout: bool = True,
            print_summary: bool = False,
            new_training_kwargs: dict = {},
            new_network_kwargs: dict = {},
            ) -> archs.scNet:
    if isinstance(new_conditions, str):
        new_conditions = [new_conditions]

    n_new_conditions = len(new_conditions)

    network_kwargs = network.network_kwargs
    training_kwargs = network.training_kwargs

    network_kwargs['n_conditions'] += n_new_conditions
    network_kwargs['n_mmd_conditions'] += n_new_conditions
    network_kwargs['freeze_expression_input'] = freeze_expression_input
    network_kwargs['mmd_computation_method'] = "general"

    if remove_dropout:
        network_kwargs['dropout_rate'] = 0.0

    for key in new_training_kwargs.keys():
        training_kwargs[key] = new_training_kwargs[key]

    for key in new_network_kwargs.keys():
        network_kwargs[key] = new_network_kwargs[key]

    # Instantiate new model with old parameters except `n_conditions`
    new_network = archs.scNet(**network_kwargs, **training_kwargs,
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


PRETRAINED_TASKS = {
    "pancreas": {
        "default_link": "",
    },
    "mouse_brain": {
        "default_link": "",
    },
    "tabula_muris_senis": {
        "default_link": "",
    },
    "hcl": {
        "default_link": "",
    },
    "hcl_mca": {
        "default_link": "",
    },
    "tabula_muris_senis_mca": {
        "default_link": "",
    },

}


def create_scNet_from_pre_trained_task(pre_trained_task: str = None,
                                       new_task: str = None,
                                       target_conditions: list = [],
                                       version: str = 'scNet',
                                       use_default_params: bool = True,
                                       downloaded_path: str = None,
                                       model_path: str = None,
                                       **kwargs,
                                       ):
    pre_trained_task = pre_trained_task.lower()
    version = version.lower()

    if version == 'scnet':
        freeze_input_expression = True
        freeze = True
    elif version == 'scnet v1':
        freeze_input_expression = False
        freeze = True
    elif version == 'scnet v2':
        freeze_input_expression = False
        freeze = False
    else:
        raise Exception("Invalid scNet version. Must be one of \'scNet\', \'scNet v1\', or \'scNet v2\'.")

    if os.path.exists(downloaded_path) and downloaded_path.endswith(".zip"):
        if model_path:
            base_path = os.path.join(os.path.dirname(model_path), f"scNet-{new_task}/")
        else:
            base_path = os.path.join(os.path.dirname(downloaded_path), f"scNet-{new_task}/")

        extract_dir = os.path.join(base_path, f"before-{pre_trained_task}/")
        unzip_model_directory(downloaded_path, extract_dir=extract_dir)

        model_path = extract_dir if model_path is None else model_path
    elif not os.path.isdir(downloaded_path):
        raise ValueError("`model_path` should be either path to downloaded zip file or scNet pre-trained directory")

    task = PRETRAINED_TASKS.get(pre_trained_task, None)
    if task:
        if use_default_params:
            kwargs.update(task.get("default_hyper_params", {}))
        kwargs.update(task.get("network_config", {}))
    else:
        raise Exception("Invalid task")

    config_path = os.path.join(extract_dir, f"scNet-{pre_trained_task}.json")
    pre_trained_scNet = archs.scNet.from_config(config_path, new_params=kwargs, construct=True, compile=True)

    pre_trained_scNet.model_path = model_path
    pre_trained_scNet.task_name = pre_trained_task

    pre_trained_scNet.restore_model_weights(compile=True)

    scNet = operate(pre_trained_scNet,
                    new_conditions=target_conditions,
                    init='Xavier',
                    freeze=freeze,
                    freeze_expression_input=freeze_input_expression,
                    remove_dropout=False,
                    print_summary=False,
                    )

    scNet.task_name = new_task
    scNet.model_path = os.path.join(base_path, f"after/")

    return scNet


def download_pretrained_scNet(task_name: str,
                              save_path: str = './',
                              make_dir=False):
    task_dict = PRETRAINED_TASKS.get(task_name, None)

    if task_dict:
        download_link = task_dict.get('default_link', '')
        if download_link != '':
            file_path, response = download_file(download_link, save_path, make_dir)
            return file_path
        else:
            raise Exception("Download link does not exist for the specified task")
    else:
        raise ValueError("Invalid task")
