import numpy as np
import torch
import anndata

from scarches.models.trvae.trvae import trVAE
from scarches.trainers.trvae.unsupervised import trVAETrainer


def trvae_operate(
        network: trVAE,
        data: anndata,
        condition_key: str = None,
        size_factor_key: str = None,
        n_epochs: int = 20,
        freeze: bool = True,
        freeze_expression: bool = True,
        remove_dropout: bool = True,
) -> [trVAE, trVAETrainer]:
    """Transfer Learning function for new data. Uses old trained Network and expands it for new conditions.
       Parameters
       ----------
       network: trVAE
            A scNet model object.
       data: Anndata
            Query anndata object.
       condition_key: String
            Key where the conditions in the data can be found.
       size_factor_key: String
            Key where the size_factors in the data can be found.
       n_epochs: Integer
            Number of epochs for training the network on query data.
       freeze: Boolean
            If 'True' freezes every part of the network except the first layers of encoder/decoder.
       freeze_expression: Boolean
            If 'True' freeze every weight in first layers except the condition weights.
       remove_dropout: Boolean
            If 'True' remove Dropout for Transfer Learning.
       Returns
       -------
       new_network: trVAE
            Newly network that got trained on query data.
       new_trainer: trVAETrainer
            Trainer for the newly network.
    """
    conditions = network.conditions
    new_conditions = []
    data_conditions = data.obs[condition_key].unique().tolist()
    # Check if new conditions are already known
    for item in data_conditions:
        if item not in conditions:
            new_conditions.append(item)

    n_new_conditions = len(new_conditions)

    # Add new conditions to overall conditions
    for condition in new_conditions:
        conditions.append(condition)

    # Update DR Rate
    new_dr = network.dr_rate
    if remove_dropout:
        new_dr = 0.0

    print("Surgery to get new Network...")
    new_network = trVAE(
        network.input_dim,
        conditions=conditions,
        hidden_layer_sizes=network.hidden_layer_sizes,
        latent_dim=network.latent_dim,
        dr_rate=new_dr,
        use_mmd=network.use_mmd,
        mmd_boundary=network.mmd_boundary,
        recon_loss=network.recon_loss,
    )

    # Expand First Layer weights of encoder/decoder of old network by new conditions
    encoder_input_weights = network.encoder.FC.L0.cond_L.weight
    to_be_added_encoder_input_weights = np.random.randn(encoder_input_weights.size()[0], n_new_conditions) * np.sqrt(
        2 / (encoder_input_weights.size()[0] + 1 + encoder_input_weights.size()[1]))
    to_be_added_encoder_input_weights = torch.from_numpy(to_be_added_encoder_input_weights).float().to(network.device)
    network.encoder.FC.L0.cond_L.weight.data = torch.cat((encoder_input_weights,
                                                          to_be_added_encoder_input_weights), 1)

    decoder_input_weights = network.decoder.FirstL.L0.cond_L.weight
    to_be_added_decoder_input_weights = np.random.randn(decoder_input_weights.size()[0], n_new_conditions) * np.sqrt(
        2 / (decoder_input_weights.size()[0] + 1 + decoder_input_weights.size()[1]))
    to_be_added_decoder_input_weights = torch.from_numpy(to_be_added_decoder_input_weights).float().to(network.device)
    network.decoder.FirstL.L0.cond_L.weight.data = torch.cat((decoder_input_weights,
                                                              to_be_added_decoder_input_weights), 1)

    # Set the weights of new network to old network weights
    new_network.load_state_dict(network.state_dict())

    # Freeze parts of the network
    if freeze:
        new_network.freeze = True
        for name, p in new_network.named_parameters():
            p.requires_grad = False
            if freeze_expression:
                if 'cond_L.weight' in name:
                    p.requires_grad = True
            else:
                if "L0" in name or "B0" in name:
                    p.requires_grad = True

    new_trainer = trVAETrainer(
        new_network,
        data,
        condition_key=condition_key,
        size_factor_key=size_factor_key,
        batch_size=1024,
        n_samples=4096
    )
    new_trainer.train(
        n_epochs=n_epochs,
        lr=0.001
    )

    return new_network, new_trainer
