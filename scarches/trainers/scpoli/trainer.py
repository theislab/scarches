import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from ..trvae._utils import make_dataset
from ..trvae.trainer import Trainer
from ._utils import cov, euclidean_dist


class scPoliTrainer(Trainer):
    """
    scPoli Trainer class. This class contains the implementation of the training routine for scPoli models

    Parameters
    ----------
    model: SCPoli
        PyTorch model to train
    adata: : `~anndata.AnnData`
        Annotated data matrix.
    condition_key: String
        column name of conditions in `adata.obs` data frame.
    cell_type_key: String
        column name of celltypes in `adata.obs` data frame.
    train_frac: Float
        Defines the fraction of data that is used for training and data that is used for validation.
    batch_size: Integer
        Defines the batch size that is used during each Iteration
    n_samples: Integer or None
        Defines how many samples are being used during each epoch. This should only be used if hardware resources
        are limited.
    clip_value: Float
        If the value is greater than 0, all gradients with an higher value will be clipped during training.
    weight decay: Float
        Defines the scaling factor for weight decay in the Adam optimizer.
    alpha_iter_anneal: Integer or None
        If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every iteration until the input
        integer is reached.
    alpha_epoch_anneal: Integer or None
        If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every epoch until the input
        integer is reached.
    use_early_stopping: Boolean
        If 'True' the EarlyStopping class is being used for training to prevent overfitting.
    early_stopping_kwargs: Dict
        Passes custom Earlystopping parameters.
    use_stratified_sampling: Boolean
        If 'True', the sampler tries to load equally distributed batches concerning the conditions in every
        iteration.
    monitor: Boolean
        If `True', the progress of the training will be printed after each epoch.
    n_workers: Integer
        Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
    labeled_indices: list
        List of integers that indicate the annotated data
    pretraining_epochs: Integer
        Number of epochs for pretraining
    clustering: String
        Clustering methodology to use, can be KMeans or Leiden.
    clustering_res: Float
        Clustering resolution to use for leiden clustering. Bigger values result in
        finer clusters.
    n_clusters: Integer
        Number of clusters to set for KMeans algorithm.
    unlabeled_weight: Float
        Weight for loss computed including unlabeled samples
    eta: Float
        Weight for the prototype loss
    prototype_training: Bool
        Boolean that can be used to turn off prototype training.
    unlabeled_prototype_training: Bool
        Boolean that can be used to turn off prototype training. This can lead to a significant speedup
        since it allows the model to skip the clustering step. Considering doing this if you do not plan
        to make use of unlabeled prototypes for novel cell type annotation.
    seed: Integer
        Define a specific random seed to get reproducable results.
    """

    def __init__(
        self,
        model,
        adata,
        labeled_indices: list = None,
        pretraining_epochs=None,
        clustering: str = "leiden",
        clustering_res: float = 2,
        n_clusters: int = None,
        unlabeled_weight: float = 0,
        eta: float = 1,
        prototype_training: bool = True,
        unlabeled_prototype_training: bool = True,
        **kwargs,
    ):
        super().__init__(model, adata, **kwargs)
        self.eta = eta
        self.clustering = clustering
        self.n_clusters = n_clusters
        self.unlabeled_weight = unlabeled_weight
        self.clustering_res = clustering_res
        self.pretraining_epochs = pretraining_epochs
        self.use_early_stopping_orig = self.use_early_stopping

        self.prototypes_labeled = None  # prototypes labeled cells (means)
        self.prototypes_labeled_cov = None  # prototypes labeled cells (cov)
        self.prototypes_unlabeled = None  # prototypes all cells (means)
        self.best_prototypes_labeled = None  # cache for ES, to use best state
        self.best_prototypes_labeled_cov = None  # cache for ES
        self.best_prototypes_unlabeled = None  # cache for ES
        self.prototype_optim = None  # prototype optimizer
        
        #set indices for labeled data
        if labeled_indices is None:
            self.labeled_indices = range(len(adata))
        else:
            self.labeled_indices = labeled_indices
        self.update_labeled_indices(self.labeled_indices)
        self.prototype_training = prototype_training
        self.unlabeled_prototype_training = unlabeled_prototype_training
        self.any_labeled_data = 1 in self.train_data.labeled_vector.unique().tolist()
        self.any_unlabeled_data = (
            0 in self.train_data.labeled_vector.unique().tolist() 
            or self.model.unknown_ct_names is not None
        )
        
        #parse prototypes from model into right format
        if self.model.prototypes_labeled["mean"] is not None:
            self.prototypes_labeled = self.model.prototypes_labeled["mean"]
            self.prototypes_labeled_cov = self.model.prototypes_labeled["cov"]
        if self.prototypes_labeled is not None:
            self.prototypes_labeled = self.prototypes_labeled.to(device=self.device)
            self.prototypes_labeled_cov = self.prototypes_labeled_cov.to(
                device=self.device
            )

    def train(self, n_epochs=400, lr=1e-3, eps=0.01):
        self.initialize_loaders()
        begin = time.time()
        self.model.train()
        self.n_epochs = n_epochs

        params_embedding = []
        params = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if "embedding" in name:
                    params_embedding.append(p)
                else:
                    params.append(p)

        self.optimizer = torch.optim.Adam(
            [
                {"params": params_embedding, "weight_decay": 0},
                {"params": params},
            ],
            lr=lr,
            eps=eps,
            weight_decay=self.weight_decay,
        )

        for self.epoch in range(n_epochs):
            self.on_epoch_begin(lr, eps)
            self.iter_logs = defaultdict(list)
            for self.iter, batch_data in enumerate(self.dataloader_train):
                for key, batch in batch_data.items():
                    batch_data[key] = batch.to(self.device)

                #loss calculation
                self.on_iteration(batch_data)

            #validation of model, monitoring, early stopping
            self.on_epoch_end()
            if self.use_early_stopping:
                if not self.check_early_stop():
                    break

        if self.best_state_dict is not None and self.reload_best:
            print("Saving best state of network...")
            print("Best State was in Epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)

        self.model.eval()
        self.after_loop()

        self.training_time += time.time() - begin

    def on_iteration(self, batch_data):
        #do not update any weight on first layers except condition weights
        if self.model.freeze:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    if not module.weight.requires_grad:
                        module.affine = False
                        module.track_running_stats = False

        #calculate loss depending on trainer/model
        self.current_loss = loss = self.loss(batch_data)
        self.optimizer.zero_grad()

        loss.backward()
        #gradient Clipping
        if self.clip_value > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
        if self.model.freeze == True:
            if self.model.embedding:
                self.model.embedding.weight.grad[
                    : self.model.n_reference_conditions
                ] = 0
        self.optimizer.step()

    def update_labeled_indices(self, labeled_indices):
        """
        Function to generate a dataset with new labeled indices after init.

        Parameters
        ==========
        labeled_indices: list
            List of integer indices for labeled samples.

        """
        self.labeled_indices = labeled_indices
        self.train_data, self.valid_data = make_dataset(
            self.adata,
            train_frac=self.train_frac,
            condition_key=self.condition_key,
            cell_type_keys=self.cell_type_keys,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=self.model.cell_type_encoder,
            labeled_indices=self.labeled_indices,
        )

    def get_latent_train(self):
        """
        Function to return the latent representation of the training dataset.

        Returns
        =======
        latent
        """
        latents = []
        indices = np.arange(len(self.train_data))
        subsampled_indices = np.array_split(indices, self.batch_size)
        for batch in subsampled_indices:
            batch_data = self.train_data[batch]
            latent = self.model.get_latent(
                batch_data["x"].to(self.device),
                batch_data["batch"].to(self.device),
            )
            latents += [latent.cpu().detach()]
        latent = torch.cat(latents)
        return latent.to(self.device)

    def initialize_prototypes(self):
        """
        Function that initializes prototypes
        """
        # Compute Latent of whole train data
        latent = self.get_latent_train()

        # Init labeled prototypes if labeled data existent
        if self.any_labeled_data is True:
            # get cell type annot
            labeled_latent = latent[torch.where(self.train_data.labeled_vector == 1)[0]]
            labeled_cell_types = self.train_data.cell_types[
                torch.where(self.train_data.labeled_vector == 1)[0], :
            ]
            # check if model already has initialized prototypes
            # and then initialize new prototypes for new or unseen cell types in query
            if self.prototypes_labeled is not None:  
                with torch.no_grad():
                    if len(self.model.new_prototypes) > 0:
                        for value in self.model.new_prototypes:
                            indices = labeled_cell_types.eq(value).nonzero(
                                as_tuple=False
                            )[:, 0]
                            prototype = labeled_latent[indices].mean(0)
                            prototype_cov = cov(labeled_latent[indices]).unsqueeze(0)
                            self.prototypes_labeled = torch.cat(
                                [self.prototypes_labeled, prototype]
                            )
                            self.prototypes_labeled_cov = torch.cat(
                                [self.prototypes_labeled_cov, prototype_cov]
                            )
            else:  
                #compute labeled prototypes
                (
                    self.prototypes_labeled,
                    self.prototypes_labeled_cov,
                ) = self.update_labeled_prototypes(
                    latent[torch.where(self.train_data.labeled_vector == 1)[0]],
                    self.train_data.cell_types[
                        torch.where(self.train_data.labeled_vector == 1)[0], :
                    ],
                    None,
                    None,
                )

        # Init unlabeled prototypes if unlabeled data exists
        # Unknown ct names: list of strings that identify cells to ignore during training
        if (self.any_unlabeled_data is True) and (self.unlabeled_prototype_training is True):
            lat_array = latent.cpu().detach().numpy()

            if self.clustering == "kmeans" and self.n_clusters is not None:
                print(
                    f"\nInitializing unlabeled prototypes with KMeans with a given number of"
                    f"{self.n_clusters} clusters."
                )
                k_means = KMeans(n_clusters=self.n_clusters).fit(lat_array)
                k_means_prototypes = torch.tensor(
                    k_means.cluster_centers_, device=self.device
                )
                # initialize tensor with zeros
                self.prototypes_unlabeled = [
                    torch.zeros(
                        size=(1, self.model.latent_dim),
                        requires_grad=False,
                        device=self.device,
                    )
                    for _ in range(self.n_clusters)
                ]

                with torch.no_grad():
                    # replace zeros with the kmeans centroids
                    [
                        self.prototypes_unlabeled[i].copy_(k_means_prototypes[i, :])
                        for i in range(k_means_prototypes.shape[0])
                    ]
            else:
                if self.clustering == "kmeans" and self.n_clusters is None:
                    print(
                        f"\nInitializing unlabeled prototypes with Leiden "
                        f"because no value for the number of clusters was given."
                    )
                else:
                    print(
                        f"\nInitializing unlabeled prototypes with Leiden "
                        f"with an unknown number of  clusters."
                    )
                lat_adata = sc.AnnData(lat_array)
                sc.pp.neighbors(lat_adata)
                sc.tl.leiden(lat_adata, resolution=self.clustering_res)

                features = pd.DataFrame(
                    lat_adata.X, index=np.arange(0, lat_adata.shape[0])
                )
                group = pd.Series(
                    np.asarray(lat_adata.obs["leiden"], dtype=int),
                    index=np.arange(0, lat_adata.shape[0]),
                    name="cluster",
                )
                merged_df = pd.concat([features, group], axis=1)
                cluster_centers = np.asarray(merged_df.groupby("cluster").mean())

                self.n_clusters = cluster_centers.shape[0]
                print(f"Clustering succesful. Found {self.n_clusters} clusters.")
                leiden_prototypes = torch.tensor(cluster_centers, device=self.device)

                self.prototypes_unlabeled = [
                    torch.zeros(
                        size=(1, self.model.latent_dim),
                        requires_grad=False,
                        device=self.device,
                    )
                    for _ in range(self.n_clusters)
                ]

                with torch.no_grad():
                    [
                        self.prototypes_unlabeled[i].copy_(leiden_prototypes[i, :])
                        for i in range(leiden_prototypes.shape[0])
                    ]

    def on_epoch_begin(self, lr, eps):
        """
        Routine that happens at the beginning of every epoch. Model update step.
        """
        if (self.epoch == self.pretraining_epochs) and (self.prototype_training is True):
            self.initialize_prototypes()
            if (self.any_unlabeled_data is True) and (self.unlabeled_prototype_training is True):
                self.prototype_optim = torch.optim.Adam(
                    params=self.prototypes_unlabeled,
                    lr=lr,
                    eps=eps,
                    weight_decay=self.weight_decay,
                )
        if self.epoch < self.pretraining_epochs:
            self.use_early_stopping = False
        if self.use_early_stopping_orig and self.epoch >= self.pretraining_epochs:
            self.use_early_stopping = True
        if self.epoch >= self.pretraining_epochs and self.epoch - 1 == self.best_epoch:
            self.best_prototypes_labeled = self.prototypes_labeled
            self.best_prototypes_labeled_cov = self.prototypes_labeled_cov
            self.best_prototypes_unlabeled = self.prototypes_unlabeled

    def loss(self, total_batch=None):
        latent, recon_loss, kl_loss, mmd_loss = self.model(**total_batch)

        #calculate classifier loss for labeled/unlabeled data
        label_categories = total_batch["labeled"].unique().tolist()
        unweighted_prototype_loss = torch.tensor(0.0, device=self.device)
        unlabeled_loss = torch.tensor(0.0, device=self.device)
        labeled_loss = torch.tensor(0.0, device=self.device)
        if self.epoch >= self.pretraining_epochs:
            #calculate prototype loss for all data
            if self.prototypes_unlabeled is not None:
                unlabeled_loss, _ = self.prototype_unlabeled_loss(
                    latent,
                    torch.stack(self.prototypes_unlabeled).squeeze(),
                )
                unweighted_prototype_loss = (
                    unweighted_prototype_loss + self.unlabeled_weight * unlabeled_loss
                )

            # Calculate prototype loss for labeled data
            if (self.any_labeled_data is True) and (self.prototype_training is True):
                labeled_loss = self.prototype_labeled_loss(
                    latent[torch.where(total_batch["labeled"] == 1)[0], :],
                    self.prototypes_labeled,
                    total_batch["celltypes"][
                        torch.where(total_batch["labeled"] == 1)[0], :
                    ],
                )
                unweighted_prototype_loss = unweighted_prototype_loss + labeled_loss

        # Loss addition and Logs
        prototype_loss = self.eta * unweighted_prototype_loss
        cvae_loss = recon_loss + self.calc_alpha_coeff() * kl_loss + mmd_loss
        loss = cvae_loss + prototype_loss
        self.iter_logs["loss"].append(loss.item())
        self.iter_logs["unweighted_loss"].append(
            recon_loss.item()
            + kl_loss.item()
            + mmd_loss.item()
            + unweighted_prototype_loss.item()
        )
        self.iter_logs["cvae_loss"].append(cvae_loss.item())
        if self.epoch >= self.pretraining_epochs:
            self.iter_logs["prototype_loss"].append(prototype_loss.item())
            if 0 in label_categories or self.model.unknown_ct_names is not None:
                self.iter_logs["unlabeled_loss"].append(unlabeled_loss.item())
            if 1 in label_categories:
                self.iter_logs["labeled_loss"].append(labeled_loss.item())
        return loss

    def on_epoch_end(self):
        """
        Routine at the end of each epoch. prototype update step.
        """
        self.model.eval()

        if (
            (self.epoch >= self.pretraining_epochs) 
            and (self.prototype_training is True)
        ):
            latent = self.get_latent_train()
            label_categories = self.train_data.labeled_vector.unique().tolist()

            # Update labeled prototype positions
            if self.any_labeled_data is True:
                (
                    self.prototypes_labeled,
                    self.prototypes_labeled_cov,
                ) = self.update_labeled_prototypes(
                    latent[torch.where(self.train_data.labeled_vector == 1)[0]],
                    self.train_data.cell_types[
                        torch.where(self.train_data.labeled_vector == 1)[0], :
                    ],
                    self.prototypes_labeled,
                    self.prototypes_labeled_cov,
                    self.model.new_prototypes,
                )

            # Update unlabeled prototype positions
            if (self.any_unlabeled_data is True) and (self.unlabeled_prototype_training is True):
                for proto in self.prototypes_unlabeled:
                    proto.requires_grad = True
                self.prototype_optim.zero_grad()
                update_loss, args_count = self.prototype_unlabeled_loss(
                    latent,
                    torch.stack(self.prototypes_unlabeled).squeeze(),
                )
                update_loss.backward()
                self.prototype_optim.step()
                for proto in self.prototypes_unlabeled:
                    proto.requires_grad = False

        self.model.train()
        super().on_epoch_end()

    def after_loop(self):
        """
        Routine at the end of training. Load best state.
        """
        if self.best_state_dict is not None and self.reload_best:
            self.prototypes_labeled = self.best_prototypes_labeled
            self.prototypes_labeled_cov = self.best_prototypes_labeled_cov
            self.prototypes_unlabeled = self.best_prototypes_unlabeled

        self.model.prototypes_labeled["mean"] = self.prototypes_labeled
        self.model.prototypes_labeled["cov"] = self.prototypes_labeled_cov

        if self.prototypes_unlabeled is not None:
            self.model.prototypes_unlabeled["mean"] = torch.stack(
                self.prototypes_unlabeled
            ).squeeze()
        else:
            self.model.prototypes_unlabeled["mean"] = self.prototypes_unlabeled

    def update_labeled_prototypes(
        self, latent, labels, previous_prototypes, previous_prototypes_cov, mask=None
    ):
        """
        Function that updates labeled prototypes.

        Parameters
        ==========
        latent: Tensor
            Latent representation of labeled batch
        labels: Tensor
            Tensor containing cell type information of the batch
        previous_prototypes: Tensor
            Tensor containing the means of the prototypes before update
        previous_prototypes_cov: Tensor
            Tensor containing the covariance matrices of the prototypes before udpate.

        """
        with torch.no_grad():
            unique_labels = torch.unique(labels, sorted=True)
            prototypes_mean = None
            prototypes_cov = None
            for value in range(self.model.n_cell_types):
                if (
                    mask is None or value in mask
                ) and value in unique_labels:  # update the prototype included in mask if there is one
                    indices = labels.eq(value).nonzero(as_tuple=False)[:, 0]
                    prototype = latent[indices, :].mean(0).unsqueeze(0)
                    prototype_cov = cov(latent[indices, :]).unsqueeze(0)
                    prototypes_mean = (
                        torch.cat([prototypes_mean, prototype])
                        if prototypes_mean is not None
                        else prototype
                    )
                    prototypes_cov = (
                        torch.cat([prototypes_cov, prototype_cov])
                        if prototypes_cov is not None
                        else prototype_cov
                    )
                else:  # do not update the prototypes (e.g. during surgery prototypes are fixed)
                    prototype = previous_prototypes[value].unsqueeze(0)
                    prototype_cov = previous_prototypes_cov[value].unsqueeze(0)
                    prototypes_mean = (
                        torch.cat([prototypes_mean, prototype])
                        if prototypes_mean is not None
                        else prototype
                    )
                    prototypes_cov = (
                        torch.cat([prototypes_cov, prototype_cov])
                        if prototypes_cov is not None
                        else prototype_cov
                    )
        return prototypes_mean, prototypes_cov

    def prototype_labeled_loss(self, latent, prototypes, labels):
        """
        Compute the labeled prototype loss. Different losses are included.

        Parameters
        ==========
        latent: Tensor
            Latent representation of labeled batch
        prototypes: Tensor
            Tensor containing the means of the prototypes
        labels: Tensor
            Tensor containing cell type information of the batch
        """
        unique_labels = torch.unique(labels, sorted=True)
        distances = euclidean_dist(latent, prototypes)
        loss = torch.tensor(0.0, device=self.device)

        # If data only contains 'unknown' celltypes
        if unique_labels.tolist() == [-1]:
            return loss

        # Basic euclidean distance loss
        for value in unique_labels:
            if value == -1:
                continue
            indices = labels.eq(value).nonzero(as_tuple=False)[:, 0]
            label_loss = distances[indices, value].sum(0) / len(indices)
            loss += label_loss

        return loss

    def prototype_unlabeled_loss(self, latent, prototypes):
        """
        Compute the unlabeled prototype loss. Different losses are included.

            Parameters
            ==========
            latent: Tensor
                Latent representation of labeled batch
            prototypes: Tensor
                Tensor containing the means of the prototypes
        """
        dists = euclidean_dist(latent, prototypes)
        min_dist, y_hat = torch.min(dists, 1)
        args_uniq = torch.unique(y_hat, sorted=True)
        args_count = torch.stack([(y_hat == x_u).sum() for x_u in args_uniq])

        loss_val = torch.stack(
            [min_dist[y_hat == idx_class].mean(0) for idx_class in args_uniq]
        ).mean()

        return loss_val, args_count
