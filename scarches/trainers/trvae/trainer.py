import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from scarches.utils.monitor import EarlyStopping
from ._utils import make_dataset, custom_collate, print_progress


class Trainer:
    """ScArches base Trainer class. This class contains the implementation of the base CVAE/TRVAE Trainer.

       Parameters
       ----------
       model: trVAE
            Number of input features (i.e. gene in case of scRNA-seq).
       adata: : `~anndata.AnnData`
            Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
            for 'mse' loss.
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
       use_stratified_split: Boolean
            If `True`, the train and validation data will be constructed in such a way that both have same distribution
            of conditions in the data.
       monitor: Boolean
            If `True', the progress of the training will be printed after each epoch.
       n_workers: Integer
            Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
       seed: Integer
            Define a specific random seed to get reproducable results.
    """
    def __init__(self,
                 model,
                 adata,
                 condition_key: str = None,
                 cell_type_key: str = None,
                 train_frac: float = 0.9,
                 batch_size: int = 128,
                 n_samples: int = None,
                 clip_value: float = 0.0,
                 weight_decay: float = 0.04,
                 alpha_iter_anneal: int = None,
                 alpha_epoch_anneal: int = None,
                 early_stopping_kwargs: dict = None,
                 **kwargs):

        self.adata = adata
        self.model = model
        self.condition_key = condition_key
        self.cell_type_key = cell_type_key
        self.train_frac = train_frac

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.clip_value = clip_value
        self.weight_decay = weight_decay
        self.alpha_iter_anneal = alpha_iter_anneal
        self.alpha_epoch_anneal = alpha_epoch_anneal

        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else dict())

        self.use_early_stopping = kwargs.pop("use_early_stopping", True)
        self.use_stratified_sampling = kwargs.pop("use_stratified_sampling", True)
        self.use_stratified_split = kwargs.pop("use_stratified_split", False)
        self.monitor = kwargs.pop("monitor", True)
        self.n_workers = kwargs.pop("n_workers", 0)
        self.seed = kwargs.pop("seed", 2020)

        self.early_stopping = EarlyStopping(**early_stopping_kwargs)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.epoch = -1
        self.n_epochs = None
        self.iter = 0
        self.best_epoch = None
        self.best_state_dict = None
        self.current_loss = None
        self.previous_loss_was_nan = False
        self.nan_counter = 0
        self.optimizer = None
        self.training_time = 0

        self.train_data = None
        self.valid_data = None
        self.sampler = None
        self.dataloader_train = None
        self.dataloader_valid = None

        self.iters_per_epoch = None
        self.val_iters_per_epoch = None

        self.logs = defaultdict(list)

    def calc_alpha_coeff(self):
        """Calculates current alpha coefficient for alpha annealing.

           Parameters
           ----------

           Returns
           -------
           Current annealed alpha value
        """
        if self.alpha_epoch_anneal is not None:
            alpha_coeff = min(self.epoch / self.alpha_epoch_anneal, 1)
        elif self.alpha_iter_anneal is not None:
            alpha_coeff = min(((self.epoch * self.iters_per_epoch + self.iter) / self.alpha_iter_anneal), 1)
        else:
            alpha_coeff = 1
        return alpha_coeff

    def train(self,
              n_epochs=400,
              lr=1e-3,
              eps=0.01):
        begin = time.time()
        self.model.train()
        self.n_epochs = n_epochs

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=self.weight_decay)
        # Initialize Train/Val Data, Sampler, Dataloader
        self.initialize_loaders()

        for self.epoch in range(n_epochs):
            self.iter_logs = defaultdict(list)
            for self.iter, batch_data in enumerate(self.dataloader_train):
                # Safe data to right device
                for key1 in batch_data:
                    for key2, batch in batch_data[key1].items():
                        batch_data[key1][key2] = batch.to(self.device)

                # Loss Calculation
                self.on_iteration(batch_data)

            # Validation of Model, Monitoring, Early Stopping
            self.on_epoch_end()
            if self.use_early_stopping:
                if not self.check_early_stop():
                    break

        if self.best_state_dict is not None:
            print("Saving best state of network...")
            print("Best State was in Epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)

        self.model.eval()

        self.training_time += (time.time() - begin)

    def initialize_loaders(self):
        """
        Initializes Train-/Test Data and Dataloaders with custom_collate and WeightedRandomSampler for Trainloader.
        Returns:

        """
        # Create Train/Valid AnnotatetDataset objects
        self.train_data, self.valid_data = make_dataset(
            self.adata,
            train_frac=self.train_frac,
            use_stratified_split=self.use_stratified_split,
            condition_key=self.condition_key,
            cell_type_key=self.cell_type_key,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=None,
        )

        if self.n_samples is None or self.n_samples > len(self.train_data):
            self.n_samples = len(self.train_data)
        self.iters_per_epoch = int(np.ceil(self.n_samples / self.batch_size))

        if self.use_stratified_sampling:
            # Create Sampler and Dataloaders
            stratifier_weights = torch.tensor(self.train_data.stratifier_weights, device=self.device)

            self.sampler = WeightedRandomSampler(stratifier_weights,
                                                 num_samples=self.n_samples,
                                                 replacement=True)
            self.dataloader_train = torch.utils.data.DataLoader(dataset=self.train_data,
                                                                batch_size=self.batch_size,
                                                                sampler=self.sampler,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
        else:
            self.dataloader_train = torch.utils.data.DataLoader(dataset=self.train_data,
                                                                batch_size=self.batch_size,
                                                                shuffle=True,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
        if self.valid_data is not None:
            val_batch_size = self.batch_size
            if self.batch_size > len(self.valid_data):
                val_batch_size = len(self.valid_data)
            self.val_iters_per_epoch = int(np.ceil(len(self.valid_data) / self.batch_size))
            self.dataloader_valid = torch.utils.data.DataLoader(dataset=self.valid_data,
                                                                batch_size=val_batch_size,
                                                                shuffle=True,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)

    def on_iteration(self, batch_data):
        # Dont update any weight on first layers except condition weights
        if self.model.freeze:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    if not module.weight.requires_grad:
                        module.affine = False
                        module.track_running_stats = False

        # Calculate Loss depending on Trainer/Model
        self.current_loss = loss = self.loss(**batch_data)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        if self.clip_value > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)

        self.optimizer.step()

    def on_epoch_end(self):
        # Get Train Epoch Logs
        for key in self.iter_logs:
            if "loss" in key:
                self.logs["epoch_" + key].append(
                    sum(self.iter_logs[key][:]).cpu().detach().numpy() / len(self.iter_logs[key][:]))

        # Validate Model
        if self.valid_data is not None:
            self.validate()

        # Monitor Logs
        if self.monitor:
            print_progress(self.epoch, self.logs, self.n_epochs)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.iter_logs = defaultdict(list)
        # Calculate Validation Losses
        for val_iter, batch_data in enumerate(self.dataloader_valid):
            for key1 in batch_data:
                for key2, batch in batch_data[key1].items():
                    batch_data[key1][key2] = batch.to(self.device)

            val_loss = self.loss(**batch_data)

        # Get Validation Logs
        for key in self.iter_logs:
            if "loss" in key:
                self.logs["val_" + key].append(
                    sum(self.iter_logs[key][:]).cpu().detach().numpy() / len(self.iter_logs[key][:]))

        self.model.train()

    def check_early_stop(self):
        # Calculate Early Stopping and best state
        early_stopping_metric = self.early_stopping.early_stopping_metric
        if self.early_stopping.update_state(self.logs[early_stopping_metric][-1]):
            self.best_state_dict = self.model.state_dict()
            self.best_epoch = self.epoch

        continue_training, update_lr = self.early_stopping.step(self.logs[early_stopping_metric][-1])
        if update_lr:
            print(f'\nADJUSTED LR')
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor

        return continue_training
