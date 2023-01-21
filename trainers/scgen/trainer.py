import torch
from scipy import sparse
from anndata import AnnData
from collections import defaultdict

from ._utils import shuffle_adata, print_progress
from ...utils.monitor import EarlyStopping


class vaeArithTrainer:
    """
    This class contains the implementation of the VAEARITH Trainer

    Parameters
    ----------
    model: vaeArith
    adata: : `~anndata.AnnData`
        Annotated Data Matrix for training VAE network.
    n_epochs: int
        Number of epochs to iterate and optimize network weights
    train_frac: Float
        Defines the fraction of data that is used for training and data that is used for validation.
    batch_size: integer
        size of each batch of training dataset to be fed to network while training.
    patience: int
        Number of consecutive epochs in which network loss is not going lower.
        After this limit, the network will stop training.
    threshold: float
        Threshold for difference between consecutive validation loss values
        if the difference is upper than this `threshold`, this epoch will not
        considered as an epoch in early stopping.
    shuffle: bool
        if `True` shuffles the training dataset
    early_stopping_kwargs: Dict
        Passes custom Earlystopping parameters
    """
    def __init__(self, model, adata, train_frac: float = 0.9, batch_size = 32, shuffle=True, early_stopping_kwargs: dict = {
            "early_stopping_metric": "val_loss",
            "threshold": 0,
            "patience": 20,
            "reduce_lr": True,
            "lr_patience": 13,
            "lr_factor": 0.1}, **kwargs): # maybe add more parameters

        self.model = model

        self.seed = kwargs.get("seed", 2021)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.model.cuda() # put model to cuda(gpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.adata = adata

        self.train_frac = train_frac
        self.shuffle = shuffle
        self.batch_size = batch_size

        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else dict())
        self.early_stopping = EarlyStopping(**early_stopping_kwargs)
        self.monitor = kwargs.pop("monitor", True)

        # Optimization attributes
        self.optim = None
        # self.weight_decay = weight_decay
        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        # self.training_time = 0
        # self.n_iter = 0
        self.best_epoch = None
        self.best_state_dict = None

        self.logs = defaultdict(list)


    @staticmethod
    def _anndataToTensor(adata: AnnData) -> torch.Tensor:
        data_ndarray = adata.X.A
        data_tensor = torch.from_numpy(data_ndarray)
        return data_tensor

    def train_valid_split(self, adata: AnnData, train_frac = 0.9):
        if train_frac == 1:
            return adata, None
        n_obs = adata.shape[0]
        shuffled = shuffle_adata(adata)

        train_adata = shuffled[:int(train_frac * n_obs)] # maybe not the best way to round
        valid_adata = shuffled[int(train_frac * n_obs):]
        return train_adata, valid_adata


    def train(self, n_epochs = 100, lr = 0.001, eps = 1e-8, **extras_kwargs):
        self.n_epochs = n_epochs
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optim = torch.optim.Adam(
            params, lr=lr, eps=eps) # consider changing the param. like weight_decay, eps, etc.

        train_data, valid_data = self.train_valid_split(self.adata) # possible bad of using static method this way. Think about adding static methods to util.py

        if self.shuffle:
            train_adata = shuffle_adata(train_data)
            valid_adata = shuffle_adata(valid_data)
            loss_hist = []
        for self.epoch in range(self.n_epochs):
            self.model.train()
            self.iter_logs = defaultdict(list)
            train_loss = 0
            loss_hist.append(0)
            for lower in range(0, train_adata.shape[0], self.batch_size):
                upper = min(lower + self.batch_size, train_adata.shape[0])
                if sparse.issparse(train_adata.X):
                    x_mb = torch.from_numpy(train_adata[lower:upper, :].X.A)
                else:
                    x_mb = torch.from_numpy(train_adata[lower:upper, :].X)
                if upper - lower > 1:
                    x_mb = x_mb.to(self.device) #to cuda or cpu
                    reconstructions, mu, logvar = self.model(x_mb)

                    loss = self.model._loss_function(x_mb, reconstructions, mu, logvar)

                    self.optim.zero_grad()

                    loss.backward()
                    self.optim.step()

                    self.iter_logs["loss"].append(loss.item())
                    train_loss += loss.item() # loss.item() contains the loss of entire mini-batch divided by the batch size

            self.on_epoch_end()

            valid_loss = 0
            train_loss_end_epoch = 0
            self.iter_logs = defaultdict(list)
            with torch.no_grad(): # disables the gradient calculation
                self.model.eval()
                for lower in range(0, train_adata.shape[0], self.batch_size):
                    upper = min(lower + self.batch_size, train_adata.shape[0])
                    if sparse.issparse(train_adata.X):
                        x_mb = torch.from_numpy(train_adata[lower:upper, :].X.A)
                    else:
                        x_mb = torch.from_numpy(train_adata[lower:upper, :].X)
                    if upper - lower > 1:
                        x_mb = x_mb.to(self.device)
                        reconstructions, mu, logvar = self.model(x_mb)
                        loss = self.model._loss_function(x_mb, reconstructions, mu, logvar)

                        train_loss_end_epoch += loss.item()
                for lower in range(0, valid_adata.shape[0], self.batch_size):
                    upper = min(lower + self.batch_size, valid_adata.shape[0])
                    if sparse.issparse(valid_adata.X):
                        x_mb = torch.from_numpy(valid_adata[lower:upper, :].X.A)
                    else:
                        x_mb = torch.from_numpy(valid_adata[lower:upper, :].X)
                    if upper - lower > 1:
                        x_mb = x_mb.to(self.device)
                        reconstructions, mu, logvar = self.model(x_mb)
                        loss = self.model._loss_function(x_mb, reconstructions, mu, logvar)

                        self.iter_logs["loss"].append(loss.item())
                        valid_loss += loss.item() # loss.item() contains the loss of entire mini-batch divided by the batch size

            # Get Validation Logs
            for key in self.iter_logs:
                if "loss" in key:
                    self.logs["val_" + key].append(
                        sum(self.iter_logs[key][:]) / len(self.iter_logs[key][:]))


            # Monitor Logs
            if self.monitor:
                print_progress(self.epoch, self.logs, self.n_epochs)

            if not self.check_early_stop():
                break

        if self.best_state_dict is not None:
            print("Saving best state of network...")
            print("Best State was in Epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)



    def on_epoch_end(self):
        # Get Train Epoch Logs
        for key in self.iter_logs:
            if "loss" in key:
                self.logs["epoch_" + key].append(
                    sum(self.iter_logs[key][:]) / len(self.iter_logs[key][:]))



    def check_early_stop(self):
        # Calculate Early Stopping and best state
        early_stopping_metric = self.early_stopping.early_stopping_metric
        if self.early_stopping.update_state(self.logs[early_stopping_metric][-1]):
            self.best_state_dict = self.model.state_dict()
            self.best_epoch = self.epoch

        continue_training, update_lr = self.early_stopping.step(self.logs[early_stopping_metric][-1])


        if update_lr:
            print(f'\nADJUSTED LR')
            for param_group in self.optim.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor

        return continue_training
