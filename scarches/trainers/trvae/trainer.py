import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from ...utils.monitor import EarlyStopping
from ._utils import make_dataset, custom_collate, print_progress




fisher_dict = {}
optpar_dict = {}
ewc_lambda = 1

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
       cell_type_keys: List
            List of column names of different celltype levels in `adata.obs` data frame.
       batch_size: Integer
            Defines the batch size that is used during each Iteration
       alpha_epoch_anneal: Integer or None
            If not 'None', the KL Loss scaling factor (alpha_kl) will be annealed from 0 to 1 every epoch until the input
            integer is reached.
       alpha_kl: Float
            Multiplies the KL divergence part of the loss.
       alpha_iter_anneal: Integer or None
            If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every iteration until the input
            integer is reached.
       use_early_stopping: Boolean
            If 'True' the EarlyStopping class is being used for training to prevent overfitting.
       reload_best: Boolean
            If 'True' the best state of the model during training concerning the early stopping criterion is reloaded
            at the end of training.
       early_stopping_kwargs: Dict
            Passes custom Earlystopping parameters.
       train_frac: Float
            Defines the fraction of data that is used for training and data that is used for validation.
       n_samples: Integer or None
            Defines how many samples are being used during each epoch. This should only be used if hardware resources
            are limited.
       use_stratified_sampling: Boolean
            If 'True', the sampler tries to load equally distributed batches concerning the conditions in every
            iteration.
       monitor: Boolean
            If `True', the progress of the training will be printed after each epoch.
       monitor_only_val: Boolean
            If `True', only the progress of the validation datset is displayed.
       clip_value: Float
            If the value is greater than 0, all gradients with an higher value will be clipped during training.
       weight decay: Float
            Defines the scaling factor for weight decay in the Adam optimizer.
       n_workers: Integer
            Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
       seed: Integer
            Define a specific random seed to get reproducable results.
    """
    def __init__(self,
                 model,
                 adata,
                 condition_key: str = None,
                 cell_type_keys: str = None,
                 batch_size: int = 128,
                 alpha_epoch_anneal: int = None,
                 alpha_kl: float = 1.,
                 use_early_stopping: bool = True,
                 reload_best: bool = True,
                 early_stopping_kwargs: dict = None,
                 **kwargs):

        self.adata = adata
        self.model = model
        self.condition_key = condition_key
        self.cell_type_keys = cell_type_keys

        self.batch_size = batch_size
        self.alpha_epoch_anneal = alpha_epoch_anneal
        self.alpha_iter_anneal = kwargs.pop("alpha_iter_anneal", None)
        self.use_early_stopping = use_early_stopping
        self.reload_best = reload_best

        self.alpha_kl = alpha_kl

        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else dict())

        self.n_samples = kwargs.pop("n_samples", None)
        self.train_frac = kwargs.pop("train_frac", 0.9)
        self.use_stratified_sampling = kwargs.pop("use_stratified_sampling", True)

        self.weight_decay = kwargs.pop("weight_decay", 0.04)
        self.clip_value = kwargs.pop("clip_value", 0.0)

        self.n_workers = kwargs.pop("n_workers", 0)
        self.seed = kwargs.pop("seed", 2020)
        self.monitor = kwargs.pop("monitor", True)
        self.monitor_only_val = kwargs.pop("monitor_only_val", True)

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

        # Create Train/Valid AnnotatetDataset objects
        self.train_data, self.valid_data = make_dataset(
            self.adata,
            train_frac=self.train_frac,
            condition_key=self.condition_key,
            cell_type_keys=self.cell_type_keys,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=self.model.cell_type_encoder,
        )

    def initialize_loaders(self):
        """
        Initializes Train-/Test Data and Dataloaders with custom_collate and WeightedRandomSampler for Trainloader.
        Returns:

        """
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

    def calc_alpha_coeff(self):
        """Calculates current alpha coefficient for alpha annealing.

           Parameters
           ----------

           Returns
           -------
           Current annealed alpha value
        """
        if self.alpha_epoch_anneal is not None:
            alpha_coeff = min(self.alpha_kl * self.epoch / self.alpha_epoch_anneal, self.alpha_kl)
        elif self.alpha_iter_anneal is not None:
            alpha_coeff = min((self.alpha_kl * (self.epoch * self.iters_per_epoch + self.iter) / self.alpha_iter_anneal), self.alpha_kl)
        else:
            alpha_coeff = self.alpha_kl
        return alpha_coeff

    def train(self,
              n_epochs=400,
              lr=1e-3,
              eps=0.01,
              ID=0,
              learning_approach=None):

        self.initialize_loaders()
        begin = time.time()
        self.model.train()
        self.n_epochs = n_epochs
    

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=self.weight_decay)

        self.before_loop()

        for self.epoch in range(n_epochs):
            self.on_epoch_begin(lr, eps)
            self.iter_logs = defaultdict(list)
            for self.iter, batch_data in enumerate(self.dataloader_train):
                for key, batch in batch_data.items():
                    batch_data[key] = batch.to(self.device)
                # Loss Calculation
                if learning_approach==None:
                    self.on_standard_learning(batch_data)
                elif learning_approach=='Surgery':
                    self.on_iteration(batch_data)               
                elif learning_approach=='ewc':
                    self.on_ewc(ID,batch_data) 
                elif learning_approach=='rehearsal':
                    self.on_rehearsal(batch_data)
                elif learning_approach == 'latent replay': 
                    if self.epoch == 0: 
                        batch_data['first_epoch'] = 1 #flag data in first epoch to be saved later in the forward function of encoder
                        for name, module in self.model.named_modules():
                            if 'encoder.FC.L1' in name:
                                batch_data['replay_layer'] = 1 #when = 1, this is the latent replay layer
                    else:
                        batch_data['first_epoch'] = 0 #flag data in other epochs NOT to be saved later in the forward function of encoder
                    freeze = True
                    if freeze and self.iter>0: #freeze the parameters of layers before the latent layer in encoder, decoder doesn't have
                        for name, module in self.model.named_modules():
                            if 'encoder.FC.L0' in name:
                                for p_name, p in module.named_parameters():
                                    p.requires_grad = False
                    self.on_latent_replay(ID,batch_data)
                    
                elif learning_approach == 'LR+EWC':
                    if self.epoch == 0: 
                        batch_data['first_epoch'] = 1 #flag data in first epoch to be saved later in the forward function of encoder
                        for name, module in self.model.named_modules():
                            if 'encoder.FC.L1' in name:
                                batch_data['replay_layer'] = 1 # flag the reply layer, when = 1, this is the latent replay layer
                    else:
                        batch_data['first_epoch'] = 0
                    freeze = True
                    if freeze and self.iter>0:
                        for name, module in self.model.named_modules():
                            if 'encoder.FC.L0' in name:
                                for p_name, p in module.named_parameters():
                                    p.requires_grad = False
                    self.on_LR_EWC(ID,batch_data)  
                               


            # Validation of Model, Monitoring, Early Stopping
            self.on_epoch_end()
            if self.use_early_stopping:
                if not self.check_early_stop():
                    break
        if learning_approach=='ewc':
            self.on_task_update(ID,batch_data) #batch_data here is not the whole data, it is only the last batch in the training                                                            #set. torch.tensor[8,1001] which does not matter for this function as its job is to store
                                               #fisher inofrmation. In this function parameters are NOT updated.
        if learning_approach=='LR+EWC': 
            self.on_task_update(ID,batch_data) #feed the last batch to to store fisher information without updating the parameters


        if self.best_state_dict is not None and self.reload_best:
            print("Saving best state of network...")
            print("Best State was in Epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)

        self.model.eval()
        self.after_loop()

        self.training_time += (time.time() - begin)

    def before_loop(self):
        pass

    def on_epoch_begin(self, lr, eps):
        pass

    def after_loop(self):
        pass

    def on_rehearsal(self, batch_data):
        self.current_loss = loss = self.loss(batch_data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
    def on_latent_replay(self, ID, batch_data):
        batch_data['external_memory'] = 1 #when = 1, save ONLY the training set WITHOUT the validation set
        batch_data['dataset_counter'] = ID              
        self.current_loss = loss = self.loss(batch_data)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True) 
        self.optimizer.step()
        
    def on_LR_EWC(self,task_id, batch_data):
        batch_data['external_memory'] = 1
        batch_data['dataset_counter'] = task_id  
        self.train_ewc(task_id,batch_data)
       
    def on_task_update(self,task_id,batch_data):
        print('Calculating fisher information!')
        self.optimizer.zero_grad()
        self.current_loss = loss = self.loss(batch_data)
        loss.backward()


        fisher_dict[task_id] = {}
        optpar_dict[task_id] = {}

        for name, param in self.model.named_parameters():
            optpar_dict[task_id][name] = param.data.clone() 
            fisher_dict[task_id][name] = param.grad.data.clone().pow(2) 

            
    def train_ewc(self,task_id,batch_data):  
        self.current_loss = loss = self.loss(batch_data)
        self.optimizer.zero_grad()

        for task in range(task_id): 
            for name, param in self.model.named_parameters():
                fisher = fisher_dict[task][name]
                optpar = optpar_dict[task][name]
                if 'theta' in name: 
                    if fisher[task].size()==torch.Size([1]):
                        loss += (fisher * (optpar - param[:,task+1]).pow(2)).sum() * ewc_lambda
                    else:
                        loss += (fisher[:,task]* (optpar[:,task] - param[:,task+1]).pow(2)).sum() * ewc_lambda
                
                elif 'bias' in name:
                    loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

                else:
                    if fisher.size()==param.size():
                        loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda 
                    else:
                        loss += (fisher[:,task] * (optpar[:,task] - param[:,task+1]).pow(2)).sum() * ewc_lambda
      
        loss.backward()
        self.optimizer.step()        
        
    def on_ewc(self,task_id,batch_data): 
        self.train_ewc(task_id,batch_data)  
                           
    def on_iteration(self, batch_data):
        # Dont update any weight on first layers except condition weights
        if self.model.freeze:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    if not module.weight.requires_grad:
                        module.affine = False
                        module.track_running_stats = False

        # Calculate Loss depending on Trainer/Model
        self.current_loss = loss = self.loss(batch_data)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        if self.clip_value > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)

        self.optimizer.step()

    def on_standard_learning(self,batch_data):
        # Calculate Loss depending on Trainer/Model
        self.current_loss = loss = self.loss(batch_data)
        self.optimizer.zero_grad()
        loss.backward()
        
    def on_epoch_end(self):
        # Get Train Epoch Logs
        for key in self.iter_logs:
            self.logs["epoch_" + key].append(np.array(self.iter_logs[key]).mean())

        # Validate Model
        if self.valid_data is not None:
            self.validate()

        # Monitor Logs
        if self.monitor:
            print_progress(self.epoch, self.logs, self.n_epochs, self.monitor_only_val)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.iter_logs = defaultdict(list)
        # Calculate Validation Losses
        for val_iter, batch_data in enumerate(self.dataloader_valid):
            for key, batch in batch_data.items():
                batch_data[key] = batch.to(self.device)
            batch_data['external_memory'] = 0 #when = 0, it is the validatin set, so do not save it in the memory
            val_loss = self.loss(batch_data)

        # Get Validation Logs
        for key in self.iter_logs:
            self.logs["val_" + key].append(np.array(self.iter_logs[key]).mean())

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
