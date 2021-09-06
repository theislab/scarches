from ..trvae.unsupervised import trVAETrainer
import torch
import sys

class ProxGroupLasso:
    def __init__(self, alpha, omega=None, inplace=True):
    # omega - vector of coefficients with size
    # equal to the number of groups
        if omega is None:
            self._group_coeff = alpha
        else:
            self._group_coeff = (omega*alpha).view(-1)

        self._inplace = inplace

    def __call__(self, W):
        if not self._inplace:
            W = W.clone()

        norm_vect = W.norm(p=2, dim=0)
        norm_g_gr_vect = norm_vect>self._group_coeff

        scaled_norm_vector = norm_vect/self._group_coeff
        scaled_norm_vector+=(~(scaled_norm_vector>0)).float()

        W-=W/scaled_norm_vector
        W*=norm_g_gr_vect.float()

        return W

class VIATrainer(trVAETrainer):
    """ScArches Unsupervised Trainer class. This class contains the implementation of the unsupervised CVAE/TRVAE
       Trainer.
           Parameters
           ----------
           model: trVAE
                Number of input features (i.e. gene in case of scRNA-seq).
           adata: : `~anndata.AnnData`
                Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
                for 'mse' loss.
           alpha: Float
                Group Lasso regularization coefficient
           omega: Tensor or None
                If not 'None', vector of coefficients for each group
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
           weight_decay: Float
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
    def __init__(
            self,
            model,
            adata,
            alpha,
            omega=None,
            print_n_deactive=False,
            **kwargs
    ):
        super().__init__(model, adata, **kwargs)

        self.alpha = alpha
        self.omega = omega
        self.prox_operator = None
        self.print_n_deactive = print_n_deactive

        self.watch_lr = None

        if self.omega is not None:
            self.omega = self.omega.to(self.device)

    def on_iteration(self, batch_data):
        if self.prox_operator is None and self.alpha is not None:
            self.watch_lr = self.optimizer.param_groups[0]['lr']
            self.prox_operator = ProxGroupLasso(self.alpha*self.watch_lr, self.omega)

        super().on_iteration(batch_data)

        if self.prox_operator is not None:
            self.prox_operator(self.model.decoder.L0.expr_L.weight.data)

    def check_early_stop(self):
        continue_training = super().check_early_stop()

        if continue_training:
            new_lr = self.optimizer.param_groups[0]['lr']
            if self.watch_lr is not None and self.watch_lr != new_lr:
                self.watch_lr = new_lr
                self.prox_operator = ProxGroupLasso(self.alpha*self.watch_lr, self.omega)

        return continue_training

    def on_epoch_end(self):
        if self.print_n_deactive:
            if self.alpha is not None:
                n_deact_terms = self.model.decoder.n_inactive_terms()
                msg = f'Number of deactivated terms: {n_deact_terms}'
                if self.epoch > 0:
                    msg = '\n' + msg
                print(msg)
        super().on_epoch_end()
