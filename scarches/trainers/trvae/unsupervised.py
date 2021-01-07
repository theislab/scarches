from .trainer import Trainer
import torch


class trVAETrainer(Trainer):
    """ScArches Unsupervised Trainer class. This class contains the implementation of the unsupervised CVAE/TRVAE
       Trainer.

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
    def __init__(
            self,
            model,
            adata,
            **kwargs
    ):
        super().__init__(model, adata, **kwargs)

    def loss(self, total_batch=None):
        recon_loss, kl_loss, mmd_loss = self.model(**total_batch)
        loss = recon_loss + self.calc_alpha_coeff()*kl_loss + mmd_loss
        self.iter_logs["loss"].append(loss)
        self.iter_logs["unweighted_loss"].append(recon_loss + kl_loss + mmd_loss)
        self.iter_logs["recon_loss"].append(recon_loss)
        self.iter_logs["kl_loss"].append(kl_loss)
        if self.model.use_mmd:
            self.iter_logs["mmd_loss"].append(mmd_loss)
        return loss