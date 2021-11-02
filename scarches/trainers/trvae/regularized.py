from .unsupervised import trVAETrainer
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

class ProxOperL1:
    def __init__(self, alpha, I=None, inplace=True):
        self._I = ~I.bool() if I is not None else None
        self._alpha=alpha
        self._inplace=inplace

    def __call__(self, W):
        if not self._inplace:
            W = W.clone()

        W_geq_alpha = W>=self._alpha
        W_leq_neg_alpha = W<=-self._alpha
        W_cond_joint = ~W_geq_alpha&~W_leq_neg_alpha

        if self._I is not None:
            W_geq_alpha &= self._I
            W_leq_neg_alpha &= self._I
            W_cond_joint &= self._I

        W -= W_geq_alpha.float()*self._alpha
        W += W_leq_neg_alpha.float()*self._alpha
        W -= W_cond_joint.float()*W

        return W

def get_prox_operator(alpha, omega, alpha_l1, mask):
    if alpha is not None:
        p_gr = ProxGroupLasso(alpha, omega)
    else:
        p_gr = lambda W: W

    prox_op = p_gr

    if alpha_l1 is not None:
        if mask is None:
            raise ValueError('Provide soft mask.')
        p_l1_annot = ProxOperL1(alpha_l1, mask)
        prox_op = lambda W: p_gr(p_l1_annot(W))

    return prox_op

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
            alpha_l1=None,
            gamma_ext=None,
            beta=1.,
            print_n_deactive=False,
            **kwargs
    ):
        super().__init__(model, adata, **kwargs)

        self.alpha = alpha
        self.omega = omega
        self.alpha_l1 = alpha_l1
        self.prox_operator_compose = None
        self.print_n_deactive = print_n_deactive

        self.gamma_ext = gamma_ext
        self.prox_operator_l1_ext = None

        self.watch_lr = None

        if self.omega is not None:
            self.omega = self.omega.to(self.device)

        if self.model.use_hsic:
            self.beta = beta
        else:
            self.beta = None

        self.compose_init = False
        self.l1_ext_init = False

    def on_iteration(self, batch_data):
        if self.prox_operator_compose is None and (self.alpha is not None or self.alpha_l1 is not None):
            self.watch_lr = self.optimizer.param_groups[0]['lr']

            if self.model.mask is not None:
                dvc = self.model.decoder.L0.expr_L.weight.device
                self.model.mask = self.model.mask.to(dvc)

            alpha_corr = self.alpha*self.watch_lr if self.alpha is not None else None
            alpha_l1_corr = self.alpha_l1*self.watch_lr if self.alpha_l1 is not None else None
            self.prox_operator_compose = get_prox_operator(alpha_corr, self.omega, alpha_l1_corr, self.model.mask)
            self.compose_init = True

        has_ext = self.model.decoder.L0.n_ext > 0
        if self.prox_operator_l1_ext is None and self.gamma_ext is not None and has_ext:
            if self.watch_lr is None:
                self.watch_lr = self.optimizer.param_groups[0]['lr']
            self.prox_operator_l1_ext = ProxOperL1(self.gamma_ext*self.watch_lr)
            self.l1_ext_init = True

        super().on_iteration(batch_data)

        if self.prox_operator_compose is not None:
            self.prox_operator_compose(self.model.decoder.L0.expr_L.weight.data)

        if self.prox_operator_l1_ext is not None:
            self.prox_operator_l1_ext(self.model.decoder.L0.ext_L.weight.data)

    def check_early_stop(self):
        continue_training = super().check_early_stop()

        if continue_training:
            new_lr = self.optimizer.param_groups[0]['lr']
            if self.watch_lr is not None and self.watch_lr != new_lr:
                self.watch_lr = new_lr
                if self.compose_init:
                    alpha_corr = self.alpha*self.watch_lr if self.alpha is not None else None
                    alpha_l1_corr = self.alpha_l1*self.watch_lr if self.alpha_l1 is not None else None
                    self.prox_operator_compose = get_prox_operator(alpha_corr, self.omega, alpha_l1_corr, self.model.mask)
                if self.l1_ext_init:
                    self.prox_operator_l1_ext = ProxOperL1(self.gamma_ext*self.watch_lr)

        return continue_training

    def on_epoch_end(self):
        if self.print_n_deactive:
            if self.alpha is not None:
                n_deact_terms = self.model.decoder.n_inactive_terms()
                msg = f'Number of deactivated terms: {n_deact_terms}'
                if self.epoch > 0:
                    msg = '\n' + msg
                print(msg)
                print('-------------------')
            if self.alpha_l1 is not None:
                share_deact_genes = (self.model.decoder.L0.expr_L.weight.data.abs()==0)&~self.model.mask.bool()
                share_deact_genes = share_deact_genes.float().sum().cpu().numpy() / self.model.n_inact_genes
                print('Share of deactivated inactive genes: %.4f' % share_deact_genes)
                print('-------------------')
            if self.l1_ext_init:
                active_genes = (self.model.decoder.L0.ext_L.weight.data.abs().cpu().numpy()>0).sum(0)
                print ('Active genes in extension terms:', active_genes)
                sparse_share = 1. - active_genes / self.model.input_dim
                print('Sparcity share in extension terms:', sparse_share)
        super().on_epoch_end()

    def loss(self, total_batch=None):
        if self.beta is None:
            return super().loss(total_batch)
        else:
            recon_loss, kl_loss, mmd_loss, hsic_loss = self.model(**total_batch)
            loss = recon_loss + self.calc_alpha_coeff()*kl_loss + mmd_loss + self.beta*hsic_loss
            self.iter_logs["loss"].append(loss)
            self.iter_logs["unweighted_loss"].append(recon_loss + kl_loss + mmd_loss + hsic_loss)
            self.iter_logs["recon_loss"].append(recon_loss)
            self.iter_logs["kl_loss"].append(kl_loss)
            self.iter_logs["hsic_loss"].append(kl_loss)
            if self.model.use_mmd:
                self.iter_logs["mmd_loss"].append(mmd_loss)
            return loss
