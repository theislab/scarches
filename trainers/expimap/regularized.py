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

        # to check for update
        self._alpha = alpha

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


class ProxL1:
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


class expiMapTrainer(trVAETrainer):
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
           beta: Float or None
                HSIC regularization coefficient for new unannotated terms.
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
           alpha_l1: Float
                L1 regularization coefficient for the soft mask of reference and new constrained terms.
                Specifies the strength for deactivating the genes which are not in the corresponding annotations \ groups
                in the mask.
           alpha_l1_epoch_anneal: Integer
                If not 'None', the alpha_l1 scaling factor will be annealed from 0 to 1 every 'alpha_l1_anneal_each' epochs
                until the input integer is reached.
           alpha_l1_anneal_each: Integer
                Anneal alpha_l1 every alpha_l1_anneal_each'th epoch, i.e. for 5 (default)
                do annealing every 5th epoch.
           gamma_ext: Float
                L1 regularization coefficient for the new unconstrained terms. Specifies the strength of
                sparcity enforcement.
           gamma_epoch_anneal: Integer
                If not 'None', the gamma_ext scaling factor will be annealed from 0 to 1 every 'gamma_anneal_each' epochs
                until the input integer is reached.
           gamma_anneal_each: Integer
                Anneal gamma_ext every gamma_anneal_each'th epoch, i.e. for 5 (default)
                do annealing every 5th epoch.
           beta: Float
                HSIC regularization coefficient for the unconstrained terms.
                Multiplies the HSIC loss terms if not 'None'.
        """
    def __init__(
            self,
            model,
            adata,
            alpha,
            omega=None,
            alpha_l1=None,
            alpha_l1_epoch_anneal=None,
            alpha_l1_anneal_each=5,
            gamma_ext=None,
            gamma_epoch_anneal=None,
            gamma_anneal_each=5,
            beta=1.,
            print_stats=False,
            **kwargs
    ):
        super().__init__(model, adata, **kwargs)

        self.print_stats = print_stats

        self.alpha = alpha
        self.omega = omega

        if self.omega is not None:
            self.omega = self.omega.to(self.device)

        self.gamma_ext = gamma_ext
        self.gamma_epoch_anneal = gamma_epoch_anneal
        self.gamma_anneal_each = gamma_anneal_each

        self.alpha_l1 = alpha_l1
        self.alpha_l1_epoch_anneal = alpha_l1_epoch_anneal
        self.alpha_l1_anneal_each = alpha_l1_anneal_each

        if self.model.use_hsic:
            self.beta = beta
        else:
            self.beta = None

        self.watch_lr = None

        self.use_prox_ops = self.check_prox_ops()
        self.prox_ops = {}

        self.corr_coeffs = self.init_anneal()

    def check_prox_ops(self):
        use_prox_ops = {}

        use_main = self.model.decoder.L0.expr_L.weight.requires_grad

        use_prox_ops['main_group_lasso'] = use_main and self.alpha is not None

        use_mask = use_main and self.model.mask is not None
        use_prox_ops['main_soft_mask'] = use_mask and self.alpha_l1 is not None

        use_ext = self.model.n_ext_decoder > 0 and self.gamma_ext is not None
        use_ext = use_ext and self.model.decoder.L0.ext_L.weight.requires_grad
        use_prox_ops['ext_unannot_l1'] = use_ext

        use_ext_m = self.model.n_ext_m_decoder > 0 and self.alpha_l1 is not None
        use_ext_m = use_ext_m and self.model.decoder.L0.ext_L_m.weight.requires_grad
        use_prox_ops['ext_soft_mask'] = use_ext_m and self.model.ext_mask is not None

        return use_prox_ops

    def init_anneal(self):
        corr_coeffs = {}

        use_soft_mask = self.use_prox_ops['main_soft_mask'] or self.use_prox_ops['ext_soft_mask']
        if use_soft_mask and self.alpha_l1_epoch_anneal is not None:
            corr_coeffs['alpha_l1'] = 1. / self.alpha_l1_epoch_anneal
        else:
            corr_coeffs['alpha_l1'] = 1.

        if self.use_prox_ops['ext_unannot_l1'] and self.gamma_epoch_anneal is not None:
            corr_coeffs['gamma_ext'] = 1. / self.gamma_epoch_anneal
        else:
            corr_coeffs['gamma_ext'] = 1.

        return corr_coeffs

    def anneal(self):
        any_change = False

        if self.corr_coeffs['gamma_ext'] < 1.:
            any_change = True
            time_to_anneal = self.epoch > 0 and self.epoch % self.gamma_anneal_each == 0
            if time_to_anneal:
                self.corr_coeffs['gamma_ext'] = min(self.epoch / self.gamma_epoch_anneal, 1.)
                if self.print_stats:
                    print('New gamma_ext anneal coefficient:', self.corr_coeffs['gamma_ext'])

        if self.corr_coeffs['alpha_l1'] < 1.:
            any_change = True
            time_to_anneal = self.epoch > 0 and self.epoch % self.self.alpha_l1_anneal_each == 0
            if time_to_anneal:
                self.corr_coeffs['alpha_l1'] = min(self.epoch / self.alpha_l1_epoch_anneal, 1.)
                if self.print_stats:
                    print('New alpha_l1 anneal coefficient:', self.corr_coeffs['alpha_l1'])

        return any_change

    def init_prox_ops(self):
        if any(self.use_prox_ops.values()) and self.watch_lr is None:
            self.watch_lr = self.optimizer.param_groups[0]['lr']

        if 'main_group_lasso' not in self.prox_ops and self.use_prox_ops['main_group_lasso']:
            print('Init the group lasso proximal operator for the main terms.')
            alpha_corr = self.alpha * self.watch_lr
            self.prox_ops['main_group_lasso'] = ProxGroupLasso(alpha_corr, self.omega)

        if 'main_soft_mask' not in self.prox_ops and self.use_prox_ops['main_soft_mask']:
            print('Init the soft mask proximal operator for the main terms.')
            main_mask = self.model.mask.to(self.device)
            alpha_l1_corr = self.alpha_l1 * self.watch_lr * self.corr_coeffs['alpha_l1']
            self.prox_ops['main_soft_mask'] = ProxL1(alpha_l1_corr, main_mask)

        if 'ext_unannot_l1' not in self.prox_ops and self.use_prox_ops['ext_unannot_l1']:
            print('Init the L1 proximal operator for the unannotated extension.')
            gamma_ext_corr = self.gamma_ext * self.watch_lr * self.corr_coeffs['gamma_ext']
            self.prox_ops['ext_unannot_l1'] = ProxL1(gamma_ext_corr)

        if 'ext_soft_mask' not in self.prox_ops and self.use_prox_ops['ext_soft_mask']:
            print('Init the soft mask proximal operator for the annotated extension.')
            ext_mask = self.model.ext_mask.to(self.device)
            alpha_l1_corr = self.alpha_l1 * self.watch_lr * self.corr_coeffs['alpha_l1']
            self.prox_ops['ext_soft_mask'] = ProxL1(alpha_l1_corr, ext_mask)

    def update_prox_ops(self):
        if 'main_group_lasso' in self.prox_ops:
            alpha_corr = self.alpha * self.watch_lr
            if self.prox_ops['main_group_lasso']._alpha != alpha_corr:
                self.prox_ops['main_group_lasso'] = ProxGroupLasso(alpha_corr, self.omega)

        if 'ext_unannot_l1' in self.prox_ops:
            gamma_ext_corr = self.gamma_ext * self.watch_lr * self.corr_coeffs['gamma_ext']
            if self.prox_ops['ext_unannot_l1']._alpha != gamma_ext_corr:
                self.prox_ops['ext_unannot_l1']._alpha = gamma_ext_corr

        for mask_key in ('main_soft_mask', 'ext_soft_mask'):
            if mask_key in self.prox_ops:
                alpha_l1_corr = self.alpha_l1 * self.watch_lr * self.corr_coeffs['alpha_l1']
                if self.prox_ops[mask_key]._alpha != alpha_l1_corr:
                    self.prox_ops[mask_key]._alpha = alpha_l1_corr

    def apply_prox_ops(self):
        if 'main_soft_mask' in self.prox_ops:
            self.prox_ops['main_soft_mask'](self.model.decoder.L0.expr_L.weight.data)
        if 'main_group_lasso' in self.prox_ops:
            self.prox_ops['main_group_lasso'](self.model.decoder.L0.expr_L.weight.data)
        if 'ext_unannot_l1' in self.prox_ops:
            self.prox_ops['ext_unannot_l1'](self.model.decoder.L0.ext_L.weight.data)
        if 'ext_soft_mask' in self.prox_ops:
            self.prox_ops['ext_soft_mask'](self.model.decoder.L0.ext_L_m.weight.data)

    def on_iteration(self, batch_data):
        self.init_prox_ops()

        super().on_iteration(batch_data)

        self.apply_prox_ops()

    def check_early_stop(self):
        continue_training = super().check_early_stop()

        if continue_training:
            new_lr = self.optimizer.param_groups[0]['lr']
            if self.watch_lr is not None and self.watch_lr != new_lr:
                self.watch_lr = new_lr
                self.update_prox_ops()

        return continue_training

    def on_epoch_end(self):
        if self.print_stats:
            if self.use_prox_ops['main_group_lasso']:
                n_deact_terms = self.model.decoder.n_inactive_terms()
                msg = f'Number of deactivated terms: {n_deact_terms}'
                if self.epoch > 0:
                    msg = '\n' + msg
                print(msg)
                print('-------------------')
            if self.use_prox_ops['main_soft_mask']:
                main_mask = self.prox_ops['main_soft_mask']._I
                share_deact_genes = (self.model.decoder.L0.expr_L.weight.data.abs()==0) & main_mask
                share_deact_genes = share_deact_genes.float().sum().cpu().numpy() / self.model.n_inact_genes
                print('Share of deactivated inactive genes: %.4f' % share_deact_genes)
                print('-------------------')
            if self.use_prox_ops['ext_soft_mask']:
                ext_mask = self.prox_ops['ext_soft_mask']._I
                share_deact_ext_genes = (self.model.decoder.L0.ext_L_m.weight.data.abs()==0) & ext_mask
                share_deact_ext_genes = share_deact_ext_genes.float().sum().cpu().numpy() / self.model.n_inact_ext_genes
                print('Share of deactivated inactive genes in extension terms: %.4f' % share_deact_ext_genes)
                print('-------------------')
            if self.use_prox_ops['ext_unannot_l1']:
                active_genes = (self.model.decoder.L0.ext_L.weight.data.abs().cpu().numpy()>0).sum(0)
                print('Active genes in unannotated extension terms:', active_genes)
                sparse_share = 1. - active_genes / self.model.input_dim
                print('Sparcity share in unannotated extension terms:', sparse_share)
                print('-------------------')

        any_change = self.anneal()
        if any_change:
            self.update_prox_ops()

        super().on_epoch_end()

    def loss(self, total_batch=None):
        recon_loss, kl_loss, hsic_loss = self.model(**total_batch)

        if self.beta is not None and self.model.use_hsic:
            weighted_hsic = self.beta * hsic_loss
            self.iter_logs["hsic_loss"].append(hsic_loss.item())
        else:
            weighted_hsic = 0.

        loss = recon_loss + self.calc_alpha_coeff()*kl_loss + weighted_hsic

        self.iter_logs["loss"].append(loss.item())
        self.iter_logs["unweighted_loss"].append((recon_loss + kl_loss + hsic_loss).item())
        self.iter_logs["recon_loss"].append(recon_loss.item())
        self.iter_logs["kl_loss"].append(kl_loss.item())

        return loss
