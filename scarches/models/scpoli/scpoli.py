import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from ._utils import one_hot_encoder
from ..trvae.losses import mse, nb, zinb, bce, poisson, nb_dist

class scpoli(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layer_sizes,
        cell_types,
        unknown_ct_names,
        conditions,
        conditions_combined,
        inject_condition,
        latent_dim,
        embedding_dims,
        embedding_max_norm,
        recon_loss,
        dr_rate,
        beta,
        use_bn,
        use_ln,
        prototypes_labeled,
        prototypes_unlabeled,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.embedding_dims = embedding_dims
        self.embedding_max_norm = embedding_max_norm
        self.cell_types = cell_types
        self.n_cell_types = len(cell_types)
        self.cell_type_encoder = {
            k: v for k, v in zip(cell_types, range(len(cell_types)))
        }
        self.n_conditions = [len(conditions[cond]) for cond in conditions.keys()]
        self.n_reference_conditions = None
        self.conditions = conditions
        self.condition_encoders = {cond: {
            k: v for k, v in zip(conditions[cond], range(len(conditions[cond])))
        } for cond in conditions.keys()}
        self.conditions_combined = conditions_combined
        self.n_conditions_combined = len(conditions_combined)
        self.conditions_combined_encoder = {
            k: v for k, v in zip(conditions_combined, range(len(conditions_combined)))
        }
        self.inject_condition = inject_condition
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.use_mmd = False
        self.recon_loss = recon_loss
        self.hidden_layer_sizes = hidden_layer_sizes
        self.freeze = False
        self.unknown_ct_names = unknown_ct_names
        if self.unknown_ct_names is not None:
            for unknown_ct in self.unknown_ct_names:
                self.cell_type_encoder[unknown_ct] = -1
        self.prototypes_labeled = (
            {"mean": None}
            if prototypes_labeled is None
            else prototypes_labeled
        )
        self.prototypes_unlabeled = (
            {"mean": None} if prototypes_unlabeled is None else prototypes_unlabeled
        )
        self.new_prototypes = None
        self.num_reference_conditions = None
        if self.prototypes_labeled["mean"] is not None:
            # Save indices of possible new prototypes to train
            self.new_prototypes = []
            for idx in range(self.n_cell_types - len(self.prototypes_labeled["mean"])):
                self.new_prototypes.append(len(self.prototypes_labeled["mean"]) + idx)

        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        if recon_loss in ["nb", "zinb", "nb_dist"]:
            self.theta = torch.nn.Parameter(
                torch.randn(self.input_dim, self.n_conditions_combined)
            )
        else:
            self.theta = None

        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.input_dim)

        self.embeddings = nn.ModuleList(nn.Embedding(
            self.n_conditions[i], self.embedding_dims[i], max_norm=self.embedding_max_norm
        ) for i in range(len(self.embedding_dims)))

        print(
            "Embedding dictionary:\n",
            f"\tNum conditions: {self.n_conditions}\n",
            f"\tEmbedding dim: {self.embedding_dims}",
        )
        self.encoder = Encoder(
            encoder_layer_sizes,
            self.latent_dim,
            self.use_bn,
            self.use_ln,
            self.use_dr,
            self.dr_rate,
            sum(self.embedding_dims) if "encoder" in self.inject_condition else None,
        )
        self.decoder = Decoder(
            decoder_layer_sizes,
            self.latent_dim,
            self.recon_loss,
            self.use_bn,
            self.use_ln,
            self.use_dr,
            self.dr_rate,
            sum(self.embedding_dims) if "decoder" in self.inject_condition else None,
        )

    def forward(
        self,
        x=None,
        batch=None,
        combined_batch=None,
        sizefactor=None,
        celltypes=None,
        labeled=None,
    ):   
        batch_embeddings = torch.hstack([self.embeddings[i](batch[:, i]) for i in range(batch.shape[1])])
        x_log = torch.log(1 + x)
        if self.recon_loss == "mse":
            x_log = x
        if "encoder" in self.inject_condition:
            z1_mean, z1_log_var = self.encoder(x_log, batch_embeddings)
        else:
            z1_mean, z1_log_var = self.encoder(x_log, batch=None)
        z1 = self.sampling(z1_mean, z1_log_var)

        if "decoder" in self.inject_condition:
            outputs = self.decoder(z1, batch_embeddings)
        else:
            outputs = self.decoder(z1, batch=None)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x_log).sum(dim=-1).mean()
        elif self.recon_loss == "zinb":
            dec_mean_gamma, dec_dropout, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(one_hot_encoder(combined_batch, self.n_conditions_combined), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = (
                -zinb(x=x, mu=dec_mean, theta=dispersion, pi=dec_dropout)
                .sum(dim=-1)
                .mean()
            )
        elif self.recon_loss == "nb":
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(one_hot_encoder(combined_batch, self.n_conditions_combined), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = -nb(x=x, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()
        elif self.recon_loss == "nb_dist":
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(
                dec_mean_gamma.size(0), dec_mean_gamma.size(1)
            )
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = nb_dist(x=x, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()
        elif self.recon_loss == 'bernoulli':
            recon_x, y1 = outputs
            recon_loss = bce(recon_x, x).sum(dim=-1).mean()
        elif self.recon_loss == 'poisson':
            recon_x, y1 = outputs
            recon_loss = poisson(recon_x, x).sum(dim=-1).mean()

        z1_var = torch.exp(z1_log_var) + 1e-4
        kl_div = (
            kl_divergence(
                Normal(z1_mean, torch.sqrt(z1_var)),
                Normal(torch.zeros_like(z1_mean), torch.ones_like(z1_var)),
            )
            .sum(dim=1)
            .mean()
        )

        mmd_loss = torch.tensor(0.0, device=z1.device)
        if self.use_mmd:
            mmd_calculator = mmd(self.n_conditions, self.beta, self.mmd_boundary)
            if self.mmd_on == "z":
                mmd_loss = mmd_calculator(z1, batch)
            else:
                mmd_loss = mmd_calculator(y1, batch)

        return z1, recon_loss, kl_div, mmd_loss

    def add_new_cell_type(self, latent, cell_type_name, prototypes, classes_list=None):
        """
        Function used to add new annotation for a novel cell type.

        Parameters
        ----------
        latent: torch.Tensor
            Latent representation of adata.
        cell_type_name: str
            Name of the new cell type
        prototypes: list
            List of indices of the unlabeled prototypes that correspond to the new cell type
        classes_list: torch.Tensor
            Tensor of prototype indices corresponding to current hierarchy

        Returns
        -------
        """
        # Update internal model parameters
        device = next(self.parameters()).device
        self.cell_types.append(cell_type_name)
        self.n_cell_types += 1
        self.cell_type_encoder = {
            k: v for k, v in zip(self.cell_types, range(len(self.cell_types)))
        }

        # Add new celltype index to hierarchy index list of prototypes
        classes_list = torch.cat(
            (
                classes_list,
                torch.tensor([self.n_cell_types - 1], device=classes_list.device),
            )
        )

        # Add new prototype mean to labeled prototype means
        new_prototype = (
            self.prototypes_unlabeled["mean"][prototypes].mean(0).unsqueeze(0)
        )
        self.prototypes_labeled["mean"] = torch.cat(
            (self.prototypes_labeled["mean"], new_prototype), dim=0
        )

        # Get latent indices which correspond to new prototype
        self.prototypes_labeled["mean"] = self.prototypes_labeled["mean"].to(device)
        latent = latent.to(device)
        dists = torch.cdist(latent, self.prototypes_labeled["mean"][classes_list, :])
        min_dist, y_hat = torch.min(dists, 1)
        y_hat = classes_list[y_hat]
        indices = y_hat.eq(self.n_cell_types - 1).nonzero(as_tuple=False)[:, 0]

    def classify(
        self,
        x,
        c=None,
        prototype=False,
        classes_list=None,
        p=2,
        get_prob=False,
        log_distance=True,
    ):
        """
        Classifies unlabeled cells using the prototypes obtained during training.
        Data handling before call to model's classify method.

        x: torch.Tensor
            Features to be classified.
        c: torch.Tensor
            Condition vector.
        prototype: Boolean
            Boolean whether to classify the gene features or prototypes stored
            stored in the model.
        classes_list: torch.Tensor
            Tensor of prototype indices corresponding to current hierarchy
        get_prob: Str
            Method to use for scaling euclidean distances to pseudo-probabilities
        """
        if prototype:
            latent = x
        else:
            latent = self.get_latent(x, c)
        device = next(self.parameters()).device
        self.prototypes_labeled["mean"] = self.prototypes_labeled["mean"].to(device)
        dists = torch.cdist(latent, self.prototypes_labeled["mean"][classes_list, :], p)

        # Idea of using euclidean distances for classification
        if get_prob == True:
            dists = F.softmax(-dists, dim=1)
            uncert, preds = torch.max(dists, dim=1)
            preds = classes_list[preds]
        else:
            uncert, preds = torch.min(dists, dim=1)
            preds = classes_list[preds]
            if log_distance == True:
                probs = torch.log1p(uncert)

        return preds, uncert, dists

    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
        It is actually sampling from latent space distributions with N(mu, var), computed by encoder.
        Parameters
        ----------
        mu: torch.Tensor
                Torch Tensor of Means.
        log_var: torch.Tensor
                Torch Tensor of log. variances.
        Returns
        -------
        Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()

    def get_latent(self, x, c=None, mean=False):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
        data.
        Parameters
        ----------
        x:  torch.Tensor
             Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
        c: torch.Tensor
             Torch Tensor of condition labels for each sample.
        mean: boolean
        Returns
        -------
        Returns Torch Tensor containing latent space encoding of 'x'.
        """
        x_ = torch.log(1 + x)
        if self.recon_loss == "mse":
            x_ = x
        if "encoder" in self.inject_condition:
            # c = c.type(torch.cuda.LongTensor)
            c = c.long()
            embed_c = torch.hstack([self.embeddings[i](c[:, i]) for i in range(c.shape[1])])
            z_mean, z_log_var = self.encoder(x_, embed_c)
        else:
            z_mean, z_log_var = self.encoder(x_)
        latent = self.sampling(z_mean, z_log_var)
        if mean:
            return z_mean
        return latent


class Encoder(nn.Module):
    """ScArches Encoder class. Constructs the encoder sub-network of TRVAE and CVAE. It will transform primary space
    input to means and log. variances of latent space with n_dimensions = z_dimension.
    Parameters
    ----------
    layer_sizes: List
         List of first and hidden layer sizes
    latent_dim: Integer
         Bottleneck layer (z)  size.
    use_bn: Boolean
         If `True` batch normalization will be applied to layers.
    use_ln: Boolean
         If `True` layer normalization will be applied to layers.
    use_dr: Boolean
         If `True` dropout will applied to layers.
    dr_rate: Float
         Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
    num_classes: Integer
         Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
         conditional VAE.
    """

    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int,
        use_bn: bool,
        use_ln: bool,
        use_dr: bool,
        dr_rate: float,
        embedding_dim: int = None,
    ):
        super().__init__()

        self.embedding_dim = 0
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        self.FC = None

        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if i == 0:
                    print(
                        "\tInput Layer in, out and cond:",
                        in_size,
                        out_size,
                        self.embedding_dim,
                    )
                    (
                        self.FC.add_module(
                            name="L{:d}".format(i),
                            module=CondLayers(
                                in_size, out_size, self.embedding_dim, bias=True
                            ),
                        )
                    )

                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    (
                        self.FC.add_module(
                            name="L{:d}".format(i),
                            module=nn.Linear(in_size, out_size, bias=True),
                        )
                    )
                if use_bn:
                    (
                        self.FC.add_module(
                            "N{:d}".format(i),
                            module=nn.BatchNorm1d(out_size, affine=True),
                        )
                    )
                elif use_ln:
                    (
                        self.FC.add_module(
                            "N{:d}".format(i),
                            module=nn.LayerNorm(out_size, elementwise_affine=False),
                        )
                    )
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if use_dr:
                    self.FC.add_module(
                        name="D{:d}".format(i), module=nn.Dropout(p=dr_rate)
                    )

        print("\tMean/Var Layer in/out:", layer_sizes[-1], latent_dim)
        self.mean_encoder = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], latent_dim)

    def forward(self, x, batch=None):
        if batch is not None:
            #    batch = one_hot_encoder(batch, n_cls=self.n_classes)
            x = torch.cat((x, batch), dim=-1)
        if self.FC is not None:
            x = self.FC(x)
        means = self.mean_encoder(x)
        log_vars = self.log_var_encoder(x)
        return means, log_vars


class Decoder(nn.Module):
    """ScArches Decoder class. Constructs the decoder sub-network of TRVAE or CVAE networks. It will transform the
    constructed latent space to the previous space of data with n_dimensions = x_dimension.
    Parameters
    ----------
    layer_sizes: List
         List of hidden and last layer sizes
    latent_dim: Integer
         Bottleneck layer (z)  size.
    recon_loss: String
         Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
    use_bn: Boolean
         If `True` batch normalization will be applied to layers.
    use_ln: Boolean
         If `True` layer normalization will be applied to layers.
    use_dr: Boolean
         If `True` dropout will applied to layers.
    dr_rate: Float
         Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
    num_classes: Integer
         Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
         conditional VAE.
    """

    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int,
        recon_loss: str,
        use_bn: bool,
        use_ln: bool,
        use_dr: bool,
        dr_rate: float,
        embedding_dim: int = None,
    ):
        super().__init__()
        self.use_dr = use_dr
        self.recon_loss = recon_loss
        self.embedding_dim = 0
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        layer_sizes = [latent_dim] + layer_sizes
        print("Decoder Architecture:")
        # Create first Decoder layer
        self.FirstL = nn.Sequential()
        print(
            "\tFirst Layer in, out and cond: ",
            layer_sizes[0],
            layer_sizes[1],
            self.embedding_dim,
        )

        (
            self.FirstL.add_module(
                name="L0",
                module=CondLayers(
                    layer_sizes[0], layer_sizes[1], self.embedding_dim, bias=False
                ),
            )
        )
        if use_bn:
            (
                self.FirstL.add_module(
                    "N0", module=nn.BatchNorm1d(layer_sizes[1], affine=True)
                )
            )
        elif use_ln:
            (
                self.FirstL.add_module(
                    "N0", module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False)
                )
            )
        (self.FirstL.add_module(name="A0", module=nn.ReLU()))
        if self.use_dr:
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dr_rate))

        # Create all Decoder hidden layers
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(
                zip(layer_sizes[1:-1], layer_sizes[2:])
            ):
                if i + 3 < len(layer_sizes):
                    print("\tHidden Layer", i + 1, "in/out:", in_size, out_size)
                    (
                        self.HiddenL.add_module(
                            name="L{:d}".format(i + 1),
                            module=nn.Linear(in_size, out_size, bias=False),
                        )
                    )
                    if use_bn:
                        (
                            self.HiddenL.add_module(
                                "N{:d}".format(i + 1),
                                module=nn.BatchNorm1d(out_size, affine=True),
                            )
                        )
                    elif use_ln:
                        (
                            self.HiddenL.add_module(
                                "N{:d}".format(i + 1),
                                module=nn.LayerNorm(out_size, elementwise_affine=False),
                            )
                        )
                    (
                        self.HiddenL.add_module(
                            name="A{:d}".format(i + 1), module=nn.ReLU()
                        )
                    )
                    if self.use_dr:
                        (
                            self.HiddenL.add_module(
                                name="D{:d}".format(i + 1), module=nn.Dropout(p=dr_rate)
                            )
                        )
        else:
            self.HiddenL = None

        # Create Output Layers
        print("\tOutput Layer in/out: ", layer_sizes[-2], layer_sizes[-1], "\n")
        if self.recon_loss == "mse":
            self.recon_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.ReLU()
            )
        elif self.recon_loss == "zinb":
            # mean gamma
            self.mean_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1)
            )
            # dropout
            self.dropout_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        elif self.recon_loss in ["nb", "nb_dist"]:
            # mean gamma
            self.mean_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1)
            )
        elif self.recon_loss == 'bernoulli':
            self.recon_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Sigmoid()
            )
        elif self.recon_loss == 'poisson':
            self.recon_decoder = nn.Sequential(
                nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1)
            )

    def forward(self, z, batch=None):
        # Add Condition Labels to Decoder Input
        if batch is not None:
            # batch = one_hot_encoder(batch, n_cls=self.n_classes)
            z_cat = torch.cat((z, batch), dim=-1)
            dec_latent = self.FirstL(z_cat)
        else:
            dec_latent = self.FirstL(z)

        # Compute Hidden Output
        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent

        # Compute Decoder Output
        if self.recon_loss == "mse":
            recon_x = self.recon_decoder(x)
            return recon_x, dec_latent
        elif self.recon_loss == "zinb":
            dec_mean_gamma = self.mean_decoder(x)
            dec_dropout = self.dropout_decoder(x)
            return dec_mean_gamma, dec_dropout, dec_latent
        elif self.recon_loss in ["nb", "nb_dist"]:
            dec_mean_gamma = self.mean_decoder(x)
            return dec_mean_gamma, dec_latent
        elif self.recon_loss == 'bernoulli':
            recon_x = self.recon_decoder(x)
        elif self.recon_loss == 'poisson':
            recon_x = self.recon_decoder(x)
            return recon_x, dec_latent
       


class CondLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cond: int,
        bias: bool,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        if self.n_cond == 0:
            out = self.expr_L(x)
        else:
            expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)
            out = self.expr_L(expr) + self.cond_L(cond)
        return out
