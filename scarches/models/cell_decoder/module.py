from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, one_hot
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi import _CONSTANTS
from torch.distributions import Normal, Poisson
import torch.nn.functional as F
import torch

class CellDecoderVAE(BaseModuleClass):
    def __init__(
        self,
        n_obs: int,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dispersion: str = "gene",
        gene_likelihood: str = "zinb",
        deeply_inject_covariates: bool = True,
        use_batch_norm_decoder: bool = True,
        use_layer_norm_decoder: bool = False
    ):

        super().__init__()
        self.dispersion = dispersion
        self.gene_likelihood = gene_likelihood

        self.latent_distribution = "normal"

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        self.z_m = torch.nn.Parameter(torch.randn(n_obs, n_latent))
        self.z_log_v = torch.nn.Parameter(torch.randn(n_obs, n_latent))

        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder
        )

    def _get_inference_input(self, tensors):
        return dict(ind_x=tensors["ind_x"])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = torch.log(tensors[_CONSTANTS.X_KEY].sum(1)).unsqueeze(1)
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        input_dict = {
            "z": z,
            "library": library,
            "batch_index": batch_index
        }
        return input_dict

    @auto_move_data
    def inference(self, ind_x):
        z_m_s = self.z_m[ind_x]
        z_v_s = torch.exp(self.z_log_v[ind_x])

        z = Normal(z_m_s, z_v_s.sqrt()).rsample()

        return dict(z=z, qz_m=z_m_s, qz_v=z_v_s)

    @auto_move_data
    def generative(self, z, library, batch_index):
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, z, library, batch_index
        )

        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[_CONSTANTS.X_KEY]

        z_m_s = inference_outputs["qz_m"]
        z_v_s = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mean = torch.zeros_like(z_m_s)
        scale = torch.ones_like(z_v_s)

        kl_divergence = kl(Normal(z_m_s, z_v_s.sqrt()), Normal(mean, scale)).sum(dim=1)
        weighted_kl_local = kl_weight * kl_divergence

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        loss = torch.mean(reconst_loss + weighted_kl_local)

        return LossRecorder(loss, reconst_loss, kl_divergence, torch.tensor(0.0))

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout) -> torch.Tensor:
        if self.gene_likelihood == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss
