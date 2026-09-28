from scvi.distributions import NegativeBinomial
import torch
from torch.autograd import Variable
from torch.distributions import Poisson
import torch.nn.functional as F

from ._utils import partition


MMD_ALPHAS = [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
    1e3, 1e4, 1e5, 1e6
]


def bce(recon_x, x):
    """Computes BCE loss between reconstructed data and ground truth data.

       Parameters
       ----------
       recon_x: torch.Tensor
            Torch Tensor of reconstructed data
       x: torch.Tensor
            Torch Tensor of ground truth data

       Returns
       -------
       MSE loss value
    """
    bce_loss = torch.nn.functional.binary_cross_entropy(recon_x, (x > 0).float(), reduction='none')
    return bce_loss

def mse(recon_x, x):
    """Computes MSE loss between reconstructed data and ground truth data.

       Parameters
       ----------
       recon_x: torch.Tensor
            Torch Tensor of reconstructed data
       x: torch.Tensor
            Torch Tensor of ground truth data

       Returns
       -------
       MSE loss value
    """
    mse_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='none')
    return mse_loss

def poisson(recon_x, x):
    """Computes Poisson NLL between reconstructed data and ground truth data.

       Parameters
       ----------
       recon_x: torch.Tensor
            Torch Tensor of reconstructed data
       x: torch.Tensor
            Torch Tensor of ground truth data

       Returns
       -------
       MSE loss value
    """
    #poisson_nll = torch.nn.functional.poisson_nll_loss(recon_x, x, reduction='none')
    poisson_loss = -Poisson(recon_x).log_prob(x)
    return poisson_loss


def nb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    """
       This negative binomial function was taken from:
       Title: scvi-tools
       Authors: Romain Lopez <romain_lopez@gmail.com>,
                Adam Gayoso <adamgayoso@berkeley.edu>,
                Galen Xing <gx2113@columbia.edu>
       Date: 16th November 2020
       Code version: 0.8.1
       Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

       Computes negative binomial loss.
       Parameters
       ----------
       x: torch.Tensor
            Torch Tensor of ground truth data.
       mu: torch.Tensor
            Torch Tensor of means of the negative binomial (has to be positive support).
       theta: torch.Tensor
            Torch Tensor of inverse dispersion parameter (has to be positive support).
       eps: Float
            numerical stability constant.

       Returns
       -------
       If 'mean' is 'True' NB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return res

def nb_dist(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    loss = -NegativeBinomial(mu=mu, theta=theta).log_prob(x)
    return loss


def zinb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps=1e-8):
    """
       This zero-inflated negative binomial function was taken from:
       Title: scvi-tools
       Authors: Romain Lopez <romain_lopez@gmail.com>,
                Adam Gayoso <adamgayoso@berkeley.edu>,
                Galen Xing <gx2113@columbia.edu>
       Date: 16th November 2020
       Code version: 0.8.1
       Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

       Computes zero inflated negative binomial loss.
       Parameters
       ----------
       x: torch.Tensor
            Torch Tensor of ground truth data.
       mu: torch.Tensor
            Torch Tensor of means of the negative binomial (has to be positive support).
       theta: torch.Tensor
            Torch Tensor of inverses dispersion parameter (has to be positive support).
       pi: torch.Tensor
            Torch Tensor of logits of the dropout parameter (real support)
       eps: Float
            numerical stability constant.

       Returns
       -------
       If 'mean' is 'True' ZINB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return res


def pairwise_distance(x, y):
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.

       Parameters
       ----------
       x: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       y: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       alphas: Tensor

       Returns
       -------
       Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1. / (2. * alphas)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss_calc(source_features, target_features):
    """Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.

       - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.

       Parameters
       ----------
       source_features: torch.Tensor
            Tensor with shape [batch_size, z_dim]
       target_features: torch.Tensor
            Tensor with shape [batch_size, z_dim]

       Returns
       -------
       Returns the computed MMD between x and y.
    """
    alphas = Variable(torch.FloatTensor(MMD_ALPHAS)).to(device=source_features.device)

    cost = torch.mean(gaussian_kernel_matrix(source_features, source_features, alphas))
    cost += torch.mean(gaussian_kernel_matrix(target_features, target_features, alphas))
    cost -= 2 * torch.mean(gaussian_kernel_matrix(source_features, target_features, alphas))

    return cost


def mmd(y,c,n_conditions, beta, boundary):
    """Initializes Maximum Mean Discrepancy(MMD) between every different condition.

       Parameters
       ----------
       n_conditions: integer
            Number of classes (conditions) the data contain.
       beta: float
            beta coefficient for MMD loss.
       boundary: integer
            If not 'None', mmd loss is only calculated on #new conditions.
       y: torch.Tensor
            Torch Tensor of computed latent data.
       c: torch.Tensor
            Torch Tensor of condition labels.

       Returns
       -------
       Returns MMD loss.
    """

    labels = c.flatten()
    # As in partition(), only labels in [0, n_conditions) take part.
    keep = (labels >= 0) & (labels < n_conditions)
    if boundary is not None:
        # The boundary route skips conditions with fewer than two cells.
        _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
        keep &= counts[inverse] >= 2
    y, labels = y[keep], labels[keep]

    groups, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    if counts.numel() < 2:
        return beta * torch.tensor(0.0, device=y.device)

    alphas = torch.tensor(MMD_ALPHAS, dtype=torch.float32, device=y.device)
    kernel = gaussian_kernel_matrix(y, y, alphas)
    membership = F.one_hot(inverse, num_classes=counts.numel()).to(kernel.dtype)
    membership = membership / counts.to(kernel.dtype)
    group_kernel = membership.T @ kernel @ membership

    # Block means include self-kernels, as in mmd_loss_calc.
    within = group_kernel.diagonal()
    pair_losses = within[:, None] + within[None, :] - 2 * group_kernel
    if boundary is None:
        # Every unordered pair of populated conditions.
        pair_mask = groups[:, None] > groups[None, :]
    else:
        # Only pairs of an old (< boundary) and a new (>= boundary) condition.
        pair_mask = (groups[:, None] < boundary) & (groups[None, :] >= boundary)
    return beta * pair_losses[pair_mask].sum()
