import torch
import numpy as np
from functools import partial
from torch.autograd import Variable
import torch.nn.functional as F

from ._utils import partition


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


def nb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    """Computes negative binomial loss.

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


def zinb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps=1e-8):
    """Computes zero inflated negative binomial loss.

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

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, sigmas):
    """Computes multiscale-RBF kernel between x and y.

       Parameters
       ----------
       x: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       y: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       sigmas: Tensor

       Returns
       -------
       Returns the computed multiscale-RBF kernel between x and y.
    """

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    """Computes Maximum Mean Discrepancy(MMD) between x and y.

       Parameters
       ----------
       x: torch.Tensor
            Tensor with shape [batch_size, z_dim]
       y: torch.Tensor
            Tensor with shape [batch_size, z_dim]
       kernel: gaussian_kernel_matrix

       Returns
       -------
       Returns the computed MMD between x and y.
    """

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost


def mmd_loss_calc(source_features, target_features):
    """Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.

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

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    if torch.cuda.is_available():
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas))
        )
    else:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas=Variable(torch.FloatTensor(sigmas))
        )
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel=gaussian_kernel)

    return loss_value


def mmd(n_conditions, beta, boundary):
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
    def mmd_loss(y, c):
        # partition separates y into num_cls subsets w.r.t. their labels c
        conditions_mmd = partition(y, c, n_conditions)
        loss = torch.tensor(0.0, device=y.device)
        if boundary is not None:
            for i in range(boundary):
                for j in range(boundary, n_conditions):
                    if conditions_mmd[i].size(0) < 2 or conditions_mmd[j].size(0) < 2:
                        continue
                    loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])
        else:
            for i in range(len(conditions_mmd)):
                if conditions_mmd[i].size(0) < 1:
                    continue
                for j in range(i):
                    if conditions_mmd[j].size(0) < 1 or i == j:
                        continue
                    loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])
        return beta * loss

    return mmd_loss
