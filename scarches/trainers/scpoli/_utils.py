import numpy as np
import torch
from ...dataset import MultiConditionAnnotatedDataset


def make_dataset(adata,
                 train_frac=0.9,
                 condition_keys=None,
                 cell_type_keys=None,
                 condition_encoders=None,
                 cell_type_encoder=None,
                 labeled_indices=None,
                 ):
    """Splits 'adata' into train and validation data and converts them into 'CustomDatasetFromAdata' objects.

       Parameters
       ----------

       Returns
       -------
       Training 'CustomDatasetFromAdata' object, Validation 'CustomDatasetFromAdata' object
    """
    # Preprare data for semisupervised learning
    labeled_array = np.zeros((len(adata), 1))
    if labeled_indices is not None:
        labeled_array[labeled_indices] = 1

    if cell_type_keys is not None:
        finest_level = None
        n_cts = 0
        for cell_type_key in cell_type_keys:
            if len(adata.obs[cell_type_key].unique().tolist()) >= n_cts:
                n_cts = len(adata.obs[cell_type_key].unique().tolist())
                finest_level = cell_type_key

        train_idx, val_idx = train_test_split(adata, train_frac, cell_type_key=finest_level,
                                              labeled_array=labeled_array)

    elif condition_key is not None:
        train_idx, val_idx = train_test_split(adata, train_frac, condition_key=condition_key)
    else:
        train_idx, val_idx = train_test_split(adata, train_frac)

    data_set_train = MultiConditionAnnotatedDataset(
        adata if train_frac == 1 else adata[train_idx],
        condition_keys=condition_keys,
        cell_type_keys=cell_type_keys,
        condition_encoders=condition_encoders,
        cell_type_encoder=cell_type_encoder,
        labeled_array=labeled_array[train_idx]
    )
    if train_frac == 1:
        return data_set_train, None
    else:
        data_set_valid = trVAEDataset(
            adata[val_idx],
            condition_keys=condition_keys,
            cell_type_keys=cell_type_keys,
            condition_encoders=condition_encoders,
            cell_type_encoder=cell_type_encoder,
            labeled_array=labeled_array[val_idx]
        )
        return data_set_train, data_set_valid

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


def t_dist(x, y, alpha):
    """student t-distribution, as same as used in t-SNE algorithm.
             q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
    Arguments:
        inputs: the variable containing data, shape=(n_samples, n_features)
    Return:
        q: student's t-distribution with degree alpha, or soft labels for each sample. shape=(n_samples, n_clusters)
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    distances = torch.pow(x - y, 2).sum(2) / alpha

    q = 1.0 / (1.0 + distances)
    q = torch.pow(q, (alpha + 1.0) / 2.0)
    q = (q.T / q.sum(1)).T
    return q


def target_distribution(q):
    weight = torch.pow(q, 2) / q.sum(0)
    return (weight.T / weight.sum(1)).T


def kl_loss(p, q):
    return (p * torch.log(p / q)).sum(1).mean()


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    d = x.size(1)
    m = y.size(0)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
