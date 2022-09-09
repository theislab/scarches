import warnings 
import os
import csv
import numpy as np
from sklearn.metrics import *
from sklearn.covariance import GraphicalLassoCV, graphical_lasso, LedoitWolf
from sklearn.preprocessing import StandardScaler
import torch
try:
    import torch_geometric.nn as pyg_nn
except:
    warnings.warn('In order to use sagenet models, please install pytorch geometric (see https://pytorch-geometric.readthedocs.io).')
    # warnings.warn('Reommended: .')
from sklearn.utils.extmath import fast_logdet
import numpy as np
from scipy import sparse



def glasso(adata, alphas=5, n_jobs=None, mode='cd'):
    """
        Recustructs the gene-gene interaction network based on gene expressions in `.X` using a guassian graphical model estimated by `glasso`. 

        Parameters
        ----------
        adata: `AnnData` 
            The annotated data matrix of shape `n_obs × n_vars`. Rows correspond to cells and columns to genes.
        alphas: int or array-like of shape (n_alphas,), dtype=`float`, default=`5`
            Non-negative. If an integer is given, it fixes the number of points on the grids of alpha to be used. If a list is given, it gives the grid to be used. 
        n_jobs: int, default `None`
            Non-negative. number of jobs.

        Returns
        -------
        adds an `csr_matrix` matrix under key `adj` to `.varm`.

        References
        -----------
        Friedman, J., Hastie, T., & Tibshirani, R. (2008). 
        Sparse inverse covariance estimation with the graphical lasso. 
        Biostatistics, 9(3), 432-441.
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(adata.X)
    cov = GraphicalLassoCV(alphas=alphas, n_jobs=n_jobs).fit(data)
    precision_matrix = cov.get_precision()
    adjacency_matrix = precision_matrix.astype(bool).astype(int)
    adjacency_matrix[np.diag_indices_from(adjacency_matrix)] = 0
    save_adata(adata, attr='varm', key='adj', data=sparse.csr_matrix(adjacency_matrix))


def compute_metrics(y_true, y_pred):
    """
        Computes prediction quality metrics.

        Parameters
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) labels.

        y_pred : 1d array-like, or label indicator array / sparse matrix
            Predicted labels, as returned by a classifier.

        Returns
        --------
        accuracy : accuracy
        conf_mat : confusion matrix
        precision : weighted precision score
        recall : weighted recall score
        f1 : weighted f1 score
    """
    accuracy  = accuracy_score(y_true, y_pred)
    conf_mat  = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')
    f1        = f1_score(y_true, y_pred, average='weighted')
    return accuracy, conf_mat, precision, recall, f1




def get_dataloader(graph, X, y, batch_size=1, undirected=True, shuffle=True, num_workers=0):
    """
        Converts a graph and a dataset to a dataloader.
        
        Parameters
        ----------
        graph : igraph object
            The underlying graph to be fed to the graph neural networks.

        X : numpy ndarray
            Input dataset with columns as features and rows as observations.

        y : numpy ndarray
            Class labels.

        batch_size: int, default=1
            The batch size.

        undirected: boolean
            if the input graph is undirected (symmetric adjacency matrix).

        shuffle: boolean, default = `True`
            Wheather to shuffle the dataset to be passed to `torch_geometric.data.DataLoader`.

        num_workers: int, default = 0
            Non-negative. Number of workers to be passed to `torch_geometric.data.DataLoader`.


        Returns
        --------
        dataloader : a pytorch-geometric dataloader. All of the graphs will have the same connectivity (given by the input graph),
        but the node features will be the features from X.
    """
    n_obs, n_features = X.shape
    rows, cols = np.where(graph == 1)
    edges      = zip(rows.tolist(), cols.tolist())
    sources    = []
    targets    = []
    for edge in edges:
        sources.append(edge[0])
        targets.append(edge[1])
        if undirected:
            sources.append(edge[0])
            targets.append(edge[1])
    edge_index  = torch.tensor([sources,targets],dtype=torch.long)

    list_graphs = []
    y = y.tolist()
    # print(y)
    for i in range(n_obs):
        y_tensor = torch.tensor(y[i])
        X_tensor = torch.tensor(X[i,:]).view(X.shape[1], 1).float()
        data     = geo_dt.Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
        list_graphs.append(data.coalesce())

    dataloader = geo_dt.DataLoader(list_graphs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    return dataloader



def kullback_leibler_divergence(X):

    """Finds the pairwise Kullback-Leibler divergence
        matrix between all rows in X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Array of probability data. Each row must sum to 1.

        Returns
        -------
        D : ndarray, shape (n_samples, n_samples)
            The Kullback-Leibler divergence matrix. A pairwise matrix D such that D_{i, j}
            is the divergence between the ith and jth vectors of the given matrix X.

        Notes
        -----
        Based on code from Gordon J. Berman et al.
        (https://github.com/gordonberman/MotionMapper)

        References
        -----------
        Berman, G. J., Choi, D. M., Bialek, W., & Shaevitz, J. W. (2014). 
        Mapping the stereotyped behaviour of freely moving fruit flies. 
        Journal of The Royal Society Interface, 11(99), 20140672.
    """

    X_log = np.log(X)
    X_log[np.isinf(X_log) | np.isnan(X_log)] = 0

    entropies = -np.sum(X * X_log, axis=1)

    D = np.matmul(-X, X_log.T)
    D = D - entropies
    D = D / np.log(2)
    D *= (1 - np.eye(D.shape[0]))

    return D

def multinomial_rvs(n, p):
    """Sample from the multinomial distribution with multiple p vectors.

        Parameters
        ----------
        n : int
            must be a scalar >=1
        p : numpy ndarray 
            must an n-dimensional 
            he last axis of p holds the sequence of probabilities for a multinomial distribution.
        
        Returns
        -------
        D : ndarray
            same shape as p
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out

def save_adata(adata, attr, key, data):
    """updates an attribute of an `AnnData` object

        Parameters
        ----------
        adata : `AnnData` 
            The annotated data matrix of shape `n_obs × n_vars`. Rows correspond to cells and columns to genes.
        attr : str
            must be an attribute of `adata`, e.g., `obs`, `var`, etc.
        key : str
            must be a key in the attr
        data : non-specific
            the data to be updated/placed

    """
    obj = getattr(adata, attr)
    obj[key] = data
