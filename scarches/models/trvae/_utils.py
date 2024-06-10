import torch
import numpy as np
import logging
import seaborn as sns
import pandas as pd
from matplotlib import pyplot
from scipy import stats

logger = logging.getLogger(__name__)


def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def partition(data, partitions, num_partitions):
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        indices = torch.nonzero((partitions == i), as_tuple=False).squeeze(1)
        res += [data[indices]]
    return res


def reg_mean_plot(
    adata,
    condition_key,
    target_condition,
    labels,
    path_to_save="./reg_mean.pdf",
    save=False,
    gene_list=None,
    show=False,
    top_100_genes=None,
    verbose=False,
    legend=True,
    title=None,
    x_coeff=0.30,
    y_coeff=0.8,
    fontsize=14,
    **kwargs,
):
    """
    Plots mean matching figure for a set of specific genes.

    Parameters
    ----------
    adata: `~anndata.AnnData`
        AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
        AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
        corresponding to batch and cell type metadata, respectively.
    axis_keys: dict
        Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
            `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
    labels: dict
        Dictionary of axes labels of the form `{"x": "x-axis-name", "y": "y-axis name"}`.
    path_to_save: basestring
        path to save the plot.
    save: boolean
        Specify if the plot should be saved or not.
    gene_list: list
        list of gene names to be plotted.
    show: bool
        if `True`: will show to the plot after saving it.

    """

    sns.set_theme()
    sns.set_theme(color_codes=True)

    axis_keys = {"x":"other", "y":target_condition}

    diff_genes = top_100_genes
    target_cd = adata[adata.obs[condition_key] == target_condition]
    other_cd = adata[adata.obs[condition_key] != target_condition]
    if diff_genes is not None:
        if hasattr(diff_genes, "tolist"):
            diff_genes = diff_genes.tolist()
        adata_diff = adata[:, diff_genes]
        target_diff = adata_diff[adata_diff.obs[condition_key] == target_condition]
        other_diff = adata_diff[adata_diff.obs[condition_key] != target_condition]
        x_diff = np.asarray(np.mean(target_diff.X, axis=0)).ravel()
        y_diff = np.asarray(np.mean(other_diff.X, axis=0)).ravel()
        m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
            x_diff, y_diff
        )
        if verbose:
            print("top_100 DEGs mean: ", r_value_diff**2)
    x = np.asarray(np.mean(other_cd.X, axis=0)).ravel()
    y = np.asarray(np.mean(target_cd.X, axis=0)).ravel()
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    if verbose: 
        print("All genes mean: ", r_value**2)
    df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
    ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
    ax.tick_params(labelsize=fontsize)
    if "range" in kwargs:
        start, stop, step = kwargs.get("range")
        ax.set_xticks(np.arange(start, stop, step))
        ax.set_yticks(np.arange(start, stop, step))
    ax.set_xlabel(labels["x"], fontsize=fontsize)
    ax.set_ylabel(labels["y"], fontsize=fontsize)
    if gene_list is not None:
        texts = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="black"))
            pyplot.plot(x_bar, y_bar, "o", color="red", markersize=5)

    if legend:
        pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if title is None:
        pyplot.title("", fontsize=fontsize)
    else:
        pyplot.title(title, fontsize=fontsize)
    ax.text(
        max(x) - max(x) * x_coeff,
        max(y) - y_coeff * max(y),
        r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
        fontsize=kwargs.get("textsize", fontsize),
    )
    if diff_genes is not None:
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - (y_coeff + 0.15) * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
            + f"{r_value_diff ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
    if save:
        pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
    if show:
        pyplot.show()
    pyplot.close()
    if diff_genes is not None:
        return r_value**2, r_value_diff**2
    else:
        return r_value**2
