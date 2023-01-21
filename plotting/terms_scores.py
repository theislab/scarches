from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

def plot_abs_bfs_key(scores, terms, key, n_points=30, lim_val=2.3, fontsize=8, scale_y=2, yt_step=0.3,
                     title=None, ax=None):
    txt_args = dict(
        rotation='vertical',
        verticalalignment='bottom',
        horizontalalignment='center',
        fontsize=fontsize,
    )

    ax = ax if ax is not None else plt.axes()
    ax.grid(False)

    bfs = np.abs(scores[key]['bf'])
    srt = np.argsort(bfs)[::-1][:n_points]
    top = bfs.max()

    ax.set_ylim(top=top * scale_y)
    yt = np.arange(0, top * 1.1, yt_step)
    ax.set_yticks(yt)

    ax.set_xlim(0.1, n_points + 0.9)
    xt = np.arange(0, n_points + 1, 5)
    xt[0] = 1
    ax.set_xticks(xt)

    for i, (bf, term) in enumerate(zip(bfs[srt], terms[srt])):
        ax.text(i+1, bf, term, **txt_args)

    ax.axhline(y=lim_val, color='red', linestyle='--', label='')

    ax.set_xlabel("Rank")
    ax.set_ylabel("Absolute log bayes factors")
    ax.set_title(key if title is None else title)

    return ax.figure

def plot_abs_bfs(adata, scores_key="bf_scores", terms: Union[str, list]="terms",
                 keys=None, n_cols=3, **kwargs):
    """\
    Plot the absolute bayes scores rankings.
    """
    scores = adata.uns[scores_key]

    if isinstance(terms, str):
        terms = np.asarray(adata.uns[terms])
    else:
        terms = np.asarray(terms)

    if len(terms) != len(next(iter(scores.values()))["bf"]):
        raise ValueError('Incorrect length of terms.')

    if keys is None:
        keys = list(scores.keys())

    if len(keys) == 1:
        keys = keys[0]

    if isinstance(keys, str):
        return plot_abs_bfs_key(scores, terms, keys, **kwargs)

    n_keys = len(keys)

    if n_keys <= n_cols:
        n_cols = n_keys
        n_rows = 1
    else:
        n_rows = int(np.ceil(n_keys / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols)
    for key, ix in zip(keys, product(range(n_rows), range(n_cols))):
        if n_rows == 1:
            ix = ix[1]
        elif n_cols == 1:
            ix = ix[0]
        plot_abs_bfs_key(scores, terms, key, ax=axs[ix], **kwargs)

    n_inactive = n_rows * n_cols - n_keys
    if n_inactive > 0:
        for i in range(n_inactive):
            axs[n_rows-1, -(i+1)].axis('off')

    return fig
