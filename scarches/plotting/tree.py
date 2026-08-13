import numpy as np
import matplotlib.pyplot as plt

from . import _tree_data as data
from . import _tree_draw as draw
from . import _tree_layout as geometry


def plot_tree(
    tree,
    adata=None,
    cell_type_key=None,
    batch_key=None,
    color=None,
    edge_color=None,
    label_color='auto',
    pie=None,
    layout='horizontal',
    order_by='label',
    show_counts=True,
    show_internal_labels=False,
    clade_annotations=None,
    clade_colors=None,
    circular_gap=40.0,
    node_size=48.0,
    linewidth=2.6,
    fontsize=12,
    cmap=None,
    figsize=None,
    dpi=200,
    title=None,
    show_legend=True,
    ax=None,
    show=True,
):
    """Plot an scHPL / treeArches cell-type hierarchy.

    Parameters
    ----------
    tree:
        Tree returned by ``scHPL.learn_tree`` (list with root at ``tree[0]``)
        or a root ``TreeNode``.
    adata: :class:`~anndata.AnnData`, optional
        Observations used for counts, diversity, and obs-based coloring.
    cell_type_key: str, optional
        ``adata.obs`` column with labels matching tree node names / aliases.
    batch_key: str, optional
        ``adata.obs`` column identifying studies. Used for sample diversity,
        for stripping ``-study`` suffixes from labels, and (by default) for
        coloring leaf names by majority study (ARBOL-style).
    color: str, optional
        Node color: ``'n'``, ``'diversity'``, or an ``adata.obs`` column.
    edge_color: str, optional
        Same options as ``color``. Edges stay neutral by default so topology
        remains the primary visual channel.
    label_color: ``{'auto', 'study', None}``
        Color leaf names by majority study (``auto`` does so when
        ``batch_key`` is set) or draw them black. Study suffixes are stripped
        from the text whenever study is encoded in color.
    pie: str, optional
        Categorical ``adata.obs`` column for composition pies. Circular
        layouts pie every node, horizontal layouts pie the leaves. When unset
        but ``batch_key`` is available, only multi-study leaves get pies.
    layout: ``{'horizontal', 'circular'}``
        Sideways cladogram, or circular rose with equal-radius tiers.
    order_by: ``{'label', 'n', None}``
        Sibling order for layout only; the tree is never mutated. Alphabetical
        labels make repeated cell types easy to compare, ``'n'`` places larger
        subtrees first.
    show_counts: bool
        Draw cell counts beside leaves.
    show_internal_labels: bool
        Also name internal branch points, above their branch and only where
        the text fits (``horizontal`` layout).
    clade_annotations: mapping, optional
        Circular-layout lineage highlights. Keys are legend labels, values are
        tree nodes, node names, or lists of either. Each selected branch is
        enclosed by a translucent annular sector, and selected branches are
        grouped together by legend order.
    clade_colors: mapping, optional
        Colors keyed by ``clade_annotations`` labels.
    circular_gap: float
        Empty angle in degrees on the right of a circular tree, which keeps
        labels clear of the legend. Set to zero for a full circle.
    node_size: float
        Marker size for leaves; internal branch points stay small.
    linewidth: float
        Base edge width; count scaling is logarithmic.
    fontsize: float
        Leaf-label font size.
    cmap:
        Matplotlib colormap for continuous channels. Categorical channels use
        a fixed high-contrast palette.
    figsize: tuple, optional
    dpi: int
        Figure DPI, which drives crispness in Jupyter.
    title: str, optional
    show_legend: bool
    ax: matplotlib axis, optional
    show: bool

    Returns
    -------
    matplotlib axis
    """
    if layout not in ('horizontal', 'circular'):
        raise ValueError("layout must be 'horizontal' or 'circular'")
    if clade_annotations and layout != 'circular':
        raise ValueError(
            "clade_annotations are currently supported only for layout='circular'"
        )

    root = tree[0] if isinstance(tree, list) else tree
    nodes = list(root.walk())
    batch_levels = data.batch_levels_of(nodes, adata, batch_key)
    if label_color == 'auto':
        label_color = 'study' if batch_key is not None else None
    if label_color not in ('study', None):
        raise ValueError("label_color must be 'auto', 'study', or None")

    masks = None
    counts = None
    if adata is not None and cell_type_key is not None:
        if cell_type_key not in adata.obs:
            raise KeyError(cell_type_key)
        masks = data.node_masks(
            root, np.asarray(adata.obs[cell_type_key].values, dtype=str),
        )
        # Counts first, so layout can order siblings by size (as in ARBOL).
        counts = data.node_values(
            nodes, adata, cell_type_key, batch_key, 'n', masks=masks,
        )

    clades = None
    if clade_annotations:
        clades = {
            label: draw.resolve_clade_nodes(root, selectors)
            for label, selectors in clade_annotations.items()
        }

    pos = geometry.layout_positions(
        root, layout,
        sort_counts=counts if order_by == 'n' else None,
        sort_labels=order_by == 'label',
        sort_ranks=_clade_ranks(clades) if clades else None,
        circular_gap=circular_gap,
    )
    max_depth = max(geometry.depth_of(node) for node in nodes) or 1

    pie_key, pie_scope = _pie_channel(
        pie, layout, adata, cell_type_key, batch_key,
    )
    values = data.channel_values(
        nodes, adata, cell_type_key, batch_key, color, masks=masks,
    )
    edge_values = data.channel_values(
        nodes, adata, cell_type_key, batch_key, edge_color, masks=masks,
    )
    pie_data = (
        data.pie_fractions(nodes, adata, cell_type_key, pie_key, masks=masks)
        if pie_key is not None else None
    )
    encodes_study = label_color == 'study' or (
        batch_key is not None and edge_color == batch_key
    )
    studies = (
        data.study_values(
            nodes, adata, cell_type_key, batch_key, batch_levels, masks=masks,
        )
        if encodes_study else None
    )
    # Pies already own the node markers, so continuous node values move to edges.
    if pie_key is not None and edge_values is None and values is not None:
        if data.is_continuous(values):
            edge_values = values

    ax = _prepare_axis(
        ax, layout, figsize, dpi,
        n_leaves=len(geometry.dfs_leaves(root)),
    )
    cmap = plt.get_cmap(cmap or 'cividis')
    empty = data.empty_mask(nodes, counts, values if color is not None else None)

    clade_handles = []
    if clades:
        clade_handles = draw.draw_clade_annotations(
            ax, root, pos, clades, clade_colors=clade_colors,
            circular_gap=circular_gap,
        )
    if layout == 'circular':
        draw.draw_tier_rings(ax, max_depth)

    draw.draw_edges(
        ax, nodes, pos, edge_values, cmap, empty, counts,
        lw=linewidth, layout=layout,
    )
    study_handles = draw.draw_nodes(
        ax, nodes, pos,
        values=values,
        counts=counts,
        pie_data=pie_data,
        pie_scope=pie_scope,
        empty=empty,
        cmap=cmap,
        studies=studies,
        color_labels_by_study=label_color == 'study',
        node_size=node_size,
        show_counts=show_counts and counts is not None,
        show_internal_labels=show_internal_labels,
        suppress_count_text=color == 'n',
        batch_levels=batch_levels,
        fontsize=fontsize if layout == 'horizontal' else fontsize * 0.85,
        layout=layout,
        leaf_x=geometry.leaf_offset(layout, max_depth),
    )

    if show_legend:
        _add_legend(
            ax, layout,
            handles=clade_handles + study_handles,
            values=values,
            edge_values=edge_values,
            pie_data=pie_data,
            cmap=cmap,
            color=color,
            edge_color=edge_color,
            title=_legend_title(
                encodes_study, batch_key, pie_key, bool(clade_handles),
            ),
        )

    if title:
        # A circular tree fills its axis, so its title belongs to the figure.
        if layout == 'circular':
            ax.figure.suptitle(title, fontsize=fontsize + 1)
        else:
            ax.set_title(title, fontsize=fontsize + 1, pad=4)
    ax.set_aspect('equal' if layout == 'circular' else 'auto')
    ax.axis('off')
    geometry.set_limits(ax, nodes, pos, layout, show_counts)

    if show:
        plt.show()
    return ax


def _prepare_axis(ax, layout, figsize, dpi, n_leaves):
    if ax is not None:
        if dpi is not None:
            ax.figure.set_dpi(dpi)
        return ax
    if figsize is None:
        figsize = (
            (7.0, 7.0) if layout == 'circular'
            else (7.6, max(4.2, 0.29 * n_leaves))
        )
    figure, ax = plt.subplots(figsize=figsize, dpi=dpi, layout='constrained')
    figure.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return ax


def _pie_channel(pie, layout, adata, cell_type_key, batch_key):
    """Resolve the pie column and which nodes receive a pie."""
    if pie is not None:
        return pie, 'all' if layout == 'circular' else 'leaves'
    auto_study_pies = (
        adata is not None and cell_type_key is not None
        and batch_key is not None and batch_key in adata.obs
    )
    if auto_study_pies:
        return batch_key, 'mixed'
    return None, None


def _clade_ranks(clades):
    """Legend order per annotated node and its ancestors, for grouped layout."""
    ranks = {}
    for rank, selected in enumerate(clades.values()):
        for node in selected:
            current = node
            while current is not None:
                ranks[id(current)] = min(rank, ranks.get(id(current), rank))
                current = current.ancestor
    return ranks


def _legend_title(encodes_study, batch_key, pie_key, has_clades):
    if encodes_study:
        return draw.legend_label(batch_key or 'study')
    if pie_key is not None:
        return draw.legend_label(pie_key)
    if has_clades:
        return 'lineage'
    return None


def _add_legend(
    ax, layout, handles, values, edge_values, pie_data, cmap, color, edge_color,
    title,
):
    """One colorbar for the continuous channel, one legend for the rest."""
    handles = list(handles)
    if pie_data is not None:
        levels = next(iter(pie_data.values()))[0]
        handles.extend(draw.category_patches(levels))
    if (
        edge_values is not None and not data.is_continuous(edge_values)
        and not handles
    ):
        handles.extend(draw.category_lines(draw.category_levels(edge_values)))

    node_values_continuous = values is not None and data.is_continuous(values)
    if node_values_continuous:
        draw.add_colorbar(ax, values, cmap, color)
    else:
        if values is not None:
            handles.extend(draw.category_patches(draw.category_levels(values)))
        if edge_values is not None and data.is_continuous(edge_values):
            draw.add_colorbar(ax, edge_values, cmap, edge_color)

    draw.add_legend(
        ax, handles, title=title,
        # Continuous node colors already occupy the right margin.
        below=layout == 'horizontal' and node_values_continuous,
    )
