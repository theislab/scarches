import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anndata import AnnData
from scHPL.utils import TreeNode, create_tree

from scarches.plotting import plot_tree
from scarches.plotting._tree_data import compact_name, node_values
from scarches.plotting._tree_draw import CLADE_ALPHA, EMPTY_COLOR
from scarches.plotting._tree_layout import dfs_leaves, layout_positions


def _tiny_tree():
    tree = create_tree('root')
    immune = TreeNode(['immune'])
    tcell = TreeNode(['T', 'T cell'])
    bcell = TreeNode(['B'])
    mono = TreeNode(['Mono'])
    tree[0].add_descendant(immune)
    tree[0].add_descendant(mono)
    immune.add_descendant(tcell)
    immune.add_descendant(bcell)
    return tree


def _tiny_adata():
    labels = np.array(['T', 'T cell', 'B', 'B', 'Mono', 'Mono', 'Mono'])
    batches = np.array(['s1', 's1', 's1', 's2', 's1', 's2', 's3'])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    return AnnData(
        X=np.zeros((len(labels), 2)),
        obs={
            'cell_type': labels,
            'study': batches,
            'score': scores,
        },
    )


def _annotation_text(ax):
    texts = []
    for child in ax.get_children():
        if hasattr(child, 'get_text'):
            try:
                texts.append(child.get_text())
            except Exception:
                pass
    return texts


def test_horizontal_layout_side_by_side_labels():
    tree = _tiny_tree()
    adata = _tiny_adata()
    ax = plot_tree(
        tree, adata=adata, cell_type_key='cell_type', batch_key='study',
        layout='horizontal', show_counts=True, show=False,
    )
    texts = _annotation_text(ax)
    blob = '\n'.join(texts)
    assert 'Mono' in blob
    assert 'n=3' in blob
    # labels are separate artists, not multiline blocks
    assert not any('\n' in t for t in texts if t)
    edge_collection = next(
        c for c in ax.collections
        if isinstance(c, matplotlib.collections.LineCollection)
    )
    # Rectangular edges are direct two-point segments, not dense repeated paths.
    assert all(len(segment) == 2 for segment in edge_collection.get_segments())
    plt.close(ax.figure)


def test_circular_equal_radius_tiers():
    tree = _tiny_tree()
    # true tier radii (no leaf outer-alignment)
    pos = layout_positions(tree[0], 'circular', align_leaves=False)
    by_depth = {}
    for node in tree[0].walk():
        d = 0
        cur = node
        while cur.ancestor is not None:
            d += 1
            cur = cur.ancestor
        r = np.hypot(*pos[id(node)])
        by_depth.setdefault(d, []).append(r)
    for rs in by_depth.values():
        assert np.allclose(rs, rs[0], atol=1e-8)
    assert np.isclose(max(max(rs) for rs in by_depth.values()), 0.72)

    ax = plot_tree(
        tree,
        adata=_tiny_adata(),
        cell_type_key='cell_type',
        color='score',
        layout='circular',
        show_counts=False,
        show=False,
    )
    assert np.allclose(ax.figure.get_size_inches(), (7.0, 7.0))
    assert any(isinstance(c, matplotlib.collections.LineCollection)
               for c in ax.collections)
    plt.close(ax.figure)


def test_circular_clade_annotation_sector_and_legend():
    tree = _tiny_tree()
    immune = next(node for node in tree[0].walk() if node.name == ['immune'])
    ax = plot_tree(
        tree,
        layout='circular',
        clade_annotations={'Immune lineage': immune},
        label_color=None,
        show_counts=False,
        show=False,
    )
    sectors = [
        patch for patch in ax.patches
        if isinstance(patch, matplotlib.patches.Wedge)
        and np.isclose(patch.get_alpha(), CLADE_ALPHA)
    ]
    assert len(sectors) == 1
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        'Immune lineage'
    ]
    plt.close(ax.figure)


def test_edge_color_and_internal_labels():
    tree = _tiny_tree()
    adata = _tiny_adata()
    ax = plot_tree(
        tree, adata=adata, cell_type_key='cell_type', batch_key='study',
        color='diversity', edge_color='n', show_internal_labels=True,
        show=False,
    )
    texts = _annotation_text(ax)
    blob = '\n'.join(texts)
    assert 'immune' in blob
    assert any(isinstance(c, matplotlib.collections.LineCollection)
               for c in ax.collections)
    # Simpson diversity drives the node colorbar: Mono spans three studies,
    # the single-study B leaf has none.
    nodes = list(tree[0].walk())
    diversity = node_values(nodes, adata, 'cell_type', 'study', 'diversity')
    mono = next(n for n in nodes if n.name == ['Mono'])
    bcell = next(n for n in nodes if n.name == ['B'])
    assert diversity[id(mono)] > diversity[id(bcell)]
    plt.close(ax.figure)


def test_unknown_label_color_raises():
    with pytest.raises(ValueError):
        plot_tree(_tiny_tree(), label_color='parent', show=False)


def test_empty_nodes_are_grey():
    tree = _tiny_tree()
    # only Mono cells -> other branches empty
    adata = AnnData(
        X=np.zeros((3, 2)),
        obs={
            'cell_type': np.array(['Mono', 'Mono', 'Mono']),
            'study': np.array(['s1', 's2', 's3']),
            'score': np.array([0.1, 0.2, 0.3]),
        },
    )
    ax = plot_tree(
        tree, adata=adata, cell_type_key='cell_type',
        color='score', edge_color='score', show_counts=True, show=False,
    )
    # scatter colors include greys for empty branches
    scatters = [c for c in ax.collections if hasattr(c, 'get_facecolors')]
    facecolors = np.vstack([
        c.get_facecolors() for c in scatters if len(c.get_facecolors())
    ])
    greys = np.all(np.isclose(facecolors[:, :3], EMPTY_COLOR[:3], atol=0.05), axis=1)
    assert greys.any()
    plt.close(ax.figure)


def test_obs_color_and_pie():
    tree = _tiny_tree()
    adata = _tiny_adata()
    ax = plot_tree(
        tree, adata=adata, cell_type_key='cell_type',
        color='score', pie='study', show=False,
    )
    boxes = [c for c in ax.get_children() if c.__class__.__name__ == 'AnnotationBbox']
    wedges = [c for c in ax.get_children() if isinstance(c, matplotlib.patches.Wedge)]
    assert len(boxes) > 0 or len(wedges) > 0
    plt.close(ax.figure)


def test_categorical_node_color_gets_patch_legend_not_colorbar():
    tree = _tiny_tree()
    adata = _tiny_adata()
    # majority region per node: T -> a, B and Mono -> b
    adata.obs['region'] = np.array(['a', 'a', 'b', 'b', 'b', 'b', 'b'])
    ax = plot_tree(
        tree, adata=adata, cell_type_key='cell_type', color='region',
        show=False,
    )
    labels = {text.get_text() for text in ax.get_legend().get_texts()}
    assert {'a', 'b'} <= labels
    # a categorical channel must not add a colorbar axis
    assert len(ax.figure.axes) == 1
    plt.close(ax.figure)


def test_alias_matching_counts():
    tree = _tiny_tree()
    adata = _tiny_adata()
    nodes = list(tree[0].walk())
    counts = node_values(nodes, adata, 'cell_type', 'study', 'n')
    t_leaf = [n for n in nodes if n.name == ['T', 'T cell']][0]
    assert counts[id(t_leaf)] == 2
    immune = [n for n in nodes if n.name == ['immune']][0]
    assert counts[id(immune)] == 4


def test_missing_obs_column_raises():
    tree = _tiny_tree()
    adata = _tiny_adata()
    with pytest.raises(KeyError):
        plot_tree(tree, adata=adata, cell_type_key='cell_type',
                  color='missing', show=False)


def test_invalid_layout_raises():
    tree = _tiny_tree()
    with pytest.raises(ValueError):
        plot_tree(tree, layout='diagonal', show=False)


def test_negative_values_colorbar():
    tree = _tiny_tree()
    adata = _tiny_adata()
    adata.obs['score'] = np.array([-0.5, -0.2, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax = plot_tree(
        tree, adata=adata, cell_type_key='cell_type',
        color='score', show_counts=True, show=False,
    )
    assert ax.figure is not None
    plt.close(ax.figure)


def test_constant_continuous_values_still_have_labeled_colorbar():
    tree = _tiny_tree()
    adata = _tiny_adata()
    adata.obs['score'] = 0.5
    ax = plot_tree(
        tree,
        adata=adata,
        cell_type_key='cell_type',
        color='score',
        show=False,
    )
    assert len(ax.figure.axes) == 2
    assert ax.figure.axes[1].get_ylabel() == 'score'
    assert np.allclose(ax.figure.axes[1].get_yticks(), [0.5])
    assert np.allclose(ax.figure.get_size_inches(), (7.6, 4.2))
    plt.close(ax.figure)


def test_compact_name_strips_batch_and_collapses_aliases():
    tree = _tiny_tree()
    tcell = [n for n in tree[0].walk() if n.name == ['T', 'T cell']][0]
    assert compact_name(tcell) == 'T / T cell'
    node = TreeNode(['CD4+ T cells-Freytag', 'CD4+ T cells-Sun'])
    assert compact_name(node, batch_levels={'Freytag', 'Sun'}) == 'CD4+ T cells'
    one = TreeNode(['CD14+ Monocytes-Oetjen'])
    assert compact_name(one, batch_levels={'Oetjen'}) == 'CD14+ Monocytes'


def test_dfs_leaves_keep_subtrees_contiguous():
    tree = _tiny_tree()
    leaves = dfs_leaves(tree[0])
    names = [n.name[0] for n in leaves]
    # immune subtree (T, B) must be contiguous; Mono is the other root child
    assert names == ['T', 'B', 'Mono'] or names == ['Mono', 'T', 'B']
    pos = layout_positions(tree[0], 'horizontal', align_leaves=True)
    ys = {n.name[0]: pos[id(n)][1] for n in leaves}
    # sibling leaves under immune stay adjacent in y
    assert abs(ys['T'] - ys['B']) == 1.0


def test_order_by_n_puts_larger_sibling_first():
    tree = _tiny_tree()
    adata = _tiny_adata()
    nodes = list(tree[0].walk())
    counts = node_values(nodes, adata, 'cell_type', 'study', 'n')
    leaves = dfs_leaves(tree[0], sort_counts=counts)
    # Mono (n=3) is a larger root child than immune (n=4)? immune=4 > mono=3
    # so immune subtree should come first when ordering by n
    assert leaves[0].name[0] in ('T', 'B')
    assert leaves[-1].name[0] == 'Mono' or leaves[0].name[0] == 'Mono'


def test_mixed_leaf_pies_for_multi_study_only():
    tree = create_tree('root')
    multi = TreeNode(['CD4+ T cells-Freytag', 'CD4+ T cells-Sun'])
    single = TreeNode(['CD14+ Monocytes-Oetjen'])
    tree[0].add_descendant(multi)
    tree[0].add_descendant(single)
    adata = AnnData(
        X=np.zeros((5, 2)),
        obs={
            'cell_type': np.array([
                'CD4+ T cells-Freytag', 'CD4+ T cells-Freytag',
                'CD4+ T cells-Sun',
                'CD14+ Monocytes-Oetjen', 'CD14+ Monocytes-Oetjen',
            ]),
            'study': np.array(['Freytag', 'Freytag', 'Sun', 'Oetjen', 'Oetjen']),
        },
    )
    ax = plot_tree(
        tree, adata=adata, cell_type_key='cell_type', batch_key='study',
        layout='horizontal', show=False,
    )
    # horizontal pies are point-space AnnotationBbox markers (stay circular)
    boxes = [
        c for c in ax.get_children()
        if c.__class__.__name__ == 'AnnotationBbox'
    ]
    assert len(boxes) >= 1
    plt.close(ax.figure)


def test_study_label_color_strips_suffix_and_legend():
    # treeArches-style CellType-Study aliases
    tree = create_tree('root')
    a = TreeNode(['CD4+ T cells-Freytag'])
    b = TreeNode(['CD14+ Monocytes-Oetjen'])
    tree[0].add_descendant(a)
    tree[0].add_descendant(b)
    adata = AnnData(
        X=np.zeros((4, 2)),
        obs={
            'cell_type': np.array([
                'CD4+ T cells-Freytag', 'CD4+ T cells-Freytag',
                'CD14+ Monocytes-Oetjen', 'CD14+ Monocytes-Oetjen',
            ]),
            'study': np.array(['Freytag', 'Freytag', 'Oetjen', 'Oetjen']),
        },
    )
    ax = plot_tree(
        tree, adata=adata, cell_type_key='cell_type', batch_key='study',
        layout='horizontal', show=False,
    )
    texts = _annotation_text(ax)
    blob = '\n'.join(texts)
    assert 'CD4+ T cells' in blob
    assert 'CD4+ T cells-Freytag' not in blob
    assert 'CD14+ Monocytes-Oetjen' not in blob
    legend = ax.get_legend()
    assert legend is not None
    assert all(text.get_text() for text in legend.get_texts())
    plt.close(ax.figure)
