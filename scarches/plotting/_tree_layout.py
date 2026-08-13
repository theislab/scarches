"""Node positions for the hierarchy plot.

Pure geometry: sibling ordering, coordinates and axis limits. Keeping this
separate from drawing makes the non-crossing property of the layout testable.
"""
import numpy as np

from ._tree_data import is_leaf, node_label

# Compact extents: the tree occupies little canvas so labels get the rest.
CIRCULAR_R_MIN = 0.12
CIRCULAR_R_MAX = 0.72
DEPTH_SCALE = {'horizontal': 0.10, 'circular': 1.0}


def depth_of(node):
    depth = 0
    current = node
    while current.ancestor is not None:
        depth += 1
        current = current.ancestor
    return depth


def ordered_children(node, sort_counts=None, sort_labels=False, sort_ranks=None):
    """Sibling order used for layout only; the tree itself is never mutated."""
    children = list(node.descendants)
    if not children:
        return children

    if sort_counts is not None:
        def secondary(child):
            return -float(sort_counts.get(id(child), 0)), node_label(child)
    elif sort_labels:
        def secondary(child):
            return node_label(child).casefold()
    else:
        def secondary(child):
            return children.index(child)

    if sort_ranks is not None:
        return sorted(
            children,
            key=lambda c: (sort_ranks.get(id(c), float('inf')), secondary(c)),
        )
    if sort_counts is not None or sort_labels:
        return sorted(children, key=secondary)
    return children


def dfs_leaves(node, sort_counts=None, sort_labels=False, sort_ranks=None):
    """Leaf order that keeps each subtree's leaves contiguous (non-crossing)."""
    if is_leaf(node):
        return [node]
    out = []
    for child in ordered_children(node, sort_counts, sort_labels, sort_ranks):
        out.extend(dfs_leaves(child, sort_counts, sort_labels, sort_ranks))
    return out


def leaf_offset(layout, max_depth):
    """x of the shared leaf column, or None outside horizontal layouts."""
    if layout != 'horizontal':
        return None
    return DEPTH_SCALE[layout] * float(max_depth)


def tier_radius(depth, max_depth):
    return CIRCULAR_R_MIN + (CIRCULAR_R_MAX - CIRCULAR_R_MIN) * (
        depth / float(max_depth)
    )


def layout_positions(
    root, layout, align_leaves=True, sort_counts=None, sort_labels=False,
    sort_ranks=None, circular_gap=40.0,
):
    """Map node id to ``(x, y)`` for a horizontal or circular cladogram."""
    leaves = dfs_leaves(root, sort_counts, sort_labels, sort_ranks)
    n_leaves = max(len(leaves), 1)
    leaf_index = {id(n): i for i, n in enumerate(leaves)}
    span = {}

    def assign_span(node):
        if is_leaf(node):
            index = leaf_index[id(node)]
            span[id(node)] = (index, index + 1)
            return span[id(node)]
        starts, ends = zip(*(
            assign_span(child)
            for child in ordered_children(
                node, sort_counts, sort_labels, sort_ranks,
            )
        ))
        span[id(node)] = (min(starts), max(ends))
        return span[id(node)]

    assign_span(root)
    max_depth = max(depth_of(n) for n in root.walk()) or 1
    depth_scale = DEPTH_SCALE[layout]
    pos = {}

    if layout == 'horizontal':
        for node in root.walk():
            start, end = span[id(node)]
            depth = float(depth_of(node))
            aligned = align_leaves and is_leaf(node)
            pos[id(node)] = (
                depth_scale * (float(max_depth) if aligned else depth),
                0.5 * (start + end - 1),
            )
        return pos

    gap = np.deg2rad(np.clip(circular_gap, 0.0, 180.0))
    span_angle = 2.0 * np.pi - gap
    for node in root.walk():
        start, end = span[id(node)]
        theta = 0.5 * gap + span_angle * ((start + end) * 0.5) / n_leaves
        depth = float(depth_of(node))
        if depth == 0:
            radius = 0.0
        elif align_leaves and is_leaf(node):
            radius = CIRCULAR_R_MAX
        else:
            radius = tier_radius(depth, max_depth)
        pos[id(node)] = (radius * np.cos(theta), radius * np.sin(theta))
    return pos


def set_limits(ax, nodes, pos, layout, has_counts):
    """Reserve room for label text outside the drawn tree."""
    all_x = [pos[id(n)][0] for n in nodes]
    all_y = [pos[id(n)][1] for n in nodes]
    if layout == 'circular':
        margin = 0.48 if has_counts else 0.42
        extent = max(
            np.max(np.abs(all_x)), np.max(np.abs(all_y)), 0.1,
        ) + margin
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        return
    # Leaf names are always drawn; counts need one more text column.
    ax.set_xlim(
        min(all_x) - 0.10 * max(DEPTH_SCALE[layout], 0.1),
        max(all_x) + (2.05 if has_counts else 1.55),
    )
    # Layout order reads top-to-bottom, like a table.
    ax.set_ylim(max(all_y) + 0.55, min(all_y) - 0.55)
