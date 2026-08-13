"""Matplotlib artists for the hierarchy plot.

Each function takes precomputed values and positions and adds one family of
artists to an axis: edges, node markers with their label fields, composition
pies, clade highlights, and the legend or colourbar.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge, Circle, Patch
from matplotlib.offsetbox import AnnotationBbox, DrawingArea

from ._tree_data import (
    alias_list, compact_name, continuous_limits, finite, is_continuous, is_leaf,
    parse_study_suffix,
)
from ._tree_layout import CIRCULAR_R_MIN, dfs_leaves, tier_radius

EMPTY_COLOR = (0.55, 0.55, 0.55, 1.0)
NEUTRAL_EDGE = (0.25, 0.25, 0.25, 1.0)
NEUTRAL_NODE = (0.18, 0.18, 0.18, 1.0)
LABEL_COLOR = (0.05, 0.05, 0.05, 1.0)
CLADE_ALPHA = 0.16

# cat palette colorblind friendly ^._.^
CAT_COLORS = [
    '#009E73', '#D55E00', '#0072B2', '#CC79A7',
    '#E69F00', '#56B4E9', '#000000', '#F0E442',
]

PIE_RADIUS = {'horizontal': 0.09, 'circular': 0.075}
_TEXT_STROKE = [patheffects.withStroke(linewidth=2.8, foreground='white')]
_METRIC_LABELS = {
    'n': 'Cell count (log1p)',
    'diversity': 'Sample diversity (Simpson)',
    'scHPL_prob': 'Mean scHPL prediction score',
}


def cat_color(index):
    return to_rgba(CAT_COLORS[index % len(CAT_COLORS)])


def category_levels(values):
    return sorted({str(v) for v in values.values() if finite(v)})


def category_patches(levels):
    return [
        Patch(facecolor=cat_color(i), edgecolor='white', label=str(level))
        for i, level in enumerate(levels)
    ]


def category_lines(levels, lw=3.0):
    return [
        Line2D([0], [0], color=cat_color(i), lw=lw, label=str(level))
        for i, level in enumerate(levels)
    ]


def legend_label(key):
    if key is None:
        return ''
    return _METRIC_LABELS.get(str(key), str(key).replace('_', ' '))


def colorize(values, cmap, empty=None):
    """Map node values to colours, greying out nodes without observations."""
    if values is None:
        return None
    empty = empty or {}
    if is_continuous(values):
        limits = continuous_limits(values)
        if limits is None:
            return {k: EMPTY_COLOR for k in values}
        norm = Normalize(vmin=limits[0], vmax=limits[1])
        return {
            k: EMPTY_COLOR if (empty.get(k, False) or not finite(v))
            else cmap(norm(float(v)))
            for k, v in values.items()
        }

    lut = {level: cat_color(i) for i, level in enumerate(category_levels(values))}
    return {
        k: EMPTY_COLOR if (empty.get(k, False) or not finite(v)) else lut[str(v)]
        for k, v in values.items()
    }


def _edge_width(count, nmax, lw):
    """Restrained log scale so small branches stay visible."""
    if count is None or count <= 0:
        return lw * 0.7
    return lw * (0.7 + 0.65 * np.log1p(count) / np.log1p(max(nmax, 1)))


def _bezier_diagonal_polar(p0, p1, n=60):
    """Cubic curve in polar space: radius eases while the angle sweeps."""
    r0, t0 = np.hypot(*p0), np.arctan2(p0[1], p0[0])
    r1, t1 = np.hypot(*p1), np.arctan2(p1[1], p1[0])
    # Unwrap the angle so a branch never sweeps the long way round the circle.
    t1 = t0 + ((t1 - t0 + np.pi) % (2.0 * np.pi) - np.pi)
    r_mid = 0.5 * (r0 + r1)

    t = np.linspace(0.0, 1.0, n)
    w0, w1, w2, w3 = (
        (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3,
    )
    r = w0 * r0 + (w1 + w2) * r_mid + w3 * r1
    theta = (w0 + w1) * t0 + (w2 + w3) * t1
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def draw_edges(ax, nodes, pos, values, cmap, empty, counts, lw, layout):
    colors = colorize(values, cmap, empty)
    nmax = max(max(counts.values()), 1) if counts else 1
    segments, seg_colors, widths = [], [], []

    def append(segment, node_id, count):
        segments.append(segment)
        if colors is not None:
            seg_colors.append(colors[node_id])
        else:
            seg_colors.append(
                EMPTY_COLOR if empty.get(node_id, False) else NEUTRAL_EDGE
            )
        widths.append(_edge_width(count, nmax, lw))

    if layout == 'horizontal':
        for parent in nodes:
            children = list(parent.descendants)
            if not children:
                continue
            px = pos[id(parent)][0]
            child_y = [pos[id(child)][1] for child in children]
            if len(children) > 1:
                # Rectangular riser spanning the children of this node.
                segments.append(np.asarray([
                    (px, min(child_y)), (px, max(child_y)),
                ]))
                seg_colors.append(
                    colors.get(id(parent), (0.35, 0.35, 0.35, 1.0))
                    if colors is not None else (0.35, 0.35, 0.35, 1.0)
                )
                widths.append(_edge_width(
                    counts.get(id(parent)) if counts else None, nmax, lw,
                ))
            for child in children:
                cx, cy = pos[id(child)]
                append(
                    np.asarray([(px, cy), (cx, cy)]), id(child),
                    counts.get(id(child)) if counts else None,
                )
    else:
        for node in nodes:
            if node.ancestor is None:
                continue
            p0 = pos[id(node.ancestor)]
            p1 = pos[id(node)]
            if np.hypot(*p0) < 1e-8:
                # Fan root edges out of a small hub instead of a single point.
                angle = np.arctan2(p1[1], p1[0])
                hub = 0.5 * CIRCULAR_R_MIN - 0.005
                p0 = (hub * np.cos(angle), hub * np.sin(angle))
            append(
                _bezier_diagonal_polar(p0, p1), id(node),
                counts.get(id(node)) if counts else None,
            )

    ax.add_collection(LineCollection(
        segments, colors=seg_colors, linewidths=widths,
        capstyle='round', joinstyle='round', zorder=1,
    ))


def draw_tier_rings(ax, max_depth):
    for depth in range(1, max_depth + 1):
        ax.add_patch(Circle(
            (0.0, 0.0), tier_radius(depth, max_depth), fill=False,
            linestyle=':', linewidth=0.7, edgecolor=(0.65, 0.65, 0.65, 1.0),
            zorder=0,
        ))


def resolve_clade_nodes(root, selectors):
    """Accept nodes, node names, or lists of either, and return tree nodes."""
    if hasattr(selectors, 'descendants') or isinstance(selectors, str):
        selectors = [selectors]

    nodes = list(root.walk())
    known = {id(node) for node in nodes}
    resolved = []
    for selector in selectors:
        if hasattr(selector, 'descendants'):
            if id(selector) not in known:
                raise ValueError('clade annotation node is not part of tree')
            resolved.append(selector)
            continue
        matches = [
            node for node in nodes
            if str(selector) in alias_list(node)
            or str(selector) == compact_name(node, strip_study=False)
        ]
        if not matches:
            raise KeyError(f'clade annotation node not found: {selector!r}')
        resolved.append(matches[0])
    return resolved


def draw_clade_annotations(ax, root, pos, clades, clade_colors=None, circular_gap=40.0):
    """Shade selected circular clades, like ggtree's ``geom_hilight``."""
    leaves = dfs_leaves(root)
    gap = np.deg2rad(np.clip(circular_gap, 0.0, 180.0))
    half_step = 0.48 * (2.0 * np.pi - gap) / max(len(leaves), 1)
    outer = max(np.hypot(*pos[id(leaf)]) for leaf in leaves) + 0.06
    colors = clade_colors or {}
    handles = []

    for i, (label, selected) in enumerate(clades.items()):
        color = colors.get(label, cat_color(i))
        for node in selected:
            angles = np.mod([
                np.arctan2(pos[id(leaf)][1], pos[id(leaf)][0])
                for leaf in dfs_leaves(node)
            ], 2.0 * np.pi)
            ancestor = node.ancestor
            inner = (
                max(np.hypot(*pos[id(ancestor)]) - 0.025, 0.06)
                if ancestor is not None else 0.06
            )
            ax.add_patch(Wedge(
                (0.0, 0.0), outer,
                np.rad2deg(max(float(np.min(angles) - half_step), 0.0)),
                np.rad2deg(min(float(np.max(angles) + half_step), 2.0 * np.pi)),
                width=outer - inner, facecolor=color, edgecolor=color,
                linewidth=0.8, alpha=CLADE_ALPHA, zorder=-2,
            ))
        handles.append(Patch(
            facecolor=to_rgba(color, CLADE_ALPHA), edgecolor=color,
            linewidth=0.8, label=str(label),
        ))
    return handles


def _draw_pie(ax, x, y, fracs, radius, size_pts=None):
    """Draw a composition pie at a node.

    Horizontal layouts have unequal data aspect, so data-coordinate wedges
    would look elliptical; ``size_pts`` switches to a point-space marker that
    stays circular on screen.
    """
    _levels, fractions = fracs
    in_points = size_pts is not None
    if in_points:
        area = DrawingArea(size_pts, size_pts, 0, 0)
        center = (0.5 * size_pts, 0.5 * size_pts)
        radius = 0.5 * size_pts - 0.6
    else:
        center = (x, y)

    start = 0.0
    for index, frac in enumerate(fractions):
        if frac <= 0:
            continue
        end = start + 360.0 * float(frac)
        wedge = Wedge(
            center, radius, start, end, facecolor=cat_color(index),
            edgecolor='white', linewidth=0.5, zorder=3,
        )
        if in_points:
            area.add_artist(wedge)
        else:
            ax.add_patch(wedge)
        start = end

    if in_points:
        ax.add_artist(AnnotationBbox(
            area, (x, y), frameon=False, pad=0.0, zorder=3,
        ))


def _pie_category_count(fracs, min_frac=0.02):
    return int(np.sum(np.asarray(fracs[1], dtype=float) > min_frac))


def _wants_pie(node, pie_data, pie_scope, leaf, empty):
    if pie_data is None or empty:
        return False
    fracs = pie_data[id(node)]
    if not np.any(fracs[1] > 0):
        return False
    if pie_scope == 'all':
        return True
    if pie_scope == 'mixed':
        return leaf and _pie_category_count(fracs) >= 2
    return leaf


def _draw_fields_horizontal(ax, x0, y, fields, fontsize, count_column=1.45):
    """Name on the left, count right-aligned in a fixed column."""
    for text, color, weight in fields:
        if text.startswith('n='):
            ax.text(
                x0 + count_column, y, text, fontsize=fontsize * 0.95, color=color,
                fontweight=weight, ha='right', va='center', zorder=5,
                clip_on=False,
            )
        else:
            ax.text(
                x0, y, text, fontsize=fontsize, color=color, fontweight=weight,
                ha='left', va='center', zorder=5, clip_on=False,
            )


def _fits_internal_label(node, y, placed, min_gap=0.45):
    """Label real branch points, and only where the text will not stack."""
    return len(node.descendants) > 1 and all(
        abs(y - other) > min_gap for other in placed
    )


def _draw_internal_label(ax, x, y, text, fontsize):
    """Internal names ride above their branch point, clear of the leaf column."""
    ax.text(
        x + 0.015, y - 0.26, text, fontsize=fontsize * 0.78,
        color=(0.32, 0.32, 0.32, 1.0), ha='left', va='center', zorder=5,
        clip_on=False, path_effects=_TEXT_STROKE,
    )


def _draw_fields_circular(ax, x, y, fields, fontsize):
    """Radial labels kept upright, with a white halo for legibility."""
    r = np.hypot(x, y)
    ux, uy = (1.0, 0.0) if r < 1e-8 else (x / r, y / r)
    tx, ty = -uy, ux
    base_x, base_y = x + ux * 0.12, y + uy * 0.12
    angle = np.degrees(np.arctan2(y, x))
    if angle > 90 or angle < -90:
        angle -= 180.0
        ha, direction = 'right', -1.0
    else:
        ha, direction = 'left', 1.0

    cursor = 0.0
    for text, color, weight in fields:
        ax.text(
            base_x + tx * cursor * direction, base_y + ty * cursor * direction,
            text, fontsize=fontsize, color=color, fontweight=weight,
            ha=ha, va='center', zorder=5, clip_on=False, rotation=angle,
            rotation_mode='anchor', path_effects=_TEXT_STROKE,
        )
        cursor += 0.030 * max(len(text), 1) + 0.018


def draw_nodes(
    ax, nodes, pos, *, values, counts, pie_data, pie_scope, empty, cmap,
    studies, color_labels_by_study, node_size, show_counts,
    show_internal_labels, suppress_count_text, batch_levels, fontsize, layout,
    leaf_x,
):
    """Draw node markers and their label fields; return study legend handles."""
    colors = colorize(values, cmap, empty)
    study_levels = category_levels(studies) if studies is not None else []
    study_lut = {level: cat_color(i) for i, level in enumerate(study_levels)}
    pie_radius = PIE_RADIUS[layout]
    text_x0 = (leaf_x if leaf_x is not None else 0.0) + 0.10

    xs, ys, sizes, facecolors = [], [], [], []
    internal_label_ys = []
    for node in nodes:
        nid = id(node)
        x, y = pos[nid]
        leaf = is_leaf(node)
        blank = empty.get(nid, False)

        if _wants_pie(node, pie_data, pie_scope, leaf, blank):
            _draw_pie(
                ax, x, y, pie_data[nid],
                pie_radius if leaf else pie_radius * 0.75,
                size_pts=(15.0 if leaf else 11.0) if layout == 'horizontal' else None,
            )
        else:
            xs.append(x)
            ys.append(y)
            # ARBOL: tiny internal branching points, emphasised leaves.
            sizes.append(node_size if leaf else max(node_size * 0.16, 6.0))
            if blank:
                facecolors.append(EMPTY_COLOR)
            elif colors is not None:
                facecolors.append(colors[nid])
            elif leaf and studies is not None and finite(studies.get(nid, np.nan)):
                facecolors.append(study_lut[str(studies[nid])])
            else:
                facecolors.append(NEUTRAL_NODE)

        if not (leaf or show_internal_labels):
            continue

        fields = []
        name = compact_name(
            node, batch_levels=batch_levels, strip_study=bool(batch_levels),
        )
        if name:
            fields.append((
                name,
                _label_color(node, nid, leaf, studies, study_lut, batch_levels,
                            color_labels_by_study, pie_data),
                'medium',
            ))
        if (
            leaf and show_counts and counts is not None
            and not (suppress_count_text and layout == 'circular')
        ):
            fields.append((f'n={counts[nid]:,}', (0.0, 0.0, 0.0, 1.0), 'semibold'))

        if not fields:
            continue
        if layout == 'horizontal' and leaf:
            _draw_fields_horizontal(ax, text_x0, y, fields, fontsize)
        elif layout == 'horizontal':
            if _fits_internal_label(node, y, internal_label_ys):
                _draw_internal_label(ax, x, y, fields[0][0], fontsize)
                internal_label_ys.append(y)
        elif leaf:
            _draw_fields_circular(ax, x, y, fields, fontsize)

    if xs:
        ax.scatter(
            xs, ys, s=sizes, c=facecolors, zorder=4, edgecolors='none',
            linewidths=0,
        )
    if color_labels_by_study and study_levels:
        return category_lines(study_levels, lw=3.5)
    return []


def _label_color(
    node, nid, leaf, studies, study_lut, batch_levels, color_labels_by_study,
    pie_data,
):
    """Colour a leaf name by its study, unless a pie already shows the mix."""
    if not (color_labels_by_study and leaf and studies is not None):
        return LABEL_COLOR
    if pie_data is not None:
        n_categories = _pie_category_count(pie_data[nid])
    elif batch_levels:
        n_categories = len({
            parse_study_suffix(alias, batch_levels) for alias in alias_list(node)
        } - {None})
    else:
        n_categories = 0
    if n_categories > 1:
        return LABEL_COLOR
    study = studies.get(nid, np.nan)
    return study_lut.get(str(study), LABEL_COLOR) if finite(study) else LABEL_COLOR


def add_colorbar(ax, values, cmap, key):
    limits = continuous_limits(values)
    if limits is None:
        return
    mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=limits[0], vmax=limits[1]), cmap=cmap,
    )
    mappable.set_array([])
    label = legend_label(key)
    bar = plt.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02, label=label)
    finite_values = np.asarray(
        [v for v in values.values() if finite(v)], dtype=float,
    )
    if finite_values.size and np.allclose(finite_values, finite_values[0]):
        # A single value would otherwise get ticks spanning the padded range.
        bar.set_ticks([float(finite_values[0])])
    bar.ax.tick_params(labelsize=9)
    bar.set_label(label, size=10)


def add_legend(ax, handles, title=None, below=False):
    seen = set()
    unique = []
    for handle in handles:
        label = handle.get_label()
        if label in seen or label.startswith('_'):
            continue
        seen.add(label)
        unique.append(handle)
    if not unique:
        return

    kwargs = dict(
        handles=unique, title=title, frameon=False, fontsize=9,
        title_fontsize=10, borderaxespad=0.0,
    )
    if below:
        ax.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.01),
            ncol=min(len(unique), 4), **kwargs,
        )
    else:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), **kwargs)
