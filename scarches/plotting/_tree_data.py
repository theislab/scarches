"""Tree and :class:`~anndata.AnnData` adapter for the hierarchy plot.

Alias handling, subtree membership and per-node aggregation live here. This
module stays free of matplotlib and of layout geometry so the numbers behind a
tree plot can be tested without drawing anything.
"""
import numpy as np


def node_label(node):
    """Every alias of a node joined, used for stable sibling ordering."""
    names = node.name if isinstance(node.name, (list, tuple)) else [node.name]
    return ' & '.join(str(n) for n in names if n is not None)


def alias_list(node):
    names = node.name if isinstance(node.name, (list, tuple)) else [node.name]
    return [str(n) for n in names if n is not None]


def is_leaf(node):
    return len(node.descendants) == 0


def finite(value):
    """True for usable numbers and for non-empty categorical values."""
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return value is not None and str(value) != 'nan'


def is_continuous(values):
    for value in values.values():
        if not finite(value):
            continue
        return isinstance(value, (int, float, np.floating, np.integer)) or (
            np.isscalar(value) and not isinstance(value, (str, bytes))
        )
    return True


def continuous_limits(values):
    """Colour limits that stay valid when every node shares one value."""
    arr = np.asarray([v for v in values.values() if finite(v)], dtype=float)
    if not arr.size:
        return None
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if np.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.05, 0.05)
        vmin -= padding
        vmax += padding
    return vmin, vmax


def strip_batch_suffix(label, batch_levels):
    if batch_levels and '-' in label:
        stem, suffix = label.rsplit('-', 1)
        if suffix in batch_levels:
            return stem
    return label


def parse_study_suffix(label, batch_levels):
    if '-' not in label:
        return None
    _stem, suffix = label.rsplit('-', 1)
    if batch_levels is None or suffix in batch_levels:
        return suffix
    return None


def infer_batch_levels(nodes):
    """Study suffixes read off treeArches ``CellType-Study`` aliases."""
    suffixes = []
    for node in nodes:
        for alias in alias_list(node):
            if '-' in alias:
                suffixes.append(alias.rsplit('-', 1)[1])
    return list(dict.fromkeys(suffixes))


def batch_levels_of(nodes, adata, batch_key):
    """Observed batch values first, then suffixes inferred from the tree."""
    if batch_key is None:
        return None
    inferred = infer_batch_levels(nodes)
    if adata is not None and batch_key in adata.obs:
        observed = [str(v) for v in adata.obs[batch_key].astype(str).unique()]
        return list(dict.fromkeys(observed + inferred))
    return inferred


def compact_name(node, batch_levels=None, strip_study=True):
    """Collapse matched aliases and drop ``-study`` suffixes when encoded."""
    aliases = alias_list(node)
    if not aliases:
        return ''

    levels = set(batch_levels) if batch_levels else None
    if strip_study and levels:
        stems = [strip_batch_suffix(a, levels) for a in aliases]
    else:
        stems = list(aliases)
    unique = list(dict.fromkeys(stems))
    if len(unique) == 1:
        return unique[0]
    joined = ' / '.join(unique)
    return joined if len(joined) <= 36 else joined[:35] + '…'


def subtree_mask(node, labels):
    aliases = set()
    for descendant in node.walk():
        if is_leaf(descendant):
            aliases |= set(alias_list(descendant))
    if not aliases:
        return np.zeros(len(labels), dtype=bool)
    return np.isin(np.asarray(labels, dtype=str), list(aliases))


def node_masks(root, labels):
    """Observation membership per node, computed once for every channel."""
    labels = np.asarray(labels, dtype=str)
    masks = {}

    def visit(node):
        if is_leaf(node):
            mask = np.isin(labels, list(set(alias_list(node))))
        else:
            mask = np.logical_or.reduce([visit(c) for c in node.descendants])
        masks[id(node)] = mask
        return mask

    visit(root)
    return masks


def _mask_of(node, masks, labels):
    if masks is not None:
        return masks[id(node)]
    return subtree_mask(node, labels)


def _simpson(values):
    if len(values) == 0:
        return np.nan
    _, counts = np.unique(np.asarray(values, dtype=str), return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def node_values(nodes, adata, cell_type_key, batch_key, key, masks=None):
    """Aggregate ``key`` over the observations belonging to each subtree."""
    if key is None:
        return None
    if adata is None:
        raise ValueError(f"adata is required for color/value '{key}'")
    if cell_type_key is None:
        raise ValueError('cell_type_key is required when adata is provided')
    if cell_type_key not in adata.obs:
        raise KeyError(cell_type_key)

    labels = np.asarray(adata.obs[cell_type_key].values, dtype=str)
    out = {}

    if key == 'n':
        for node in nodes:
            out[id(node)] = int(_mask_of(node, masks, labels).sum())
        return out

    if key == 'diversity':
        if batch_key is None:
            raise ValueError('batch_key is required for diversity')
        if batch_key not in adata.obs:
            raise KeyError(batch_key)
        batches = np.asarray(adata.obs[batch_key].values)
        for node in nodes:
            mask = _mask_of(node, masks, labels)
            out[id(node)] = _simpson(batches[mask]) if np.any(mask) else np.nan
        return out

    if key not in adata.obs:
        raise KeyError(key)

    col = adata.obs[key]
    for node in nodes:
        mask = _mask_of(node, masks, labels)
        if not np.any(mask):
            out[id(node)] = np.nan
            continue
        vals = col.iloc[np.where(mask)[0]] if hasattr(col, 'iloc') else col[mask]
        arr = np.asarray(vals)
        if np.issubdtype(arr.dtype, np.number):
            out[id(node)] = float(np.nanmean(arr.astype(float)))
        else:
            levels, counts = np.unique(arr.astype(str), return_counts=True)
            out[id(node)] = levels[np.argmax(counts)]
    return out


def channel_values(nodes, adata, cell_type_key, batch_key, key, masks=None):
    """``node_values`` on a colour channel; counts are log1p for readability."""
    values = node_values(nodes, adata, cell_type_key, batch_key, key, masks=masks)
    if key == 'n' and values is not None:
        return {k: (np.log1p(v) if finite(v) else v) for k, v in values.items()}
    return values


def pie_fractions(nodes, adata, cell_type_key, pie_key, masks=None):
    """Per-node category fractions, with levels fixed by the whole tree."""
    if adata is None or cell_type_key is None:
        raise ValueError('adata and cell_type_key are required for pie')
    if pie_key not in adata.obs:
        raise KeyError(pie_key)

    labels = np.asarray(adata.obs[cell_type_key].values, dtype=str)
    cats = np.asarray(adata.obs[pie_key].values, dtype=str)
    matched = masks[id(nodes[0])] if (masks is not None and nodes) else None
    level_values = cats[matched] if matched is not None else cats
    levels = sorted(str(x) for x in np.unique(level_values))

    out = {}
    for node in nodes:
        mask = _mask_of(node, masks, labels)
        if not np.any(mask):
            out[id(node)] = (levels, np.zeros(len(levels)))
            continue
        counts = np.array(
            [(cats[mask].astype(str) == level).sum() for level in levels],
            dtype=float,
        )
        total = counts.sum()
        out[id(node)] = (levels, counts / total if total > 0 else counts)
    return out


def study_values(nodes, adata, cell_type_key, batch_key, batch_levels, masks=None):
    """Majority study per node from ``adata``, else from alias suffixes."""
    usable = (
        adata is not None and cell_type_key is not None and batch_key is not None
        and batch_key in adata.obs and cell_type_key in adata.obs
    )
    if not usable:
        return {
            id(node): _majority_study_from_aliases(node, batch_levels)
            for node in nodes
        }

    labels = np.asarray(adata.obs[cell_type_key].values, dtype=str)
    batches = np.asarray(adata.obs[batch_key].values, dtype=str)
    out = {}
    for node in nodes:
        mask = _mask_of(node, masks, labels)
        if not np.any(mask):
            out[id(node)] = _majority_study_from_aliases(node, batch_levels)
            continue
        levels, counts = np.unique(batches[mask], return_counts=True)
        out[id(node)] = str(levels[np.argmax(counts)])
    return out


def _majority_study_from_aliases(node, batch_levels):
    counts = {}
    for alias in alias_list(node):
        suffix = parse_study_suffix(alias, batch_levels)
        if suffix is None:
            continue
        counts[suffix] = counts.get(suffix, 0) + 1
    if not counts:
        return np.nan
    return max(counts.items(), key=lambda item: item[1])[0]


def empty_mask(nodes, counts, values):
    """Nodes with no matching observations, drawn grey instead of coloured."""
    empty = {}
    for node in nodes:
        nid = id(node)
        if counts is not None and counts.get(nid, 0) == 0:
            empty[nid] = True
        elif values is not None and not finite(values.get(nid, np.nan)):
            empty[nid] = True
        else:
            empty[nid] = False
    return empty
