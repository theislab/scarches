from importlib import import_module

from .terms_scores import plot_abs_bfs
from .sankey import sankey_diagram
from .tree import plot_tree

__all__ = ('plot_abs_bfs', 'sankey_diagram', 'SCVI_EVAL', 'TRVAE_EVAL', 'plot_tree')

_LAZY_MODULES = {'SCVI_EVAL': '.scvi_eval', 'TRVAE_EVAL': '.trvae_eval'}


def __getattr__(name):
    """Load the model-evaluation plotters on first use.

    They import torch and override global scanpy, torch and numpy defaults at
    module level, which should not happen merely because something imported
    ``scarches``.
    """
    if name in _LAZY_MODULES:
        return getattr(import_module(_LAZY_MODULES[name], __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(__all__)
