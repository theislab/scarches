API
===

The API reference contains detailed descriptions of the different end-user classes, functions, methods, etc.


.. note::

    This API reference only contains end-user documentation.
    If you are looking to hack away at scArches' internals, you will find more detailed comments in the source code.

Import scarches as::

    import scarches as sca

After reading the data (``sca.data.read``), you can you can instantiate one of the implemented models from ``sca.models`` module (currently we support ``trVAE``,
``scVI``, ``scANVI``, and ``TotalVI``) and train it on your dataset.

.. toctree::
    :glob:
    :maxdepth: 2

    *
