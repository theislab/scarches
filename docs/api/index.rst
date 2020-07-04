API
===

The API reference contains detailed descriptions of the different end-user classes, functions, methods, etc.


.. note::

    This API reference only contains end-user documentation.
    If you are looking to hack away at scArches' internals, you will find more detailed comments in the source code.

Import scArches as::

    import scarches as sca

After reading the data (``sca.data.read``), you can normalize your data with our ``sca.data.normalize_hvg`` function.
Then, you can instantiate one of the implemented models from ``sca.models`` module (currently we support ``scArches``,
``scArches``, ``scArchesNB``, and ``scArchesZINB``) and train it on your dataset. Finally, after training a model on your task, You can
share your trained model via ``sca.zenodo`` functions. Multiple examples are provided in `here`.

.. toctree::
    :glob:
    :maxdepth: 2

    *