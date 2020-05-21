API
===

The API reference contains detailed descriptions of the different end-user classes, functions, methods, etc.


.. note::

    This API reference only contains end-user documentation.
    If you are looking to hack away at scNet's internals, you will find more detailed comments in the source code.

Import scNet as::

    import scnet as sn

After reading the data (``sn.data.read``), you can normalize your data with our ``sn.data.normalize_hvg`` function.
Then, you can instantiate one of the implemented models from ``sn.models`` module (currently we support ``scNet``,
``CVAE``, ``CVAE_NB``, and ``CVAE_ZINB``) and train it on your dataset. Finally, after training a model on your task, You can
share your trained model via ``sn.zenodo`` functions. Multiple examples are provided in `here`.

.. toctree::
    :glob:
    :maxdepth: 2

    *