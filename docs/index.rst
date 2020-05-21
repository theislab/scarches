|PyPI| |PyPIDownloads| |Docs|

scNet - Query to reference single-cell integration with transfer learning
=========================================================================

scNet is novel pipe-line which uses transfer learning and architectural surgery techniques in deep learning to address
the challenge of integrating query datasets with reference atlases.

.. image:: https://raw.githubusercontent.com/theislab/scNet/master/sketch.png
   :width: 500px
   :align: center

.. toctree::
   :maxdepth: 1
   :caption: Main
   :hidden:

   about
   installation
   api/index.rst
   release_notes
   references

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   pancreas_from_pretrained
   pancreas_from_scratch


Main Principles
---------------

scNet has the following main principles:

* **User Friendly**: scNet is an API designed for human beings, not machines. scNet offers consistent & simple APIs, it minimizes the number of user actions required for a common use case, and it provides clear feedback upon user error.

* **Extendability**: It is very simple to add new modules, or extend a module in scNet. All modules are implemented in the way to work correctly independent of each other. Being able to easily create new modules allows scNet to be more suitable for advanced research.

* **Python**: scNet is package implemented with Python language, which is compact, easier to debug, and allows for ease of extensibility.

Support
-------

Please feel free to ask questions:

* `Mohammad Lotfollahi <mohammad.lotfollahi@helmholtz-muenchen.de>`_

* `Mohsen Naghipourfar <naghipourfar@ce.sharif.edu>`_

You can also post bug reports and feature requests in `GitHub issues <https://github.com/theislab/scNet/issues>`_.
Please Make sure read our guidelines first.

.. |PyPI| image:: https://img.shields.io/pypi/v/scnet.svg
   :target: https://pypi.org/project/scnet

.. |PyPIDownloads| image:: https://pepy.tech/badge/scnet
   :target: https://pepy.tech/project/scnet

.. |Docs| image:: https://readthedocs.org/projects/scnet/badge/?version=latest
   :target: https://scnet.readthedocs.io

.. |travis| image:: https://travis-ci.org/theislab/scnet.svg?branch=master
   :target: https://travis-ci.org/theislab/scNet

.. _scanpy: https://scanpy.readthedocs.io


