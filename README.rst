|PyPI| |travis| |Docs| |PyPIDownloads|

scArches - single-cell architecture surgery
=========================================================================
.. raw:: html

 <img src="https://user-images.githubusercontent.com/33202701/89729020-15f7c200-da32-11ea-989b-1b9a3283f642.png" width="900px" align="center">

scArches is a package to integrate newly produced single-cell datasets into integrated reference atlases. Our method can facilitate large collaborative projects with decentralise training and integration of multiple datasets by different groups. scArches is compatible with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_, and hosts efficient implementations of all conditional generative models for single-cell data.

What can you do with scArches?
-------------------------------
- Integrate many single-cell datasets and share the trained model and the data (if possible).
- Download a pre-trained model for your atlas of interest, update it with new datasets and share with your collaborators.
- Construct a customized reference by downloading a reference atlas, add a few  pre-trained adaptors (datasets) and project your own data in to this customized reference atlas.
- Project and integrate query datasets on the top of a reference and use latent representation for downstream tasks, e.g.: diff testing, clustering.

Usage and installation
-------------------------------
See `here <https://scarches.readthedocs.io/>`_ for documentation and tutorials.

Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an `issue <https://github.com/theislab/scarches/issues/new>`__. Our package supports tf/keras now but pytorch version will be added very soon.


Reference
-------------------------------
If scArches is useful in your research, please consider citing this `preprint <https://www.biorxiv.org/content/10.1101/2020.07.16.205997v1/>`_.


.. |PyPI| image:: https://img.shields.io/pypi/v/scarches.svg
   :target: https://pypi.org/project/scarches

.. |PyPIDownloads| image:: https://pepy.tech/badge/scarches
   :target: https://pepy.tech/project/scarches

.. |Docs| image:: https://readthedocs.org/projects/scarches/badge/?version=latest
   :target: https://scarches.readthedocs.io

.. |travis| image:: https://travis-ci.com/theislab/scarches.svg?branch=master
    :target: https://travis-ci.com/theislab/scarches
