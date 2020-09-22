|PyPI| |travis| |Docs|

scArches - single-cell architecture surgery
=========================================================================
.. raw:: html

 <img src="https://user-images.githubusercontent.com/33202701/89729020-15f7c200-da32-11ea-989b-1b9a3283f642.png" width="700px" align="center">

scArches is a package to integrate newly produced single-cell datasets into integrated references atlases. Our method can facilitate large collaborative projects with decentralised training and integration of multiple datasets by different groups. scArches is compatible with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_. and hosts efficient implementations of all conditional generative models for single-cell data. 

What can you do with scArches?
-------------------------------
- Integrate many single-cell datasets and share the trained model and the data (if possible).
- Download a pre-trained model for your atlas of interest, update it wih new datasets and share with your collaborators.
- Construct a customized reference by downloading a reference atlas, add a few  pre-trained adaptors (datasets) and project your own data in to this customized reference atlas.
- Project and integrate query datasets on the top of a reference and use latent representation for downstream tasks, e.g.:diff testing, clustering.

What is an adaptor?
--------------------------------
.. raw:: html

    <img src="https://user-images.githubusercontent.com/33202701/89730296-bdc6bd00-da3d-11ea-9012-410e22fa200a.png" width="200px" height="200px" align="right">

In scArches, each query datasets is added to the reference model by
training a set of weights called `adaptor`. Each `adaptor` is a sharable
object. This will enable users to download a reference model, customise
that reference model with a set of `adaptors` (datasets) and finally
add user data as a new `adaptor` and also share this adaptor with others.


Where to start?
---------------
To get a sense of how the model works please go through `this <https://scarches.readthedocs.io/en/latest/pancreas_pipeline.html>`__ tutorial.
To find out how to construct and share or use pre-trained models example sections. Check `this <https://scarches.readthedocs.io/en/latest/zenodo_intestine.html>`__ example to learn how to start with a raw data  and pre-process data for the model.

It is always good to have a look at our `training tips <https://scarches.readthedocs.io/en/latest/training_tips.html>`__ to
help you train more optimal models.


Reference
-------------------------------
If scArches is useful in your research, please consider to cite the `preprint <https://www.biorxiv.org/content/10.1101/2020.07.16.205997v1/>`_.





.. |PyPI| image:: https://img.shields.io/pypi/v/scarches.svg
   :target: https://pypi.org/project/scarches

.. |PyPIDownloads| image:: https://pepy.tech/badge/scarches
   :target: https://pepy.tech/project/scarches

.. |Docs| image:: https://readthedocs.org/projects/scarches/badge/?version=latest
   :target: https://scarches.readthedocs.io

.. |travis| image:: https://travis-ci.com/theislab/scarches.svg?branch=master
    :target: https://travis-ci.com/theislab/scarches


.. toctree::
   :maxdepth: 1
   :caption: Main
   :hidden:

   about
   installation
   api/index.rst
   model_sharing
   training_tips

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   pancreas_pipeline
   zenodo_pancreas_from_pretrained
   zenodo_pancreas_from_scratch
   zenodo_intestine


