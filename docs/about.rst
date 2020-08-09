|PyPI| |travis| |Docs|

scArches - single-cell architecture surgery
=========================================================================
.. raw:: html

 <img src="https://user-images.githubusercontent.com/33202701/89729020-15f7c200-da32-11ea-989b-1b9a3283f642.png" width="400px" align="center">

scArches is a package to integrate newly produced single-cell datasets into integrated references atlases. Our method can facilitate large collaborative projects with decentralise training and integration of multiple datasets by different groups. scArches is compatible with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_. and hosts efficient implementations of all conditional generative models for single-cell data. 

What can you do with scArches?
--------------------------------
- Integrate many single-cell datasets and share the trained model and the data (if possible).
- Download a pre-trained model for your atlas of interest, update it wih new datasets and share with your collaborators.
- Construct a customized reference by downloading a reference atlas, add a few  pre-trained adaptors (datasets) and project your own data in to this customized reference atlas.
- Project and integrate query datasets on the top of a reference and use latent representation for downstream tasks,e.g.:diff testing, clustering.

Where to start?
--------------------------------
To get a sense of how the model works please go through `this <https://scarches.readthedocs.io/en/latest/pancreas_pipeline.html`__.
For examples on how to use or construct and share pre-trained models check examples.

What is an adaptor?
--------------------------------
In scArches ecosystem, we can have a reference model trained on multiple datasets. People can update
this reference model by adding new query datasets. Each query datasets is added to the reference
model by training a set of weights called `adaptor`. Each `adaptor` is a sharable object and is only usable
with a reference model which it was trained for. Therefore, we have a fix reference, and a set of
available `adaptors` for that reference. This will enable users to download a reference model, customise
that reference model with a set of `adaptors` (datasets) and finally user can add her own data as a new
`adaptor` and also share this adaptor for others.

 <img src="https://user-images.githubusercontent.com/33202701/89729666-1eeb9200-da38-11ea-8eea-c4373529355e.png" width="400px" align="left">




.. |PyPI| image:: https://img.shields.io/pypi/v/scarches.svg
   :target: https://pypi.org/project/scarches

.. |PyPIDownloads| image:: https://pepy.tech/badge/scarches
   :target: https://pepy.tech/project/scarches

.. |Docs| image:: https://readthedocs.org/projects/scarches/badge/?version=latest
   :target: https://scarches.readthedocs.io

.. |travis| image:: https://travis-ci.com/theislab/scarches.svg?branch=master
    :target: https://travis-ci.com/theislab/scarches
