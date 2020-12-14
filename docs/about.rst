|PyPI| |travis| |Docs|

scArches (PyTorch) - single-cell architecture surgery
=========================================================================
.. raw:: html

 <img src="https://user-images.githubusercontent.com/33202701/89729020-15f7c200-da32-11ea-989b-1b9a3283f642.png" width="700px" align="center">

scArches is a package to integrate newly produced single-cell datasets into integrated references atlases. Our method can facilitate large collaborative projects with decentralised training and integration of multiple datasets by different groups. scArches is compatible with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_. and hosts efficient implementations of all conditional generative models for single-cell data.


What can you do with scArches?
-------------------------------
- Construct single or multi-modal (CITE-seq) reference atlases and share the trained model and the data (if possible).
- Download a pre-trained model for your atlas of interest, update it wih new datasets and share with your collaborators.
- Project and integrate query datasets on the top of a reference and use latent representation for downstream tasks, e.g.:diff testing, clustering, classification


What are different models?
---------------
scArches is itself and algorithm to map to project query on the top of reference datasets and is applicable
to different models. Here we provide a short explanation and hints when to use which model. Our models are divided into
three categories:


What are different models?
---------------
scArches is itself and algorithm to map to project query on the top of reference datasets and is applicable
to different models. Here we provide a short explanation and hints when to use which model. Our models are divided into
three categories:

Unsupervised
 This class of algortihms need no `cell type` labels, meaning that you can creat a reference and project a query without having access to cell type labeles.
 We implemented two algorithms:

 - **scVI**  (`Lopez et al.,2018 <https://www.nature.com/articles/s41592-018-0229-2>`_.): Requires access to raw counts values for data integration and assumes
 count distribution on the data (NB, ZINB, Poission).

 - **trVAE** (`Lotfollahi et al.,2019 <https://arxiv.org/abs/1910.01791>`_.): It supports both normalized log tranformed or count data as input and applies additional MMD loss to have better mearging in the latent space.

Supervised and Semi-supervised
 This class of algorithmes assume the user has access to `cell type` labels when creating the reference data and usaully perfomr better integration
 compared to. unsupervised methods. However, the query data still can be unlabaled. In addition to integration , you can classify your query cells using
 these methods.

 - **scANVI** (`Xu et al.,2019 <https://www.biorxiv.org/content/10.1101/532895v1>`_.): It neeeds cell type labels for reference data. Your query data can be either   unlabeled or labeled. In case of unlabeled query data you can use this method to also classify your query cells using reference labels.

Multi-modal
 These algorithms can be used to contstruct multi-modal references atlas and map query data from either modalities on the top of the reference.

 - **totalVI** (`Gayoso al.,2019 <https://www.biorxiv.org/content/10.1101/532895v1>`_.): This model can be used to build multi-modal  CITE-seq reference atalses.
   Query datasets can be either from sc-RNAseq or CITE-seq. In addition to integrating query with reference one can use this model to impute the Proteins
   in the query datasets.


Where to start?
---------------
To get a sense of how the model works please go through `this <https://scarches.readthedocs.io/en/latest/pancreas_pipeline.html>`__ tutorial.
To find out how to construct and share or use pre-trained models example sections. Check `this <https://scarches.readthedocs.io/en/latest/zenodo_intestine.html>`__ example to learn how to start with a raw data  and pre-process data for the model.

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

