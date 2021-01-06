|PyPI| |PyPIDownloads| |Docs| |travis|


scArches - single-cell architecture surgery
=========================================================================
.. raw:: html

 <img src="https://user-images.githubusercontent.com/33202701/89729020-15f7c200-da32-11ea-989b-1b9a3283f642.png" width="700px" align="center">

scArches is a package to integrate newly produced single-cell datasets into integrated reference atlases. Our method can facilitate large collaborative projects with decentralized training and integration of multiple datasets by different groups. scArches is compatible with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_. and hosts efficient implementations of several conditional generative models for single-cell data.

What can you do with scArches?
-------------------------------
- Construct single or multi-modal (CITE-seq) reference atlases and share the trained model and the data (if possible).
- Download a pre-trained model for your atlas of interest, update it with new datasets and share with your collaborators.
- Project and integrate query datasets on the top of a reference and use latent representation for downstream tasks, e.g.:diff testing, clustering, classification

What are the different models?
---------------
scArches is itself an algorithm to map to project query on the top of reference datasets and applies
to different models. Here we provide a short explanation and hints on when to use which model. Our models are divided into
three categories:

Unsupervised
 This class of algorithms require no `cell type` labels, meaning that you can create a reference and project a query without having access to cell type labels.
 We implemented two algorithms:

 - **scVI**  (`Lopez et al., 2018 <https://www.nature.com/articles/s41592-018-0229-2>`_): Requires access to raw counts values for data integration and assumes
 count distribution on the data (NB, ZINB, Poisson).

 - **trVAE** (`Lotfollahi et al.,2020 <https://academic.oup.com/bioinformatics/article/36/Supplement_2/i610/6055927?guestAccessKey=71253caa-1779-40e8-8597-c217db539fb5>`_): It supports both normalized log transformed or count data as input and applies additional MMD loss to have better merging in the latent space.

Supervised and Semi-supervised
 This class of algorithms assumes the user has access to `cell type` labels when creating the reference data and usually perform better integration compared to. unsupervised methods. However, query data still can be unlabeled. In addition to integration, you can classify your query cells using
 these methods.

 - **scANVI** (`Xu et al., 2019 <https://www.biorxiv.org/content/10.1101/532895v1>`_): It needs cell type labels for reference data. Your query data can be either unlabeled or labeled. In the case of unlabeled query data, you can use this method to also classify your query cells using reference labels.

Multi-modal
 These algorithms can be used to construct multi-modal references atlas and map query data from either modality on the top of the reference.

 - **totalVI** (`Gayoso al., 2019 <https://www.biorxiv.org/content/10.1101/532895v1>`_): This model can be used to build multi-modal  CITE-seq reference atalses.
   Query datasets can be either from sc-RNAseq or CITE-seq. In addition to integrating query with reference, one can use this model to impute the Proteins
   in the query datasets.

How to choose the right model
---------------

- If your reference data is labeled (cell-type labels) and you have unlabeled or labeled query then use **scArches scANVI**


- If your reference and query are both unlabeled our preferred model is **scArches scVI** and if it did not work for you try **scArches trVAE**


- If you have CITE-seq data and you want to integrate RNA-seq as query and denoise the proteins in the RNA-seq then use **scArches totalVI**

Usage and installation
-------------------------------
See `here <https://scarches.readthedocs.io/>`_ for documentation and tutorials.

Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an `issue <https://github.com/theislab/scarches/issues/new>`__ or reach us by `email <mailto:cottoneyejoe.server@gmail.com,mo.lotfollahi@gmail.com,mohsen.naghipourfar@gmail.com>`_.

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
