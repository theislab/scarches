|PyPI| |travis| |Docs|

scArches (PyTorch) - single-cell architecture surgery
=========================================================================
.. raw:: html

 <img src="https://user-images.githubusercontent.com/33202701/89729020-15f7c200-da32-11ea-989b-1b9a3283f642.png" width="700px" align="center">

scArches is a package to integrate newly produced single-cell datasets into integrated reference atlases. Our method can facilitate large collaborative projects with decentralized training and integration of multiple datasets by different groups. scArches is compatible with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_. and hosts efficient implementations of all conditional generative models for single-cell data.

.. note::

  expiMap has been added to scArches code base. It allows interpretable representation learning from scRNA-seq data and also reference mapping. Try it in the tutorial section.

What can you do with scArches?
-------------------------------
- Construct single or multi-modal (CITE-seq) reference atlases and share the trained model and the data (if possible).
- Download a pre-trained model for your atlas of interest, update it with new datasets and share with your collaborators.
- Project and integrate query datasets on the top of a reference and use latent representation for downstream tasks, e.g.:diff testing, clustering, classification

What are the different models?
---------------
scArches is itself an algorithm to map to project query on the top of reference datasets and applies
to different models. Here we provide a short explanation and hints on when to use which model. Our models are:

- **scVI**  (`Lopez et al., 2018 <https://www.nature.com/articles/s41592-018-0229-2>`_): Requires access to raw counts values for data integration and assumes count distribution on the data (NB, ZINB, Poisson).

- **trVAE** (`Lotfollahi et al.,2020 <https://academic.oup.com/bioinformatics/article/36/Supplement_2/i610/6055927?guestAccessKey=71253caa-1779-40e8-8597-c217db539fb5>`_): It supports both normalized log-transformed or count data as input and applies additional MMD loss to have better merging in the latent space.

- **scANVI** (`Xu et al., 2019 <https://www.biorxiv.org/content/10.1101/532895v1>`_): It needs cell type labels for reference data. Your query data can be either unlabeled or labeled. In the case of unlabeled query data, you can use this method also to classify your query cells using reference labels.

- **scGen** (`Lotfollahi et al., 2019 <https://www.nature.com/articles/s41592-019-0494-8>`_): This method requires cell-type labels for both reference building and Mapping. The reference mapping for this method solely relies on the integrated reference and requires no fine-tuning.

- **expiMap** (`Lotfollahi*, Rybakov* et al., 2023 <https://www.nature.com/articles/s41556-022-01072-x>`_): This method takes prior knowledge from gene sets databases or users allowing to analyze your query data in the context of known gene programs.

- **totalVI** (`Gayoso al., 2019 <https://www.biorxiv.org/content/10.1101/532895v1>`_): This model can be used to build multi-modal  CITE-seq reference atalses.

- **treeArches** (`Michielsen*, Lotfollahi* et al., 2022 <https://www.biorxiv.org/content/10.1101/2022.07.07.499109v1>`_): This model builds a hierarchical tree for cell-types in the reference atlas and when mapping the query data can annotate and also identify novel cell-states and populations present in the query data.

- **SageNet** (`Heidari et al., 2022 <https://www.biorxiv.org/content/10.1101/2022.04.14.488419v1>`_): This model allows constrcution of a spatial atlas by mapping query dissociated single cells/spots (e.g., from  scRNAseq or visium datasets) into a common coordinate framework using one or more spatially resolved reference datasets.

- **mvTCR** (`Drost et al., 2022 <https://www.biorxiv.org/content/10.1101/2021.06.24.449733v2.abstract?%3Fcollection=>`_): Using this model you will be able to integrate T-cell receptor (TCR, treated as a sequence) and scRNA-seq dataset across multiple donors into a joint representation capturing information from both modalities.

- **scPoli** (`De Donno et al., 2022 <https://www.biorxiv.org/content/10.1101/2022.11.28.517803v1>`_): This model allows data integration of scRNA-seq and also scATAC-seq dataset, prototype-based label transfer and reference mapping. scPoli learns both sample embeddings and integrated cell embeddings, thus providing the user with a multi-scale view of the data, especially useful in the case of many samples to integrate.

Where to start?
---------------
To get a sense of how the model works please go through `this <https://scarches.readthedocs.io/en/latest/trvae_surgery_pipeline.html>`__ tutorial.
To find out how to construct and share or use pre-trained models example sections.

Reference
-------------------------------
If scArches is useful in your research, please consider citing the `paper <https://www.nature.com/articles/s41587-021-01001-7>`_.


.. |PyPI| image:: https://img.shields.io/pypi/v/scarches.svg
   :target: https://pypi.org/project/scarches

.. |PyPIDownloads| image:: https://pepy.tech/badge/scarches
   :target: https://pepy.tech/project/scarches

.. |Docs| image:: https://readthedocs.org/projects/scarches/badge/?version=latest
   :target: https://scarches.readthedocs.io

.. |travis| image:: https://travis-ci.com/theislab/scarches.svg?branch=master
    :target: https://travis-ci.com/theislab/scarches
