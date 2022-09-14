.. raw:: html

 <img src="https://user-images.githubusercontent.com/33202701/187203672-e0415eec-1278-4b2a-a097-5bb8b6ab694f.svg" width="300px" height="200px" align="center">

|PyPI| |PyPIDownloads| |Docs| |travis|


Single-cell architecture surgery (scArches) is a package for reference-based analysis of single-cell data.

Updates
-------------------------------

- **(7.07.2022)** We have added `treeArches <https://www.biorxiv.org/content/10.1101/2022.07.07.499109v1>`_ to scArches code base. treeArches enables building cell-type hierarchies to identify novel states (e.g., disease, subpopulations) in the query data when mapped to the reference. See tutorials `here <https://scarches.readthedocs.io/>`_ .

- **(6.02.2022)** We have added `expiMap <https://www.biorxiv.org/content/10.1101/2022.02.05.479217v1>`_ to scArches code base. expiMap allows interpretable reference mapping. See tutorials `here <https://scarches.readthedocs.io/>`_ .

What is scArches?
-------------------------------
scArches allows analysis of your single-cell query data by integrating it into a reference atlas. To map your data you need an integrated atlas using one of the reference building methods for deifferent applications that are supported by scArches whcih are , inlcuding:


  
- **scVI**  (`Lopez et al., 2018 <https://www.nature.com/articles/s41592-018-0229-2>`_): Requires access to raw counts values for data integration and assumes count distribution on the data (NB, ZINB, Poisson).

- **trVAE** (`Lotfollahi et al.,2020 <https://academic.oup.com/bioinformatics/article/36/Supplement_2/i610/6055927?guestAccessKey=71253caa-1779-40e8-8597-c217db539fb5>`_): It supports both normalized log-transformed or count data as input and applies additional MMD loss to have better merging in the latent space.

- **scANVI** (`Xu et al., 2019 <https://www.biorxiv.org/content/10.1101/532895v1>`_): It needs cell type labels for reference data. Your query data can be either unlabeled or labeled. In the case of unlabeled query data, you can use this method also to classify your query cells using reference labels.

- **scGen** (`Lotfollahi et al., 2019 <https://www.nature.com/articles/s41592-019-0494-8>`_): This method requires cell-type labels for both reference building and Mapping. The reference mapping for this method solely relies on the integrated reference and requires no fine-tuning.

- **expiMap** (`Lotfollahi*, Rybakov* et al., 2022 <https://www.biorxiv.org/content/10.1101/2022.02.05.479217v1>`_): This method takes prior knowledge from gene sets databases or users allowing to analyze your query data in the context of known gene programs.  

- **totalVI** (`Gayoso al., 2019 <https://www.biorxiv.org/content/10.1101/532895v1>`_): This model can be used to build multi-modal  CITE-seq reference atalses.

- **treeArches** (`Michielsen*, Lotfollahi* et al., 2022 <https://www.biorxiv.org/content/10.1101/2022.07.07.499109v1>`_): This model builds a hierarchical tree for cell-types in the reference atlas and when mapping the query data can annotate and also identify novel cell-states and populations present in the query data.

- **SageNet** (`Heidari et al., 2022 <https://www.biorxiv.org/content/10.1101/2022.07.07.499109v1>`_): This model allows constrcution of a spatial atlas by mapping query dissociated single cells (e.g., from an scRNAseq data) into a common coordinate framework using one or more spatially resolved reference datasets.

Usage and installation
-------------------------------
See `here <https://scarches.readthedocs.io/>`_ for documentation and tutorials.

Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an `issue <https://github.com/theislab/scarches/issues/new>`__ or reach us by `email <mailto:cottoneyejoe.server@gmail.com,mo.lotfollahi@gmail.com,mohsen.naghipourfar@gmail.com>`_.

Reference
-------------------------------
If scArches is helpful in your research, please consider citing the following `paper <https://www.nature.com/articles/s41587-021-01001-7>`_:
::


       @article{lotfollahi2021mapping,
         title={Mapping single-cell data to reference atlases by transfer learning},
         author={Lotfollahi, Mohammad and Naghipourfar, Mohsen and Luecken, Malte D and Khajavi,
         Matin and B{\"u}ttner, Maren and Wagenstetter, Marco and Avsec, {\v{Z}}iga and Gayoso,
         Adam and Yosef, Nir and Interlandi, Marta and others},
         journal={Nature Biotechnology},
         pages={1--10},
         year={2021},
         publisher={Nature Publishing Group}}




.. |PyPI| image:: https://img.shields.io/pypi/v/scarches.svg
   :target: https://pypi.org/project/scarches

.. |PyPIDownloads| image:: https://pepy.tech/badge/scarches
   :target: https://pepy.tech/project/scarches

.. |Docs| image:: https://readthedocs.org/projects/scarches/badge/?version=latest
   :target: https://scarches.readthedocs.io

.. |travis| image:: https://travis-ci.com/theislab/scarches.svg?branch=master
    :target: https://travis-ci.com/theislab/scarches
