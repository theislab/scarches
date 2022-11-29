.. raw:: html

 <img src="https://user-images.githubusercontent.com/33202701/187203672-e0415eec-1278-4b2a-a097-5bb8b6ab694f.svg" width="300px" height="200px" align="center">


Single-cell architecture surgery (scArches) is a package for reference-based analysis of single-cell data.


.. note::

- **(22.10.2022)** We have added `mvTCR <https://www.biorxiv.org/content/10.1101/2021.06.24.449733v2.abstract?%3Fcollection=>`_ and `SageNet <https://www.biorxiv.org/content/10.1101/2022.04.14.488419v1>`_ enabling mapping multimodal immune profiling (TCR+scRNAreq) and scRNA-seq to spatial atlases, respectively.

  **(7.07.2022)** We have added `treeArches <https://www.biorxiv.org/content/10.1101/2022.07.07.499109v1>`_ to scArches code base. treeArches enables building cell-type hierarchies to identify novel states (e.g., disease, subpopulations) in the query data when mapped to the reference. See tutorials `here <https://scarches.readthedocs.io/>`_ .

  **(6.02.2022)** We have added `expiMap <https://www.biorxiv.org/content/10.1101/2022.02.05.479217v1>`_ to scArches code base. expiMap allows interpretable reference mapping. Try it in the tutorials section.

What is scArches?
-------------------------------
scArches allows analysis of your single-cell query data by integrating it into a reference atlas. To map your data you need an integrated atlas using one of the reference building methods for different applications that are supported by scArches which are, including:



- **scVI**  (`Lopez et al., 2018 <https://www.nature.com/articles/s41592-018-0229-2>`_): Requires access to raw counts values for data integration and assumes count distribution on the data (NB, ZINB, Poisson).

- **trVAE** (`Lotfollahi et al.,2020 <https://academic.oup.com/bioinformatics/article/36/Supplement_2/i610/6055927?guestAccessKey=71253caa-1779-40e8-8597-c217db539fb5>`_): It supports both normalized log-transformed or count data as input and applies additional MMD loss to have better merging in the latent space.

- **scANVI** (`Xu et al., 2019 <https://www.biorxiv.org/content/10.1101/532895v1>`_): It needs cell type labels for reference data. Your query data can be either unlabeled or labeled. In the case of unlabeled query data, you can use this method also to classify your query cells using reference labels.

- **scGen** (`Lotfollahi et al., 2019 <https://www.nature.com/articles/s41592-019-0494-8>`_): This method requires cell-type labels for both reference building and Mapping. The reference mapping for this method solely relies on the integrated reference and requires no fine-tuning.

- **expiMap** (`Lotfollahi*, Rybakov* et al., 2022 <https://www.biorxiv.org/content/10.1101/2022.02.05.479217v1>`_): This method takes prior knowledge from gene sets databases or users allowing to analyze your query data in the context of known gene programs.

- **totalVI** (`Gayoso al., 2019 <https://www.biorxiv.org/content/10.1101/532895v1>`_): This model can be used to build multi-modal  CITE-seq reference atalses.

- **treeArches** (`Michielsen*, Lotfollahi* et al., 2022 <https://www.biorxiv.org/content/10.1101/2022.07.07.499109v1>`_): This model builds a hierarchical tree for cell-types in the reference atlas and when mapping the query data can annotate and also identify novel cell-states and populations present in the query data.

- **SageNet** (`Heidari et al., 2022 <https://www.biorxiv.org/content/10.1101/2022.04.14.488419v1>`_): This model allows constrcution of a spatial atlas by mapping query dissociated single cells/spots (e.g., from  scRNAseq or visium datasets) into a common coordinate framework using one or more spatially resolved reference datasets.

- **mvTCR** (`Drost et al., 2022 <https://www.biorxiv.org/content/10.1101/2021.06.24.449733v2.abstract?%3Fcollection=>`_): Using this model you will be able to integrate T-cell receptor (TCR, treated as a sequence) and scRNA-seq dataset across multiple donors into a joint representation capturing information from both modalities.

Which model to choose?
---------------

- If your reference data is labeled (cell-type labels) and you have an unlabeled or labeled query, then use **scArches scANVI** or **treeArrches**  .

- If your reference data is labeled (cell-type labels) and you have a labeled query, then use **scGen**.

- If your reference and query are unlabeled, our preferred model is **scArches scVI** and if it did not work for you, try **scArches trVAE**, which gives you better integration but is a bit slower.

- If you have CITE-seq data and want to integrate RNA-seq as a query and impute missing proteins in query scRNA-seq data, then use **scArches totalVI**.

- If you scRANseq data and want to analyze your data in the context of gene programs to answer a question such as what pathways have changed after a disease or which genes are causing my new disease state in the query separate from others, then use **expiMap**.


- If you want to build a cellular hierarchy and continuously update the hierarchy using new query datasets, see how your query populations compare to the original hierarchy to identify new subpopulations or disease states in your query, then use **treeArches**.

- If you have scRNA seq data and want to map it to a reference spatial atlas to infer the spatial location and perform cell-cell interaction analysis then use **SageNet**.



Where to start?
---------------
To get a sense of how the model works please go through `this <https://scarches.readthedocs.io/en/latest/trvae_surgery_pipeline.html>`__ tutorial.
To find out how to construct and share or use pre-trained models example sections.

Reference
-------------------------------
If scArches is useful in your research, please consider citing the `preprint <https://www.biorxiv.org/content/10.1101/2020.07.16.205997v1/>`_.


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

   scvi_surgery_pipeline
   scanvi_surgery_pipeline
   totalvi_surgery_pipeline
   trvae_surgery_pipeline
   trVAE_zenodo_pipeline
   reference_building_from_scratch
   pbmc_pipeline_trvae_scvi_scanvi
   scgen_map_query
   expimap_surgery_pipeline_basic
   expimap_surgery_pipeline_advanced
   treeArches_pbmc
   treeArches_identifying_new_ct
   SageNet_mouse_embryo
   mvTCR_borcherding
   multigrate
