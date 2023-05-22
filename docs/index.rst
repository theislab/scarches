.. raw:: html

 <img src="https://user-images.githubusercontent.com/33202701/187203672-e0415eec-1278-4b2a-a097-5bb8b6ab694f.svg" width="200px" height="200px" align="center">


What is scArches?
-------------------------------
scArches allows your single-cell query data to be analyzed by integrating it into a reference atlas. To map your data, you need an integrated atlas using one of the reference-building methods for different applications that are supported by scArches which are, including:

- **Annotating a single-cell dataset using a reference atlas**: You can check following models/tutorials using  `scPoli (De Donno et al., 2022) <https://docs.scarches.org/en/latest/scpoli_surgery_pipeline.html>`_ or `scANVI (Xu et al., 2019 ) <https://docs.scarches.org/en/latest/scanvi_surgery_pipeline.html>`_.


- **Identify novel cell states present in your data by mapping to an atlas**: If you want to detect cell-states affected by disease or novel subpopulations see `treeArches (Michielsen*, Lotfollahi* et al., 2022) <https://docs.scarches.org/en/latest/treeArches_identifying_new_ct.html>`_ and also similar use case by mapping to `Human Lung cell atlas <https://docs.scarches.org/en/latest/hlca_map_classify.html>`_.


- **Multimodal single-cell atlases**: You can check the tutorial for `Multigrate (Litinetskaya*, Lotfollahi* et al., 2022) <https://docs.scarches.org/en/latest/multigrate.html>`_ to work with CITE-seq + Multiome (ATAC+ RNA). Additionally, you can check `mvTCR (Drost et al., 2022) <https://docs.scarches.org/en/latest/mvTCR_borcherding.html>`_ for joint analysis of T-cell Receptor (TCR) and scRNAseq data. To impute missing surface proteins for your query single-cell RNAseq data using a CITE-seq reference, see  `totalVI (Gayoso et al., 2019) <https://docs.scarches.org/en/latest/totalvi_surgery_pipeline.html>`_.


- **Data integration/batch correction**: For integration of multiple scRNAseq datasets see  `scVI (Lopez et al, 2018) <https://docs.scarches.org/en/latest/scvi_surgery_pipeline.html>`_ or `trVAE (Lotfollahi et al, 2020) <https://docs.scarches.org/en/latest/trvae_surgery_pipeline.html>`_. In case of strong batch effect and access to cell-type labels, consider using `scGen (Lotfollahi et al., 2019) <https://docs.scarches.org/en/latest/scgen_map_query.html>`_.


- **Spatial transcriptomics**: To map scRNAseq data to a spatial reference and infer spatial locations check `SageNet (Heidari et al., 2022) <https://docs.scarches.org/en/latest/SageNet_mouse_embryo.html>`_.


- **Querying gene programs in single-cell atlases**: Using gene programs (GPs), you can embed your datasets into known subspaces (e.g., interferon signaling) and see the activity of your query dataset within desired GPs. You can use available GP databases (e.g, GO pathways) or your curated GPs, see `expiMap (Lotfollahi*, Rybakov*  et al., 2023) <https://docs.scarches.org/en/latest/expimap_surgery_pipeline_basic.html>`_. One can also learn novel GPs as shown `here <https://docs.scarches.org/en/latest/expimap_surgery_pipeline_advanced.html>`_.


**Links to the papers** can be found `here <https://docs.scarches.org/en/latest/about.html>`_.


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
   training_tips

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   scvi_surgery_pipeline
   scanvi_surgery_pipeline
   totalvi_surgery_pipeline
   trvae_surgery_pipeline
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
   scpoli_surgery_pipeline
   hlca_map_classify
