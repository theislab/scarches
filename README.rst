.. raw:: html

 <img src="https://user-images.githubusercontent.com/33202701/187203672-e0415eec-1278-4b2a-a097-5bb8b6ab694f.svg" width="300px" height="200px" align="center">

|PyPI| |PyPIDownloads| |Docs|


Single-cell architecture surgery (scArches) is a package for reference-based analysis of single-cell data.


What is scArches?
-------------------------------
scArches allows your single-cell query data to be analyzed by integrating it into a reference atlas. By mapping your data into an integrated reference you can transfer cell-type annotation from reference to query, identify disease states by mapping to healthy atlas, and advanced applications such as imputing missing data modalities or spatial locations.


Usage and installation
-------------------------------
See `here <https://scarches.readthedocs.io/>`_ for documentation and tutorials.

Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an `issue <https://github.com/theislab/scarches/issues/new>`__ or reach us by `email <mo.lotfollahi@gmail.com>`_.

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
