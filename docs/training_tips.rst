A few tips on training models
===================

trVAE
 - We recommend you to set `recon_loss` = `nb` or `zinb`. These loss functions require access to count and not normalized data. You need to have normalized log-transformed data in `adata.X` and raw count data in `adata.raw.X`. You also need to have normalization factors for each cell in `adata.obs[scale_factors]`.        These normalization factors can be obtained with `scanpy.pp.normalize_total <https://github.com/theislab/scarches/blob/master/requirements.txt>`_  or other normalization methods such as `scran <https://bioconductor.org/packages/devel/bioc/vignettes/scran/inst/doc/scran.html>`_.
  
 - If you don't have access to count data and have normalized data then set `recon_loss` to  `mse`.

 - trVAE relies on an extra MMD term to force further integration of data sets. There is a parameter called `beta` (default=1) which regulates MMD effect in training. Higher values of `beta` will force extra mixing (might remove biological variation if too big!) while smaller values might result in less mixing (still batch effect). If you set   `beta` = `0` the model reduces to a Vanilla CVAE.

 - It is important to use highly variable genes for training. We recommend to use at least 2000 HVGs and if you have more complicated datasets, conditions then try  to increase it to 5000 or so to include enough information for the model.

 - Regarding `architecture` always try with the default one ([128,128], `z_dimension`=10) and check the results. If you have more complicated data sets with many datasets and conditions and etc then you can increase the depth ([128,128,128] or [128,128,128,128]).  According to our experiments, small values of `z_dimension` between  10 (default) and 20 are good.

scVI 
   - scVI require access to raw count data.
   - scVI already has a default good parameter the only thing you might change is `n_hidden` which we suggest increasing to 2 (min) and max 4-5 for more
   complicated datasets.

   
scANVI 
  - It requires access to raw count data.
  - If you have query data the query data should be treated as unlabelled (Unknown) or have the same set of cell-types labels as reference. If you have a new cell-type label that is in the query data but not in reference and you want to use this in the training query you will get an error! We will fix this in future releases.






