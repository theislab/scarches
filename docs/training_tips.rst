A few tips on training models
===================

trVAE
 - We recommend you to set `recon_loss` = `nb` or `zinb`. These loss functions require access to count data. You need to have raw count data in `adata.raw.X`.
  
 - If you don't have access to count data and have normalized log-transformed data then set `recon_loss` to  `mse`.

 - trVAE relies on an extra MMD term to force further integration of data sets. There is a parameter called `beta` (default=1) which regulates MMD effect in training. Higher values of `beta` will force extra mixing (might remove biological variation if too big!) while smaller values might result in less mixing (still batch effect). If you set   `beta` = `0` the model reduces to a Vanilla CVAE, but it is better to set 'use_mmd' to 'False' when MMD should not be used.

 - It is important to use highly variable genes for training. We recommend to use at least 2000 HVGs and if you have more complicated datasets, conditions then try  to increase it to 5000 or so to include enough information for the model.

 - Regarding `architecture` always try with the default one ([128,128], `z_dimension`=10) and check the results. If you have more complicated data sets with many datasets and conditions and etc then you can increase the depth ([128,128,128] or [128,128,128,128]).  According to our experiments, small values of `z_dimension` between  10 (default) and 20 are good.

scVI 
   - scVI require access to raw count data.
   - scVI already has a default good parameter the only thing you might change is `n_layers` which we suggest increasing to 2 (min) and max 4-5 for more
   complicated datasets.

   
scANVI 
  - It requires access to raw count data.
  - If you have query data the query data should be treated as unlabelled (Unknown) or have the same set of cell-types labels as reference. If you have a new cell-type label that is in the query data but not in reference and you want to use this in the training query you will get an error! We will fix this in future releases.






