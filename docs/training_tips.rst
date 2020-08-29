A few tips on training models
===================

- We recommend you to set `loss_fn` = `nb` or `zinb`. These loss functions require the access to count and not normalized data. You need to have normalized log-transformed data in `adata.X` and raw count data in `adata.raw.X`. You also need to have normalization factors for each cell in `adata.obs[scale_factors]`. These normalization factors can be obtained with `scanpy.pp.normalize_total <https://github.com/theislab/scarches/blob/master/requirements.txt>`_  or other normalization methods such as `scran <https://bioconductor.org/packages/devel/bioc/vignettes/scran/inst/doc/scran.html>`_.



- If you don't have access to count data and have normalized data then set `loss_fn` to `sse` or `mse`.



- If you want better separation of cell types you can increase the `n_epochs`. 100 epochs in most cases yield good quality but you can increase uo to 200. If some cell types are merged which should not be try to increase `n_epochs` and decrease `alpha` (see next tip).



- If you want to increase the mixing of the different batches then try to increase `alpha` when tou construct the the model. Maximum value of `alpha` can be 1. Increasing alpha will give you better mixing but it is a trade off! Increase `alpha` might also merge some small cell types or conditions. You can start with very small values (e.g 0.0001) and then increase that (0.001 ->0.005->0.01 and even 0.1 and finally 0.5).


- It is important to use highly variable genes for training. We recommend to use at least 2000 hvgs and if you have more complicated datasets, conditions then try to increase it to 5000 or so to include enough information for the model.

- Regarding `architecture` always try with the  default one ([128,128], `z_dimension`=10) and check the results. If you have more complicated data sets with many datasets and conditions and etc then you can increase the depth ([128,128,128] or [128,128,128,128]).  According to our experiments small values of `z_dimension` between  10 (default) and 20 are good.
