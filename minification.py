from anndata import AnnData
from scipy.sparse import csr_matrix
import torch
from scipy import sparse
import numpy as np
import scanpy as sc
import scarches as sca



#should be a method of scPoli 
def get_latent(module, x, c=None, mean=False, mean_var=False):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
        data.
        Parameters
        ----------
        x:  torch.Tensor
             Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
        c: torch.Tensor
             Torch Tensor of condition labels for each sample.
        mean: boolean
        Returns
        -------
        Returns Torch Tensor containing latent space encoding of 'x'.
        """
        #compute latent representation
        x_ = torch.log(1 + x)
        if module.recon_loss == "mse":
            x_ = x
        if "encoder" in module.inject_condition:
            # c = c.type(torch.cuda.LongTensor)
            c = c.long()
            embed_c = torch.hstack([module.embeddings[i](c[:, i]) for i in range(c.shape[1])])
            z_mean, z_log_var = module.encoder(x_, embed_c)
        else:
            z_mean, z_log_var = module.encoder(x_)
        latent = module.sampling(z_mean, z_log_var)
        if mean:
            return z_mean
        elif mean_var:
            return (z_mean, z_log_var)
        return latent

#should be a method of scPoli 
def get_latent_representation(
        model,
        adata,
        mean: bool = False,
        mean_var: bool = False
    ):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
        data.

         Parameters
         ----------
         x
             Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
         c
             `numpy nd-array` of original (unencoded) desired labels for each sample.
         mean
             return mean instead of random sample from the latent space

        Returns
        -------
             Returns array containing latent space encoding of 'x'.
        """
        device = next(model.model.parameters()).device
        x = adata.X
        c = {k: adata.obs[k].values for k in model.condition_keys_}

        if isinstance(c, dict):
            label_tensor = []
            for cond in c.keys():
                query_conditions = c[cond]
                if not set(query_conditions).issubset(model.conditions_[cond]):
                    raise ValueError("Incorrect conditions")
                labels = np.zeros(query_conditions.shape[0])
                for condition, label in model.model.condition_encoders[cond].items():
                    labels[query_conditions == condition] = label
                label_tensor.append(labels)
            c = torch.tensor(label_tensor, device=device).T
        if sparse.issparse(x):
            x = x.toarray()
        x = torch.tensor(x, dtype=torch.float32)

        latents = []
        # batch the latent transformation process
        indices = torch.arange(x.size(0))
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            latent = get_latent(model.model,
                x[batch, :].to(device), c[batch, :].to(device), mean, mean_var
            )
            latent = (latent,) if not isinstance(latent, tuple) else latent
            latents += [tuple(l.cpu().detach() for l in latent)]

        result = tuple(torch.cat(l) for l in zip(*latents))
        result = result[0] if len(result) == 1 else result

        return result


def get_minified_adata_scrna(
    adata: AnnData,
) -> AnnData:
    """Returns a minified adata that works for most scrna models (such as SCVI, SCANVI).

    Parameters
    ----------
    adata
        Original adata, of which we to create a minified version.

    """

    all_zeros = csr_matrix(adata.X.shape)
    layers = {layer: all_zeros.copy() for layer in adata.layers}
    bdata = AnnData(
        X=all_zeros,
        layers=layers,
        uns=adata.uns.copy(),
        obs=adata.obs,
        var=adata.var,
        varm=adata.varm,
        obsm=adata.obsm,
        obsp=adata.obsp,
    )

    return bdata


     
def minify_adata(model, adata):

    """
    This function is adapted from scvi-tools 
    https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html#scvi.model.SCVI.minify_adata

    minify adata using latent posterior parameters:

    * the original count data is removed (`adata.X`, adata.raw, and any layers)
    * the parameters of the latent representation of the original data is stored
    * everything else is left untouched
    """

    #get the latent representation and store it in the adata
    qzm, qzv = model.get_latent_representation(adata, mean_var=True)

    adata.obsm["X_latent_qzm"] = qzm
    adata.obsm["X_latent_qzv"] = qzv

    #we cannot minify data where we do not use observed library size for gene count generation. 
    #In SCVI model, the library size can be modelled as a latent variable. However in scPoli it is set
    #to be observed (equal to the total UMI RNA count of a cell).


    minified_adata = get_minified_adata_scrna(adata)
    minified_adata.obsm["X_latent_qzm"] = adata.obsm["X_latent_qzm"]
    minified_adata.obsm["X_latent_qzv"] = adata.obsm["X_latent_qzv"]
    counts = adata.X
    minified_adata.obs["observed_lib_size"] = np.squeeze(
        np.asarray(counts.sum(axis=1))
    )

    #TODO: set is_minified attribute to True

    
    minified_adata.write("adata.h5ad")

def main():
     
    adata = sc.read("atlas_646ddf52fd46b85aafce28c2_data_not_minifiied.h5ad")

    model =sca.models.scPoli.load("model", adata)

    minify_adata(adata, model)

if __name__ == "__main__":
     main()
     

# import scarches as sca
# from scanpy.datasets import pbmc3k_processed, pbmc3k #replace with stored atlas trained on scPoli

# @pytest.mark.parametrize('get_adata', ["path/to/atlas1", "path/to/atlas2"])
# def test_minification(get_adata):
#      adata = get_adata()
#      model = scarches.models.scPoli.load(path = "path/to/model", adata=adata)

#      minify_adata(model, adata)
    
    




        




    



