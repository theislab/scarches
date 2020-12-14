import scanpy as sc
import torch
import os
import scarches as sca
from scarches.plotting import TRVAE_EVAL

save_path = os.path.expanduser(f'~/Documents/benchmark_results/trvae_umaps/testing/')
if not os.path.exists(save_path):
    os.makedirs(save_path)

results = dict()

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

condition_key = 'study'
size_factor_key = 'size_factors'
cell_type_key = 'cell_type'
show_plots = False
trvae_epochs = 200
surgery_epochs = 100

target_conditions = ["Pancreas SS2", "Pancreas CelSeq2"]
adata = sc.read(os.path.expanduser(f'~/Documents/datasets/pancreas_normalized.h5ad'))
source_adata = adata[~adata.obs[condition_key].isin(target_conditions)]
target_adata = adata[adata.obs[condition_key].isin(target_conditions)]
source_conditions = source_adata.obs[condition_key].unique().tolist()

network = sca.models.trVAE(
    source_adata.n_vars,
    conditions=source_conditions,
    hidden_layer_sizes=[256, 128, 32],
    latent_dim=10,
    dr_rate=0.05,
    use_mmd=True,
    recon_loss="zinb",
)

trainer = sca.trainers.trVAETrainer(
    network,
    source_adata,
    alpha_epoch_anneal=200,
    condition_key=condition_key,
    size_factor_key=size_factor_key,
    batch_size=1024,
    n_samples=4096
)

trainer.train(
    n_epochs=trvae_epochs,
    lr=0.001
)

trvae_eval = TRVAE_EVAL(
    model=network,
    adata=source_adata,
    trainer=trainer,
    condition_key=condition_key,
    cell_type_key=cell_type_key
)
results["reference_ebm"] = trvae_eval.get_ebm()
results["reference_knn"] = trvae_eval.get_knn_purity()
results["reference_asw_b"], results["reference_asw_c"] = trvae_eval.get_asw()
results["reference_nmi"] = trvae_eval.get_nmi()
trvae_eval.plot_history(show=show_plots, save=True, dir_path=f'{save_path}reference_history')
trvae_eval.plot_latent(show=show_plots, save=True, dir_path=f'{save_path}reference_latent')
trvae_eval.adata_latent.write_h5ad(filename=f'{save_path}reference_latent_data.h5ad')
torch.save(network.state_dict(), f'{save_path}reference_model')

# --------------------------------------------------TRANSFER LEARNING---------------------------------------------------
new_network, new_trainer = sca.trvae_operate(
    network,
    target_adata,
    condition_key=condition_key,
    size_factor_key=size_factor_key,
    n_epochs=surgery_epochs,
    freeze=True,
    freeze_expression=True,
    remove_dropout=False
)
surgery_eval = TRVAE_EVAL(
    model=new_network,
    adata=target_adata,
    trainer=new_trainer,
    condition_key=condition_key,
    cell_type_key=cell_type_key
)
results["query_ebm"] = surgery_eval.get_ebm()
results["query_knn"] = surgery_eval.get_knn_purity()
results["query_asw_b"], results["query_asw_c"] = surgery_eval.get_asw()
results["query_nmi"] = surgery_eval.get_nmi()
surgery_eval.plot_history(show=show_plots, save=True, dir_path=f'{save_path}surgery_history')
surgery_eval.plot_latent(show=show_plots, save=True, dir_path=f'{save_path}query_latent')
surgery_eval.adata_latent.write_h5ad(filename=f'{save_path}query_latent_data.h5ad')
torch.save(new_network.state_dict(), f'{save_path}surgery_model')

# -------------------------------------------------Plotting Whole Data--------------------------------------------------
full_eval = TRVAE_EVAL(
    model=new_network,
    adata=adata,
    condition_key=condition_key,
    cell_type_key=cell_type_key
)

full_eval.get_latent_score()
if show_plots:
    full_eval.plot_latent()
full_eval = TRVAE_EVAL(
    model=new_network,
    adata=adata,
    trainer=new_trainer,
    condition_key=condition_key,
    cell_type_key=cell_type_key
)
results["full_ebm"] = full_eval.get_ebm()
results["full_knn"] = full_eval.get_knn_purity()
results["full_asw_b"], results["full_asw_c"] = full_eval.get_asw()
results["full_nmi"] = full_eval.get_nmi()
full_eval.plot_latent(show=show_plots, save=True, dir_path=f'{save_path}full_latent')
full_eval.adata_latent.write_h5ad(filename=f'{save_path}full_latent_data.h5ad')
