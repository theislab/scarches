# Architecture Surgery

Tranfer learning using architecture surgery for scRNA-seq

## Working with Kipoi

In order to use your model 
1. Train your model with your own dataset 
3. Use the following script to add model to kipoi
```python
import surgeon
data_path = "./data/data.h5ad"
adata = surgeon.dl.read_h5ad(data_path)
condition_key = "condition"
train_adata, valid_adata = surgeon.utils.train_test_split(adata, 0.80)
conditions = adata.obs[condition_key].unique().tolist()
n_conditions = len(conditions)
condition_encoder = surgeon.tl.create_dictionary(conditions, [])
network = surgeon.archs.CVAE(x_dimension=train_adata.shape[1], 
                             z_dimension=10,
                             n_conditions=n_conditions,
                             lr=0.001,
                             alpha=0.001,
                             eta=1.0,
                             clip_value=3.0,
                             loss_fn='nb',
                             model_path="./models/CVAE/data_name/",
                             dropout_rate=0.2
                             )
                             
network.train(train_adata,
              valid_adata,
              condition_key,
              cell_type_key=None,
              le=condition_encoder,
              n_epochs=300,
              batch_size=32,
              early_stop_limit=20,
              lr_reducer=15,
              n_per_epoch=0,
              save=True,
)
ACCESS_TOKEN = "csN42Z7FXyNPwi8xOV9LDXtKI2cezQVjv6hoRrri5X6edFGhVrJEPiZQwBZG"
surgeon.kp.create_kipoi_model(
    model=network,
    model_name="surgeon",
    data_path=data_path,
    access_token=ACCESS_TOKEN,
)

