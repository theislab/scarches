# Architecture Surgery

Tranfer learning using architecture surgery for scRNA-seq

## Working with Kipoi

In order to use your model 
1. Train your model with your own dataset 
3. Use the following script to add model to kipoi
```python
import surgeon
data_path = "./data/data.h5ad"
model_path = "./models/CVAE/cvae.h5"
ACCESS_TOKEN = "csN42Z7FXyNPwi8xOV9LDXtKI2cezQVjv6hoRrri5X6edFGhVrJEPiZQwBZG"
surgeon.kp.create_kipoi_model(
    model_name="surgeon",
    model_path=model_path,
    data_path=data_path,
    access_token=ACCESS_TOKEN,
)

```


