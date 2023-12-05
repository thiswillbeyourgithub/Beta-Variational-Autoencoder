# Beta-Variational-Autoencoder
Simple Beta VAE made mostly with GPT-4. The goal was to use it in scikit-learn to do dimension reduction.

A wrapper called `OptimizedBVAE` can be used to do a grid search over the hidden_dim parameters then return the best model after further training.

The optimizer used is AdamW, but `VeLO` can be used from [this repo](https://github.com/janEbert/PyTorch-VeLO).

## Usage
```
from bvae import ReducedBVAE
model = ReducedBVAE(
    input_dim,
    z_dim,
    hidden_dim,
    dataset_size,
    lr=1e-3,
    epochs=1000,
    beta=1.0,
    weight_decay=0.01,
    use_VeLO=False,
    )
model.prepare_dataset(dataset, val_ratio=0.2, batch_size=500)
model.train_bvae(patience=100)
projection = model.transform(datapoints)
```
