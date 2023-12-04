# Beta-Variational-Autoencoder
Simple Beta VAE made mostly with GPT-4. The goal was to use it in scikit-learn to do dimension reduction.

A wrapper called `OptimizedBVAE` can be used to do a grid search over the hidden_dim parameters then return the best model after further training.

The optimizer used is AdamW, but `VeLO` can be used from [this repo](https://github.com/janEbert/PyTorch-VeLO).

## Usage
```
from bvae import ReducedBVAE
model = ReducedBVAE()
model.train_bvae(dataset, batch_size=batch_size)
projection = model.transform(datapoints)
```
