# Beta-Variational-Autoencoder
Simple Beta VAE made mostly with GPT-4. The goal was to use it in scikit-learn to do dimension reduction.

## Usage
```
from bvae import ReducedBVAE
model = ReducedBVAE()
model.train_bvae(dataset, batch_size=batch_size)
projection = model.transform(datapoints)
```
