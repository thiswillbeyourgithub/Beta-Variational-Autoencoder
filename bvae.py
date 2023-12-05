import os
import time
import torch
from torch import nn, optim
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from pytorch_velo import VeLO  # https://github.com/janEbert/PyTorch-VeLO

# whi and red are functions I used for printing. Can be ignored.
try:
    from .misc import whi, red
except Exception as err:
    try:
        from misc import whi, red
        print(f"Exception when loading from .misc: '{err}'")
    except:
        whi = print
        red = print


class ReducedBVAE(nn.Module):
    """A reduced Variational Autoencoder for dimensionality reduction."""
    def __init__(self, input_dim, z_dim, hidden_dim, dataset_size, lr=1e-3, epochs=1000, beta=1.0, weight_decay=0.0, use_VeLO=False):
        """
        Initialize the ReducedBVAE model with the specified parameters.

        :param input_dim: Dimensionality of the input data.
        :param z_dim: Dimensionality of the latent space.
        :param hidden_dim: Number of neurons in the hidden layer.
        :param dataset_size: number of points in the dataset
        :param beta: Scaling factor for KL divergence. Default is 1.0.
        :param epochs: Number of training epochs.
        :param use_VeLO: optimizer. If False will use AdamW
        :param lr: Learning rate for the AdamW optimizer.
        :param weight_decay: Weight decay for regularization in VeLO.
        """
        super(ReducedBVAE, self).__init__()
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_VeLO = use_VeLO
        self.beta = beta
        self.scaler = MinMaxScaler()

        # Model architecture
        mean_dim = (hidden_dim + z_dim) // 2
        self.to(self.device)
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, mean_dim).to(self.device)

        self.fc_mu = nn.Linear(mean_dim, z_dim).to(self.device)
        self.fc_std = nn.Linear(mean_dim, z_dim).to(self.device)
        self.fc3 = nn.Linear(z_dim, mean_dim).to(self.device)

        self.fc4 = nn.Linear(mean_dim, hidden_dim).to(self.device)
        self.fc5 = nn.Linear(hidden_dim, input_dim).to(self.device)

        if not self.use_VeLO:
            self.optimizer = optim.AdamW(self.parameters(), lr=lr)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=10,
                    verbose=False,
                    )
        else:
            self.optimizer = VeLO(self.parameters(), weight_decay=weight_decay, num_training_steps=epochs * dataset_size, device=self.device, seed=424242)

        self._dataset_loaded = False

    def encode(self, x):
        """
        Encode the input into two latent variables, mean and log variance.

        :param x: Input tensor to encode.
        :return: A tuple of two tensors, mean and log variance.
        """
        h = torch.relu(self.fc2(torch.relu(self.fc1(x))))
        return self.fc_mu(h), self.fc_std(h)

    def reparameterize(self, mu, logvar):
        """
        Reparameterize the mean and log variance to the latent variable z.

        :param mu: Tensor containing the means.
        :param logvar: Tensor containing the log variances.
        :return: Latent variable z.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        """
        Decode the latent variable z back to the reconstructed input.

        :param z: Latent variable to decode.
        :return: Reconstructed input tensor.
        """
        return torch.sigmoid(self.fc5(torch.relu(self.fc4(torch.relu(self.fc3(z))))))

    def forward(self, x):
        """
        Forward pass of the B-VAE model.

        :param x: Input tensor to process.
        :return: A tuple of three tensors, reconstructed input, mean, and log variance.
        """
        mu, logvar = self.encode(x.view(-1, len(x[0])))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, x, recon_x, mu, logvar):
        """
        Compute the B-VAE loss function.

        :param x: Original input tensor.
        :param recon_x: Reconstructed input tensor.
        :param mu: Tensor containing the means.
        :param logvar: Tensor containing the log variances.
        :return: Scalar loss value.
        """
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, len(x[0])), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + self.beta * KLD
        if loss < 0:
            red(f"Negative loss value: '{loss}'")
        return loss

    def prepare_dataset(self, dataset, val_ratio=0.2, batch_size=500):
        """
        Prepare the dataset for the model. This is a separate method from
        self.train() so that the dataset is consistently preprocessed, split
        and shuffled if the OptimizedBVAE wrapper is used.

        :param dataset: Dataset to train on.
        :param val_ratio: Ratio of the dataset to use for validation.
        :param batch_size: Batch size for training.
        """
        train_size = int((1 - val_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Fit and transform the scaler only on the training data. Convert to Tensor after scaling.
        self.scaler.fit([data.numpy() for data in train_dataset])
        train_dataset = torch.stack([torch.tensor(self.scaler.transform(data.numpy().reshape(1, -1))).float().squeeze(0) for data in train_dataset])
        val_dataset = torch.stack([torch.tensor(self.scaler.transform(data.numpy().reshape(1, -1))).float().squeeze(0) for data in val_dataset])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self._dataset_loaded = True

    def train(self, patience=100):
        """
        Train the B-VAE model with early stopping.

        :param patience: Number of epochs to wait for improvement before stopping. None to disable.
        """
        assert self._dataset_loaded, (
                "self.train can only be called after "
                "self.prepare_dataset was used")
        best_val_loss = float('inf')
        no_improvement = 0
        start = time.time()
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            for i, data in enumerate(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self(data)
                loss = self.loss_function(data, recon, mu, logvar)
                loss.backward()
                if self.use_VeLO:
                    self.optimizer.step(lambda: self.loss_function(data, *self(data)))
                else:
                    self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader.dataset)
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for data in self.val_loader:
                    data = data.to(self.device)
                    recon, mu, logvar = self(data)
                    loss = self.loss_function(data, recon, mu, logvar)
                    val_loss += loss.item()
            val_loss /= len(self.val_loader.dataset)
            whi(f'Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if patience and no_improvement >= patience:
                    whi(f'Early stopping at epoch {epoch} due to no improvement in validation loss.')
                    break
            if not self.use_VeLO:
                self.scheduler.step()
        whi(f"Training complete in {int(time.time()-start)}s")
        return best_val_loss

    def transform(self, x):
        """
        Transform the input data into the latent space representation.

        :param x: Input data to transform.
        :return: Latent space representation of the input data.
        """
        self.eval()
        with torch.no_grad():
            x = self.scaler.transform(x)
            x = torch.FloatTensor(x).to(self.device)
            mu, _ = self.encode(x)
            return mu.cpu().numpy()


class OptimizedBVAE(BaseEstimator):
    """A sklearn-compatible wrapper for the ReducedBVAE model."""
    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, dataset):
        """Fit the model according to the given training data."""
        hidden_dim_candidates = [int(len(dataset) * ratio) for ratio in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]]
        beta_candidates = [5]
        batch_size = dataset.shape[0]
        best_params = {}
        self.params['use_VeLO'] = False
        stored_loaders = []  # stores the dataset

        best_loss = float('inf')

        pbar = tqdm(
                total=len(hidden_dim_candidates) * len(beta_candidates),
                desc="Finding the best beta and hidden_dim",
                unit="training",
                colour="red")
        for hidden_dim in hidden_dim_candidates:
            for beta in beta_candidates:
                whi(f"Training with hidden_dim {hidden_dim}, beta {beta}, batch_size {batch_size}")
                self.params['hidden_dim'] = hidden_dim
                self.params['beta'] = beta

                model = ReducedBVAE(**self.params)

                # reusing the same splits
                if stored_loaders:
                    model.scaler, model.train_loader, model.val_loader, model._dataset_loaded = *stored_loaders
                else:
                    model.prepare_dataset(
                            dataset=dataset,
                            )
                    stored_loaders = [
                            model.scaler,
                            model.train_loader,
                            model.val_loader,
                            model._dataset_loaded,
                            ]

                loss = model.train(
                        patience=10,
                        )
                if loss < best_loss:
                    best_loss = loss
                    best_params["beta"] = beta
                    best_params["hidden_dim"] = hidden_dim
                pbar.update(1)
        pbar.close()

        self.params['hidden_dim'] = best_params["hidden_dim"]
        self.params['beta'] = best_params["beta"]
        self.params['use_VeLO'] = False
        # batch_size = max(1, dataset.shape[0] // 4)
        red(f"Real training with beta {best_params['beta']}, hidden_dim {best_params['hidden_dim']}/{len(dataset)}, batch_size {batch_size}. Best loss was {best_loss}")
        model = ReducedBVAE(**self.params)
        model.train_bvae(dataset, batch_size=batch_size, patience=None)

        return model
