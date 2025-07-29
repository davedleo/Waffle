from torch import Tensor, exp, randn_like
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, ReLU, Conv1d, AdaptiveMaxPool1d, Dropout, Conv2d, MaxPool2d, Flatten





class LeNet5(Module): 

    def __init__(self, in_channels: int = 1, in_padding: int = 2, num_classes: int = 10): 
        super(LeNet5, self).__init__()

        self._fmap = Sequential(
            Conv2d(in_channels = in_channels, out_channels = 6, kernel_size = 5, stride = 1, padding = in_padding),
            MaxPool2d(kernel_size = 2, stride = 2), 
            ReLU(),
            Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1),
            MaxPool2d(kernel_size = 2, stride = 2), 
            ReLU()
        )

        self._mlp = Sequential(
            Flatten(),
            Linear(400, 128), 
            ReLU(),
            Linear(128, 84),
            ReLU(),
            Linear(84, num_classes)
        )


    def forward(self, X: Tensor) -> Tensor: 
        X = self._fmap(X)
        y = self._mlp(X)
        return y
    




class NewsCNNClassifier(Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.conv1 = Conv1d(in_channels=50, out_channels=128, kernel_size=5, padding=2)
        self.conv2 = Conv1d(128, 128, kernel_size=5, padding=2)
        self.dropout = Dropout(0.3)
        self.pool = AdaptiveMaxPool1d(1)
        self.fc = Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))     
        x = F.relu(self.conv2(x))     
        x = self.pool(x).squeeze(-1)   
        x = self.dropout(x)
        return self.fc(x)





class VAE(Module): 
    def __init__(self, num_features: int, latent_dim: int = 100): 
        super().__init__()
        
        # Encoder maps input -> hidden -> latent mean & logvar
        self.encoder = Sequential(
            Linear(num_features, 500), ReLU()
        )
        self.fc_mu = Linear(500, latent_dim)
        self.fc_logvar = Linear(500, latent_dim)

        # Decoder maps latent -> hidden -> output
        self.decoder = Sequential(
            Linear(latent_dim, 500), ReLU(),
            Linear(500, num_features)
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = exp(0.5 * logvar)
        eps = randn_like(std)
        return mu + eps * std  # z ~ N(mu, sigma^2)

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.encoder(X)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
