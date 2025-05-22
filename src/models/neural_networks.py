from torch import Tensor, no_grad
from torch.nn import Module, Sequential, Linear, ReLU, Conv2d, MaxPool2d, Flatten





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





class VAE(Module): 


    def __init__(
            self, 
            num_features: int
    ): 
        self.encoder = Sequential(
            Linear(num_features, 500), ReLU(),
            Linear(500, 100)
        )
        self.decoder = Sequential(
            Linear(100, 500), ReLU(),
            Linear(500, num_features)
        )
        return
    

    def forward(
            self, 
            X: Tensor
    ) -> Tensor: 
        return self.decoder(self.encoder(X))
