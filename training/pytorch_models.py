import torch
from torch import nn


# import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, dim, shape):
        super(View, self).__init__()
        self.dim = dim
        self.shape = shape

    def forward(self, input):
        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(input.shape)[self.dim + 1:]
        return input.view(*new_shape)


nn.Unflatten = View


# Define music autoencoder model
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # cnn layers
            nn.Conv2d(1, 32, 3, padding=1),  # 32: 5.28600556
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # float32[8,16,5,4]
            # linear layers
            nn.Flatten(start_dim=1),
            nn.Linear(480, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            # linear layers
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 480),
            nn.ReLU(),
            nn.Unflatten(1, (16, 6, 5)),
            # cnn layers
            nn.ConvTranspose2d(16, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
