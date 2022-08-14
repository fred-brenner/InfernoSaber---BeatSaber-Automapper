import torch
from torch import nn
# import torch.nn.functional as F


# Define music autoencoder model
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 20, 3, padding=1),     # 32: 5.28600556
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 5, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 20, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
