import torch
from torch import nn

from tools.config import config


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
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # linear layers
            nn.Flatten(start_dim=1),
            nn.Dropout(0.1),
            nn.Linear(480, 128),
            nn.ReLU(),
            nn.Linear(128, config.bottleneck_len),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            # linear layers
            nn.Linear(config.bottleneck_len, 128),
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


# define automapper model
class AutoMapper(nn.Module):
    def __init__(self):
        super(AutoMapper, self).__init__()

        # setup variables
        input_dim = 1
        hidden_dim = 10
        n_lstm_layer = 32
        self.n_lstm_layer = n_lstm_layer
        self.hidden_dim = hidden_dim

        # song input layer

        # recurrent time lstm layer
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm_time = nn.LSTM(input_dim, hidden_dim, n_lstm_layer, batch_first=True)
        self.fc_time = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_lstm_layer, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.n_lstm_layer, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out
