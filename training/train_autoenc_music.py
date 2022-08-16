import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from pytorch_models import ConvAutoencoder
from torchviz import make_dot

from helpers import *
from preprocessing.music_processing import run_music_preprocessing
from tools.config import config, paths


check_cuda_device()

# Setup configuration
#####################
min_bps_limit = config.min_bps_limit
max_bps_limit = config.max_bps_limit
learning_rate = config.learning_rate
n_epochs = config.n_epochs
batch_size = config.batch_size
test_samples = config.test_samples
np.random.seed(3)

# Data Preprocessing
####################
# get name array
name_ar, diff_ar = filter_by_bps(min_bps_limit, max_bps_limit)
print(f"Importing {len(name_ar)} songs")

# load song input
song_ar = run_music_preprocessing(name_ar, save_file=False, song_combined=True)

# sample into train/val/test
test_loader = DataLoader(song_ar[:test_samples], batch_size=test_samples)
song_ar = song_ar[test_samples:]

# shuffle and split
np.random.shuffle(song_ar)
split = int(song_ar.shape[0] * 0.85)

# setup data loaders
train_loader = DataLoader(song_ar[:split], batch_size=batch_size)
val_loader = DataLoader(song_ar[split:], batch_size=batch_size)


# Model Building
################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Building model on device <{torch.cuda.get_device_name(0)}>")
# input_shape = song_ar.shape[1] * song_ar.shape[2]
model = ConvAutoencoder().to(device)

if True:
    # print model details
    print(model)

# visualize model details
img_batch = next(iter(train_loader)).to(device)
# yhat = model(img_batch)
# make_dot(yhat, params=dict(list(model.named_parameters()))).render("autoencoder_music", format="png")
input_names = [f'Image Input (batch_size={batch_size})']
output_names = ['Image Output']
save_path = paths.model_autoenc_music_file + '.onnx'
torch.onnx.export(model, img_batch, save_path, input_names=input_names, output_names=output_names)

# Loss function
criterion = nn.MSELoss()
# criterion = nn.BCELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Model Training
################
min_val_loss = np.inf
for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    # Training
    model.train()
    for images in train_loader:
        # images, _ = data
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader)
    # print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # Validation
    val_loss = calculate_loss_score(model, device, val_loader, criterion)
    print(f'Epoch {epoch} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss}')
    if min_val_loss > val_loss:
        print(f'Validation Loss Decreased({min_val_loss:.8f}->{val_loss:.8f}) \t Saving The Model')
        min_val_loss = val_loss
        # Saving State Dict
        save_path = paths.model_autoenc_music_file + '_saved_model.pth'
        torch.save(model.state_dict(), save_path)


# Model Evaluation
##################
run_plot_autoenc(model, device, test_loader, test_samples)


print("Finished Training")
