import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from helpers import *
from preprocessing.music_processing import run_music_preprocessing
from pytorch_models import ConvAutoencoder
import matplotlib.pyplot as plt

check_cuda_device()

# Setup configuration
#####################
min_bps_limit = 5
max_bps_limit = 5.1
learning_rate = 0.005
n_epochs = 60
batch_size = 4
test_samples = 5
np.random.seed(3)

# Data Preprocessing
####################
# get name array
name_ar, diff_ar = filter_by_bps(min_bps_limit, max_bps_limit)

# load song input
song_ar = run_music_preprocessing(name_ar, save_file=False, song_combined=True)

# scale song to 0-1
song_ar = np.asarray(song_ar)
song_ar = song_ar.clip(min=0)
song_ar /= song_ar.max()

# reshape image into 3D tensor
song_ar = song_ar.reshape((song_ar.shape[0], 1, song_ar.shape[1], song_ar.shape[2]))

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
print(model)

# Loss function
criterion = nn.MSELoss()

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
    val_loss = 0.0
    model.eval()
    for val_images in val_loader:
        val_images = val_images.to(device)
        val_output = model(val_images)
        loss = criterion(val_output, val_images)
        val_loss = loss.item() * val_images.size(0)

    val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss}')
    if min_val_loss > val_loss:
        print(f'Validation Loss Decreased({min_val_loss:.8f}--->{val_loss:.8f}) \t Saving The Model')
        min_val_loss = val_loss
        # Saving State Dict
        # torch.save(model.state_dict(), 'saved_model.pth')

# Model Evaluation
##################
# Batch of test images
dataiter = iter(test_loader)
images = dataiter.next()
images = images.to(device)

# Sample outputs
output = model(images)
images = images.cpu().numpy()

# output = output.view(batch_size, 3, 32, 32)
output = output.cpu().detach().numpy()

# Original Images
print("Original Images vs. Reconstruction")
fig, axes = plt.subplots(nrows=2, ncols=test_samples, sharex=True, sharey=True, figsize=(12, 4))
for idx in np.arange(test_samples):
    ax = fig.add_subplot(2, test_samples, idx + 1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
for idx in np.arange(test_samples):
    ax = fig.add_subplot(2, test_samples, idx + test_samples + 1, xticks=[], yticks=[])
    plt.imshow(np.transpose(output[idx], (1, 2, 0)))
plt.show()

# Model Saving
##############
