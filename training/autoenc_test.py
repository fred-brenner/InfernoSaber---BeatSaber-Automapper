import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_models import ConvAutoencoder

from helpers import *
from preprocessing.music_processing import run_music_preprocessing
from tools.config import config, paths


# Setup configuration
#####################
min_bps_limit = 5
max_bps_limit = 5.2
learning_rate = 0.005
n_epochs = 50
batch_size = 4
test_samples = 5
np.random.seed(3)


# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = paths.model_autoenc_music_file + '_saved_model.pth'

model = ConvAutoencoder()
model.load_state_dict(torch.load(save_path))
model = model.to(device)
model.eval()

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


# Model Evaluation
##################
# Batch of test images
dataiter = iter(test_loader)
images = dataiter.next()
images = images.to(device)

# Sample outputs
output = model(images)
repr_out = model.encoder(images)
images = images.cpu().numpy()
output = output.cpu().detach().numpy()
repr_out = repr_out.cpu().detach().numpy()

plot_autoenc_results(images, repr_out, output, test_samples)
