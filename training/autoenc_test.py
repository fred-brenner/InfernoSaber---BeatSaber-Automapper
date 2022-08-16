import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_models import ConvAutoencoder

from helpers import *
from preprocessing.music_processing import run_music_preprocessing
from tools.config import config, paths


# Setup configuration
#####################
min_bps_limit = config.min_bps_limit
max_bps_limit = config.max_bps_limit
learning_rate = config.learning_rate
batch_size = config.batch_size
test_samples = config.test_samples
np.random.seed(3)
criterion = nn.MSELoss()


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
# Calculate validation score
val_loss = calculate_loss_score(model, device, val_loader, criterion)
print(f"Validation loss: {val_loss:.6f}")

# Calculate test score
test_loss = calculate_loss_score(model, device, test_loader, criterion)
print(f"Test loss: {test_loss:.6f}")

# Plot first batch of test images
run_plot_autoenc(model, device, test_loader, test_samples)
