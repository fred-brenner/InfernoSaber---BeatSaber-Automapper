import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_models import ConvAutoencoder

from helpers import *
from preprocessing.bs_mapper_pre import load_ml_data
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

auto_model = ConvAutoencoder()
auto_model.load_state_dict(torch.load(save_path))
auto_model = auto_model.to(device)

# Data Preprocessing
####################

ml_input, ml_output = load_ml_data()
tds_test = TensorDataset(torch.tensor(ml_input[:test_samples]), torch.tensor(ml_output[:test_samples]))
tds_train = TensorDataset(torch.tensor(ml_input[test_samples:]), torch.tensor(ml_output[test_samples:]))

# sample into train/val/test
test_loader = DataLoader(tds_test, batch_size=test_samples)
train_loader = DataLoader(tds_train, batch_size=batch_size)

# shuffle and split
split = int(song_ar.shape[0] * 0.85)

# setup data loaders
train_loader = DataLoader(song_ar[:split], batch_size=batch_size, num_workers=8, pin_memory=True)
val_loader = DataLoader(song_ar[split:], batch_size=batch_size, num_workers=8, pin_memory=True)



# # Model Evaluation
# ##################
# # Calculate validation score
# val_loss = calculate_loss_score(model, device, val_loader, criterion)
# print(f"Validation loss: {val_loss:.6f}")
#
# # Calculate test score
# test_loss = calculate_loss_score(model, device, test_loader, criterion)
# print(f"Test loss: {test_loss:.6f}")
