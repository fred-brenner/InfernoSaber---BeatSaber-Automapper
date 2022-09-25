import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_models import ConvAutoencoder, AutoMapper

from helpers import *
from preprocessing.bs_mapper_pre import load_ml_data, lstm_shift
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

# Load pretrained model
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = paths.model_autoenc_music_file + '_saved_model.pth'

pre_model = ConvAutoencoder()
print(f"Loading pretrained autoencoder model with bottleneck len: {config.bottleneck_len}")
pre_model.load_state_dict(torch.load(save_path))
pre_model = pre_model.to(device)

# Data Preprocessing
####################

ml_input, ml_output = load_ml_data()
ml_input, ml_output = lstm_shift(ml_input[0], ml_input[1], ml_output)
[in_song, in_time_l, in_class_l] = ml_input

# apply autoencoder to input
dl_song = DataLoader(in_song)
pre_model.eval()
in_song_l = []
for images in dl_song:
    images = images.to(device)
    output = pre_model.encoder(images).cpu().detach().numpy()
    in_song_l.append(output.reshape(-1))
in_song_l = np.asarray(in_song_l)

# sample into train/val/test
tds_test = TensorDataset(torch.tensor(in_song_l[:test_samples]),
                         torch.tensor(in_time_l[:test_samples]),
                         torch.tensor(in_class_l[:test_samples]),
                         torch.tensor(ml_output[:test_samples]))
tds_train = TensorDataset(torch.tensor(in_song_l[test_samples:]),
                          torch.tensor(in_time_l[test_samples:]),
                          torch.tensor(in_class_l[test_samples:]),
                          torch.tensor(ml_output[test_samples:]))

test_loader = DataLoader(tds_test, batch_size=test_samples)
train_loader = DataLoader(tds_train, batch_size=batch_size)

# setup complete model
save_path = paths.model_automap_file + '_saved_model.pth'
auto_model = AutoMapper()
auto_model = auto_model.to(device)
torch.save(auto_model.state_dict(), save_path)


# # Model Evaluation
# ##################
# # Calculate validation score
# val_loss = calculate_loss_score(model, device, val_loader, criterion)
# print(f"Validation loss: {val_loss:.6f}")
#
# # Calculate test score
# test_loss = calculate_loss_score(model, device, test_loader, criterion)
# print(f"Test loss: {test_loss:.6f}")
