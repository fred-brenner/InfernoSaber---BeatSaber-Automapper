import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Input, LSTM, CuDNNLSTM, Flatten, Dropout, MaxPooling2D, Conv2D, BatchNormalization, SpatialDropout2D, concatenate
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
import os
import h5py
import glob
import pickle
import keras.backend as K
from keras.utils import to_categorical

from helpers import *
from tensorflow_models import *
from preprocessing.music_processing import run_music_preprocessing
from tools.config import config, paths


# Check Cuda compatible GPU
if not test_gpu_tf():
    exit()

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
song_ar, _ = run_music_preprocessing(name_ar, save_file=False, song_combined=True)

# sample into train/val/test
ds_test = song_ar[:test_samples]
song_ar = song_ar[test_samples:]

# shuffle and split
np.random.shuffle(song_ar)
split = int(song_ar.shape[0] * 0.85)

# setup data loaders
ds_train = song_ar[:split]
ds_val = song_ar[split:]


# Model Building
################
# create timestamp
dateTimeObj = datetime.now()
timestamp = 'm_' + str(dateTimeObj.month) + '_d_' + str(dateTimeObj.day) + '_h_' + str(dateTimeObj.hour) + '_min_' + str(dateTimeObj.minute)
save_model_name = 'keras_model_ep_' + timestamp + ".h5"
# save_model_name = "old"

# load model
model, save_model_name = load_keras_model(save_model_name)
# create model
if model is None:
    encoder = create_keras_model('enc1', learning_rate,
                                 ds_in=None, ds_out=None)
    decoder = create_keras_model('dec1', learning_rate,
                                 ds_in=None, ds_out=None)
    auto_input = Input(shape=(28, 28, 1))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto_encoder = Model(auto_input, decoded)

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
