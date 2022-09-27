import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Model
from keras.optimizers import adam_v2
# from keras.callbacks import ReduceLROnPlateau
# from keras.models import load_model
# import os
# import h5py
# import glob
# import pickle
from datetime import datetime

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
# epochs_per_input = config.epochs_per_input
# n_epochs = int(n_epochs/epochs_per_input)
batch_size = config.batch_size
test_samples = config.test_samples
np.random.seed(3)

# Data Preprocessing
####################
# get name array
name_ar, diff_ar = filter_by_bps(min_bps_limit, max_bps_limit)
print(f"Importing {len(name_ar)} songs")

# load song input
song_ar, _ = run_music_preprocessing(name_ar, save_file=False,
                                     song_combined=True, channels_last=True)

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
timestamp = f"{dateTimeObj.month}_{dateTimeObj.day}__{dateTimeObj.hour}_{dateTimeObj.minute}"
save_model_name = f"tf_model_autoenc_{config.bottleneck_len}bneck_{timestamp}.h5"
save_enc_name = f"tf_model_enc_{config.bottleneck_len}bneck_{timestamp}.h5"
# save_model_name = "old"

# load model
auto_encoder, save_model_name = load_keras_model(save_model_name)
# create model
if auto_encoder is None:
    encoder = create_keras_model('enc1', learning_rate)
    decoder = create_keras_model('dec1', learning_rate)
    auto_input = Input(shape=(24, 20, 1))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto_encoder = Model(auto_input, decoded)

    adam = adam_v2.Adam(learning_rate=learning_rate, decay=learning_rate/n_epochs)
    auto_encoder.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    encoder.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

# Model Training
################
# min_val_loss = np.inf
# for epoch in range(1, n_epochs + 1):

# Training
training = auto_encoder.fit(x=ds_train, y=ds_train, validation_data=(ds_val, ds_val),
                            epochs=n_epochs, batch_size=batch_size,
                            shuffle=False, verbose=1)

# Model Evaluation
##################
print("\nEvaluating test data...")
eval = auto_encoder.evaluate(ds_test, ds_test)
# print(f"Test loss: {eval[0]:.4f}, test accuracy: {eval[1]:.4f}")

run_plot_autoenc(encoder, auto_encoder, ds_test)

# Save Model
############
print(f"Saving model at: {paths.model_path}")
auto_encoder.save(paths.model_path + save_model_name)
encoder.save(paths.model_path + save_enc_name)

print("\nFinished Training")
