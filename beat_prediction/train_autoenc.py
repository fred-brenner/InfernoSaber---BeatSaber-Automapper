import numpy as np
import os
from datetime import datetime
from tensorflow import keras
from keras.optimizers import adam_v2
# from keras.optimizers import Adam
# from tabulate import tabulate
from PIL import Image
import matplotlib.pyplot as plt

from training.helpers import *
from find_beats import *
from preprocessing.bs_mapper_pre import load_beat_data

from training.tensorflow_models import *
from beat_prediction.beat_to_lstm import *
from preprocessing.bs_mapper_pre import load_ml_data, lstm_shift
from tools.config import config, paths


def autoencoder_input_resample(ar_list):
    ar_all = None
    for ar in ar_list:
        if ar_all is None:
            ar_all = ar
        else:
            ar_all = np.hstack((ar_all, ar))
    return ar_all


# Setup configuration
#####################
# # Check Cuda compatible GPU
# if not test_gpu_tf():
#     exit()
#
# # Load pretrained model
# encoder_path = paths.model_path + config.enc_version
# encoder_model = keras.models.load_model(encoder_path)

# gather input
##############
name_ar, _ = filter_by_bps(config.min_bps_limit, config.max_bps_limit)
song_input, pitch_input = find_beats(name_ar, train_data=True)

# calculate discrete timings
pitch_times = []
n_x = song_input[0].shape[0]
for idx in range(len(pitch_input)):
    pitch_times.append(get_pitch_times(pitch_input[idx]))
    # resize song input to fit pitch algorithm
    im = Image.fromarray(song_input[idx])
    im = im.resize((len(pitch_input[idx]), n_x))
    song_input[idx] = np.asarray(im)
    # # test song input
    # plt.imshow(song_input[idx])
    # plt.show()

# load real beats
_, real_beats = load_beat_data(name_ar)

beat_resampled = samplerate_beats(real_beats, pitch_times)

# data generation
x_song = autoencoder_input_resample(song_input)
x_song = x_song.T
x_song = minmax_3d(x_song)
y_beat = autoencoder_input_resample(beat_resampled)

# create timestamp
dateTimeObj = datetime.now()
timestamp = f"{dateTimeObj.month}_{dateTimeObj.day}__{dateTimeObj.hour}_{dateTimeObj.minute}"
save_model_name = f"tf_model_autoenc_single_{config.bottleneck_len}bneck_{timestamp}.h5"
save_enc_name = f"tf_model_enc_single_{config.bottleneck_len}bneck_{timestamp}.h5"

# setup ML model
################
learning_rate = config.learning_rate
tcn_len = config.tcn_len
# load model
auto_encoder, save_model_name = load_keras_model(save_model_name)
# create model
if auto_encoder is None:
    encoder = create_music_model('enc1', dim_in, tcn_len)
    decoder = create_keras_model('dec1', dim_in, tcn_len)
    auto_input = Input(shape=(264, 1, 1))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto_encoder = Model(auto_input, decoded)

    adam = adam_v2.Adam(learning_rate=learning_rate, decay=learning_rate * 2 / config.n_epochs)
    auto_encoder.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    encoder.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

print(auto_encoder.summary())

x, y = beat_to_lstm(song_input, beat_resampled)
cw = calc_class_weight(y)

auto_encoder.fit(x=x, y=y, epochs=config.n_epochs, shuffle=False,
                 batch_size=config.batch_size, verbose=1, class_weight=cw)

# y_pred = auto_encoder.predict(x)
#
# fig = plt.figure()
# plt.plot(y, label='original')
# plt.plot(y_pred, label='prediction')
# plt.legend()
# plt.show()

print("")
