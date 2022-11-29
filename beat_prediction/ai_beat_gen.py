import numpy as np
import os
from datetime import datetime
from tensorflow import keras
from keras.optimizers import adam_v2
# from keras.optimizers import Adam
# from tabulate import tabulate
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from training.helpers import *
from find_beats import *
from preprocessing.bs_mapper_pre import load_beat_data

from training.tensorflow_models import *
from beat_prediction.beat_to_lstm import *
from beat_prediction.beat_prop import get_beat_prop
from preprocessing.bs_mapper_pre import load_ml_data, lstm_shift
from tools.config import config, paths

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
tcn_len = config.tcn_len

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

#
[x_beat_prop, x_onset] = get_beat_prop(song_input[0])

# load real beats
_, real_beats = load_beat_data(name_ar)

beat_resampled = samplerate_beats(real_beats, pitch_times)

x_song, y = beat_to_lstm(song_input, beat_resampled)

x_song = minmax_3d(x_song)
cw = calc_class_weight(y)

x_last_beats = last_beats_to_lstm(y)

# setup ML model
################
model = create_music_model('tcn', song_input[0].shape[0], tcn_len)
adam = adam_v2.Adam(learning_rate=config.learning_rate, decay=config.learning_rate * 2 / config.beat_n_epochs)
model.compile(loss='binary_crossentropy', optimizer=adam,
              metrics=['binary_accuracy', 'accuracy'])

print(model.summary())
# x_input = [x_song]
x_input = [x_song, x_beat_prop, x_onset]
model.fit(x=x_input, y=y, epochs=config.beat_n_epochs, shuffle=False,
          batch_size=config.batch_size, verbose=1, class_weight=cw)

y_pred = model.predict(x_input)
# bound prediction to 0 or 1
thresh = 0.6
y_pred[y_pred > thresh] = 1
y_pred[y_pred <= thresh] = 0

fig = plt.figure()
# plt.plot(y, 'b-', label='original')
y_count = np.arange(0, len(y), 1)
y_count = y * y_count

plt.vlines(y_count, ymin=-0.1, ymax=1.1, colors='k', label='original', linewidth=2)
plt.plot(y_pred, 'b-', label='prediction', linewidth=1)
plt.legend()
plt.show()

print("")
