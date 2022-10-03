import numpy as np
import os
from datetime import datetime
from tensorflow import keras
from keras.optimizers import adam_v2
from tabulate import tabulate
from PIL import Image
import matplotlib.pyplot as plt

from training.helpers import *
from find_beats import *
from preprocessing.bs_mapper_pre import load_beat_data

from training.tensorflow_models import *
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

# setup ML model
################


print("")
