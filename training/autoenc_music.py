import numpy as np
import torch
from helpers import *
from preprocessing.music_processing import run_music_preprocessing


check_cuda_device()

# Setup configuration
#####################
min_bps_limit = 5
max_bps_limit = 5.1

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

# Model Building
################


# Model Training
################

# Model Evaluation
##################

# Model Saving
##############
