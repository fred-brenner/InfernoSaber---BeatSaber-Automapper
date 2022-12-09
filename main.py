###################################################
# This file is needed to find the working directory
###################################################
import os

from tools.config import paths
import map_creation.gen_beats as beat_generator


# MAP GENERATOR
###############
# TODO: iterate over songs
for song_name in os.listdir(paths.songs_pred):
    beat_generator.main(song_name)


# TRAINING
##########
# run bs_shift / shift.py
# run training / train_autoenc_music.py
# run training / train_bs_automapper.py
# run beat_prediction / ai_beat_gen.py
