###################################################
# This file is needed to find the working directory
###################################################
import os
import shutil
import time
import sys

from tools.config import paths, config
import map_creation.gen_beats as beat_generator
from bs_shift.export_map import *

import tensorflow as tf


def main(diff: float, export_results_to_bs=True, quick_start=None):
    # change difficulty
    if diff is not None:
        config.max_speed = diff
        config.max_speed_orig = diff
    if quick_start is not None:
        config.quick_start = quick_start

    # limit gpu ram usage
    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=conf)
    tf.compat.v1.keras.backend.set_session(sess)

    # MAP GENERATOR
    ###############
    song_list = os.listdir(paths.songs_pred)
    song_list = check_music_files(song_list, paths.songs_pred)
    print(f"Found {len(song_list)} songs. Iterating...")
    if len(song_list) == 0:
        print("No songs found!")

    for i, song_name in enumerate(song_list):
        start_time = time.time()
        song_name = song_name[:-4]
        print(f"Analyzing song: {song_name} ({i + 1} of {len(song_list)})")
        beat_generator.main([song_name])
        end_time = time.time()
        print(f"Time needed: {end_time - start_time}s")

        # create zip archive for online viewer
        shutil.make_archive(f'{paths.new_map_path}{config.max_speed}_{song_name}',
                            'zip', f'{paths.new_map_path}1234_{song_name}')
        # export map to beat saber
        if export_results_to_bs:
            shutil_copy_maps(song_name)

    print("Finished map generator")


# ############################################################

# ############################################################
# if fails, rerun train_bs_automapper with correct min/max_bps
# until training is started (cancel after data import)
##############################################################

# TRAINING
##########
# run bs_shift / shift.py
# run training / train_autoenc_music.py
# run training / train_bs_automapper.py
# run beat_prediction / ai_beat_gen.py
# run lighting_prediction / train_lighting.py

if __name__ == "__main__":
    diff = os.environ.get('max_speed')
    if diff is not None:
        diff = float(diff)
        print(f"Set max speed to {diff}")
    else:
        print("Use default values")

    qs = os.environ.get('quick_start')
    if qs is not None:
        qs = float(qs)
        print(f"Set quick_start to {qs}")

    export_results_to_bs = True
    main(diff, export_results_to_bs, qs)
