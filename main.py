###################################################
# This file is needed to find the working directory
###################################################
import os
import shutil

from tools.config import paths
import map_creation.gen_beats as beat_generator

import tensorflow as tf

# limit gpu ram usage
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


# MAP GENERATOR
###############
print(f"Found {len(os.listdir(paths.songs_pred))} songs. Iterating...")
song_list = os.listdir(paths.songs_pred)
for i, song_name in enumerate(song_list):
    song_name = song_name[:-4]
    print(f"Analyzing song: {song_name} ({i+1} of {len(song_list)})")
    beat_generator.main([song_name])
    shutil.make_archive(f'{paths.new_map_path}1234_{song_name}',
                        'zip', f'{paths.new_map_path}1234_{song_name}')

print("Finished map generator")

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
