###################################################
# This file is needed to find the working directory
###################################################
import os

from tools.config import paths
import map_creation.gen_beats as beat_generator


# ############################################################
# if fails, rerun train_bs_automapper with correct min/max_bps
# until training is started (then cancel)
##############################################################

# MAP GENERATOR
###############
print(f"Found {len(os.listdir(paths.songs_pred))} songs. Iterating...")
for song_name in os.listdir(paths.songs_pred):
    song_name = song_name[:-4]
    print(f"Analyzing song: {song_name}")
    beat_generator.main([song_name])

print("Finished map generator")

# TRAINING
##########
# run bs_shift / shift.py
# run training / train_autoenc_music.py
# run training / train_bs_automapper.py
# run beat_prediction / ai_beat_gen.py
