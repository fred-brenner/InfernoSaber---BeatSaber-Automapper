# Run all training scripts for a new model
# import os
# import shutil
# import sys

from tools.config import paths, config
import map_creation.gen_beats as beat_generator
from bs_shift.export_map import *

print(f"use_mapper_selection value: {config.use_mapper_selection}")
print(f"use_bpm_selection value: {config.use_bpm_selection}")
input("Adapted the mapper_selection and use_bpm_selection in the config file?\n"
      "Press enter to continue...")

# TRAINING
##########
# run bs_shift / shift.py
input("Did you run shift.py?")

# you can skip this step
# run training / train_autoenc_music.py
# os.system("training/train_autoenc_music.py")
# TODO: subprocess this bitch

# run training / train_bs_automapper.py
# os.system(f"{paths.main_path}training/train_bs_automapper.py")

# run beat_prediction / ai_beat_gen.py

# run lighting_prediction / train_lighting.py


