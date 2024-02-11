# Run all training scripts for a new model
import os
import shutil
import sys
import subprocess

from tools.config import paths, config


from training.helpers import test_gpu_tf
# import map_creation.gen_beats as beat_generator
# from bs_shift.export_map import *
#

# Check Cuda compatible GPU
if not test_gpu_tf():
    exit()

print(f"use_mapper_selection value: {config.use_mapper_selection}")
print(f"use_bpm_selection value: {config.use_bpm_selection}")
if config.use_bpm_selection:
    print(f"with bpm limits [{config.min_bps_limit}, {config.max_bps_limit}]")
input("Adapted the mapper_selection and use_bpm_selection in the config file?\n"
      "Press enter to continue...")

# create folder if required
if not os.path.isdir(paths.model_path):
    print(f"Creating model folder: {config.use_mapper_selection}")
    os.makedirs(paths.model_path)

else:
    if len(os.listdir(paths.model_path)) > 1:
        print("Model folder already available. Exit manually to change folder in config.")
        input("Continue with same model folder?")

# TRAINING
##########
print("Which trainings do you want to start? Reply with y or n for each model.")
run_list = input("1. shift music | 2. music autoencoder | 3. song mapper | 4. beat generator | 5. lights generator | ")
if len(run_list) != 5:
    print("Wrong input format. Exit")
    exit()

# run bs_shift / shift.py
if run_list[0].lower() == 'y':
    print(f"Analyzing BS music files from folder: {paths.bs_input_path}")
    subprocess.call(['python3', './bs_shift/shift.py'])

# run training / train_autoenc_music.py
# os.system("training/train_autoenc_music.py")
if run_list[1].lower() == 'y':
    subprocess.call(['python3', './training/train_autoenc_music.py'])

# run training / train_bs_automapper.py
if run_list[2].lower() == 'y':
    subprocess.call(['python3', './training/train_bs_automapper.py'])

# run beat_prediction / ai_beat_gen.py
if run_list[3].lower() == 'y':
    subprocess.call(['python3', './beat_prediction/ai_beat_gen.py'])

# run lighting_prediction / train_lighting.py
if run_list[4].lower() == 'y':
    subprocess.call(['python3', './lighting_prediction/train_lighting.py'])
