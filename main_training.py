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
input("Adapted the mapper_selection and use_bpm_selection in the config file?\n"
      "Press enter to continue...")

# TRAINING
##########
# run bs_shift / shift.py
input("Did you run shift.py?")

print("Which trainings do you want to start? Reply with y or n for each model.")
run_list = input("1. music autoencoder | 2. song mapper | 3. beat generator | 4. lights generator | ")
if len(run_list) != 4:
    print("Wrong input format. Exit")
    exit()
# create folder if required
if not os.path.isdir(paths.model_path):
    print(f"Creating model folder: {config.use_mapper_selection}")
    os.makedirs(paths.model_path)

else:
    if len(os.listdir(paths.model_path)) > 1:
        print("Model folder already available. Exit manually to change folder in config.")
        input("Continue with same model folder?")

# you can skip this step
# run training / train_autoenc_music.py
# os.system("training/train_autoenc_music.py")
if run_list[0].lower() == 'y':
    subprocess.call(['python', './training/train_autoenc_music.py'])

# run training / train_bs_automapper.py
if run_list[1].lower() == 'y':
    subprocess.call(['python', './training/train_bs_automapper.py'])

# run beat_prediction / ai_beat_gen.py
if run_list[2].lower() == 'y':
    subprocess.call(['python', './beat_prediction/ai_beat_gen.py'])

# run lighting_prediction / train_lighting.py
if run_list[3].lower() == 'y':
    subprocess.call(['python', './lighting_prediction/train_lighting.py'])
