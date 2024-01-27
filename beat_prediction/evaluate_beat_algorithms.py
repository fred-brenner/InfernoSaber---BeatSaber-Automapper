import matplotlib.pyplot as plt
import sys, os

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

from tools.config import paths, config

# overwrite test path and config
################################
config.training_songs_diff = 'ExpertPlus'
config.allow_training_diff2 = True
config.training_songs_diff2 = 'Expert'

paths.bs_input_path = r'C:\Users\frede\Desktop\BS_Automapper\Data\training\evaluate_beats\bs_map_input' + '/'
paths.copy_path_map = r'C:\Users\frede\Desktop\BS_Automapper\Data\training\evaluate_beats\maps' + '/'
paths.dict_all_path = r'C:\Users\frede\Desktop\BS_Automapper\Data\training\evaluate_beats\maps_dict_all' + '/'
paths.copy_path_song = r'C:\Users\frede\Desktop\BS_Automapper\Data\training\evaluate_beats\songs_egg' + '/'
paths.diff_ar_file = r'C:\Users\frede\Desktop\BS_Automapper\Data\training\evaluate_beats\songs_diff' + "/diff_ar.npy"
paths.name_ar_file = r'C:\Users\frede\Desktop\BS_Automapper\Data\training\evaluate_beats\songs_diff' + "/name_ar.npy"

from beat_prediction.validate_find_beats import plot_beat_vs_real
from bs_shift.shift import shift_bs_songs, delete_old_files
from bs_shift.bps_find_songs import bps_find_songs
from bs_shift.map_to_dict_all import map_to_dict_all
from bs_shift.cleanup_n_format import clean_songs


# Shift stuff
#############
clean_songs()

delete_old_files()

try:
    shift_bs_songs()
except:
    print("Error while analyzing song")

# Start casting to dictionary (notes, events, etc)
map_to_dict_all()

# Calculate notes per sec for each song
bps_find_songs()


# Run analysis
##############


# Compare results
#################
