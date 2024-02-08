import matplotlib.pyplot as plt
from itertools import product
import numpy as np
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
paths.songs_pred = paths.copy_path_song

from beat_prediction.validate_find_beats import plot_beat_vs_real
from bs_shift.shift import shift_bs_songs, delete_old_files
from bs_shift.bps_find_songs import bps_find_songs
from bs_shift.map_to_dict_all import map_to_dict_all
from bs_shift.cleanup_n_format import clean_songs
from training.helpers import filter_by_bps
from preprocessing.bs_mapper_pre import load_beat_data
from map_creation.gen_beats import main as gen_beats_main


# allowed time difference in seconds
tolerance = 0.05


def calculate_beat_accuracy(beat_pred, beat_real, tolerance=0.05):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for beat_time_pred in beat_pred:
        # Find the nearest beat in the ground truth
        closest_beat_real = min(beat_real, key=lambda x: abs(x - beat_time_pred))

        # Check if the difference is within the tolerance
        if abs(beat_time_pred - closest_beat_real) <= tolerance:
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(beat_real) - true_positives

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f_measure


if False:
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

# Import map data
#################
name_ar, _ = filter_by_bps(0.1, 50)
_, real_beats = load_beat_data(name_ar)
name_ar = [name_ar[0]]
real_beats = real_beats[0]

# Set tuning parameters
#######################
bool_options = [True]
# float_1_options = np.arange(60, 120, 10).tolist()
float_1_options = [0.22]
# float_2_options = np.arange(0, 0.51, 0.03).tolist()
float_2_options = [0.4]
# float_3_options = np.arange(0.15, 0.81, 0.05).tolist()
float_3_options = [0.4, 0.7, 1]
float_4_options = [0.3, 0.45, 0.6]
float_5_options = [1]
float_6_options = [None]
# int_1_options = list(range(1, 50))
int_1_options = [None]

# Set Tuning function
#####################
# Perform grid search
best_accuracy = 0.0
best_parameters = None

total_iterations = len(bool_options)
total_iterations *= len(float_1_options)
total_iterations *= len(float_2_options)
total_iterations *= len(float_3_options)
total_iterations *= len(float_4_options)
total_iterations *= len(float_5_options)
total_iterations *= len(float_6_options)
total_iterations *= len(int_1_options)

iteration = 0

# Run analysis
##############
for (bool_1,
     float_1,
     float_2,
     float_3,
     float_4,
     float_5,
     float_6,
     int_1) in product(bool_options,
                       float_1_options,
                       float_2_options,
                       float_3_options,
                       float_4_options,
                       float_5_options,
                       float_6_options,
                       int_1_options):
    iteration += 1
    print(f"Iteration {iteration} of {total_iterations}")
    # overwrite parameters
    config.add_silence_flag = bool_1
    config.add_beat_intensity_orig = 50
    config.silence_threshold_orig = float_1
    config.thresh_beat = float_2
    # config.map_filler_iters = int_1
    config.thresh_pitch = float_3
    config.factor_pitch_certainty = float_4
    config.factor_pitch_meanmax = float_5

    # Call your beat algorithm with the current parameter values and calculate accuracy
    beats_algo = gen_beats_main(name_ar, debug_beats=True)
    # Calculate accuracy
    precision, recall, f_measure = calculate_beat_accuracy(beats_algo, real_beats, tolerance)
    # accuracy = (precision + recall) / 2
    accuracy = f_measure

    # Update the best parameters if the current combination performs better
    parameters = (bool_1, float_1, float_2, float_3, float_4, float_5, float_6, int_1)
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        best_parameters = parameters
        print("Current Parameters:", best_parameters)
        print("Current Accuracy:", best_accuracy)
    else:
        print("Bad Parameters:", parameters)

print("Best Parameters:", best_parameters)
print("Best Accuracy:", best_accuracy)

if False:
    # Compare results
    #################
    precision, recall, f_measure = calculate_beat_accuracy(beats_algo, real_beats, tolerance=tolerance)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-measure: ", f_measure)

    plot_beat_vs_real(beats_algo, real_beats)
