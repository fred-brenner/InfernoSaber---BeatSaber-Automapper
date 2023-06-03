import numpy as np
########################################
# config file for all important values 
# used in multiple codes
########################################

# Map creation model configuration
"""Do change"""
max_speed = 4 * 5.6         # set around 5-40 (normal-expert++)
add_beat_intensity = 110    # try to match bps by x% [80, 110]
expert_fact = 0.63          # expert plus to expert factor [0.6, 0.7]
create_expert_flag = True   # create second expert map
thresh_beat = 0.48          # minimum beat response required to trigger generator
thresh_pitch = 1.00         # minimum beat for pitch check (0.8,low-1.5,high)
threshold_end = 1.1         # factor for start and end threshold
random_note_map_factor = 0.3    # stick note map to random song/center (set to 0 to disable)
random_note_map_change = 3      # change frequency for center (1-5)
quick_start = 1.9           # map quick start mode (0 off, 1-3 on)
t_diff_bomb = 1.5           # minimum time between notes to add bomb
t_diff_bomb_react = 0.3     # minimum time between finished added bombs
allow_mismatch_flag = False     # if True, wrong turned notes won't be removed
flow_model_flag = True      # use improved direction flow
furious_lighting_flag = False   # increase frequency of light effects
normalize_song_flag = True  # normalize song volume
increase_volume_flag = True     # increase song volume (only used in combination with normalize flag)
audio_rms_goal = 0.50
allow_dot_notes = True      # if False, all notes must have a cut direction
jump_speed_offset = -0.4    # general offset for jump speed (range [-2, 2])
map_filler_iters = 10       # max iterations for map filler
add_dot_notes = 2           # add dot notes for fastest patterns in percent [0-10]
add_breaks_flag = True      # add breaks after strong patterns
silence_threshold = 0.17    # silence threshold [0.0, 0.3]
silence_thresh_hard = 0.2   # add fixed threshold to dynamic value [0-2]
add_silence_flag = True     # whether to apply silence threshold
emphasize_beats_flag = True     # emphasize beats into double notes
add_obstacle_flag = True    # add obstacles in free areas
obstacle_time_gap = [0.3, 0.7]     # time gap before [0.2-1] after [0.5-2]
obstacle_min_duration = 0.1  # minimum duration for each obstacle [0.1-2]
obstacle_max_count = 2      # maximum appearance count for obstacles
sporty_obstacles = False

"""Caution on changes"""
obstacle_crouch_width = 4
if not sporty_obstacles:
    obstacle_allowed_types = [0, 1]     # 0wall, 1ceiling, 2jump, 3onesaber
    obstacle_positions = [[0], [3]]     # outside position of notes
else:
    obstacle_allowed_types = [0, 1]    # (ceiling walls for crouch are fixed)
    obstacle_positions = [[0, 1, 1], [2, 2, 3]]     # inside position of notes
obstacle_width = 1
check_silence_flag = True   # check for extremely silent songs
check_silence_value = -14.2  # value in dB [-13 to -15]
jsb_offset = [0.21, 0.15]   # note jump speed offset for Expert, Expert+ (range [-0.5, 0.5])
jsb_offset_factor = 0.011   # note jump factor for high difficulties
use_fixed_bpm = 100         # use fixed bpm or set to None for the song bpm
max_njs = 24.5              # maximum Note Jump Speed allowed
decr_speed_range = 20       # range for start and end (n first and last notes)
decr_speed_val = 0.28       # decrease max speed at start
reaction_time = 1.1         # reaction time (0.5-2)
reaction_time_fact = 0.013  # factor including max_speed
jump_speed = 12.5           # jump speed from beat saber (10-15)
jump_speed_fact = 0.310     # factor including max_speed
cdf = 1.2                   # cut director factor (to calculate speed, ~0.5)
min_beat_time = 1/16        # in seconds (first sanity check)
beat_spacing = 5587/196     # 5587/196s = 28.5051 steps/s
# favor_last_class = 0.15     # set factor to favor the next beat class (0.0-0.3)
max_double_note_speed = 25  # set maximum speed difference between double notes (10-30)
emphasize_beats_3 = 0.040   # fraction beats to triple
emphasize_beats_3_fact = 0.001   # factor incl max_speed
emphasize_beats_2 = 0.68    # fraction beats to double
emphasize_beats_2_fact = 0.002   # factor incl max_speed
emphasize_beats_quantile = 0.8
shift_beats_fact = 0.30     # fraction beats to shift in cut direction
add_beat_fact = 0.90        # fraction add beats (beat_generator)
add_beat_max_bounds = [0.1, 0.5, 0.8, 1.6]
pitches_allowed = [40, 50]  # percentage of pitches to be over threshold


"""Do not change"""
# Data Processing configuration
general_diff = 'ExpertPlus'
random_seed = 3
min_time_diff = 0.01        # minimum time between cuts, otherwise synchronized
samplerate_music = 14800    # samplerate for the music import
hop_size = 512
window = 2.0                # window in seconds for each song to spectrum picture (from wav_to_pic)
specgram_res = 24           # y resolution of the spectrogram (frequency subdivisions)

min_bps_limit = 7           # minimum beats_per_second value for training
max_bps_limit = 10          # maximum beats_per_second value for training

# Model versions
enc_version = 'tf_model_enc_16bneck_12_8__16_48.h5'
autoenc_version = 'tf_model_autoenc_16bneck_12_8__16_48.h5'
# mapper_version = 'tf_model_mapper_5-10_1_21__13_26.h5'
# beat_gen_version = 'tf_beat_gen_7.5_10_1_21__16_27.h5'
# event_gen_version = 'tf_event_gen_7.5_10_1_21__16_6.h5'
mapper_version = 'tf_model_mapper_7-10_1_29__19_34.h5'
beat_gen_version = 'tf_beat_gen_7_10_1_29__19_39.h5'
event_gen_version = 'tf_event_gen_7_10_1_29__19_44.h5'

# Autoencoder model configuration
learning_rate = 3e-4        # model learning rate
n_epochs = 50               # number of total epochs
batch_size = 128            # batch size
test_samples = 10           # number of test files to plot (excluded from training)
bottleneck_len = 16         # size of bottleneck distribution (1D array)

# Mapper model configuration
map_learning_rate = 4e-4    # model learning rate
map_n_epochs = 180          # number of total epochs
map_batch_size = 128        # batch size
map_test_samples = 10       # number of test files to plot (excluded from training)
lstm_len = 16

# Beat prediction model configuration
beat_learning_rate = 5e-4
beat_n_epochs = 80
beat_batch_size = 128
tcn_len = 24
tcn_test_samples = 350
delete_offbeats = 0.6      # < 1 delete non-beats to free ram
# tcn_skip = 10

# Event prediction model configuration
event_learning_rate = 1e-3
event_n_epochs = 180
event_lstm_len = 16
event_batch_size = 128

# needed for reset
max_speed_orig = max_speed
add_beat_intensity_orig = add_beat_intensity
silence_threshold_orig = silence_threshold
jump_speed_offset_orig = jump_speed_offset
obstacle_time_gap = np.asarray(obstacle_time_gap)
obstacle_time_gap_orig = obstacle_time_gap
