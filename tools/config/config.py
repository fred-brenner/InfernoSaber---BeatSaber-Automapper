########################################
# config file for all important values 
# used in multiple codes
########################################

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

ram_limit = 20              # free ram roughly in GB

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

# Map creation model configuration
"""Do change"""
max_speed = 4 * 7.5         # set around 5-40 (normal-expert++)
add_beat_intensity = 86    # try to match bps by x%
expert_fact = 0.64          # expert plus to expert factor
create_expert_flag = True   # create second expert map
thresh_beat = 0.45          # minimum beat response required to trigger generator
thresh_pitch = 0.39         # minimum beat for pitch check (0.01,low-1,high)
threshold_end = 1.4         # factor for start and end threshold
random_note_map_factor = 0.3    # stick note map to random song/center (set to 0 to disable)
random_note_map_change = 2      # change frequency for center (1-20)
quick_start = 1.9           # map quick start mode (0 off, 1-3 on)
t_diff_bomb = 1.5           # minimum time between notes to add bomb
t_diff_bomb_react = 0.3     # minimum time between finished added bombs
allow_mismatch_flag = False     # if True, wrong turned notes won't be removed
flow_model_flag = True      # use improved direction flow
furious_lighting_flag = False   # increase frequency of light effects
normalize_song_flag = True  # normalize song volume

"""Caution on changes"""
decr_speed_range = 20       # range for start and end (n first and last notes)
decr_speed_val = 0.28       # decrease max speed at start
reaction_time = 1.1         # reaction time (0.5-2)
reaction_time_fact = 0.013  # factor including max_speed
jump_speed = 15             # jump speed from beat saber (15-22)
jump_speed_fact = 0.215     # factor including max_speed
cdf = 1.2                   # cut director factor (to calculate speed, ~0.5)
min_beat_time = 1/16        # in seconds (first sanity check)
beat_spacing = 5587/196     # 5587/196s = 28.5051 steps/s
# favor_last_class = 0.15     # set factor to favor the next beat class (0.0-0.3)
max_double_note_speed = 25  # set maximum speed difference between double notes (10-30)
emphasize_beats_wait = 0.2  # minimum time in seconds
emphasize_beats_3 = 0.023   # fraction beats to triple
emphasize_beats_3_fact = 0.004   # factor incl max_speed
emphasize_beats_2 = 0.23    # fraction beats to double
emphasize_beats_2_fact = 0.0085  # factor incl max_speed
shift_beats_fact = 0.30     # fraction beats to shift in cut direction
add_beat_low_bound = 0.20   # in seconds (beat_generator)
add_beat_hi_bound = 0.90    # in seconds (beat_generator)
add_beat_fact = 0.90        # fraction add beats (beat_generator)
add_beat_max_bounds = [0.1, 0.5, 0.8, 1.6]

# Postprocessing model configuration
lstm_len_post = 10
n_epochs_post = 10

# Event prediction model configuration
event_learning_rate = 1e-3
event_n_epochs = 180
event_lstm_len = 16
event_batch_size = 128

# # cutout_window = 0.1       # window in seconds for cutout
# # limit_mem = 10            # in GigaByte (free ram | needs about double memory for FAST saving at the end)
# # rec_t = 64                # recurrent timesteps for lstm deep learning model
# max_pause_sec = 20          # max pause allowed for second ML Model in seconds
# min_pause_sec = 0.03        # min pause allowed between beats
# max_velocity = 25           # max speed for cutting notes for expert+, others are lower (still depending on max velocity), 30 means ca max 10 notes (with low distance) per sec per hand!

# bps_borders = 0.7     #set range for bps values to be classified within following variables
# # 0.15xsong_bps is added onto the border value automatically!
# bps_normal = 4
# bps_hard = 6
# bps_expert = 8
# bps_expertplus = 10

max_speed_orig = max_speed      # needed for reset
