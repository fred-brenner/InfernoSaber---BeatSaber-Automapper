########################################
# config file for all important values 
# used in multiple codes
########################################

# Data Processing configuration
random_seed = 3
min_time_diff = 0.01        # minimum time between cuts, otherwise synchronized
samplerate_music = 14800    # samplerate for the music import
hop_size = 512
window = 2.0                # window in seconds for each song to spectrum picture (from wav_to_pic)
specgram_res = 24           # y resolution of the spectrogram (frequency subdivisions)

min_bps_limit = 8           # minimum beats_per_second value for training
max_bps_limit = 10          # maximum beats_per_second value for training

ram_limit = 20              # free ram roughly in GB

# Model versions
enc_version = 'tf_model_enc_16bneck_12_8__16_48.h5'
autoenc_version = 'tf_model_autoenc_16bneck_12_8__16_48.h5'
mapper_version = 'tf_model_mapper_8-10_1_20__15_10.h5'
beat_gen_version = 'tf_beat_gen_8_10_1_19__16_27.h5'
event_gen_version = 'tf_event_gen_8_10_1_19__23_49.h5'

# Autoencoder model configuration
learning_rate = 0.0003      # model learning rate
n_epochs = 50               # number of total epochs
batch_size = 128            # batch size
test_samples = 10           # number of test files to plot (excluded from training)
bottleneck_len = 16         # size of bottleneck distribution (1D array)

# Mapper model configuration
map_learning_rate = 6e-4       # model learning rate
map_n_epochs = 140             # number of total epochs
map_batch_size = 128           # batch size
map_test_samples = 10          # number of test files to plot (excluded from training)
lstm_len = 8

# Beat prediction model configuration
beat_learning_rate = 5e-4
beat_n_epochs = 80
beat_batch_size = 256
tcn_len = 24
tcn_test_samples = 600
delete_offbeats = 0.8      # < 1
# tcn_skip = 10

# Map creation model configuration
thresh_beat = 0.6           # minimum beat response required to trigger generator
cdf = 0.7                   # cut director factor (to calculate speed, ~0.5)
min_beat_time = 0.04        # in seconds (first sanity check)
beat_spacing = 5587/196     # 5587/196s = 28.5051 steps/s
max_speed = 7.0             # set around 3-12 (normal-expert+)
favor_last_class = 0.15     # set factor to favor the next beat class (0.0-0.3)
max_double_note_speed = 20  # set maximum speed difference between double notes (10 or 15 or 20)
emphasize_beats_wait = 0.2  # minimum time in seconds
emphasize_beats_3 = 0.08    # fraction beats to triple
emphasize_beats_2 = 0.3     # fraction beats to double
shift_beats_fact = 0.7      # fraction beats to shift in cut direction
add_beat_low_bound = 0.12   # in seconds (beat_generator)
add_beat_hi_bound = 0.60    # in seconds (beat_generator)
add_beat_fact = 0.78        # fraction add beats (beat_generator)

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