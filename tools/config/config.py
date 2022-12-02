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
# max_filter_size = 3       # maxpool filter for spectrogram preprocessing
specgram_res = 24           # y resolution of the spectrogram (frequency subdivisions)

min_bps_limit = 5           # minimum beats_per_second value for training
max_bps_limit = 5.1         # maximum beats_per_second value for training

# Autoencoder model configuration
learning_rate = 0.004       # model learning rate
n_epochs = 300              # number of total epochs
# epochs_per_input = 5        # number of stacked epochs
batch_size = 128             # batch size
test_samples = 10           # number of test files to plot (excluded from training)
bottleneck_len = 16         # size of bottleneck distribution (1D array)

# Mapper model configuration
map_learning_rate = 0.005       # model learning rate
map_n_epochs = 300              # number of total epochs
map_batch_size = 128             # batch size
map_test_samples = 20           # number of test files to plot (excluded from training)
lstm_len = 8

enc_version = 'tf_model_enc_16bneck_9_27__14_33.h5'
mapper_version = 'tf_model_mapper_9_28__15_48.h5'
beat_gen_version = 'tf_beat_gen_5_5.1_11_29__12_15.h5'

# Beat prediction model configuration
beat_n_epochs = 10
tcn_len = 100
tcn_test_samples = 1000
# tcn_skip = 10

# Map creation model configuration
thresh_beat = 0.4           # minimum beat response required to trigger generator
cdf = 0.55                  # cut director factor (to calculate speed)
min_beat_time = 0.04        # in seconds (first sanity check)
beat_spacing = 28.505102    # 5587/196s = 28.5051 steps/s
max_speed = 15.0            # set around 5-15 (normal-expert+)
favor_last_class = 0.2      # set factor to favor the next beat class (0.0-0.3)
max_double_note_speed = 5   # set maximum speed difference between double notes (0-10)

# # Postprocesing configuration
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