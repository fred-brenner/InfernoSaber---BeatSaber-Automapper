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

# Model versions
enc_version = 'tf_model_enc_16bneck_12_8__16_48.h5'
autoenc_version = 'tf_model_autoenc_16bneck_12_8__16_48.h5'
mapper_version = 'tf_model_mapper_8-10_12_9__0_0.h5'
# beat_gen_version = 'tf_beat_gen_5_5.1_11_29__12_15.h5'
beat_gen_version = 'tf_beat_gen_9.5_10_12_9__1_6.h5'

# Autoencoder model configuration
learning_rate = 0.0003      # model learning rate
n_epochs = 50               # number of total epochs
batch_size = 128            # batch size
test_samples = 10           # number of test files to plot (excluded from training)
bottleneck_len = 16         # size of bottleneck distribution (1D array)

# Mapper model configuration
map_learning_rate = 0.005       # model learning rate
map_n_epochs = 120              # number of total epochs
map_batch_size = 128            # batch size
map_test_samples = 20           # number of test files to plot (excluded from training)
lstm_len = 8

# Beat prediction model configuration
beat_learning_rate = 0.003
beat_n_epochs = 20
tcn_len = 100
tcn_test_samples = 1000
# tcn_skip = 10

# Map creation model configuration
thresh_beat = 0.3           # minimum beat response required to trigger generator
cdf = 0.55                  # cut director factor (to calculate speed)
min_beat_time = 0.04        # in seconds (first sanity check)
beat_spacing = 28.505102    # 5587/196s = 28.5051 steps/s
max_speed = 6.0            # set around 3-15 (normal-expert+)
favor_last_class = 0.2      # set factor to favor the next beat class (0.0-0.3)
max_double_note_speed = 15  # set maximum speed difference between double notes (10 or 15 or 20)

# Postprocessing model configuration
lstm_len_post = 10
n_epochs_post = 10


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