########################################
# config file for all important values 
# used in multiple codes
########################################

# Data Processing configuration
random_seed = 3

min_time_diff = 0.01        # minimum time between cuts, otherwise synchronized

samplerate_music = 14800    # samplerate for the music import
window = 2.0                # window in seconds for each song to spectrum picture (from wav_to_pic)
# max_filter_size = 3       # maxpool filter for spectrogram preprocessing
specgram_res = 24           # y resolution of the spectrogram (frequency subdivisions)

min_bps_limit = 5           # minimum beats_per_second value for training
max_bps_limit = 5.1         # maximum beats_per_second value for training

# Pytorch model configuration
learning_rate = 0.004       # model learning rate
n_epochs = 300              # number of total epochs
epochs_per_input = 5        # number of stacked epochs
batch_size = 128             # batch size
test_samples = 10           # number of test files to plot (excluded from training)
bottleneck_len = 16         # size of bottleneck distribution (1D array)

lstm_len = 8
enc_version = 'tf_model_enc_16bneck_9_27__14_33.h5'

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