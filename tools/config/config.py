########################################
# config file for all important values 
# used in multiple codes
########################################

# Data Processing configuration
max_song_bps = 8            # max average beats per second in song
# beat_resolution = 0.002     # in seconds
samplerate_music = 14800     # samplerate for the music import

window = 2.0                # window in seconds for each song to spectrum picture (from wav_to_pic)
# max_filter_size = 3       # maxpool filter for spectrogram preprocessing
specgram_res = 24           # y resolution of the spectrogram
# cutout_window = 0.1       # window in seconds for cutout
# limit_mem = 10            # in GigaByte (free ram | needs about double memory for FAST saving at the end)
# rec_t = 64                # recurrent timesteps for lstm deep learning model
max_pause_sec = 20          # max pause allowed for second ML Model in seconds
min_pause_sec = 0.03        # min pause allowed between beats
max_velocity = 25           # max speed for cutting notes for expert+, others are lower (still depending on max velocity), 30 means ca max 10 notes (with low distance) per sec per hand!

# Training configuration
batch_size = 32         # batch_size for ML models
learning_rate = 0.01
model_name = 'model'

# bps_borders = 0.7     #set range for bps values to be classified within following variables
# # 0.15xsong_bps is added onto the border value automatically!
# bps_normal = 4
# bps_hard = 6
# bps_expert = 8
# bps_expertplus = 10