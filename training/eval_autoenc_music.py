from helpers import *
from plot_model import run_plot_autoenc
from tensorflow_models import *
from preprocessing.music_processing import run_music_preprocessing
from tools.config import config, paths
from tools.config.mapper_selection import get_full_model_path

# Setup configuration
#####################
min_bps_limit = config.min_bps_limit
max_bps_limit = config.max_bps_limit
test_samples = config.test_samples
np.random.seed(3)

# Data Preprocessing
####################
# get name array
name_ar, _ = filter_by_bps(min_bps_limit, max_bps_limit)
name_ar = [name_ar[0]]
print(f"Importing {len(name_ar)} song")

# load song input
song_ar, _ = run_music_preprocessing(name_ar, save_file=False,
                                     song_combined=True, channels_last=True)

# sample into train/val/test
ds_test = song_ar[:test_samples]

# Model Building
################

auto_encoder, _ = load_keras_model(get_full_model_path(config.autoenc_version))

encoder, _ = load_keras_model(get_full_model_path(config.enc_version))

# create model
# if auto_encoder is None:
#     encoder = create_keras_model('enc1', learning_rate)
#     decoder = create_keras_model('dec1', learning_rate)
#     auto_input = Input(shape=(24, 20, 1))
#     encoded = encoder(auto_input)
#     decoded = decoder(encoded)
#     auto_encoder = Model(auto_input, decoded)
#
#     adam = Adam(learning_rate=learning_rate, decay=learning_rate/n_epochs)
#     auto_encoder.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
#     encoder.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])


# Model Evaluation
##################
print("\nEvaluating test data...")
eval = auto_encoder.evaluate(ds_test, ds_test)
# print(f"Test loss: {eval[0]:.4f}, test accuracy: {eval[1]:.4f}")

run_plot_autoenc(encoder, auto_encoder, ds_test, save=False)

print("\nFinished Evaluation")
