from datetime import datetime
from tensorflow import keras
from keras.optimizers import adam_v2
from tabulate import tabulate

from training.helpers import *
from training.tensorflow_models import *
from preprocessing.bs_mapper_pre import load_ml_data, lstm_shift
from tools.config import config, paths


# Setup configuration
#####################
# Check Cuda compatible GPU
if not test_gpu_tf():
    exit()

# Load pretrained model
encoder_path = paths.model_path + config.enc_version
encoder_model = keras.models.load_model(encoder_path)
mapper_path = paths.model_path + config.mapper_version
mapper_model = keras.models.load_model(mapper_path)

# Data Preprocessing
####################
ml_input, ml_output = load_ml_data()
ml_input, ml_output = lstm_shift(ml_input[0], ml_input[1], ml_output)
[in_song, in_time_l, in_class_l] = ml_input

# apply autoencoder to input
in_song_l = encoder_model.predict(in_song)

# load mapper model
mapper_model, save_model_name = load_keras_model(config.mapper_version)
if mapper_model is None:
    print("Could not find automapper savefile. Check <config.mapper_version>. Exit.")
    exit()

# Evaluate model
################
command_len = 10
test_samples = 30
lstm_len = config.lstm_len
val_total = []
pred_class = []

in_class_idx = in_class_l[0].reshape(1, lstm_len, -1)
for idx in range(len(in_song_l)):
    ml_input_idx = [in_song_l[idx].reshape(1, -1),
                    in_time_l[idx].reshape(1, lstm_len, -1),
                    in_class_idx]

    if idx > 0:
        in_class_idx

    ml_output_idx = ml_output[idx].reshape(1, -1)
    val_idx = mapper_model.evaluate(x=ml_input_idx, y=ml_output_idx)
    val_total.append[val_idx]

    if idx < test_samples:
        pred_idx = mapper_model.predict(x=ml_input_idx)
        pred_class.append(categorical_to_class(pred_idx))

pred_class = np.asarray(pred_total)
real_class = categorical_to_class(ml_output[:test_samples])

if test_samples % command_len == 0:
    pred_class = pred_class.reshape(-1, command_len)
    real_class = real_class.reshape(-1, command_len)

print(tabulate([['Pred', pred_class], ['Real', real_class]],
               headers=['Type', 'Result (test data)']))

print("Finished Evaluation")
