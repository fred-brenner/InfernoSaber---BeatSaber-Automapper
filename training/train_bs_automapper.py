import gc
import numpy as np
from datetime import datetime
# from tensorflow import keras
from keras.optimizers import Adam
from tabulate import tabulate

from helpers import test_gpu_tf, ai_encode_song, load_keras_model, categorical_to_class
from tensorflow_models import create_keras_model
from preprocessing.bs_mapper_pre import load_ml_data, lstm_shift
from tools.config import config, paths
from lighting_prediction.train_lighting import lstm_shift_events_half


# # Check Cuda compatible GPU
# if not test_gpu_tf():
#     exit()

# Setup configuration
#####################
min_bps_limit = config.min_bps_limit
max_bps_limit = config.max_bps_limit
learning_rate = config.map_learning_rate
n_epochs = config.map_n_epochs
batch_size = config.map_batch_size
test_samples = config.map_test_samples
np.random.seed(3)

# Data Preprocessing
####################
ml_input, ml_output = load_ml_data()
# ml_input, ml_output = lstm_shift(ml_input[0], ml_input[1], ml_output)
# [in_song, in_time_l, in_class_l] = ml_input
# in_song_l = ai_encode_song(in_song)

in_song_l = ai_encode_song(ml_input[0])
ml_input, ml_output = lstm_shift_events_half(in_song_l, ml_input[1], ml_output, config.lstm_len)
[in_song_l, in_time_l, in_class_l] = ml_input

# Sample into train/val/test
############################
last_test_samples = len(in_song_l) - test_samples
# use last samples as test data
in_song_test = in_song_l[last_test_samples:]
in_time_test = in_time_l[last_test_samples:]
in_class_test = in_class_l[last_test_samples:]
out_class_test = ml_output[last_test_samples:]

in_song_train = in_song_l[:last_test_samples]
in_time_train = in_time_l[:last_test_samples]
in_class_train = in_class_l[:last_test_samples]
out_class_train = ml_output[:last_test_samples]

#                normal         lstm          lstm
ds_train = [in_song_train, in_time_train, in_class_train]
ds_test = [in_song_test, in_time_test, in_class_test]

ds_train_sample = [in_song_train[:test_samples], in_time_train[:test_samples], in_class_train[:test_samples]]

# dim_in = [in_song_train[0].shape, in_time_train[0].shape, in_class_train[0].shape]
# dim_out = out_class_train.shape[1]
dim_in = [x.shape for x in ds_train]
dim_out = out_class_train.shape

# delete variables to free ram
# keras.backend.clear_session()
# del encoder
del in_class_l
del in_class_test
# del in_song
del in_song_l
del in_song_test
del in_song_train
del in_time_l
del in_time_test
del in_time_train
del ml_input
del ml_output
gc.collect()


# Create model
##############
# create timestamp
dateTimeObj = datetime.now()
timestamp = f"{dateTimeObj.month}_{dateTimeObj.day}__{dateTimeObj.hour}_{dateTimeObj.minute}"
save_model_name = f"tf_model_mapper_{min_bps_limit}-{max_bps_limit}_{timestamp}.h5"
# load model
mapper_model, save_model_name = load_keras_model(save_model_name)
# create model
if mapper_model is None:
    mapper_model = create_keras_model('lstm_half', dim_in, dim_out)
    adam = Adam(learning_rate=learning_rate, weight_decay=2*learning_rate / n_epochs)
    # mapper_model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    mapper_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

mapper_model.summary()

# Model training
################
training = mapper_model.fit(x=ds_train, y=out_class_train,
                            epochs=n_epochs, batch_size=batch_size,
                            shuffle=True, verbose=1)

# Evaluate model
################
try:
    command_len = 10
    print("Validate model with test data...")
    validation = mapper_model.evaluate(x=ds_test, y=out_class_test)
    pred_result = mapper_model.predict(x=ds_test, verbose=0)

    pred_class = categorical_to_class(pred_result)
    real_class = categorical_to_class(out_class_test)

    pred_class = pred_class.flatten().tolist()
    real_class = real_class.flatten().tolist()

    print(tabulate([['Pred', pred_class], ['Real', real_class]], headers=['Type', 'Result (test data)']))

    print("Validate model with train data...")
    validation = mapper_model.evaluate(x=ds_train_sample, y=out_class_train[:test_samples])

    pred_result = mapper_model.predict(x=ds_train_sample, verbose=0)
    pred_class = categorical_to_class(pred_result)
    real_class = categorical_to_class(out_class_train[:test_samples])

    pred_class = pred_class.flatten().tolist()
    real_class = real_class.flatten().tolist()

    print(tabulate([['Pred', pred_class], ['Real', real_class]], headers=['Type', 'Result (train data)']))

except Exception as e:
    print(f"Error: {type(e).__name__}")
    print(f"Error message: {e}")
    print("Error in displaying mapper evaluation. Continue with saving.")

# Save Model
############
print(f"Saving model at: {paths.model_path + save_model_name}")
mapper_model.save(paths.model_path + save_model_name)

print("Finished Training")
