from datetime import datetime
from tensorflow import keras
from keras.optimizers import adam_v2
from tabulate import tabulate

from helpers import *
from tensorflow_models import *
from preprocessing.bs_mapper_pre import load_ml_data, lstm_shift
from tools.config import config, paths

# Check Cuda compatible GPU
if not test_gpu_tf():
    exit()

# Setup configuration
#####################
min_bps_limit = config.min_bps_limit
max_bps_limit = config.max_bps_limit
learning_rate = config.map_learning_rate
n_epochs = config.map_n_epochs
batch_size = config.map_batch_size
test_samples = config.map_test_samples
np.random.seed(3)

# Load pretrained model
encoder_path = paths.model_path + config.enc_version
encoder = keras.models.load_model(encoder_path)

# Data Preprocessing
####################

ml_input, ml_output = load_ml_data()
ml_input, ml_output = lstm_shift(ml_input[0], ml_input[1], ml_output)
[in_song, in_time_l, in_class_l] = ml_input

# apply autoencoder to input
in_song_l = encoder.predict(in_song)

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

# Create model
##############
# create timestamp
dateTimeObj = datetime.now()
timestamp = f"{dateTimeObj.month}_{dateTimeObj.day}__{dateTimeObj.hour}_{dateTimeObj.minute}"
save_model_name = f"tf_model_mapper_{timestamp}.h5"
# load model
mapper_model, save_model_name = load_keras_model(save_model_name)
# create model
if mapper_model is None:
    dim_in = [in_song_train[0].shape, in_time_train[0].shape, in_class_train[0].shape]
    dim_out = out_class_train.shape[1]
    mapper_model = create_keras_model('lstm1', dim_in, dim_out)
    adam = adam_v2.Adam(learning_rate=learning_rate, decay=learning_rate / n_epochs)
    mapper_model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

# Model training
################
#                normal         lstm          lstm
ds_train = [in_song_train, in_time_train, in_class_train]
ds_test = [in_song_test, in_time_test, in_class_test]

training = mapper_model.fit(x=ds_train, y=out_class_train,
                            epochs=n_epochs, batch_size=batch_size,
                            shuffle=False, verbose=1)

# Evaluate model
################
command_len = 10
print("Validate model with test data...")
validation = mapper_model.evaluate(x=ds_test, y=out_class_test)
pred_result = mapper_model.predict(x=ds_test)

pred_class = categorical_to_class(pred_result)
real_class = categorical_to_class(out_class_test)

if test_samples % command_len == 0:
    pred_class = pred_class.reshape(-1, command_len)
    real_class = real_class.reshape(-1, command_len)

print(tabulate([['Pred', pred_class], ['Real', real_class]],
               headers=['Type', 'Result (test data)']))

print("Validate model with train data...")
ds_train_sample = [in_song_train[:test_samples], in_time_train[:test_samples], in_class_train[:test_samples]]
validation = mapper_model.evaluate(x=ds_train_sample, y=out_class_train[:test_samples])

pred_result = mapper_model.predict(x=ds_train_sample)
pred_class = categorical_to_class(pred_result)
real_class = categorical_to_class(out_class_train[:test_samples])

if test_samples % command_len == 0:
    pred_class = pred_class.reshape(-1, command_len)
    real_class = real_class.reshape(-1, command_len)

print(tabulate([['Pred', pred_class], ['Real', real_class]],
               headers=['Type', 'Result (train data)']))

# Save Model
############
print(f"Saving model at: {paths.model_path}")
mapper_model.save(paths.model_path + save_model_name)

print("Finished Training")
