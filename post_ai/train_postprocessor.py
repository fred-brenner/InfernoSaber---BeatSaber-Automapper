from datetime import datetime
from tensorflow import keras
from keras.optimizers import adam_v2
from tabulate import tabulate

from training.helpers import *
from training.tensorflow_models import *
from preprocessing.bs_mapper_pre import load_beat_data, lstm_shift
from tools.config import config, paths
from map_creation.sanity_check import split_notes_rl
from post_ai.helpers import *

# Check Cuda compatible GPU
if not test_gpu_tf():
    exit()

# Data Preprocessing
####################
name_ar, _ = filter_by_bps(config.min_bps_limit, config.max_bps_limit)
notes_ar, time_ar = load_beat_data(name_ar, return_notes=True)
# reshape notes
ml_input_l = []
ml_input_r = []
ml_output_r = []
ml_input_b = []
for n in range(len(notes_ar)):
    notes_ar[n] = [list(x.reshape(-1)) for x in notes_ar[n]]
    # split notes
    [notes_r, notes_l, notes_b] = split_notes_rl(notes_ar[n])
    timings = np.concatenate((np.ones(1), np.diff(time_ar[n])))
    for n2 in range(len(notes_r)):
        for i2 in range(int(len(notes_r[n2]) / 4)):
            ml_input_r.append(notes_r[n2][4*i2+2:4*i2+4])
            ml_output_r.append(notes_r[n2][4*i2:4*i2+2])
            if i2 == 0:
                ml_input_r[-1].append(np.round(timings[n2], 3))
            else:
                ml_input_r[-1].append(0)

ml_input_r = np.asarray(ml_input_r)
ml_output_r = np.asarray(ml_output_r)

ml_input_r, ml_output_r = lstm_shift_post(ml_input_r, ml_output_r)



# Sample into train/val/test
############################
# last_test_samples = len(in_song_l) - test_samples
# # use last samples as test data
# in_song_test = in_song_l[last_test_samples:]
# in_time_test = in_time_l[last_test_samples:]
# in_class_test = in_class_l[last_test_samples:]
# out_class_test = ml_output[last_test_samples:]
#
# in_song_train = in_song_l[:last_test_samples]
# in_time_train = in_time_l[:last_test_samples]
# in_class_train = in_class_l[:last_test_samples]
# out_class_train = ml_output[:last_test_samples]

# Setup configuration
#####################
learning_rate = config.map_learning_rate
n_epochs = config.n_epochs_post
batch_size = config.map_batch_size
np.random.seed(3)

# Create model
##############
# create timestamp
dateTimeObj = datetime.now()
timestamp = f"{dateTimeObj.month}_{dateTimeObj.day}__{dateTimeObj.hour}_{dateTimeObj.minute}"
save_model_name = f"tf_model_postprocess_{timestamp}.h5"
# load model
mapper_model, save_model_name = load_keras_model(save_model_name)
# create model
if mapper_model is None:
    # dim_in = list(ml_input_r.shape)
    # dim_out = list(ml_output_r.shape)
    mapper_model = create_post_model('lstm1', config.lstm_len_post, dim_out=2)
    adam = adam_v2.Adam(learning_rate=learning_rate, decay=learning_rate / n_epochs)
    mapper_model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

# Model training
################
#                normal         lstm          lstm
# ds_train = [in_song_train, in_time_train, in_class_train]
# ds_test = [in_song_test, in_time_test, in_class_test]

training = mapper_model.fit(x=ml_input_r, y=ml_output_r,
                            epochs=n_epochs, batch_size=batch_size,
                            shuffle=False, verbose=1, validation_split=0.2)

# Evaluate model
################
eval_len = 10
print("Validate model with test data...")
validation = mapper_model.evaluate(x=ml_input_r[:eval_len],
                                   y=ml_output_r[:eval_len])
pred_result = mapper_model.predict(x=ml_input_r[:eval_len])

print(tabulate([['Pred', pred_result], ['Real', ml_output_r]],
               headers=['Type', 'Result (train data)']))
#
# # Save Model
# ############
# print(f"Saving model at: {paths.model_path}")
# mapper_model.save(paths.model_path + save_model_name)

print("Finished Training")
