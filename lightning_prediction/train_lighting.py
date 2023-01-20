import numpy as np
import matplotlib.pyplot as plt
import gc
import pickle

from datetime import datetime
from keras.optimizers import adam_v2
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate

from lightning_prediction.tf_lighting import create_tf_model

from preprocessing.beat_data_helper import load_raw_beat_data
from preprocessing.music_processing import run_music_preprocessing

from tools.config import config, paths
from tools.utils import numpy_shorts

from training.helpers import *


def lstm_shift_events_half(song_in, time_in, ml_out):
    n_samples = len(time_in)
    lstm_len = config.event_lstm_len
    delete = n_samples % lstm_len
    start = lstm_len + 1

    # ml_out
    if ml_out is None:
        l_ml_out = None
        l_ml_in = []
    else:
        l_ml_in = ml_out[:-delete].reshape(-1, lstm_len, ml_out.shape[1])[:-1]
        l_ml_out = ml_out[:-delete].reshape(-1, lstm_len, ml_out.shape[1])[1:]
    # shape(samples, lstm, features)
    l_song_in = song_in[:-delete].reshape(-1, lstm_len, song_in.shape[1], 1)[:-1]
    l_time_in = time_in[:-delete].reshape(-1, lstm_len, 1)[:-1]

    return [l_song_in, l_time_in, l_ml_in], l_ml_out


def lstm_shift_events(song_in, time_in, ml_out):
    n_samples = len(time_in)
    lstm_len = config.event_lstm_len
    start = lstm_len + 1

    # ml_out
    if ml_out is None:
        l_ml_out = None
    else:
        l_ml_out = ml_out[start:]
    l_out_in = []
    # time in
    l_time_in = []

    for idx in range(start, n_samples):
        if ml_out is not None:
            l_out_in.append(ml_out[idx-start:idx-1])
        l_time_in.append(time_in[idx-start:idx-1])

    l_time_in = np.asarray(l_time_in).reshape((-1, lstm_len, 1))

    l_out_in = np.asarray(l_out_in)
    # l_out_in = l_out_in.reshape(l_out_in.shape[0], 1, lstm_len, -1)
    # song_in
    song_in = song_in[start:]

    return [song_in, l_time_in, l_out_in], l_ml_out


def onehot_encode_events(in_event):
    ml_input = [f'{ev[0]};{ev[1]}' for ev in in_event.astype(int)]
    ml_input = np.asarray(ml_input).reshape(-1, 1)
    encoder = OneHotEncoder(dtype=int)
    encoder.fit(ml_input)
    ml_output = encoder.transform(ml_input).toarray()

    # save onehot encoder
    with open(paths.events_classify_encoder_file, "wb") as enc_file:
        pickle.dump(encoder, enc_file)
    # return ml data
    return ml_output


def get_time_from_events(events, diff=False):
    time_ar = []
    rm_idx = []
    for idx, ev in enumerate(events):
        if not diff:
            if len(ev) == 0:
                rm_idx.append(idx)
                # time_ar.append([])
            else:
                time_ar.append(ev[0])
        else:
            temp = np.diff(ev[0])
            temp = np.concatenate(([1], temp), axis=0)
            time_ar.append(temp)

    rm_idx.reverse()
    return time_ar, rm_idx


def start_training():
    # Setup configuration
    #####################
    # Check Cuda compatible GPU
    if not test_gpu_tf():
        exit()

    # model name setup
    ##################
    # create timestamp
    dateTimeObj = datetime.now()
    timestamp = f"{dateTimeObj.month}_{dateTimeObj.day}__{dateTimeObj.hour}_{dateTimeObj.minute}"
    save_model_name = f"tf_event_gen_{config.min_bps_limit}_{config.max_bps_limit}_{timestamp}.h5"

    # gather input
    ##############
    print("Gather input data:", end=' ')

    name_ar, _ = filter_by_bps(config.min_bps_limit, config.max_bps_limit)
    # if len(name_ar) > int(config.ram_limit * 1.25):
    #     name_ar = name_ar[:int(config.ram_limit * 1.25)]
    #     print("Info: Loading reduced song number into generator to not overload the RAM")
    print(f"Importing {len(name_ar)} songs")

    # load map data
    _, events, _ = load_raw_beat_data(name_ar)  # time, type, value
    time_ar, rm_idx = get_time_from_events(events, diff=False)
    [name_ar.pop(rm) for rm in rm_idx]
    _, events, _ = load_raw_beat_data(name_ar)

    # load song data
    song_ar, rm_idx = run_music_preprocessing(name_ar, time_ar, save_file=False,
                                              song_combined=False)

    print("Reshape input for AI model...")
    # remove wrong time indices
    for idx in range(len(rm_idx)):
        if len(rm_idx[idx]) > 0:
            events[idx] = np.delete(events[idx], rm_idx[idx], axis=-1)

    time_ar, _ = get_time_from_events(events, diff=True)
    time_ar = np.concatenate(time_ar, axis=0)
    events = np.concatenate(events, axis=1)
    in_event = events[1:].T
    y_out = onehot_encode_events(in_event)

    # encode song data
    song_ar = np.concatenate(song_ar, axis=0)
    in_song = ai_encode_song(song_ar)

    # x_input, y_out = lstm_shift_events(in_song, time_ar, y_out)
    x_input, y_out = lstm_shift_events_half(in_song, time_ar, y_out)

    # only use song and time data, not class as input
    # x_input = x_input[:2]

    # [in_song_l, in_time_l, in_class_l] = x_input
    # input_song_enc, input_time_lstm, input_class_lstm
    x_input_shape = [x.shape for x in x_input]

    # setup ML model
    ################
    model = create_tf_model('lstm_half', x_input_shape, y_out.shape)
    adam = adam_v2.Adam(learning_rate=config.event_learning_rate,
                        decay=config.event_learning_rate * 2 / config.event_n_epochs)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=['accuracy'])

    print(model.summary())

    model.fit(x=x_input, y=y_out, epochs=config.event_n_epochs, shuffle=True,
              batch_size=config.event_batch_size, verbose=1)

    # save model
    ############
    print(f"Saving model at: {paths.model_path + save_model_name}")
    model.save(paths.model_path + save_model_name)

    # Evaluate model
    ################
    test_samples = 30
    command_len = 10
    x_test = [x[:test_samples] for x in x_input]
    y_test = y_out[:test_samples]
    print("Validate model...")
    validation = model.evaluate(x=x_test, y=y_test)
    pred_result = model.predict(x=x_test)

    pred_class = categorical_to_class(pred_result)
    real_class = categorical_to_class(y_test)

    if test_samples % command_len == 0:
        pred_class = pred_class.reshape(-1, command_len)
        real_class = real_class.reshape(-1, command_len)

    print(tabulate([['Pred', pred_class], ['Real', real_class]],
                   headers=['Type', 'Result (test data)']))

    print("Finished lighting generator training")


if __name__ == '__main__':
    start_training()
