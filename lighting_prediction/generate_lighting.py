# import numpy as np
# import matplotlib.pyplot as plt
# import gc
# import pickle

# from datetime import datetime
# from keras.optimizers import adam_v2
# from sklearn.preprocessing import OneHotEncoder

# from lighting_prediction.tf_lighting import create_tf_model
from lighting_prediction.train_lighting import lstm_shift_events_half
from map_creation.class_helpers import get_class_size, update_out_class, add_favor_factor_next_class, \
    cast_y_class, decode_onehot_class

# from preprocessing.music_processing import run_music_preprocessing

# from tools.config import config, paths
# from tools.utils import numpy_shorts

from training.helpers import *


def decode_class_string(y_class_num):
    y_class = np.zeros((len(y_class_num), 2))
    for idx in range(len(y_class_num)):
        y_class[idx] = [y_class_num[idx][0].split(';')][0]
    return y_class


def generate(l_in_song, time_ar, save_model_name, lstm_len, encoder_file):
    class_size = get_class_size(encoder_file)
    # gather input
    ##############
    # some timings may have been removed in sanity check
    l_in_song = l_in_song[:len(time_ar)]

    time_diff = np.concatenate(([1], np.diff(time_ar)), axis=0)

    x_input, _ = lstm_shift_events_half(l_in_song, time_diff, None, lstm_len)
    [in_song_l, in_time_l, _] = x_input
    # x_input = x_input[:2]

    # setup ML model
    ################
    model, _ = load_keras_model(save_model_name)
    if model is None:
        print(f"Error. Could not load model {save_model_name}")

    """Model light"""
    # y_class = model.predict(x_input)
    # # cast to 0 and 1
    # y_arg_max = np.argmax(y_class, axis=1)
    # y_class_map = np.zeros(y_class.shape, dtype=int)
    # for idx in range(len(y_arg_max)):
    #     y_class_map[idx][y_arg_max[idx]] = 1

    """Model full"""
    # # apply event model
    # ###################
    # y_class = None
    # y_class_map = []
    # y_class_last = None
    # class_size = get_class_size(paths.events_classify_encoder_file)
    # for idx in range(len(in_song_l)):
    #     if y_class is None:
    #         in_class_l = np.zeros((len(in_song_l), config.event_lstm_len, class_size))
    #
    #     in_class_l = update_out_class(in_class_l, y_class, idx)
    #
    #     #             normal      lstm       lstm
    #     ds_train = [in_song_l[idx:idx + 1], in_time_l[idx:idx + 1], in_class_l[idx:idx + 1]]
    #     y_class = model.predict(x=ds_train)
    #
    #     # add factor to NEXT class
    #     y_class = add_favor_factor_next_class(y_class, y_class_last)
    #
    #     # find class winner
    #     y_class = cast_y_class(y_class)
    #
    #     y_class_last = y_class.copy()
    #     y_class_map.append(y_class)

    """Model half"""
    # apply note/event model
    ###################
    y_class = None
    rd_counter = 0
    rd_distribution = None
    y_class_map = np.zeros((in_time_l.shape[0], in_time_l.shape[1], class_size), dtype=int)
    for idx in range(len(in_song_l)):
        if y_class is None:
            in_class_l = np.zeros((len(in_song_l), lstm_len, class_size))
        else:
            in_class_l[idx] = y_class_map[idx - 1]

        #             normal      lstm       lstm
        ds_train = [in_song_l[idx:idx + 1], in_time_l[idx:idx + 1], in_class_l[idx:idx + 1]]
        y_class = model.predict(x=ds_train, verbose=0)

        y_class, rd_distribution, rd_counter = apply_random_mapper(y_class, rd_distribution, rd_counter)

        # TODO: add favor_bombs flag
        # find class winner
        y_arg_max = np.argmax(y_class, axis=2)[0]
        for imax in range(len(y_arg_max)):
            y_class_map[idx, imax][y_arg_max[imax]] = 1

    # decode event class output
    y_class_map = y_class_map.reshape(-1, y_class_map.shape[2])
    y_class_num = decode_onehot_class(y_class_map, encoder_file)
    if encoder_file == paths.events_classify_encoder_file:
        y_class_num = decode_class_string(y_class_num)
        # print("Finished lighting generator")
    # events_out = np.concatenate((time_ar[config.event_lstm_len+1:].reshape(-1, 1), y_class_num), axis=1)
    else:
        # print("Finished mapping generator")
        pass
    return y_class_num


def apply_random_mapper(y_class, rd_distribution, rd_counter):
    # Warning: Prediction is not stable enough
    # May lead to random resampling of ml output

    # initiate random map center
    c_window = 60
    c_val = config.random_note_map_factor
    if c_val == 0:
        return y_class, rd_distribution, rd_counter

    # scale batch to [0, 1]
    y_class_sc = y_class / np.max(y_class)

    if rd_counter <= 0:
        # initialization
        rd_distribution = np.random.rand(y_class.shape[0], y_class.shape[1], y_class.shape[2]) * c_val
        center = int(np.random.rand(1)[0] * y_class.shape[2])
        # check center is in bounds
        if center < c_window:
            center = c_window
        elif center > y_class.shape[2] - c_window - 1:
            center = y_class.shape[2] - c_window - 1
        center_start = center - c_window
        center_end = center + c_window
        # shift emphasis towards center
        rd_distribution[:, :, center_start:center_end] += c_val
        rd_distribution += 1
        rd_counter = config.random_note_map_change

    rd_counter -= 1

    y_class = y_class_sc * rd_distribution

    return y_class, rd_distribution, rd_counter


if __name__ == '__main__':
    # generate()
    pass
