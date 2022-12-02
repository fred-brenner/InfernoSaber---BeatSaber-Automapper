import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib

from tools.config import config, paths


def update_out_class(in_class_l, y_class, idx):
    if y_class is None:
        return in_class_l

    last_class_lstm = in_class_l[idx-1]
    new_class_lstm = np.concatenate((last_class_lstm[1:], y_class), axis=0)
    in_class_l[idx] = new_class_lstm

    return in_class_l


def get_class_size():
    enc = joblib.load(paths.beats_classify_encoder_file)
    size = len(enc.categories_[0])
    return size


def cast_y_class(y_class):
    max_idx = np.argmax(y_class)
    y_class = np.zeros_like(y_class)
    y_class[0, max_idx] = 1

    return y_class


def decode_onehot_class(y_class_map):
    # test = np.argmax(y_class_map, axis=-1).reshape(-1)
    enc = joblib.load(paths.beats_classify_encoder_file)

    y = np.asarray(y_class_map).reshape((len(y_class_map), -1))
    y_class_num = enc.inverse_transform(y)

    return y_class_num


def add_favor_factor_next_class(y_class, y_class_last):
    if y_class_last is None:
        return y_class

    next_class = np.argmax(y_class_last) + 1
    if next_class < y_class_last.shape[-1]:
        y_class[:, next_class] += config.favor_last_class

    return y_class
