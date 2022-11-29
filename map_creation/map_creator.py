import numpy as np
import pickle

from tools.config import config, paths


def create_map(y_class_num, timings):
    # load notes classify keys
    with open(paths.notes_classify_dict_file, 'rb') as f:
        class_keys = pickle.load(f)

    notes = decode_beats(y_class_num, class_keys)

    ###########
    # write map
    ###########
    notes
    timings


def decode_beats(y_class_num, class_keys):
    notes = []
    for idx in range(len(y_class_num)):
        y = int(y_class_num[idx])
        encoded = class_keys[y]
        notes.append(decode_class_keys(encoded))
    return notes


def decode_class_keys(encoded):
    # encoding:
    # beat = list(beat.reshape(-1))
    # beat_f = ""
    # for el in beat:
    #   beat_f += f"{el}"
    # return beat_f
    decoded = []
    for enc in encoded:
        decoded.append(int(enc))
    return decoded
