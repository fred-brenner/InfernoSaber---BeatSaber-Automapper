import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

from preprocessing.beat_data_helper import *
from tools.config import paths, config
from training.helpers import filter_by_bps
from training.get_song_repr import get_song_repr
from preprocessing.music_processing import run_music_preprocessing


# Setup configuration
np.random.seed(config.random_seed)
min_bps_limit = config.min_bps_limit
max_bps_limit = config.max_bps_limit


def lstm_shift(song_in, time_in, ml_out):
    n_samples = len(time_in)
    lstm_len = config.lstm_len
    start = lstm_len + 1

    # ml_out
    l_ml_out = ml_out[start:]
    l_out_in = []
    # time in
    l_time_in = []

    for idx in range(start, n_samples):
        l_out_in.append(ml_out[idx-start:idx-1])
        l_time_in.append(time_in[idx-start:idx-1])

    l_time_in, l_out_in = np.asarray(l_time_in), np.asarray(l_out_in)

    # song_in
    song_in = song_in[start:]

    return [song_in, l_time_in, l_out_in], l_ml_out


def load_beat_data(name_ar):
    print("Loading maps input data")
    map_dict_notes, _, _ = load_raw_beat_data(name_ar)
    notes_ar, time_ar = sort_beats_by_time(map_dict_notes)
    beat_class = cluster_notes_in_classes(notes_ar)

    return beat_class, time_ar


def load_ml_data():
    # get name array
    name_ar, diff_ar = filter_by_bps(min_bps_limit, max_bps_limit)

    # load beats (output)
    beat_ar, time_ar = load_beat_data(name_ar)

    # # load song (input)
    # song_ar = get_song_repr(name_ar)

    # load song (input)
    song_ar, rm_index = run_music_preprocessing(name_ar, time_ar, save_file=False, song_combined=False)

    # filter invalid indices
    idx = 0
    for rm_idx in rm_index:
        if len(rm_idx) > 0:
            # remove invalid songs
            name_ar.pop(idx)
            diff_ar.pop(idx)
            beat_ar.pop(idx)
            song_ar.pop(idx)
            time_ar.pop(idx)
        else:
            idx += 1

    # calculate time between
    timing_ar = calc_time_between_beats(time_ar)

    song_input = song_ar[0]
    time_input = np.asarray(timing_ar[0], dtype='float16')
    ml_output = np.asarray(beat_ar[0])

    for idx in range(1, len(song_ar)):
        song_input = np.vstack((song_input, song_ar[idx]))
        time_input = np.hstack((time_input, np.asarray(timing_ar[idx], dtype='float16')))
        ml_output = np.hstack((ml_output, np.asarray(beat_ar[idx])))

    # onehot encode output
    ml_output = ml_output.reshape(-1, 1)
    ml_output = onehot_encode(ml_output)
    ml_output = ml_output.toarray()

    return [song_input, time_input], ml_output


def onehot_encode(ml_output):
    encoder = OneHotEncoder(dtype=int)
    encoder.fit(ml_output)
    ml_output = encoder.transform(ml_output)

    # save onehot encoder
    with open(paths.beats_classify_encoder_file, "wb") as enc_file:
        pickle.dump(encoder, enc_file)
    # return ml data
    return ml_output


def calc_time_between_beats(time_ar):
    # default time for start
    dft_time = 1
    timing_input = []

    for song in time_ar:
        temp = []
        for idx in range(len(song)):
            if idx == 0:
                timing = dft_time
            else:
                timing = song[idx] - song[idx-1]
            temp.append(timing)
        timing_input.append(temp)
    return timing_input


if __name__ == '__main__':
    load_ml_data()
