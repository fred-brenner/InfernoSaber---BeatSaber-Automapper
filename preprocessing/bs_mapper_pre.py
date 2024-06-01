import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

from map_creation.sanity_check import split_notes_rl
from preprocessing.beat_data_helper import (sort_beats_by_time, remove_duplicate_notes,
                                            cluster_notes_in_classes, load_raw_beat_data, get_pos_and_dir_from_notes)
from tools.config import paths, config
from tools.utils.numpy_shorts import reduce_number_of_songs
from training.helpers import filter_by_bps
from preprocessing.music_processing import run_music_preprocessing

# from tools.utils import numpy_shorts


# Setup configuration
np.random.seed(config.random_seed)
min_bps_limit = config.min_bps_limit
max_bps_limit = config.max_bps_limit


def lstm_shift(song_in, time_in, ml_out):
    n_samples = len(time_in)
    lstm_len = config.lstm_len
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
            l_out_in.append(ml_out[idx - start:idx - 1])
        l_time_in.append(time_in[idx - start:idx - 1])

    l_time_in = np.asarray(l_time_in).reshape((-1, lstm_len, 1))

    l_out_in = np.asarray(l_out_in)
    # l_out_in = l_out_in.reshape(l_out_in.shape[0], 1, lstm_len, -1)
    # song_in
    song_in = song_in[start:]

    return [song_in, l_time_in, l_out_in], l_ml_out


def load_beat_data_v2(name_ar: list):
    print("Loading maps input data")
    map_dict_notes, _, _ = load_raw_beat_data(name_ar)
    notes_ar, time_ar = sort_beats_by_time(map_dict_notes)

    notes_ar = remove_duplicate_notes(notes_ar)
    # get positions only
    beats_ar, _, note_ar = get_pos_and_dir_from_notes(notes_ar)
    beat_class = cluster_notes_in_classes(beats_ar)

    return beat_class, time_ar, note_ar


def load_beat_data(name_ar: list, return_notes=False):
    print("Loading maps input data")
    map_dict_notes, _, _ = load_raw_beat_data(name_ar)
    notes_ar, time_ar = sort_beats_by_time(map_dict_notes)
    if return_notes:
        return notes_ar, time_ar

    beat_class = cluster_notes_in_classes(notes_ar)
    return beat_class, time_ar


def get_arrow_dataset(beat_ar, time_ar):
    arrow_l = []
    arrow_r = []
    mask_l_full = []
    mask_r_full = []
    timing_ar = []
    for idx, notes in enumerate(beat_ar):
        notes_split = split_notes_rl(notes)
        arrow_r.append(notes_split[0])
        arrow_l.append(notes_split[1])

        mask_r = [1 if len(x) > 0 else 0 for x in arrow_r[-1]]
        mask_r_full.append(mask_r)
        time_selection = [time_ar[idx][x] for x in range(len(mask_r)) if mask_r[x] > 0]
        timing_ar.append(calc_time_between_beats([time_selection])[0])
        arrow_r[-1] = [arrow_r[-1][x] for x in range(len(mask_r)) if mask_r[x] > 0]
        arrow_r[-1] = np.asarray(arrow_r[-1])[:, 3]

        mask_l = [1 if len(x) > 0 else 0 for x in arrow_l[-1]]
        mask_l_full.append(mask_l)
        time_selection = [time_ar[idx][x] for x in range(len(mask_l)) if mask_l[x] > 0]
        timing_ar[-1].extend(calc_time_between_beats([time_selection])[0])
        arrow_l[-1] = [arrow_l[-1][x] for x in range(len(mask_l)) if mask_l[x] > 0]

        arrow_all = np.hstack([arrow_r[-1], np.asarray(arrow_l[-1])[:, 3]])
        # make sure that every arrow direction is recognized
        arrow_all = np.hstack([np.arange(0, 9), arrow_all])
        # onehot encode output
        arrow_all = arrow_all.reshape(-1, 1)
        arrow_enc = onehot_encode(arrow_all, paths.arrows_classify_encoder_file)
        arrow_all = arrow_enc.toarray()
        # delete added arrows
        arrow_r[-1] = arrow_all[9:]

    return [timing_ar, arrow_r, mask_r_full, mask_l_full]


def load_ml_data():
    # get name array
    name_ar, _ = filter_by_bps(min_bps_limit, max_bps_limit)

    # Reduce amount of songs
    name_ar = reduce_number_of_songs(name_ar, hard_limit=config.mapper_song_limit)

    # load beats (output)
    beat_class, time_ar, note_ar = load_beat_data_v2(name_ar)

    # load song (input)
    song_ar, rm_index = run_music_preprocessing(name_ar, time_ar, save_file=False,
                                                song_combined=False)

    # filter invalid indices
    idx = 0
    for rm_idx in rm_index:
        if len(rm_idx) > 0:
            # remove invalid songs
            name_ar.pop(idx)
            # diff_ar.pop(idx)
            beat_class.pop(idx)
            song_ar.pop(idx)
            time_ar.pop(idx)
            note_ar.pop(idx)
        else:
            idx += 1

    # calculate time between
    timing_ar = calc_time_between_beats(time_ar)

    song_input = song_ar[0]
    time_input = np.asarray(timing_ar[0], dtype='float16')
    ml_output_notes = np.asarray(beat_class[0])

    for idx in range(1, len(song_ar)):
        song_input = np.vstack((song_input, song_ar[idx]))
        time_input = np.hstack((time_input, np.asarray(timing_ar[idx], dtype='float16')))
        ml_output_notes = np.hstack((ml_output_notes, np.asarray(beat_class[idx])))

    # onehot encode output
    ml_output_notes = ml_output_notes.reshape(-1, 1)
    ml_output_notes = onehot_encode(ml_output_notes, paths.beats_classify_encoder_file)
    ml_output_notes = ml_output_notes.toarray()

    ml_arrow = get_arrow_dataset(note_ar, time_ar)

    return [song_input, time_input], ml_output_notes, ml_arrow


def onehot_encode(ml_output, save_path):
    encoder = OneHotEncoder(dtype=int)
    encoder.fit(ml_output)
    ml_output = encoder.transform(ml_output)

    # save onehot encoder
    with open(save_path, "wb") as enc_file:
        pickle.dump(encoder, enc_file)
    # return ml data
    return ml_output


def calc_time_between_beats(time_ar):
    # default time for start
    dft_time = 1
    timing_input = []

    for song in time_ar:
        temp = np.concatenate(([dft_time], np.diff(song)), axis=0)
        # temp = []
        # for idx in range(len(song)):
        #     if idx == 0:
        #         timing = dft_time
        #     else:
        #         timing = song[idx] - song[idx-1]
        #     temp.append(timing)
        timing_input.append(list(temp))
    return timing_input


if __name__ == '__main__':
    load_ml_data()
