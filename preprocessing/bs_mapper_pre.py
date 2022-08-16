import numpy as np

from preprocessing.beat_data_helper import *
from tools.config import paths, config
from training.helpers import filter_by_bps
from training.get_song_repr import get_song_repr
from preprocessing.music_processing import run_music_preprocessing


# Setup configuration
np.random.seed(config.random_seed)
min_bps_limit = config.min_bps_limit
max_bps_limit = config.max_bps_limit


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
            beat_ar.pop(idx)
            song_ar.pop(idx)
            time_ar.pop(idx)
        else:
            idx += 1

    ml_input, ml_output = song_ar[0], np.asarray(beat_ar[0])
    for idx in range(1, len(song_ar)):
        ml_input = np.vstack((ml_input, song_ar[idx]))
        ml_output = np.hstack((ml_output, np.asarray(beat_ar[idx])))

    return ml_input, ml_output


if __name__ == '__main__':
    load_ml_data()
