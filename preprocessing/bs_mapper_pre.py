import numpy as np

from beat_data_helper import *
from tools.config import paths, config
from training.helpers import filter_by_bps
from training.get_song_repr import get_song_repr

# Setup configuration
np.random.seed(config.random_seed)
min_bps_limit = config.min_bps_limit
max_bps_limit = config.max_bps_limit


def load_beat_data(name_ar):
    print("Loading maps input data")
    map_dict_notes, _, _ = load_raw_beat_data(name_ar)
    notes_ar, time_ar = sort_beats_by_time(map_dict_notes)
    beat_class = cluster_notes_in_classes(notes_ar)

    return beat_class


def load_song_data(name_ar):
    # # load song input
    # song_ar = run_music_preprocessing(name_ar, save_file=False, song_combined=True)
    # # reshape image into 3D tensor
    # song_ar = song_ar.reshape((song_ar.shape[0], 1, song_ar.shape[1], song_ar.shape[2]))
    #
    # # sample into train/val/test
    # test_loader = DataLoader(song_ar[:test_samples], batch_size=test_samples)
    # song_ar = song_ar[test_samples:]
    #
    # # shuffle and split
    # np.random.shuffle(song_ar)
    # split = int(song_ar.shape[0] * 0.85)
    return 0


def load_ml_data():
    # get name array
    name_ar, diff_ar = filter_by_bps(min_bps_limit, max_bps_limit)

    # load beats (output)
    beat_ar = load_beat_data(name_ar)

    # load song (input)
    song_ar = get_song_repr(name_ar)

    ml_input, ml_output = song_ar, beat_ar

    # # shuffle and split
    # np.random.shuffle(beat_ar)
    # split = int(beat_ar.shape[0] * 0.85)

    return ml_input, ml_output


if __name__ == '__main__':
    load_ml_data()
