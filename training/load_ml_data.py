import numpy as np
from tools.config import config, paths
from tools.utils.load_and_save import load_npy


def load_ml_arrays():
    # load song input
    song_ar = load_npy(paths.ml_input_song_file)
    # load beat input
    beat_ar = load_npy(paths.ml_input_beat_file)

    len_diff = len(song_ar) - len(beat_ar)
    if len_diff < 0:
        print("Warning: Notes after song found.")
        exit()

    if len_diff > 0:
        # fill up beat array with zeros
        zero_ar = np.zeros((len_diff, 1))
        beat_ar = np.concatenate((beat_ar, zero_ar), axis=0)

        return song_ar, beat_ar


if __name__ == '__main__':
    song_ar, beat_ar = load_ml_arrays()
