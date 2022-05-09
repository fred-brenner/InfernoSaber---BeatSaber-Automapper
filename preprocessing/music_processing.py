import numpy as np
import aubio
from tools.utils.load_and_save import load_npy, save_npy, filter_max_bps
import tools.config.paths as paths
import tools.config.config as config

import matplotlib.pyplot as plt


def load_song(data_path: str) -> np.array:
    total_read = 0
    samples_list = []
    src = aubio.source(data_path, channels=1, samplerate=config.samplerate_music)
    while True:
        samples, read = src()
        samples_list.extend(samples)
        total_read += read
        if read < src.hop_size:
            break
    samples_ar = np.asarray(samples_list)

    return samples_ar


def process_song(song_ar: np.array) -> np.array:
    song_ar = np.abs(song_ar)

    # plt.figure()
    # plt.plot(song_ar)
    # plt.show()

    return song_ar


def main():
    # load song difficulty
    diff_ar = load_npy(paths.diff_ar_file)
    name_ar = load_npy(paths.name_ar_file)

    # filter by max song diff
    names = filter_max_bps(diff_ar, name_ar)

    # load song notes
    ending = ".egg"
    for n in names:
        song = load_song(paths.copy_path_song + n + ending)
        ml_input_song = process_song(song)
        break

    save_npy(ml_input_song, paths.ml_input_song_file)


if __name__ == '__main__':
    main()
    print("Finished")
