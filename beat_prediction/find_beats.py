import os
import aubio
import madmom
import numpy as np
import matplotlib.pyplot as plt

from tools.config import config, paths
# from preprocessing.music_processing import load_song


def find_beats():
    aubio_pitch = aubio.pitch()
    # import song from disk
    #######################
    names_ar = os.listdir(paths.songs_pred)
    ending = ".egg"
    song_ar = []
    pitch_times_ar = []
    for idx, n in enumerate(names_ar):
        n = paths.songs_pred + n
        if not n.endswith(ending):
            n += ending
        # song = load_song(n, time_ar=[], return_raw=True)
        # song_ar.append(song)

        # analyze song pitches
        total_read = 0
        pitch_list = []
        src = aubio.source(n, channels=1, samplerate=config.samplerate_music)
        while True:
            samples, read = src()
            pit = aubio_pitch(samples)
            pitch_list.extend(pit)
            total_read += read
            if read < src.hop_size:
                break
        # get pitch times
        # pitch_times = get_pitch_times(pitch_list, src.hop_size)
        pitch_times_ar.append(pitch_list)

    return pitch_times_ar


def get_pitch_times(pitch_list, hop_size):
    # plt.figure()
    # plt.plot(pitch_list)
    # plt.show()
    pitch_thresh = np.mean(pitch_list)

    pitch_times = []
    # filter by threshold
    for idx, pit in enumerate(pitch_list):
        if pit > pitch_thresh:
            seconds = idx * hop_size / config.samplerate_music
            pitch_times.append(seconds)
    return pitch_times


if __name__ == '__main__':

    find_beats()
