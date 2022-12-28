import os
import aubio
# import madmom
import numpy as np
# import matplotlib.pyplot as plt

from tools.config import config, paths
from preprocessing.music_processing import log_specgram


def find_beats(name_ar, train_data=True):
    aubio_pitch = aubio.pitch(samplerate=config.samplerate_music)
    if train_data:
        folder_path = paths.copy_path_song
    else:
        folder_path = paths.songs_pred
    # import song from disk
    #######################
    ending = ".egg"
    song_ar = []
    pitch_times_ar = []
    for idx, n in enumerate(name_ar):
        n = folder_path + n
        if not n.endswith(ending):
            n += ending

        # analyze song pitches
        total_read = 0
        pitch_list = []
        samples_list = []
        src = aubio.source(n, channels=1, samplerate=config.samplerate_music)
        while True:
            samples, read = src()
            pit = aubio_pitch(samples)
            samples_list.extend(samples)
            pitch_list.extend(pit)
            total_read += read
            if read < src.hop_size:
                break

        # get pitch times
        # pitch_times = get_pitch_times(pitch_list, src.hop_size)
        pitch_times_ar.append(pitch_list)

        # logarithmic spectrogram of song
        window_size = 35.608
        step_size = 1
        _, spect = log_specgram(np.asarray(samples_list), config.samplerate_music, window_size, step_size=step_size)
        spect = spect.T
        song_ar.append(spect)
        # # test spectogram
        # plt.imshow(spect)
        # plt.show()

    return song_ar, pitch_times_ar


def get_pitch_times(pitch_list, pitch_thresh=-123):
    # plt.figure()
    # plt.plot(pitch_list)
    # plt.show()

    pitch_times = []
    # filter by threshold
    for idx, pit in enumerate(pitch_list):
        if pit > pitch_thresh:
            seconds = idx * config.hop_size / config.samplerate_music
            pitch_times.append(seconds)
    return pitch_times


def samplerate_beats(real_beats, timing):
    beat_resampled_ar = []
    for idx in range(len(real_beats)):
        cur_timing = np.asarray(timing[idx])
        beat_resampled = np.zeros(len(timing[idx]))
        for beat in real_beats[idx]:
            beat_idx = np.argmin(abs(cur_timing - beat))
            beat_resampled[beat_idx] = 1
        beat_resampled_ar.append(beat_resampled)

    return beat_resampled_ar


if __name__ == '__main__':
    name_ar = os.listdir(paths.songs_pred)
    find_beats(name_ar)
