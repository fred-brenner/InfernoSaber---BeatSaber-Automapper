import numpy as np
import aubio
from tools.utils.load_and_save import save_npy
from tools.config import paths, config

import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image, ImageFilter


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

    # resample song into window size
    window_size = int(config.window * config.samplerate_music)
    windows_counts = int(len(samples_ar) / window_size)
    samples_ar_split = samples_ar[:int(windows_counts * window_size)]
    samples_ar_split = samples_ar_split.reshape((windows_counts, window_size))
    # x=time slot, y=window size

    return samples_ar_split


def log_specgram(audio, sample_rate, window_size,
                 step_size=1, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)


def process_song(song_ar: np.array) -> np.array:
    # use absolut values
    song_ar = np.abs(song_ar)
    # amplify signal
    song_ar *= 100

    # convert into spectrogram
    window_size = 100
    sample_rate = int(config.samplerate_music / 1)

    spectrogram_ar = []
    n_x = None
    for n in range(song_ar.shape[0]):
        _, spectrogram = log_specgram(song_ar[n], sample_rate, window_size)
        # shift spectrogram to 0+
        spectrogram -= spectrogram.min()

        # plt.imshow(spectrogram.T, aspect='auto', origin='lower')
        # plt.show()

        # resize and filter spectrogram
        im = Image.fromarray(spectrogram)
        # im = im.filter(ImageFilter.MaxFilter(config.max_filter_size))
        if n_x is None:
            n_x = spectrogram.shape[0]
        im = im.resize((config.specgram_res, n_x))

        # transpose and save spectrogram
        im = np.asarray(im).T
        spectrogram_ar.append(im)

        # plt.imshow(im, aspect='auto', origin='lower')
        # plt.show()
    return spectrogram_ar


def run_music_preprocessing(names_ar: list, save_file=True, song_combined=True):
    # load song notes
    ending = ".egg"
    song_ar = []
    for n in names_ar:
        song = load_song(paths.copy_path_song + n + ending)
        ml_input_song = process_song(song)
        if song_combined:
            song_ar.extend(ml_input_song)
        else:
            song_ar.append(ml_input_song)

    # scale song to 0-1
    song_ar = np.asarray(song_ar)
    song_ar = song_ar.clip(min=0)
    song_ar /= song_ar.max()

    if save_file:
        save_npy(song_ar, paths.ml_input_song_file)
    else:
        return song_ar
