import numpy as np
import aubio
import madmom
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter

from tools.config import config
from tools.utils import numpy_shorts


def delete_offbeats(beat_resampled, song_input, x_volume, x_onset):
    for i_song in range(len(beat_resampled)):
        rd = np.random.rand(len(beat_resampled[i_song]))
        rd += beat_resampled[i_song]

        beat_resampled[i_song] = beat_resampled[i_song][rd > config.delete_offbeats]
        song_input[i_song] = song_input[i_song][:, rd > config.delete_offbeats]

        x_volume[i_song] = x_volume[i_song][rd > config.delete_offbeats]
        x_onset[i_song] = x_onset[i_song][rd > config.delete_offbeats]

    return beat_resampled, song_input, x_volume, x_onset


def get_beat_prop(x_song):
    # get volume through absolute frequency values
    # beat_a = None
    # beat_b = None
    # for song in x_song:
    #     set_a = volume_check(song)
    #     set_a = tcn_reshape(set_a)
    #     beat_a = numpy_shorts.np_append(beat_a, set_a, 0)
    #
    #     set_b = onset_detection(song)
    #     set_b = tcn_reshape(set_b)
    #     beat_b = numpy_shorts.np_append(beat_b, set_b, 0)
    beat_a = []
    beat_b = []
    for song in x_song:
        set_a = volume_check(song)
        beat_a.append(set_a)

        set_b = onset_detection(song)
        beat_b.append(set_b)

    return [beat_a, beat_b]


def tcn_reshape(x_input):
    x_tcn = None
    for ar in x_input:
        tcn_len = config.tcn_len
        ar_out = np.zeros((len(ar) - tcn_len, tcn_len))
        for idx in range(len(ar) - tcn_len):
            ar_out[idx] = ar[idx:idx+tcn_len]

        ar_out = ar_out.reshape(ar_out.shape[0], ar_out.shape[1], 1)
        x_tcn = numpy_shorts.np_append(x_tcn, ar_out, 0)
    return x_tcn


def volume_check(x_song):
    volume = np.zeros(x_song.shape[1])
    for idx in range(len(volume)):
        volume[idx] = x_song[:, idx].sum()
    # normalize
    volume = numpy_shorts.minmax_3d(volume)
    return volume


def onset_detection(x_song):
    x_song = x_song.T
    # sf = madmom.features.onsets.spectral_flux(x_song)
    # calculate the difference
    diff = np.diff(x_song, axis=0)
    # keep only the positive differences
    pos_diff = np.maximum(0, diff)
    # sum everything to get the spectral flux
    sf = np.sum(pos_diff, axis=1)
    sf = numpy_shorts.minmax_3d(sf)

    sf = np.hstack((np.zeros(1), sf))

    # # maximum filter size spreads over 3 frequency bins
    # size = (1, 3)
    # max_spec = maximum_filter(x_song, size=size)
    # diff = np.zeros_like(x_song)
    # diff[1:] = (x_song[1:] - max_spec[: -1])
    # pos_diff = np.maximum(0, diff)
    # superflux = np.sum(pos_diff, axis=1)
    # superflux = numpy_shorts.minmax_3d(superflux)
    #
    # fig = plt.figure()
    # plt.plot(sf, label='sf')
    # plt.plot(superflux, linestyle='dashed', label='superflux')
    # plt.legend()
    # plt.show()

    return sf
