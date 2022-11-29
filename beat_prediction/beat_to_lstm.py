import numpy as np
from tools.config import config
from tools.utils import numpy_shorts


def beat_to_lstm(song_input, beat_resampled):
    tcn_len = config.tcn_len

    # # beta: use first sample only
    # x = song_input[0]
    # y = beat_resampled[0]
    x_tcn_all = None
    y_tcn_all = None

    for i_song in range(len(song_input)):
        x = song_input[i_song]
        y = beat_resampled[i_song]

        x_tcn = []
        y_tcn = []
        for i in range(x.shape[1] - tcn_len):
            x_tcn.append(x[:, i:i+tcn_len])
            y_tcn.append(y[i+tcn_len-1])

        x_tcn = np.asarray(x_tcn)
        # 3D tensor with shape (batch_size, time_steps, seq_len)
        x_tcn = x_tcn.reshape(x_tcn.shape[0], tcn_len, -1)

        y_tcn = np.asarray(y_tcn)

        x_tcn_all = numpy_shorts.np_append(x_tcn_all, x_tcn, 0)
        y_tcn_all = numpy_shorts.np_append(y_tcn_all, y_tcn, 0)

    return x_tcn_all, y_tcn_all


def beat_to_tcn(song_input, beat_resampled):
    tcn_len = config.tcn_len

    # beta: use first sample only
    x = song_input[0]
    y = beat_resampled[0]

    x_tcn = []
    y_tcn = []
    for i in range(x.shape[1] - tcn_len):
        x_tcn.append(x[:, i:i+tcn_len])
        y_tcn.append(y[i+tcn_len-1])

    x_tcn = np.asarray(x_tcn)
    # 3D tensor with shape (batch_size, timesteps, input_dim)
    x_tcn = x_tcn.reshape(x_tcn.shape[0], tcn_len, -1)

    y_tcn = np.asarray(y_tcn)

    return x_tcn, y_tcn


def last_beats_to_lstm(beats):
    tcn_len = config.tcn_len
    x_beats = []
    for idx in range(len(beats)):
        if idx < tcn_len:
            b = np.zeros(tcn_len)
            if idx > 0:
                b[-idx:] = beats[:idx]
        else:
            b = beats[idx-tcn_len:idx]
        x_beats.append(b)

    x_beats = np.asarray(x_beats)
    x_beats = x_beats.reshape(x_beats.shape[0], x_beats.shape[1], 1)
    return x_beats
