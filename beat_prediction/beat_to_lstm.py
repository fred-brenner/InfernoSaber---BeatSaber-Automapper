import numpy as np
from tools.config import config


def beat_to_lstm(song_input, beat_resampled):
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
    # 3D tensor with shape (batch_size, time_steps, seq_len)
    x_tcn = x_tcn.reshape(x_tcn.shape[0], tcn_len, -1)

    y_tcn = np.asarray(y_tcn)

    return x_tcn, y_tcn


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


def minmax_3d(ar: np.array) -> np.array:
    ar -= ar.min()
    ar /= ar.max()
    return ar
