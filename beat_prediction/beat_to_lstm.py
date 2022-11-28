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
    y_tcn = np.asarray(y_tcn)

    return x_tcn, y_tcn

