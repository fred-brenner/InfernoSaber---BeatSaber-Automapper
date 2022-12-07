import numpy as np
from tools.config import config


def lstm_shift_post(ml_input, ml_output):
    lstm_len = config.lstm_len_post
    lstm_in = np.zeros((ml_input.shape[0]-lstm_len, lstm_len, ml_input.shape[1]))
    for idx in range(lstm_len):
        if idx == 0:
            lstm_in[:, idx, :] = ml_input[10-idx:]
        else:
            lstm_in[:, idx, :] = ml_input[10-idx:-idx]

    lstm_out = np.zeros((ml_output.shape[0]-lstm_len, lstm_len, ml_output.shape[1]))
    for idx in range(lstm_len):
        lstm_out[:, idx, :] = ml_output[9-idx:-idx-1]

    ml_out = ml_output[10:]

    lstm_in = np.concatenate((lstm_in, lstm_out), axis=-1)

    return lstm_in, ml_out
