import numpy as np
import matplotlib.pyplot as plt
import os

# from preprocessing.bs_mapper_pre import load_beat_data
# from beat_prediction.find_beats import find_beats
# from tools.config import paths


def plot_beat_vs_real(beat_pred, beat_real):
    plt.figure()
    plt.vlines(beat_pred, 0, 1, colors='k', linestyles='solid', linewidth=0.2)
    plt.scatter(beat_real, [0.5] * len(beat_real))
    plt.show()

#
# if __name__ == '__main__':
#     name_ar = os.listdir(paths.songs_pred)
#     pitch_list = find_beats(name_ar, train_data=False)
#
#     name_ar = ['Born This Way', 'Dizzy']
#     _, real_beats = load_beat_data(name_ar)
#
#     idx = 0
#     plot_beat_vs_real(pitch_list[idx], real_beats[idx])
#
#     print("")
