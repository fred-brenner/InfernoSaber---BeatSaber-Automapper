import numpy as np
import tools.config.config as config


def divide_in_voxels(map_dict_notes):
    # create matrix
    min_time = 0
    max_time = map_dict_notes[0, -1]
    time_total = max_time - min_time
    time_total *= config.samplerate_music
    time_total = int(time_total) + 2
    voxel_map_dict = np.zeros((time_total, 1))

    # fill matrix
    for i, el in enumerate(map_dict_notes[0, :].tolist()):
        new_i = el * config.samplerate_music
        new_i = int(round(new_i, 0))
        voxel_map_dict[new_i] = 1

    return voxel_map_dict
