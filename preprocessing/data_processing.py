import numpy as np
import tools.config.paths as paths
import tools.config.config as config
from tools.utils.load_and_save import load_npy, save_npy, filter_max_bps
from preprocessing.time_voxel import divide_in_voxels


def main():
    # load song difficulty
    diff_ar = load_npy(paths.diff_ar_file)
    name_ar = load_npy(paths.name_ar_file)

    # filter by max song diff
    names = filter_max_bps(diff_ar, name_ar)

    # load song notes
    ending = "_notes.dat"
    for n in names:
        map_dict_notes = load_npy(paths.dict_all_path + n + ending)
        break

    # divide the notes into time voxels
    voxel_map = divide_in_voxels(map_dict_notes)

    # save ML input
    save_npy(voxel_map, paths.ml_input_beat_file)


if __name__ == '__main__':
    main()
    print("Finished")
