# saves numpy arrays of difficulty per song
import os
import numpy as np


def bps_find_songs(paths) -> None:
    diff_array = []
    name_array = []

    for i in os.listdir(paths.dict_all_path):
        if i.endswith("_notes.dat"):
            # notes file
            map_dict_notes = np.load(paths.dict_all_path + i, allow_pickle=True)
            if len(map_dict_notes[0]) < 1:
                print(f"No notes found in {i}")

            # get song time
            diff_time = max(map_dict_notes[0]) - min(map_dict_notes[0])

            # get beats per second
            bps = round(len(map_dict_notes[0]) / diff_time, 2)

            # append to array with name in front
            diff_array.append(bps)
            name_array.append(i[:-10])

    print("Max song bps found: " + str(max(diff_array)))

    # save arrays
    diff_array = np.asarray(diff_array)
    name_array = np.asarray(name_array)
    if len(diff_array) != len(name_array):
        print("Error in bps_find_songs.py")
        exit()

    np.save(paths.diff_path + "diff_ar.npy", diff_array)
    np.save(paths.diff_path + "name_ar.npy", name_array)

    # Finished
    # print("Finished notes per second calculation")
