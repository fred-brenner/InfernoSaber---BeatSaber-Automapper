"""
This script checks the folder structure.
Missing folders can be automatically created.
"""

import os
import tools.config.paths as paths
from tools.utils.ask_parameter import ask_parameter


# generic check if folder exists
def check_exists(data_path, create=True):
    if data_path.endswith('/'):
        # check if folder exists
        exist_flag = os.path.isdir(data_path)
    else:
        print(f'Missing "/" at the end of folder: {data_path}. Exit')
        exit()

    if not exist_flag:
        if create:
            if ask_parameter(f'Create folder <{data_path}> ? [y or n]', param_type='bool'):
                # create missing folder
                print(f"Creating folder: {data_path}")
                os.makedirs(data_path)
        else:
            print(f"Missing folder: {data_path}")
            return False

    return True


# iterate through folder structure
if not check_exists(paths.dir_path, create=False):
    print("Adjust input directory path (dir_path) in config")
    exit()
if not check_exists(paths.bs_song_path, create=False):
    print("Adjust BeatSaber path in config")
    exit()


check_exists(paths.model_path)
check_exists(paths.pred_path)
check_exists(paths.train_path)
check_exists(paths.temp_path)

check_exists(paths.copy_path_song)
check_exists(paths.copy_path_map)

check_exists(paths.dict_all_path)

check_exists(paths.songs_pred)

check_exists(paths.pred_input_path)
check_exists(paths.new_map_path)

# check_exists(paths.keras_path)

check_exists(paths.fail_path)
check_exists(paths.diff_path)
check_exists(paths.song_data)

# check_exists(paths.class_maps)
check_exists(paths.ml_input_path)

print("Finished folder setup.")

