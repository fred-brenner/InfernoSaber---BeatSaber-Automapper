# import glob
import json
import os
import shutil
import sys
from progressbar import ProgressBar

# Get the main script's directory
import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)


# sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.utils.str_compare import str_compare
from tools.fail_list.black_list import append_fail, delete_fails
from tools.utils.index_find_str import return_find_str

# set folder paths
from tools.config import paths, config
# import exclusion names
from tools.config import exclusion

from bps_find_songs import bps_find_songs
from map_to_dict_all import map_to_dict_all

# paths
copy_path_map = paths.copy_path_map
copy_path_song = paths.copy_path_song

if not os.path.isdir(paths.bs_input_path):
    print("Could not find Beat Saber path! Exit")
    exit()
if not os.path.isdir(copy_path_map) or not os.path.isdir(copy_path_song):
    print("Could not find copy path! Exit")
    exit()


def read_dat_file(file_path: str, filename="") -> list[str]:
    if filename != "":
        file_path = os.path.join(file_path, filename)
    with open(file_path) as f:
        dat_content = f.readlines()
    return dat_content


def read_json_content_file(file_path: str, filename="") -> list[str]:
    if filename != "":
        file_path = os.path.join(file_path, filename)
    with open(file_path) as f:
        dat_content = json.load(f)
    return dat_content


def delete_old_files():
    # delete old files
    print("Delete old files")
    for de in os.listdir(copy_path_song):
        # input("Del? " + de)
        os.remove(copy_path_song + de)
    for de in os.listdir(copy_path_map):
        # input("Del? " + de)
        os.remove(copy_path_map + de)


def shift_bs_songs(allow_diff2=False):
    # difficulty setup
    diff = "ExpertPlus"
    diff2 = "Expert"

    # variables setup
    num_cur = 0
    num_all = len(os.listdir(paths.bs_input_path)) + 1
    print("Check songs - may take a while")
    count = 0
    song_name_list = []
    song_name = None

    bar = ProgressBar(max_value=num_all)

    # walk through bs directory
    for root, dirs, files in os.walk(paths.bs_input_path):
        both = False
        excl_true = str_compare(str=os.path.basename(root), str_list=exclusion.exclusion, return_str=False,
                                silent=False)
        if excl_true:
            continue

        exp_plus_name = "ExpertPlus.dat"
        exp_name = "Expert.dat"
        for file in files:
            # get only ExpertPlus (or Expert for allow_diff2 == True)
            # if file.endswith(diff + ".dat") or (allow_diff2 and not both and file.endswith(diff2 + ".dat")):
            if file == exp_plus_name or (allow_diff2 and not both and file == exp_name):
                both = True
                # print(os.path.join(root, file))
                # print(files)

                # get song name
                info_file = False
                try:
                    for n_file in files:
                        if n_file.lower() == "info.dat":
                            num_cur += 1
                            bar.update(num_cur)
                            # import dat file
                            dat_content = read_dat_file(os.path.join(root, n_file))

                            if config.exclude_requirements:
                                search_string = '"_requirements":'
                                for s in dat_content:
                                    if search_string in s:
                                        if '[]' not in s:
                                            print("Excluding maps with custom mod requirements.")
                                            append_fail(os.path.basename(root))

                            search_string = '"_songName"'
                            # get name line
                            for s in dat_content:
                                if search_string in s:
                                    song_name, _ = return_find_str(0, s, search_string, True)
                                    info_file = True
                                    break

                            while song_name.lower() in song_name_list:
                                # song found in different versions
                                song_name += "_2"
                            song_name_list.append(song_name.lower())
                    # Finished name
                except:
                    print("Could not understand .info formatting: " + os.path.basename(root))
                    # Append to blacklist
                    append_fail(os.path.basename(root))
                    break
                # print(song_name)

                if not info_file:
                    print("Skipping... No info.dat file found in " + root)
                    break
                # copy files
                count += 1
                test_copy = 0
                for copy_file in files:
                    if copy_file.endswith(".egg"):
                        shutil.copyfile(root + "/" + copy_file, copy_path_song + song_name + copy_file[-4:])
                        test_copy += 2
                    elif copy_file.endswith(exp_name) and allow_diff2:
                        shutil.copyfile(root + "/" + copy_file, copy_path_map + song_name + copy_file[-4:])
                        test_copy += 1
                    elif copy_file.lower().endswith("info.dat"):
                        copy_file = "info.dat"
                        shutil.copyfile(root + "/" + copy_file, copy_path_map + song_name + "_" + copy_file)
                        test_copy += 2

                # Overwrite Expert with ExpertPlus if available
                for copy_file in files:
                    if copy_file.endswith(exp_plus_name):
                        shutil.copyfile(root + "/" + copy_file, copy_path_map + song_name + copy_file[-4:])
                        test_copy += 1
                        break

                # print(song_name)

                # test complete copy process
                if test_copy < 5 or test_copy > 6:
                    if test_copy > 6:
                        print("Too many files copied - probably two music files found.")
                    else:
                        print("Not enough files copied")
                    print("Error: Copy process failed at: " + os.path.basename(root))
                    append_fail(song_name)
                if len(song_name_list) != count:
                    print("Error: count != song_name_list")
                    exit()

                if len(os.listdir(copy_path_map)) / 2 != count:
                    print("Count {} vs directory: {}".format(count, len(os.listdir(copy_path_map)) / 2))
                    exit()
                if len(os.listdir(copy_path_song)) != count:
                    print("Count {} vs directory: {}".format(count, len(os.listdir(copy_path_song))))
                    exit()

    print("\nFinished Shift from BS directory to project")

    # Delete uncompleted samples
    delete_fails()


if __name__ == '__main__':
    delete_old_files()

    shift_bs_songs(allow_diff2=False)

    # Start casting to dictionary (notes, events, etc)
    map_to_dict_all()

    # Calculate notes per sec for each song
    bps_find_songs()

    print("Finished shifting")
