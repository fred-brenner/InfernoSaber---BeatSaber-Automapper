# import glob
import os
import shutil
import sys
from progressbar import ProgressBar

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.utils.str_compare import str_compare
from tools.fail_list.black_list import append_fail, delete_fails
from tools.utils.index_find_str import return_find_str

# set folder paths
import tools.config.paths as paths
# import exclusion names
import tools.config.exclusion as exclusion

from bps_find_songs import bps_find_songs
from map_to_dict_all import map_to_dict_all

# paths
copy_path_map = paths.copy_path_map
copy_path_song = paths.copy_path_song

if not os.path.isdir(paths.bs_song_path):
    print("Could not find Beat Saber path! Exit")
    exit()
if not os.path.isdir(copy_path_map) or not os.path.isdir(copy_path_song):
    print("Could not find copy path! Exit")
    exit()


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
    num_all = len(os.listdir(paths.bs_song_path)) + 1
    print("Check songs - may take a while")
    count = 0
    song_name_list = []
    song_name = None

    bar = ProgressBar(max_value=num_all)

    # walk through bs directory
    for root, dirs, files in os.walk(paths.bs_song_path):
        both = False
        excl_true = str_compare(str=os.path.basename(root), str_list=exclusion.exclusion, return_str=False,
                                silent=False)
        if excl_true:
            continue
        for file in files:
            # get only ExpertPlus (or Expert for allow_diff2 == True)
            if file.endswith(diff + ".dat") or (allow_diff2 and not both and file.endswith(diff2 + ".dat")):
                # or ( file.endswith(diff + ".dat") and allow_diff2 is False):
                both = True
                # print(os.path.join(root, file))
                # print(files)

                # get song name
                info_file = False
                try:
                    for n_file in files:
                        if n_file == "info.dat":
                            num_cur += 1
                            bar.update(num_cur)
                            # import dat file
                            dat_content = open(os.path.join(root, n_file)).readlines()

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
                    # Append to black list
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
                    elif copy_file.endswith("Expert.dat") and allow_diff2:
                        shutil.copyfile(root + "/" + copy_file, copy_path_map + song_name + copy_file[-4:])
                        test_copy += 1
                    elif copy_file.endswith("info.dat"):
                        shutil.copyfile(root + "/" + copy_file, copy_path_map + song_name + "_" + copy_file)
                        test_copy += 2

                # Overwrite Expert with ExpertPlus if available
                for copy_file in files:
                    if copy_file.endswith("ExpertPlus.dat"):
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
    map_to_dict_all(paths)

    # Calculate notes per sec for each song
    bps_find_songs(paths)

    print("Finished shifting")
