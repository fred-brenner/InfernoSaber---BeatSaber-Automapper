# This script shifts the songs (.egg) to another folder (.ogg)
# shift.py needs to be run first
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import config.paths as paths
# import os
import shutil

# paths
copy_path_new = "C:/Users/frede/Music/"
copy_path_origin = paths.copy_path_song

# folder check
if not os.path.isdir(copy_path_new):
    print("Could not find new song folder! Exit")
    exit()
if not os.path.isdir(copy_path_origin):
    print("Could not find song origin folder! Exit")
    exit()

counter = 0
for song_file in os.listdir(copy_path_origin):
    counter += 1
    if not song_file.endswith(".egg"):
        print(f"Warning: unknown file type: {song_file}")
    else:
        # copy in new directory and rename to .ogg
        new_name = song_file[:-4] + ".ogg"
        shutil.copyfile(copy_path_origin + song_file, copy_path_new + new_name)
    # print(song_file)

print(f"Finished shift to ogg of {counter} song files")