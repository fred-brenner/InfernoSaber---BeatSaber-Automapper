# This script shifts the songs (.egg) to another folder (.ogg)
# shift.py needs to be run first
import argparse
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import config.paths as paths
# import os
import shutil

def main(copy_path_new):
    copy_path_origin = paths.copy_path_song

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
            new_name = song_file[:-4] + ".ogg"
            shutil.copyfile(copy_path_origin + song_file, os.path.join(copy_path_new, new_name))

    print(f"Finished shift to ogg of {counter} song files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy .egg files to another folder as .ogg files.")
    parser.add_argument("target_dir", help="Destination folder for converted files")
    args = parser.parse_args()
    main(args.target_dir)
