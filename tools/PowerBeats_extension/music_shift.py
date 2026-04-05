# This script shifts the songs (.egg) to another folder (.mp3)

import argparse
import os
import shutil

import tools.config.paths as paths
from bs_shift.export_map import convert_music_file


def main(copy_path_new):
    copy_path_origin = paths.songs_pred

    if not os.path.isdir(copy_path_new):
        print("Could not find new song folder! Exit")
        exit()
    if not os.path.isdir(copy_path_origin):
        print("Could not find song origin folder! Exit")
        exit()

    counter = 0
    for root, _, files in os.walk(copy_path_origin):
        for song_file in files:
            if song_file.endswith(".egg"):
                counter += 1
                print(f"Converting song: {song_file}")
                output_file = f"{song_file[:-4]}.mp3"
                convert_music_file(os.path.join(root, song_file),
                                   os.path.join(copy_path_new, output_file))

    print(f"Finished shift to mp3 of {counter} song files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .egg files to .mp3 files in another folder.")
    parser.add_argument("target_dir", help="Destination folder for converted files")
    args = parser.parse_args()
    main(args.target_dir)
