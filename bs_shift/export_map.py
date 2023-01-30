import shutil
import os

from tools.config import paths


def shutil_copy_maps(song_name):
    if not os.path.isdir(paths.bs_song_path):
        print("Warning: Beatsaber folder not found, automatic export disabled.")
        return 0

    src = f'{paths.new_map_path}1234_{song_name}'
    dst = f'{paths.bs_song_path}1234_{song_name}'
    shutil.copytree(src=src, dst=dst, dirs_exist_ok=True)
