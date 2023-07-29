import os
import time
import numpy as np
import json
import tensorflow as tf
from multiprocessing import Pool
from functools import partial

from tools.config import paths, config
import map_creation.gen_beats as beat_generator
from bs_shift.export_map import *


def stack_info_data(new_info_file: list, content: list, diff_str: str, diff_num: int) -> list:
    if len(new_info_file) == 0:
        new_info_file = content[:19]
        new_info_file[3] = '"_songSubName": "",\n'
    new_info_file.append('{\n')
    new_info_file.append(f'"_difficulty": "{diff_str}",\n')
    new_info_file.append(f'"_difficultyRank": {diff_num},\n')
    new_info_file.append(f'"_beatmapFilename": "{diff_str}.dat",\n')
    new_info_file.extend(content[30:32])
    if diff_str != "ExpertPlus":
        new_info_file.append('},\n')
    else:
        new_info_file.append('}\n')
    return new_info_file


def process_song(song_list_worker, diff_list, total_runs):
    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=conf)
    tf.compat.v1.keras.backend.set_session(sess)

    counter = 0
    time_per_run = 20
    for song_name in song_list_worker:

        for diff in diff_list:
            print(f"Running difficulty: {diff / 4:.1f}")
            if diff is not None:
                config.max_speed = diff
                config.max_speed_orig = diff

            print(f"### ETA: {(total_runs - counter) * time_per_run / 60:.1f} minutes. ###")
            counter += 1
            start_time = time.time()
            if song_name.endswith(".egg"):
                song_name = song_name[:-4]
            fail_flag = beat_generator.main([song_name])
            if fail_flag:
                print("Continue with next song")
                continue
            end_time = time.time()
            time_per_run = (4 * time_per_run + (end_time - start_time)) / 5


def main_multi_par(n_workers: int, diff_list: list, export_results_to_bs=True):
    diff_list = np.sort(diff_list)
    diff_list *= 4

    print("Starting multi map generator.")
    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=conf)
    tf.compat.v1.keras.backend.set_session(sess)

    # MAP GENERATOR
    ###############
    song_list = os.listdir(paths.songs_pred)
    song_list = check_music_files(song_list, paths.songs_pred)
    print(f"Found {len(song_list)} songs. Iterating...")
    if len(song_list) == 0:
        print("No songs found!")

    total_runs = int(np.ceil(len(diff_list) * len(song_list) / n_workers))
    # Divide the song_list into chunks for each worker
    chunks = np.array_split(song_list, n_workers)
    # Create a partial function with fixed arguments
    process_partial = partial(process_song, diff_list=diff_list, total_runs=total_runs)
    # Create a pool of workers to execute the process_song function in parallel
    with Pool(processes=n_workers) as pool:
        pool.starmap(process_partial, zip(chunks))

    combine_maps(song_list, diff_list, export_results_to_bs)


def main_multi(diff_list: list, export_results_to_bs=True):
    diff_list = np.sort(diff_list)
    diff_list *= 4

    print("Starting multi map generator.")
    # limit gpu ram usage
    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=conf)
    tf.compat.v1.keras.backend.set_session(sess)

    counter = 0
    # MAP GENERATOR
    ###############
    song_list = os.listdir(paths.songs_pred)
    song_list = check_music_files(song_list, paths.songs_pred)
    print(f"Found {len(song_list)} songs. Iterating...")
    if len(song_list) == 0:
        print("No songs found!")

    total_runs = len(diff_list) * len(song_list)
    for diff in diff_list:
        print(f"Running difficulty: {diff / 4:.1f}")
        # change difficulty
        if diff is not None:
            config.max_speed = diff
            config.max_speed_orig = diff

        time_per_run = 20  # time needed in seconds (first guess)
        for song_name in song_list:
            print(f"### ETA: {(total_runs - counter) * time_per_run / 60:.1f} minutes. ###")
            counter += 1
            start_time = time.time()
            song_name = song_name[:-4]
            # print(f"Analyzing song: {song_name} ({counter + 1} of {len(song_list)})")
            fail_flag = beat_generator.main([song_name])
            if fail_flag:
                print("Continue with next song")
                continue
            end_time = time.time()
            time_per_run = (4 * time_per_run + (end_time - start_time)) / 5
    combine_maps(song_list, diff_list, export_results_to_bs)


def combine_maps(song_list, diff_list, export_results_to_bs):
    print("Running map combination")
    for song_name in song_list:
        song_name = song_name[:-4]
        overall_folder = f"{paths.new_map_path}12345_{song_name}"
        # create new folder
        os.makedirs(overall_folder, exist_ok=True)
        # copy redundant data
        src = f"{paths.new_map_path}cover.jpg"
        shutil.copy(src, overall_folder)
        src = f"{paths.songs_pred}{song_name}.egg"
        shutil.copy(src, overall_folder)
        # iterate over diffs
        new_info_file = []
        for i, diff in enumerate(diff_list):
            diff = f"{diff:.1f}"
            src = f"{paths.new_map_path}1234_{diff}_{song_name}/ExpertPlus.dat"
            src_info = f"{paths.new_map_path}1234_{diff}_{song_name}/info.dat"
            with open(src_info) as fp:
                content = fp.readlines()
            if i == 0:
                dst = f"{overall_folder}/Easy.dat"
                new_info_file = stack_info_data(new_info_file, content, "Easy", 1)
            if i == 1:
                dst = f"{overall_folder}/Normal.dat"
                new_info_file = stack_info_data(new_info_file, content, "Normal", 3)
            elif i == 2:
                dst = f"{overall_folder}/Hard.dat"
                new_info_file = stack_info_data(new_info_file, content, "Hard", 5)
            elif i == 3:
                dst = f"{overall_folder}/Expert.dat"
                new_info_file = stack_info_data(new_info_file, content, "Expert", 7)
            elif i == 4:
                dst = f"{overall_folder}/ExpertPlus.dat"
                new_info_file = stack_info_data(new_info_file, content, "ExpertPlus", 9)
            shutil.copy(src, dst)
        # write info file
        new_info_file.extend(content[33:])
        with open(f"{overall_folder}/info.dat", 'w') as fp:
            fp.writelines(new_info_file)
        # create zip archive for online viewer
        shutil.make_archive(f'{paths.new_map_path}12345_{song_name}',
                            'zip', f'{paths.new_map_path}12345_{song_name}')
        # export map to beat saber
        if export_results_to_bs:
            shutil_copy_maps(song_name, index="12345_")
            # print("Successfully exported full difficulty maps to BS")
    print("Finished multi-map generator")


if __name__ == "__main__":
    diff_list = os.environ.get('diff_list')
    if diff_list is None:
        diff_list = [3, 5, 6.5, 7.5, 8.5]
    else:
        diff_list = json.loads(diff_list)
    if len(diff_list) != 5:
        print(f"Error: Did not get 5 difficulties: {diff_list}")
    print(f"Using difficulties: {diff_list}")
    # main_multi(diff_list, True)

    # each worker needs ~5gb of ram memory
    # each worker needs ~4gb of gpu memory
    n_workers = 3
    main_multi_par(n_workers, diff_list, True)
