import json
import numpy as np
import tensorflow as tf
import time
from functools import partial
from multiprocessing import Pool

import map_creation.gen_beats as beat_generator
from tools.config import paths, config
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


def process_song(song_list_worker, total_runs):
    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=conf)
    tf.compat.v1.keras.backend.set_session(sess)

    # counter = 0
    # time_per_run = 20
    for song_name in song_list_worker:
        diff = float(song_name[1])
        print(f"Running difficulty: {diff / 4:.1f}")
        config.max_speed = diff
        config.max_speed_orig = diff
        song_name = song_name[0]

        # print(f"### ETA: {(total_runs - counter) * time_per_run / 60:.1f} minutes. ###")
        # counter += 1
        # start_time = time.time()
        if song_name.endswith(".egg"):
            song_name = song_name[:-4]
        fail_flag = beat_generator.main([song_name])
        if fail_flag:
            print("Unknown error in map generator. Continue with next song")
            continue
        # end_time = time.time()
        # time_per_run = (4 * time_per_run + (end_time - start_time)) / 5


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
    song_list_files = os.listdir(paths.songs_pred)
    song_list_files = check_music_files(song_list_files, paths.songs_pred)
    print(f"Found {len(song_list_files)} songs. Iterating...")
    if len(song_list_files) == 0:
        print("No songs found!")

    song_list = []
    for song in song_list_files:
        for diff in diff_list:
            song_list.append([song, diff])
    total_runs = int(np.ceil(len(song_list) / n_workers))
    processed_count = 0
    processed_count_real = 0
    time_per_run = 20

    # Divide the song_list into chunks for each worker
    chunks = np.array_split(song_list, len(song_list))
    # Create a partial function with fixed arguments
    process_partial = partial(process_song, total_runs=total_runs)
    # Create a pool of workers to execute the process_song function in parallel
    start_time = time.time()
    with Pool(processes=n_workers) as pool:
        for _ in pool.imap(process_partial, chunks):
            # pass
            processed_count += 1  # Increment the counter
            if processed_count % len(diff_list) == 0:
                combine_maps([song_list_files[processed_count_real]], diff_list, export_results_to_bs)
                processed_count_real += 1
            new_time_per_run = (time.time() - start_time) / processed_count
            time_per_run = (time_per_run + new_time_per_run) / 2
            print(f"### ETA: {(len(song_list) - processed_count) * time_per_run / 60:.1f} minutes. ###")

            # Check if there are remaining elements not processed in a batch of 5
        if processed_count % len(diff_list) != 0:
            print("Error: Found remaining maps which have not been combined.")
            combine_maps([song_list_files[-1]], diff_list, export_results_to_bs)

    # combine_maps(song_list_files, diff_list, export_results_to_bs)


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
    # if len(song_list) > 1:
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
            if not os.path.isfile(src):
                print(f"Could not find all files for: 1234_{diff}_{song_name}")
                return 0
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
    if len(song_list) == 1:
        print(f"Finished map combination for {song_list[0]}")
    else:
        print("Finished multi-map generator")


if __name__ == "__main__":
    diff_list = os.environ.get('diff_list')
    if diff_list is None:
        diff_list = [3.5, 4.5, 6.5, 7.5, 8.5]
    else:
        diff_list = json.loads(diff_list)
    # if len(diff_list) != 5:
    #     print(f"Warning: Did not get 5 difficulties: {diff_list}")

    config.create_expert_flag = False
    print(f"Using difficulties: {diff_list}")

    if paths.IN_COLAB:
        print("Multi-processing on colab notebook not supported :|\n"
              "Running single process.")
        main_multi(diff_list, False)
    else:
        # main_multi(diff_list, True)
        # each worker needs ~5gb of ram memory (15gb / 3)
        # each worker needs ~4gb of gpu memory (11gb / 3)
        n_workers = 3
        main_multi_par(n_workers, diff_list, True)
