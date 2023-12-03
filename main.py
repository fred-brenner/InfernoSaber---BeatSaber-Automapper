###################################################
# This file is needed to find the working directory
###################################################
# import os
# import shutil
import time
# import sys

from tools.config import paths, config
import map_creation.gen_beats as beat_generator
from bs_shift.export_map import *

import tensorflow as tf


def main(use_model=None, diff=None, export_results_to_bs=True,
         gimme_more=None, quick_start=None, beat_intensity=None,
         random_factor=None, js_offset=None,
         allow_no_dir_flag=None, silence_factor=None,
         add_obstacles=None, sporty_obstacles=None,
         add_sliders=None, slider_start_time=None,
         slider_end_time=None, slider_probability=None,
         slider_movement_min=None):

    if use_model is not None:
        config.use_mapper_selection = use_model
    # change difficulty
    if diff is not None:
        config.max_speed = diff
        config.max_speed_orig = diff
    if gimme_more is not None:
        config.gimme_more_notes_prob = gimme_more
    if quick_start is not None:
        config.quick_start = quick_start
    if beat_intensity is not None:
        config.add_beat_intensity = beat_intensity
        config.add_beat_intensity_orig = beat_intensity
    if random_factor is not None:
        config.random_note_map_factor = random_factor
    if js_offset is not None:
        config.jump_speed_offset += js_offset
        config.jump_speed_offset_orig += js_offset
    if allow_no_dir_flag is not None:
        config.allow_dot_notes = allow_no_dir_flag
    if silence_factor is not None:
        config.silence_threshold *= silence_factor
        config.silence_threshold_orig *= silence_factor
    if add_obstacles is not None:
        config.add_obstacle_flag = add_obstacles
    if sporty_obstacles is not None:
        config.sporty_obstacles = sporty_obstacles
    if add_sliders is not None:
        config.add_slider_flag = add_sliders
    if slider_start_time is not None:
        config.slider_time_gap[0] = slider_start_time
    if slider_end_time is not None:
        config.slider_time_gap[1] = slider_end_time
    if slider_probability is not None:
        config.slider_probability = slider_probability
    if slider_movement_min is not None:
        config.slider_movement_minimum = slider_movement_min

    # limit gpu ram usage
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

    for i, song_name in enumerate(song_list):
        start_time = time.time()
        song_name = song_name[:-4]
        print(f"Analyzing song: {song_name} ({i + 1} of {len(song_list)})")
        fail_flag = beat_generator.main([song_name])
        if fail_flag:
            print("Continue with next song")
            continue
        end_time = time.time()
        print(f"Time needed: {end_time - start_time}s")

        # create zip archive for online viewer
        shutil.make_archive(f'{paths.new_map_path}{config.max_speed_orig:.1f}_{song_name}',
                            'zip', f'{paths.new_map_path}1234_{config.max_speed_orig:.1f}_{song_name}')
        # export map to beat saber
        if export_results_to_bs:
            shutil_copy_maps(f"{config.max_speed_orig:.1f}_{song_name}")

    print("Finished map generator")


# ############################################################

# ############################################################
# if fails, rerun train_bs_automapper with correct min/max_bps
# until training is started (cancel after data import)
##############################################################

# TRAINING
##########
# run bs_shift / shift.py
# run training / train_autoenc_music.py
# run training / train_bs_automapper.py
# run beat_prediction / ai_beat_gen.py
# run lighting_prediction / train_lighting.py

if __name__ == "__main__":
    use_model = os.environ.get('inferno_model')

    diff = os.environ.get('max_speed')
    if diff is not None:
        diff = float(diff)
        print(f"Set BPS difficulty to {diff}")
        diff = diff * 4  # calculate bps to max_speed
    else:
        print("Use default difficulty values")

    gm = os.environ.get('gimme_more')
    if gm is not None:
        gm = float(gm)
        print(f"Set density to {gm}")

    qs = os.environ.get('quick_start')
    if qs is not None:
        qs = float(qs)
        print(f"Set quick_start to {qs}")

    bi = os.environ.get('beat_intensity')
    if bi is not None:
        bi = float(bi)
        print(f"Set beat intensity to {bi}")

    rf = os.environ.get('random_factor')
    if rf is not None:
        rf = float(rf)
        print(f"Set random factor to {rf}")

    jso = os.environ.get('jump_speed_offset')
    if jso is not None:
        jso = float(jso)
        print(f"Set jump speed offset to {jso}")

    ndf = os.environ.get('allow_no_direction_flag')
    if ndf is not None:
        if ndf == 'True':
            ndf = True
        else:
            ndf = False
        print(f"Set allow_no_direction_flag to {ndf}")

    sf = os.environ.get('silence_factor')
    if sf is not None:
        sf = float(sf)
        print(f"Set silence factor to {sf}")

    # obstacles
    aof = os.environ.get('add_obstacle_flag')
    if aof is not None:
        if aof == 'True':
            aof = True
        else:
            aof = False
        print(f"Set add_obstacle_flag to {aof}")

    sof = os.environ.get('sporty_obstacle_flag')
    if sof is not None:
        if sof == 'True':
            sof = True
        else:
            sof = False
        print(f"Set sporty_obstacle_flag to {sof}")

    # sliders
    asf = os.environ.get('add_slider_flag')
    if asf is not None:
        if asf == 'True':
            asf = True
        else:
            asf = False
        print(f"Set add_slider_flag to {asf}")

    sst = os.environ.get('slider_start_time')
    if sst is not None:
        sst = float(sst)
        print(f"Set slider_start_time to {sst}")

    se = os.environ.get('slider_end_time')
    if se is not None:
        se = float(se)
        print(f"Set slider_end_time to {se}")

    sp = os.environ.get('slider_probability')
    if sp is not None:
        sp = float(sp)
        print(f"Set slider_probability to {sp}")

    smm = os.environ.get('slider_movement_min')
    if smm is not None:
        smm = float(smm)
        print(f"Set slider_movement_min to {smm}")

    export_results_to_bs = True
    if paths.IN_COLAB:
        export_results_to_bs = False

    main(use_model, diff, export_results_to_bs, gm, qs, bi, rf, jso,
         ndf, sf, aof, sof, asf, sst, se, sp, smm)
