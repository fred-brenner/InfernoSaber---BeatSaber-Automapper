import numpy as np
import gc
from keras import backend as K
from PIL import Image
# from line_profiler_pycharm import profile

from beat_prediction.find_beats import find_beats, get_pitch_times, get_silent_times
from beat_prediction.beat_to_lstm import beat_to_lstm
from beat_prediction.beat_prop import get_beat_prop, tcn_reshape

from map_creation.sanity_check import *
# from map_creation.class_helpers import *
from map_creation.map_creator import create_map
from map_creation.find_bpm import get_file_bpm
from map_creation.bpm_optimizer import align_beats_on_bpm

from preprocessing.music_processing import run_music_preprocessing
from preprocessing.bs_mapper_pre import calc_time_between_beats
from tools.utils.numpy_shorts import get_factor_from_max_speed
# from preprocessing.bs_mapper_pre import lstm_shift

from training.helpers import *
from training.tensorflow_models import *

from lighting_prediction.generate_lighting import generate

from tools.config import config, paths
from tools.utils import numpy_shorts
from tools.config.mapper_selection import get_full_model_path


# @profile
def add_start_end_beats(map_times, x_volume, x_onset):
    if config.add_start_end_beats is False:
        return map_times

    x_onset_flat = x_onset.max(axis=1).reshape(-1) + x_onset.mean(axis=1).reshape(-1)
    x_volume_flat = x_volume.max(axis=1).reshape(-1) + x_volume.mean(axis=1).reshape(-1)
    x_flat = x_onset_flat + 3 * x_volume_flat

    limit_low = x_flat.mean()
    map_times_start_index = int(map_times[0] * config.beat_spacing)
    map_times_end_index = int(map_times[-1] * config.beat_spacing)
    map_times = list(map_times)
    # x_mask = np.zeros(len(x_flat))
    for idx in range(len(x_flat)):
        if idx < map_times_start_index or idx > map_times_end_index:
            if x_flat[idx] > limit_low:
                limit_low = x_flat[idx] + 0.1
                map_times.append(idx / config.beat_spacing)
            else:
                limit_low -= 0.1

    # timing_ar = x_mask * np.arange(0, len(x_mask), 1)
    # timing_ar = timing_ar.astype(float) / config.beat_spacing
    # timing_ar = timing_ar[timing_ar > 0]

    map_times = np.asarray(map_times)
    map_times.sort()

    return map_times


def main(name_ar: list, debug_beats=False) -> bool:
    if len(name_ar) > 1:
        print("Multi-core song generation currently not implemented!")
        exit()

    # update configuration
    if config.add_silence_flag:
        config.add_beat_intensity = config.add_beat_intensity_orig + 10
        config.silence_threshold = (1 - 0.5 * (config.max_speed_orig / 40)) * \
                                   config.silence_threshold_orig
        if config.silence_threshold < 0.02:
            config.silence_threshold = 0.02

    if config.emphasize_beats_flag:
        if config.add_silence_flag:
            config.add_beat_intensity = config.add_beat_intensity_orig
        else:
            config.add_beat_intensity = config.add_beat_intensity_orig - 10

    config.obstacle_time_gap = config.obstacle_time_gap_orig * (1 - config.max_speed / 80)
    if min(config.obstacle_time_gap) <= 0.1:
        config.obstacle_time_gap = np.asarray([0.1, 0.25])
    if config.sporty_obstacles:
        config.jump_speed_offset = config.jump_speed_offset_orig - 0.3
        if config.add_silence_flag or config.emphasize_beats_flag:
            config.add_beat_intensity -= 5
        else:
            config.add_beat_intensity = config.add_beat_intensity_orig - 5

    factor = get_factor_from_max_speed(config.max_speed, 1.5, 0.5)
    config.thresh_beat = config.thresh_beat_orig * factor
    factor = get_factor_from_max_speed(config.max_speed, 1.3, 0.1)
    config.thresh_onbeat = config.thresh_onbeat_orig * factor

    # print(f"Beat intensity: {config.add_beat_intensity}")

    # load song data
    song_input, pitch_input = find_beats(name_ar, train_data=False)

    ################
    # beat generator
    ################

    # calculate discrete timings
    silent_times = []
    pitch_times = []
    n_x = song_input[0].shape[0]
    for idx in range(len(pitch_input)):
        pitch_times.append(get_pitch_times(pitch_input[idx]))
        # resize song input to fit pitch algorithm
        im = Image.fromarray(song_input[idx])
        im = im.resize((len(pitch_input[idx]), n_x))
        song_input[idx] = np.asarray(im)
        if config.add_silence_flag:
            # remember silent timings
            silent_times.append(get_silent_times(song_input[idx], pitch_times[-1]))
        # # test song input
        # plt.imshow(song_input[idx])
        # plt.show()

    # calculate beat proposals
    [x_volume, x_onset] = get_beat_prop(song_input)
    # plt.plot(x_volume[0], label="volume", color="blue")
    # plt.plot(x_onset[0], label="onset", color="red")
    # plt.scatter(np.arange(len(pitch_times[0])), pitch_times[0], label="pitch", color="green")
    # plt.legend()
    # plt.show()
    x_volume = tcn_reshape(x_volume)
    x_onset = tcn_reshape(x_onset)

    # calculate song input
    x_song, _ = beat_to_lstm(song_input, None)
    x_song = numpy_shorts.minmax_3d(x_song)
    x_input = [x_song, x_volume, x_onset]

    # load pretrained generator model
    model_path = get_full_model_path(config.beat_gen_version)
    beat_model = load_model(model_path, custom_objects={'TCN': TCN})

    # apply beat generator
    y_beat = beat_model.predict(x_input, verbose=0)

    K.clear_session()
    gc.collect()
    del beat_model

    y_beat = apply_first_beat_thresholding(y_beat)

    # apply beat sanity check (min time diff)
    y_beat = sanity_check_beat(y_beat)

    ######################
    # sanity check timings
    ######################
    # get beat times
    timing_ar = y_beat * np.arange(0, len(y_beat), 1)
    timing_ar = timing_ar.astype(float) / config.beat_spacing
    timing_ar = timing_ar[timing_ar > config.window]
    # add beats between far beats
    if config.max_speed >= 5.5 * 4:
        timing_ar = fill_map_times_scale(timing_ar, scale_index=int(config.map_filler_iters / 2) + 1)
    if config.max_speed >= 8 * 4:
        timing_ar = fill_map_times_scale(timing_ar, scale_index=int(config.map_filler_iters - 1))
    # time_input = [timing_ar]

    # return timing_ar

    # calculate bpm
    if config.use_fixed_bpm is None or config.use_fixed_bpm <= 0:
        file = paths.songs_pred + name_ar[0] + '.egg'
        bpm, _ = get_file_bpm(file)  # 1.6
        # align beats on bpm
        timing_ar = align_beats_on_bpm(timing_ar, bpm)
        # # average bpm for songs to make more similar (jump) speeds
        # bpm = int((bpm + 120) / 2)
    else:
        bpm = config.use_fixed_bpm

    # sanity check timings
    # map_times, pitch_algo = sanity_check_timing(name_ar[0], timing_ar, song_duration)  # 3.9
    map_times = sanity_check_timing2(name_ar[0], timing_ar)
    map_times = map_times[map_times > 0]
    if len(map_times) < 3 * config.lstm_len:
        print(f"Could not match enough beats for song {name_ar[0]}")
        return 1
    # map_times = fill_map_times(map_times)
    add_beats_min_bps = config.max_speed * 10 / 40  # max_speed=40 -> min_bps = 10

    # fill start and end by onset detection
    map_times = add_start_end_beats(map_times, x_volume, x_onset)

    scale_idx = 0
    while scale_idx < config.map_filler_iters:
        if len(map_times) > add_beats_min_bps * map_times[-1] * config.add_beat_intensity / 100:
            break
        map_times = fill_map_times_scale(map_times, scale_idx)
        scale_idx += 1
    if config.verbose_level > 3:
        print(f"Map filler iterated {scale_idx}/{config.map_filler_iters} times.")
    if config.add_silence_flag:
        # remove silent parts
        map_times = remove_silent_times(map_times, silent_times[0])

    if config.use_fixed_bpm is None or config.use_fixed_bpm <= 0:
        map_times = align_beats_on_bpm(map_times, bpm)

    if debug_beats:
        return map_times

    # # compensate for lstm cutoff
    # map_times = add_lstm_prerun(map_times)

    # # calculate time between beats
    # timing_diff_ar = calc_time_between_beats([map_times])

    # map_times = add_start_end_beats(map_times, x_volume, x_onset)

    #####################
    # apply map generator
    #####################

    # load song data
    song_ar, rm_index = run_music_preprocessing(name_ar, time_ar=[map_times], save_file=False,
                                                song_combined=False, predict_path=True)  # 7.8

    # filter invalid indices
    for rm_idx in rm_index[0][::-1]:
        # timing_diff_ar[0].pop(rm_idx)   # not used right now
        # timing_ar = np.delete(timing_ar, rm_idx)  # not working/unused
        map_times = np.delete(map_times, rm_idx)

    # Load pretrained encoder model
    model_path = get_full_model_path(config.enc_version)
    enc_model = load_model(model_path)
    in_song_l = enc_model.predict(song_ar[0], verbose=0)

    mapper_model = get_full_model_path(config.mapper_version)
    # section into len(lstm) batches and calculate mapping for each batch
    y_class_map = generate(in_song_l, map_times, mapper_model, config.lstm_len,
                           paths.beats_classify_encoder_file)  # 45.2

    K.clear_session()
    gc.collect()
    del mapper_model, enc_model

    ############
    # create map
    ############
    # TODO: replace with default content instead of deletion
    map_times = map_times[config.lstm_len:]     # required by lstm start-up
    map_times = map_times[:len(y_class_map)]    # rest from sectioning into lstm len batches

    ############
    # add events
    ############
    if True:
        # (TODO: add furious_lighting to increase effect frequency)
        event_model = get_full_model_path(config.event_gen_version)
        events = generate(in_song_l, map_times, event_model, config.event_lstm_len,
                          paths.events_classify_encoder_file)  # 23.7 (47.0 -> 3.8)

        K.clear_session()
        gc.collect()
        del event_model
    else:
        events = []

    if config.bs_mapping_version != "v3":
        print(f"Warning: Using old mapping version: {config.bs_mapping_version}")
        from map_creation.map_creator_deprecated import create_map_depr
        create_map_depr(y_class_map, map_times, events, name_ar[0], bpm,
                        pitch_input[-1], pitch_times[-1])
    else:
        create_map(y_class_map, map_times, events, name_ar[0], bpm,
                   pitch_input[-1], pitch_times[-1])

    return False  # success


def apply_first_beat_thresholding(y_beat):
    lenni = len(y_beat)
    thresh_ar = np.ones(lenni) * config.thresh_beat

    # change threshold for start
    # thresh_ar[:int(lenni/5)] *= config.threshold_start
    thresh_ar[:int(lenni/10)] *= config.threshold_start

    # change threshold for end
    thresh_ar[-int(lenni/5):] *= config.threshold_end
    thresh_ar[-int(lenni/10):] *= config.threshold_end

    # run thresholding
    y_beat = y_beat.reshape(-1)
    result = (y_beat > thresh_ar).astype(int)
    result = result.reshape((-1, 1))

    return result


if __name__ == '__main__':
    # choose song
    name_ar, _ = filter_by_bps(config.min_bps_limit, config.max_bps_limit)
    name_ar = [name_ar[0]]

    main(name_ar)

    print("Finished map generator")
