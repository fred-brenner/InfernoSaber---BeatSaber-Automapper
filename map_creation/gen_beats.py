import numpy as np
from PIL import Image
# from line_profiler_pycharm import profile

from beat_prediction.find_beats import find_beats, get_pitch_times
from beat_prediction.beat_to_lstm import beat_to_lstm
from beat_prediction.beat_prop import get_beat_prop, tcn_reshape

from map_creation.sanity_check import *
# from map_creation.class_helpers import *
from map_creation.map_creator import create_map
from map_creation.find_bpm import get_file_bpm

from preprocessing.music_processing import run_music_preprocessing
from preprocessing.bs_mapper_pre import calc_time_between_beats
# from preprocessing.bs_mapper_pre import lstm_shift

from training.helpers import *
from training.tensorflow_models import *

from lighting_prediction.generate_lighting import generate

from tools.config import config, paths
from tools.utils import numpy_shorts


# @profile
def main(name_ar: list) -> None:

    if len(name_ar) > 1:
        print("Multiple song generation currently not implemented!")
        exit()

    # load song data
    song_input, pitch_input = find_beats(name_ar, train_data=False)     # 0.9

    ################
    # beat generator
    ################

    # calculate discrete timings
    pitch_times = []
    n_x = song_input[0].shape[0]
    for idx in range(len(pitch_input)):
        pitch_times.append(get_pitch_times(pitch_input[idx]))
        # resize song input to fit pitch algorithm
        im = Image.fromarray(song_input[idx])
        im = im.resize((len(pitch_input[idx]), n_x))
        song_input[idx] = np.asarray(im)
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
    model_path = paths.model_path + config.beat_gen_version
    beat_model = load_model(model_path, custom_objects={'TCN': TCN})    # 2.9

    # apply beat generator
    y_beat = beat_model.predict(x_input, verbose=0)    # 12

    y_beat[y_beat > config.thresh_beat] = 1
    y_beat[y_beat <= config.thresh_beat] = 0

    # apply beat sanity check (min time diff)
    y_beat = sanity_check_beat(y_beat)

    #####################
    # map class generator
    #####################
    # get beat times
    timing_ar = y_beat * np.arange(0, len(y_beat), 1)
    timing_ar /= config.beat_spacing
    timing_ar = timing_ar[timing_ar > config.window]
    # add beats between far beats
    fill_map_times_scale(timing_ar, scale_index=7)
    time_input = [timing_ar]

    # calculate bpm
    file = paths.songs_pred + name_ar[0] + '.egg'
    bpm, song_duration = get_file_bpm(file)     # 1.6
    # average bpm for songs to make more similar (jump) speeds
    bpm = int((bpm + 120) / 2)

    # sanity check timings
    map_times, pitch_algo = sanity_check_timing(name_ar[0], timing_ar, song_duration)   # 3.9
    map_times = map_times[map_times > 0]
    # map_times = fill_map_times(map_times)
    add_beats_min_bps = config.max_speed * 10 / 40  # max_speed=40 -> min_bps = 10
    scale_idx = 0
    while scale_idx < 10:
        if len(map_times) > add_beats_min_bps*map_times[-1]*config.add_beat_intensity/100:
            break
        map_times = fill_map_times_scale(map_times, scale_idx)
        scale_idx += 1
    print(f"Map filler iterated {scale_idx}/10 times.")
    map_times = add_lstm_prerun(map_times)

    # calculate time between beats
    timing_diff_ar = calc_time_between_beats([map_times])

    # load song data
    song_ar, rm_index = run_music_preprocessing(name_ar, time_ar=[map_times], save_file=False,
                                                song_combined=False, predict_path=True)     # 7.8

    # filter invalid indices
    for rm_idx in rm_index[0][::-1]:
        # timing_diff_ar[0].pop(rm_idx)   # not used right now
        # timing_ar = np.delete(timing_ar, rm_idx)  # not working/unused
        map_times = np.delete(map_times, rm_idx)

    # Load pretrained encoder model
    model_path = paths.model_path + config.enc_version
    enc_model = load_model(model_path)      # 0.4
    in_song_l = enc_model.predict(song_ar[0], verbose=0)       # 0.8

    y_class_map = generate(in_song_l, map_times, config.mapper_version, config.lstm_len,
                           paths.beats_classify_encoder_file)       # 45.2

    ############
    # create map
    ############
    map_times = map_times[config.lstm_len:]
    map_times = map_times[:len(y_class_map)]

    ############
    # add events
    ############
    if True:
        events = generate(in_song_l, map_times, config.event_gen_version, config.event_lstm_len,
                          paths.events_classify_encoder_file)     # 23.7 (47.0 -> 3.8)
    else:
        events = []

    create_map(y_class_map, map_times, events, name_ar[0], bpm, pitch_algo, pitch_times)     # 0.5


if __name__ == '__main__':
    # choose song
    name_ar, _ = filter_by_bps(config.min_bps_limit, config.max_bps_limit)
    name_ar = [name_ar[0]]

    main(name_ar)

    print("Finished map generator")
