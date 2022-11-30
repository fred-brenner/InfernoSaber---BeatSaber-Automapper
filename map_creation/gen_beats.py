import numpy as np
from PIL import Image
from keras.models import load_model
from tcn import TCN  # pip install keras-tcn

from training.helpers import *
from beat_prediction.find_beats import find_beats, get_pitch_times
from map_creation.sanity_check import *
from map_creation.class_helpers import *
from map_creation.map_creator import create_map
from map_creation.find_bpm import get_file_bpm
from preprocessing.music_processing import run_music_preprocessing
from preprocessing.bs_mapper_pre import calc_time_between_beats

from training.tensorflow_models import *
from beat_prediction.beat_to_lstm import beat_to_lstm
from beat_prediction.beat_prop import get_beat_prop
from preprocessing.bs_mapper_pre import load_ml_data, lstm_shift
from tools.config import config, paths
from tools.utils import numpy_shorts

tcn_len = config.tcn_len
# choose song
name_ar, _ = filter_by_bps(config.min_bps_limit, config.max_bps_limit)
name_ar = [name_ar[0]]

if len(name_ar) > 1:
    print("Multiple song generation currently not implemented!")
    exit()

# load song data
song_input, pitch_input = find_beats(name_ar, train_data=True)

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

# calculate song input
x_song, _ = beat_to_lstm(song_input, None)
x_song = numpy_shorts.minmax_3d(x_song)
x_input = [x_song, x_volume, x_onset]

# load pretrained generator model
model_path = paths.model_path + config.beat_gen_version
beat_model = load_model(model_path, custom_objects={'TCN': TCN})

# apply beat generator
y_beat = beat_model.predict(x_input)

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
timing_ar = timing_ar[timing_ar > 0]
time_input = [timing_ar]

# calculate time between beats
timing_diff_ar = calc_time_between_beats(time_input)

# load song data
song_ar, rm_index = run_music_preprocessing(name_ar, time_ar=time_input, save_file=False,
                                            song_combined=False)

# apply lstm shift
ml_input, _ = lstm_shift(song_ar[0], timing_diff_ar[0], None)
[in_song, in_time_l, _] = ml_input

# filter invalid indices
idx = 0
for rm_idx in rm_index:
    if len(rm_idx) > 0:
        print(f"Unknown problem with song: {name_ar[rm_idx]}. Remove song!")
        exit()
        # # remove invalid songs
        # name_ar.pop(idx)
        # song_ar.pop(idx)
    else:
        idx += 1

# Load pretrained encoder model
model_path = paths.model_path + config.enc_version
enc_model = load_model(model_path)

# apply encoder model
#####################
in_song_l = enc_model.predict(in_song)

# load pretrained automapper model
model_path = paths.model_path + config.mapper_version
mapper_model = load_model(model_path)

# song_input = None
# # time_input = None
# for idx in range(len(song_ar)):
#     song_input = numpy_shorts.np_append(song_input, song_ar[idx])
#     # time_input = numpy_shorts.np_append(time_input, np.asarray(timing_ar[idx], dtype='float16'))

# apply automapper
##################
y_class = None
y_class_map = []
for idx in range(len(in_song_l)):

    class_size = get_class_size()
    if y_class is None:
        in_class_l = np.zeros((len(in_song_l), config.lstm_len, class_size))

    in_class_l = update_out_class(in_class_l, y_class, idx)

    #             normal      lstm       lstm
    ds_train = [in_song_l[idx:idx+1], in_time_l[idx:idx+1], in_class_l[idx:idx+1]]
    y_class = mapper_model.predict(x=ds_train)

    y_class = cast_y_class(y_class)
    y_class_map.append(y_class)

# calculate bpm
file = paths.pred_input_path + name_ar[0] + '.egg'
bpm, song_duration = get_file_bpm(file)

# sanity check timings
map_times = sanity_check_timing(name_ar[0], timing_ar[config.lstm_len+1:], song_duration)

############
# create map
############
y_class_num = decode_onehot_class(y_class_map)
y_class_num = y_class_num[map_times > 0]
map_times = map_times[map_times > 0]

create_map(y_class_num, map_times, name_ar[0], bpm)

print("Finished")
