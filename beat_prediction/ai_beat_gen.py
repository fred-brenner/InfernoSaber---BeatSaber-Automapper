import gc
from datetime import datetime
from keras.optimizers import adam_v2
from PIL import Image
# import matplotlib.pyplot as plt

from beat_prediction.find_beats import *
from beat_prediction.beat_to_lstm import *
from beat_prediction.beat_prop import *

from preprocessing.bs_mapper_pre import load_beat_data

from training.helpers import *
from training.tensorflow_models import *

from tools.config import config, paths
from tools.utils import numpy_shorts


def main():

    # Setup configuration
    #####################
    # Check Cuda compatible GPU
    if not test_gpu_tf():
        exit()

    # model name setup
    ##################
    # create timestamp
    dateTimeObj = datetime.now()
    timestamp = f"{dateTimeObj.month}_{dateTimeObj.day}__{dateTimeObj.hour}_{dateTimeObj.minute}"
    save_model_name = f"tf_beat_gen_{config.min_bps_limit}_{config.max_bps_limit}_{timestamp}.h5"

    # gather input
    ##############
    print("Gather input data:", end=' ')

    # ram_limit = int(11 * config.ram_limit)      # 100 songs ~9gb
    name_ar, _ = filter_by_bps(config.min_bps_limit, config.max_bps_limit)
    # if len(name_ar) > ram_limit:
    #     print(f"Info: Loading reduced song number into generator to not overload the RAM "
    #           f"(previous {len(name_ar)}")
    #     name_ar = name_ar[:ram_limit]

    print(f"Importing {len(name_ar)} songs")
    song_input, pitch_input = find_beats(name_ar, train_data=True)

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

    # get beat proposals
    [x_volume, x_onset] = get_beat_prop(song_input)

    # load real beats
    _, real_beats = load_beat_data(name_ar)

    beat_resampled = samplerate_beats(real_beats, pitch_times)

    # free ram
    del _, real_beats
    del im
    del pitch_input, pitch_times
    gc.collect()

    # delete a fraction of the offbeats to balance the dataset (n_beats << n_empty)
    beat_resampled, song_input, x_volume, x_onset = delete_offbeats(beat_resampled, song_input,
                                                                    x_volume, x_onset)

    print("Reshape input for AI model...")

    def lstm_reshape_half(song_list, y=False):
        tcn_len = config.tcn_len
        song_ar = None
        for song in song_list:
            delete = song.shape[-1] % tcn_len
            if delete != 0:
                if len(song.shape) == 2:
                    song = song[:, :-delete]
                else:
                    song = song[:-delete]
            if len(song.shape) == 2:
                song = song.reshape(-1, tcn_len, song.shape[0])
            elif len(song.shape) == 1 and not y:
                song = song.reshape(-1, tcn_len, 1)
            elif len(song.shape) == 1 and y:
                song = song[tcn_len-1::tcn_len]
            song_ar = numpy_shorts.np_append(song_ar, song, axis=0)
        return song_ar

    x_volume = lstm_reshape_half(x_volume)
    x_onset = lstm_reshape_half(x_onset)
    x_song = lstm_reshape_half(song_input)
    y = lstm_reshape_half(beat_resampled, y=True)

    # x_volume = tcn_reshape(x_volume)
    # x_onset = tcn_reshape(x_onset)
    # x_song, y = beat_to_lstm(song_input, beat_resampled)

    x_song = numpy_shorts.minmax_3d(x_song)
    cw = calc_class_weight(y)

    x_input = [x_song, x_volume, x_onset]
    test_len = config.tcn_test_samples
    x_part = [x_song[:test_len], x_volume[:test_len], x_onset[:test_len]]
    y_part = y[:test_len]

    del x_song, x_volume, x_onset
    gc.collect()

    # x_last_beats = last_beats_to_lstm(y)

    # setup ML model
    ################
    model = create_music_model('tcn', song_input[0].shape[0], config.tcn_len)
    adam = adam_v2.Adam(learning_rate=config.beat_learning_rate,
                        decay=config.beat_learning_rate * 2 / config.beat_n_epochs)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=['accuracy'])

    print(model.summary())

    model.fit(x=x_input, y=y, epochs=config.beat_n_epochs, shuffle=True,
              batch_size=config.beat_batch_size, verbose=1, class_weight=cw)

    # save model
    ############
    print(f"Saving model at: {paths.model_path + save_model_name}")
    model.save(paths.model_path + save_model_name)

    # plot test result
    ##################
    if True:
        y_pred = model.predict(x_part, verbose=0)
        # bound prediction to 0 or 1
        thresh = 0.5
        y_pred[y_pred > thresh] = 1
        y_pred[y_pred <= thresh] = 0

        fig = plt.figure()
        # plt.plot(y, 'b-', label='original')
        y_count = np.arange(0, len(y_part), 1)
        y_count = y_part * y_count

        plt.vlines(y_count, ymin=-0.1, ymax=1.1, colors='k', label='original', linewidth=2)
        plt.plot(y_pred, 'b-', label='prediction', linewidth=1)
        plt.legend()
        plt.show()

    print("Finished beat generator training")


if __name__ == '__main__':
    main()
