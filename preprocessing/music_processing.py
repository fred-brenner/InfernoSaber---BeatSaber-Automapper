from PIL import Image, ImageFilter
# from progressbar import ProgressBar
from scipy import signal
import aubio
import numpy as np

from bs_shift.bps_find_songs import bps_find_songs
from tools.config import paths, config
from tools.fail_list.black_list import append_fail, delete_fails
from tools.utils import numpy_shorts
from tools.utils.load_and_save import save_npy

# from line_profiler_pycharm import profile


def load_song(data_path: str, time_ar: list, return_raw=False) -> np.array:
    total_read = 0
    samples_list = []
    src = aubio.source(data_path, channels=1, samplerate=config.samplerate_music)

    while True:
        samples, read = src()
        samples_list.extend(samples)
        total_read += read
        if read < src.hop_size:
            break

    samples_ar = np.asarray(samples_list)
    if return_raw:
        return samples_ar

    # x=time slot, y=window size
    if time_ar is None:
        # resample song into window size
        window_size = int(config.window * config.samplerate_music)
        windows_counts = int(len(samples_ar) / window_size)
        samples_ar_split = samples_ar[:int(windows_counts * window_size)]
        samples_ar_split = samples_ar_split.reshape((windows_counts, window_size))
        remove_idx = []
    else:
        window_size = int(config.window * config.samplerate_music)
        time_ar = np.asarray(time_ar) * config.samplerate_music
        time_ar = np.around(time_ar).astype('int')
        samples_ar_split = []
        remove_idx = []
        max_time = len(samples_ar)
        for idx, sample in enumerate(time_ar):
            # sanity check
            start_idx = sample - int(window_size/2)
            if start_idx < 0:
                remove_idx.append(idx)
                continue
            end_idx = sample + int(window_size/2)
            if end_idx >= max_time:
                remove_idx.append(idx)
                continue
            # add time window
            samples_ar_split.append(samples_ar[start_idx:end_idx])
        samples_ar_split = np.asarray(samples_ar_split)

    return samples_ar_split, remove_idx


def log_specgram(audio, sample_rate, window_size,
                 step_size=1, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)


def process_song(song_ar: np.array) -> np.array:
    # use absolut values
    song_ar = np.abs(song_ar)
    # amplify signal
    song_ar *= 100

    # convert into spectrogram
    window_size = 100
    sample_rate = int(config.samplerate_music / 1)

    spectrogram_ar = []
    n_x = None
    for n in range(song_ar.shape[0]):
        _, spectrogram = log_specgram(song_ar[n], sample_rate, window_size)
        # shift spectrogram to 0+
        spectrogram -= spectrogram.min()

        # plt.imshow(spectrogram.T, aspect='auto', origin='lower')
        # plt.show()

        # resize and filter spectrogram
        im = Image.fromarray(spectrogram)
        # im = im.filter(ImageFilter.MaxFilter(config.max_filter_size))
        if n_x is None:
            n_x = spectrogram.shape[0]
        im = im.resize((config.specgram_res, n_x))

        # transpose and save spectrogram
        im = np.asarray(im).T
        spectrogram_ar.append(im)

        # plt.imshow(im, aspect='auto', origin='lower')
        # plt.show()
    return np.asarray(spectrogram_ar)


def run_music_preprocessing(names_ar: list, time_ar=None, save_file=True, song_combined=True,
                            channels_last=True, predict_path=False):
    # load song notes
    ending = ".egg"
    song_ar = []
    rm_index_ar = []
    errors_appeared = 0
    # rm_index = None

    # bar = ProgressBar(max_value=len(names_ar))

    # print(f"Importing {len(names_ar)} songs")
    for idx, n in enumerate(names_ar):
        # bar.update(idx+1)
        if time_ar is None:
            time = None
        else:
            time = time_ar[idx]
        if not n.endswith(ending):
            n += ending
        try:
            if predict_path:
                song, remove_idx = load_song(paths.songs_pred + n, time_ar=time)
            else:
                song, remove_idx = load_song(paths.copy_path_song + n, time_ar=time)
        except Exception as e:
            print(f"Problem with song: {n}")
            print(f"Exception details: {str(e)}")
            # print(paths.copy_path_song)
            # exit()
            append_fail(n[:-4])
            errors_appeared += 1
            continue

        rm_index_ar.append(remove_idx)
        ml_input_song = process_song(song)

        # if song_combined:   # does not work
        #     song_ar.extend(ml_input_song)
        # else:
        song_ar.append(ml_input_song)
    if errors_appeared > 0:
        delete_fails()
        bps_find_songs()
        print("Deleted failed maps, please re-run!")
        exit()
    # scale song to 0-1
    # if len(np.asarray(song_ar).shape) > 1:
    #     song_ar = np.asarray(song_ar)
    #     # song_ar = song_ar.clip(min=0)
    #     # song_ar /= song_ar.max()
    #     song_ar = numpy_shorts.minmax_3d(song_ar)
    #     if channels_last:
    #         song_ar = song_ar.reshape((song_ar.shape[0], song_ar.shape[1], song_ar.shape[2], 1))
    #     else:
    #         song_ar = song_ar.reshape((song_ar.shape[0], 1, song_ar.shape[1], song_ar.shape[2]))
    # else:
    for idx, song in enumerate(song_ar):
        song = numpy_shorts.minmax_3d(song)
        if channels_last:
            song_ar[idx] = song.reshape((song.shape[0], song.shape[1], song.shape[2], 1))
        else:
            song_ar[idx] = song.reshape((song.shape[0], 1, song.shape[1], song.shape[2]))

    if song_combined:
        song_ar = np.concatenate(song_ar, axis=0)

    if save_file:
        save_npy(song_ar, paths.ml_input_song_file)
    else:
        return song_ar, rm_index_ar
