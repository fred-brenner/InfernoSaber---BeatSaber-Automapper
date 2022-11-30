import numpy as np
import aubio
import matplotlib.pyplot as plt

from tools.config import config, paths


def sanity_check_beat(beat):
    beat = beat.reshape(len(beat))
    beat_counts = beat.sum()
    min_count = config.beat_spacing * config.min_beat_time
    min_count = int(np.round(min_count, 0))

    found_last_beat = -10
    for idx in range(len(beat)):
        if beat[idx] == 1:
            if found_last_beat < idx - min_count:
                found_last_beat = idx
            else:
                # found too many beats
                beat[idx] = 0

    # print result
    print(f"Got {beat.sum()} beats after sanity check"
          f" (removed {beat_counts-beat.sum()})")

    return beat


def sanity_check_timing(name, timings, song_duration):
    samplerate_music = 44100

    #####################################
    # import song to analyze volume peaks
    #####################################
    file = paths.pred_input_path + name + ".egg"

    # analyze song pitches
    total_read = 0
    pitch_list = []
    tempo_list = []
    samples_list = []
    src = aubio.source(file, channels=1, samplerate=samplerate_music)
    aubio_pitch = aubio.pitch(samplerate=samplerate_music)
    aubio_tempo = aubio.tempo(samplerate=samplerate_music)
    while True:
        samples, read = src()
        pit = aubio_pitch(samples)
        tempo = aubio_tempo(samples)
        samples_list.extend(samples)
        pitch_list.extend(pit)
        tempo_list.extend(tempo)
        total_read += read
        if read < src.hop_size:
            break

    # calc volume peaks
    pitches = np.asarray(pitch_list)
    # len(pitch_list) * 512 / samplerate_music = time in seconds
    # plt.plot(pitches)
    # plt.show()

    last_pitch = 0
    threshold = 0.0
    beat_flag = False
    beat_pos = np.zeros_like(pitches)
    for idx in range(len(pitches)):
        if pitches[idx] > last_pitch and pitches[idx] > threshold:
            beat_flag = True
        else:
            if beat_flag:
                beat_pos[idx-1] = 1
                beat_flag = False
        last_pitch = pitches[idx]

    # plt.plot(beat_pos)
    # plt.show()

    allowed_timings = beat_pos * np.arange(0, len(beat_pos), 1)
    allowed_timings *= 512 / samplerate_music
    allowed_timings = allowed_timings[allowed_timings > 0]

    # match timing from beat generator
    max_time_diff = 0.3
    last_beat = 0
    for i in range(len(timings)):
        diff = np.abs(allowed_timings - timings[i])
        min_diff = diff.min()
        if min_diff < max_time_diff:
            cur_beat = allowed_timings[np.argmin(diff)]
            if last_beat < cur_beat < song_duration:
                timings[i] = cur_beat
                last_beat = cur_beat
            else:
                timings[i] = 0
        else:
            timings[i] = 0

    return timings


def sanity_check_notes(notes, timings):
    notes
    timings


# if __name__ == '__main__':
#     timings = []
#     name = 'Bodied'
#     sanity_check_timing(name, timings)
