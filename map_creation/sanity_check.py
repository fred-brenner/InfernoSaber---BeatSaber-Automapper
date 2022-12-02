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
          f" (removed {beat_counts - beat.sum()})")

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
                beat_pos[idx - 1] = 1
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
    # TODO: print counter for removing notes
    [notes_r, notes_l, notes_b] = split_notes_rl(notes)
    # test = unpslit_notes(notes_r, notes_l, notes_b)

    notes_r = correct_notes(notes_r, timings)   # TODO: allow double notes
    notes_l = correct_notes(notes_l, timings)

    # TODO: correct both notes together

    # rebuild notes
    new_notes = unpslit_notes(notes_r, notes_l, notes_b)
    return new_notes


def correct_notes(notes, timings):
    nl_last = None
    last_time = 0
    rm_counter = 0
    for idx in range(len(notes)):
        if len(notes[idx]) == 0:
            continue
        # elif len(notes[idx]) == 4:
        elif len(notes[idx]) >= 4:
            # check movement up and down
            notes[idx] = check_note_movement(nl_last, notes[idx])

            # calculate movement speed
            new_time = timings[idx]
            speed = calc_note_speed(nl_last, notes[idx], new_time - last_time)
            last_time = new_time
            nl_last = notes[idx]

            # remove too fast elements
            if speed > config.max_speed:
                rm_counter += int(len(notes[idx]) / 4)
                notes[idx] = []
                continue

        # else:
        #     notes[idx] = []
        #     print("Found multiple notes in one beat at sanity_check. Skipping.")
        #     continue
    print(f"Sanity check note speed removed {rm_counter} elements")
    return notes


def check_note_movement(notes_last, notes_new):
    if notes_last is None:
        return notes_new

    cut_x_last, cut_y_last = get_cut_dir_xy(notes_last[3])
    cut_x_new, cut_y_new = get_cut_dir_xy(notes_new[3])
    dist_x = int(np.abs(cut_x_last - cut_x_new))
    dist_y = int(np.abs(cut_y_last - cut_y_new))

    if dist_x != 2 and dist_y != 2:
        if dist_x == dist_y == 1:
            return notes_new

        new_cut = reverse_cut_dir_xy(notes_last[3])
        notes_new[3] = new_cut

    return notes_new


def calc_note_speed(notes_last, notes_new, time_diff):
    if notes_last is None:
        return 0

    # cut director factor
    cdf = config.cdf

    dist = 0
    cut_x_last, cut_y_last = get_cut_dir_xy(notes_last[3])
    cut_x_new, cut_y_new = get_cut_dir_xy(notes_new[3])
    # x direction
    dist += np.abs((notes_last[0] - cdf * cut_x_last) -
                   (notes_new[0] + cdf * cut_x_new))
    # y direction
    dist += np.abs((notes_last[1] - cdf * cut_y_last) -
                   (notes_new[1] + cdf * cut_y_new))

    speed = dist / time_diff

    return speed


def split_notes_rl(notes):
    notes_r = []
    notes_l = []
    notes_b = []
    for i in range(len(notes)):
        temp_r = []
        temp_l = []
        temp_b = []
        for n in range(int(len(notes[i]) / 4)):
            index = notes[i][0 + 4 * n]
            layer = notes[i][1 + 4 * n]
            typ = notes[i][2 + 4 * n]
            cut = notes[i][3 + 4 * n]

            if typ == 1:
                # right
                temp_r.extend([index, layer, typ, cut])
            elif typ == 0:
                # left
                temp_l.extend([index, layer, typ, cut])
            elif typ == 3:
                temp_b.extend([index, layer, typ, cut])
        notes_r.append(temp_r)
        notes_l.append(temp_l)
        notes_b.append(temp_b)

    return [notes_r, notes_l, notes_b]


def unpslit_notes(notes_r, notes_l, notes_b):
    note_counter = 0
    bomb_counter = 0
    notes = []
    for idx in range(len(notes_r)):
        temp = []
        temp.extend(notes_l[idx])
        temp.extend(notes_r[idx])
        temp.extend(notes_b[idx])

        if len(notes_l[idx]) > 0:
            note_counter += int(len(notes_l[idx]) / 4)
        if len(notes_r[idx]) > 0:
            note_counter += int(len(notes_r[idx]) / 4)
        if len(notes_b[idx]) > 0:
            bomb_counter += int(len(notes_b[idx]) / 4)

        notes.append(temp)
    print(f"Generating map with {note_counter} notes and {bomb_counter} bombs")
    return notes


def get_cut_dir_xy(cut_dir):
    cut_dir_x = np.asarray([[1, 0, -1]] * 3).flatten()
    cut_dir_y = np.asarray([[-1] * 3, [0] * 3, [1] * 3]).flatten()
    cut_dir_order = np.asarray([[4, 0, 5], [2, 8, 3], [6, 1, 7]]).flatten()
    mov_x = cut_dir_x[cut_dir_order == cut_dir]
    mov_y = cut_dir_y[cut_dir_order == cut_dir]

    return mov_x, mov_y


def reverse_cut_dir_xy(old_cut):
    cut_dir_order = np.asarray([[4, 0, 5], [2, 8, 3], [6, 1, 7]]).flatten()
    new_cat = cut_dir_order[cut_dir_order[::-1] == old_cut]
    return int(new_cat)


if __name__ == '__main__':
    timings = [16.9970068, 17.38013605, 17.77487528, 18.21605442, 18.77333333, 19.40027211, 19.80662132, 20.20136054,
               20.61931973, 21.06049887, 21.43201814, 21.74548753, 22.01251701, 22.2214966, 22.67428571, 23.05741497,
               23.46376417, 23.8585034, 24.06748299, 24.26485261, 24.4738322]
    notes = [[0, 0, 0, 6, 3, 2, 1, 5], [2, 2, 0, 5, 3, 1, 1, 5], [0, 0, 0, 6, 3, 2, 1, 5], [1, 0, 1, 1, 2, 0, 0, 1],
             [1, 0, 1, 1], [2, 0, 1, 0], [2, 0, 1, 1], [2, 0, 1, 1], [2, 0, 1, 0], [1, 0, 0, 1], [2, 0, 1, 0],
             [1, 0, 0, 1, 2, 0, 1, 1], [1, 0, 0, 3, 2, 1, 1, 2], [1, 0, 0, 1], [0, 0, 0, 0, 1, 0, 1, 1], [3, 1, 0, 5],
             [2, 0, 1, 2], [0, 0, 0, 6], [1, 0, 1, 1], [1, 0, 1, 0], [3, 0, 1, 1], [0, 0, 1, 0],
             [0, 0, 0, 1, 1, 0, 1, 1], [2, 0, 0, 7], [1, 0, 0, 0, 3, 0, 1, 1], [2, 0, 0, 7], [3, 1, 1, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [1, 0, 1, 6], [1, 1, 0, 7], [2, 0, 0, 1], [0, 0, 0, 6, 2, 0, 1, 6], [2, 0, 0, 7],
             [2, 0, 0, 1], [2, 0, 0, 1], [3, 1, 1, 7], [1, 1, 1, 0], [0, 1, 0, 2, 3, 1, 1, 3], [0, 1, 0, 4],
             [2, 0, 0, 1], [2, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 1], [3, 1, 1, 0], [2, 0, 0, 0], [2, 0, 0, 1],
             [2, 0, 1, 1], [2, 0, 1, 0], [2, 0, 0, 0], [0, 0, 0, 1], [2, 0, 0, 0], [1, 0, 1, 0], [2, 0, 1, 1],
             [2, 0, 1, 0], [0, 1, 0, 3], [1, 0, 0, 1], [1, 0, 1, 0], [3, 2, 1, 8], [3, 1, 1, 0], [0, 2, 0, 3],
             [1, 2, 0, 0], [2, 0, 0, 1], [2, 0, 1, 1], [1, 0, 0, 1], [0, 0, 0, 0], [1, 0, 1, 1], [3, 1, 1, 0],
             [1, 0, 1, 1], [3, 0, 1, 0], [1, 0, 1, 1], [2, 0, 1, 0], [2, 0, 1, 1], [2, 0, 1, 0], [3, 0, 1, 1],
             [1, 0, 1, 0], [1, 0, 0, 1], [3, 1, 1, 4], [3, 0, 1, 7], [2, 0, 0, 1], [2, 0, 0, 0],
             [2, 2, 0, 0, 3, 2, 1, 0], [2, 2, 0, 5, 3, 1, 1, 5], [0, 1, 0, 2, 3, 1, 1, 5], [1, 2, 1, 5, 3, 0, 0, 6],
             [1, 0, 0, 1, 2, 2, 1, 0], [0, 1, 0, 4, 3, 1, 1, 7], [0, 1, 0, 2, 3, 1, 1, 3], [0, 1, 0, 0, 1, 0, 1, 1],
             [1, 0, 0, 1], [0, 1, 0, 4], [3, 0, 1, 1], [2, 0, 0, 1], [1, 0, 0, 0], [3, 2, 1, 5], [2, 0, 1, 7],
             [1, 0, 0, 0], [1, 0, 0, 1], [3, 0, 1, 1]]
    # name = 'Bodied'
    # sanity_check_timing(name, timings)
    sanity_check_notes(notes, timings)
