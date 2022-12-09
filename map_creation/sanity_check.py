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
    print(f"Got {beat.sum()} beats after sanity check beat"
          f" (removed {beat_counts - beat.sum()})")

    return beat


def sanity_check_timing(name, timings, song_duration):
    samplerate_music = 44100

    #####################################
    # import song to analyze volume peaks
    #####################################
    file = paths.songs_pred + name + ".egg"

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


def sanity_check_notes(notes: list, timings):
    [notes_r, notes_l, notes_b] = split_notes_rl(notes)
    # test = unpslit_notes(notes_r, notes_l, notes_b)

    # notes_r = correct_cut_dir(notes_r, timings)
    # notes_l = correct_cut_dir(notes_l, timings)

    print("Right notes:", end=' ')
    notes_r = correct_notes(notes_r, timings)
    print("Left notes: ", end=' ')
    notes_l = correct_notes(notes_l, timings)

    # TODO: (emphasize important beats with double notes)

    time_diffs = np.concatenate((np.ones(1), np.diff(timings)), axis=0)
    notes_r, notes_l, notes_b = correct_notes_all(notes_r, notes_l, notes_b, time_diffs)

    # rebuild notes
    new_notes = unpslit_notes(notes_r, notes_l, notes_b)
    return new_notes


def correct_notes_all(notes_r, notes_l, notes_b, time_diff):
    pos_r_last = []
    pos_l_last = []
    pos_b_last = []
    last_bomb_idx = 0
    rm_counter = 0
    for idx in range(len(notes_r)):
        nb = notes_b[idx]
        nr = notes_r[idx]
        nl = notes_l[idx]

        def calc_note_pos(n, add_cut=True, inv=False):
            position = []
            for i in range(int(len(n) / 4)):
                pos = n[0 + i * 4:2 + i * 4]
                position.append(pos)

                if add_cut:
                    cut_x, cut_y = get_cut_dir_xy(n[3 + i * 4])
                    if cut_x == cut_y == 0:
                        if not inv:
                            cut_pos = [pos[0] - int(cut_x), pos[1] - int(cut_y)]
                        else:
                            cut_pos = [pos[0] + int(cut_x), pos[1] + int(cut_y)]
                        if 0 <= cut_pos[0] < 4:  # x axis
                            if 0 <= cut_pos[1] < 3:  # y axis
                                position.append(cut_pos)
            return position

        pos_r = calc_note_pos(nr)
        pos_l = calc_note_pos(nl)
        pos_b = calc_note_pos(nb, add_cut=False)

        # check bombs
        if len(pos_b) > 0:
            # calculate next beat for notes
            if idx < len(notes_r):
                pos_r_next = calc_note_pos(notes_r[idx + 1], inv=True)
                pos_l_next = calc_note_pos(notes_l[idx + 1], inv=True)
            else:
                pos_r_next = []
                pos_l_next = []
            # compare bombs and notes
            rm_b = []
            for i in range(len(pos_b)):
                if pos_b[i] in pos_l or pos_b[i] in pos_r or \
                        pos_b[i] in pos_l_last or pos_b[i] in pos_r_last or \
                        pos_b[i] in pos_l_next or pos_b[i] in pos_r_next:
                    # remove bomb
                    rm_b.extend(list(range(i * 4, (i + 1) * 4)))
            rm_b = rm_b[::-1]
            for i in rm_b:
                notes_b[idx].pop(i)
        pos_r_last = pos_r
        pos_l_last = pos_l

        # check left notes
        if len(pos_l) > 0:
            for pl in pos_l:
                if pl in pos_r:
                    # remove left note(s)
                    rm_counter += len(pos_l)
                    notes_l[idx] = []
                    break
    print(f"Static sanity check removed {rm_counter} notes.")

    return notes_r, notes_l, notes_b


def correct_notes(notes, timings):
    nl_last = None
    last_time = 0
    rm_counter = 0
    for idx in range(len(notes)):
        if len(notes[idx]) == 0:
            continue
        # elif len(notes[idx]) == 4:
        elif len(notes[idx]) >= 4:
            # check cut direction movement (of first element)
            notes[idx] = check_note_movement(nl_last, notes[idx])

            # # notes[idx] = optimize_note_movement(nl_last, notes[idx])
            # notes[idx] = check_border_notes(notes, timings, idx)

            # TODO: (change dot notes to cut notes) - schlupfloch aktuell
            # TODO: check notes direction vs position with probabilistic factor (single)
            #       alle am rand richtig drehen, zwischendrinnen löschen...
            # TODO: apply reversed movement to next note (e.g. last[2, 1, 1, 2];new[1, 0, 1, 1])
            #       not really possible without changing the whole track?

            # calculate movement speed (of first element)
            new_time = timings[idx]
            speed = calc_note_speed(nl_last, notes[idx], new_time - last_time,
                                    config.cdf)

            # remove too fast elements
            if speed > config.max_speed:
                rm_counter += int(len(notes[idx]) / 4)
                notes[idx] = []
                continue
            # update last correct note
            else:
                last_time = new_time
                nl_last = notes[idx]

            # check double notes
            if len(notes[idx]) > 4:
                rm_temp = np.zeros_like(notes[idx])
                for n in range(int(len(notes[idx]) / 4) - 1):
                    speed = calc_note_speed(notes[idx][n:n + 4],
                                            notes[idx][n + 4:n + 8],
                                            time_diff=0.05, cdf=0.5)
                    if speed > config.max_double_note_speed:
                        # try to fix second notes
                        try_notes = notes[idx][n + 4:n + 8]
                        try_notes[3] = notes[idx][n:n + 4][3]
                        speed1 = calc_note_speed(notes[idx][n:n + 4],
                                                 try_notes,
                                                 time_diff=0.05, cdf=0.65)
                        speed2 = calc_note_speed(try_notes,
                                                 notes[idx][n:n + 4],
                                                 time_diff=0.05, cdf=0.65)
                        speed = np.min([speed1, speed2])
                        if speed > config.max_double_note_speed:
                            # remove sec notes
                            rm_temp[n + 4:n + 8] = 1
                        else:
                            # include try notes
                            notes[idx][n + 4:n + 8] = try_notes

                # remove unsuited notes
                if rm_temp.sum() > 0:
                    rm_counter += int(rm_temp.sum() / 4)
                    for rm in range(len(rm_temp))[::-1]:
                        if rm_temp[rm]:
                            notes[idx].pop(rm)

    print(f"Sanity check note speed removed {rm_counter} elements")
    return notes


# def optimize_note_movement(notes_last, notes_new):
#     if notes_last is None:
#         return notes_new
#
#     def calc_pos_allowed(notes_last):
#         cx, cy = get_cut_dir_xy(notes_last[3])
#
#         if cx == cy == 0:
#             return None
#
#         x = list(notes_last[0] - np.arange(1, 3)*cx)
#         x.extend(list(notes_last[0] + np.arange(0, 3)*cx))
#         y = list(notes_last[1] - np.arange(1, 3)*cy)
#         y.extend(list(notes_last[1] + np.arange(0, 3)*cy))
#
#         return np.asarray([x, y]).T
#
#     pos_allowed = calc_pos_allowed(notes_last)
#     if pos_allowed is None:
#         return notes_new
#     if list(pos_allowed[0]) == notes_new[0:2] or list(pos_allowed[1]) == notes_new[0:2]:
#         return notes_new
#     random_ar = np.arange(5, 0, -1) * np.random.rand(5)
#     random_idx = np.argsort(random_ar)[::-1]
#     for idx in range(5):
#         x, y = list(pos_allowed[random_idx][idx])
#         if 0 <= x < 4 and 0 <= y < 3:
#             notes_new[0:2] = [x, y]
#             return notes_new


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

        # change cut direction
        # TODO: (check if new cut direction needs more speed
        #       only if timing < 0.5)

        new_cut = reverse_cut_dir_xy(notes_last[3])
        notes_new[3] = new_cut

    return notes_new


def calc_note_speed(notes_last, notes_new, time_diff, cdf):
    if notes_last is None:
        return 0

    # cut director factor
    # cdf = config.cdf

    dist = 0.5      # reaction time
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
    notes = np.load(paths.temp_path + 'notes.npy', allow_pickle=True)
    timings = np.load(paths.temp_path + 'timings.npy', allow_pickle=True)
    notes = list(notes)

    sanity_check_notes(notes, timings)
