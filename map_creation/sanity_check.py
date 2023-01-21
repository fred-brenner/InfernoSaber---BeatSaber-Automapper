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
    threshold = pitches.mean() * config.thresh_pitch * 3
    threshold_end = 2.2 * threshold
    idx_end = int(len(pitches) / 30)
    idx_end_list = list(range(idx_end))
    idx_end_list.extend(list(range(len(pitches)-idx_end, len(pitches))))
    beat_flag = False
    beat_pos = np.zeros_like(pitches)
    for idx in range(len(pitches)):
        if idx in idx_end_list:
            cur_thresh = threshold_end
        else:
            cur_thresh = threshold
        if pitches[idx] > last_pitch and pitches[idx] > cur_thresh:
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
    max_time_diff = 0.5
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


def emphasize_beats(notes, timings):

    def calc_new_note(note, new_pos):
        new_note = note * len(new_pos)
        for i in range(len(new_pos)):
            new_note[i * 4:i * 4 + 2] = new_pos[i]
        return new_note

    def update_new_note(notes, n, new_note):
        notes[n] = new_note
        return notes

    for n in range(len(notes)):
        if timings[n:n+1].max() >= config.emphasize_beats_wait:
            note = notes[n]
            if len(note) > 0:
                rd = np.random.random()
                if rd > 1 - config.emphasize_beats_3:
                    new_pos = calc_note_pos(note)
                    new_note = calc_new_note(note, new_pos)
                    if len(new_pos) < 3:
                        new_pos = calc_note_pos(new_note)[:3]
                        new_note = calc_new_note(note, new_pos)
                    update_new_note(notes, n, new_note)
                elif rd > 1 - config.emphasize_beats_3 - config.emphasize_beats_2:
                    new_pos = calc_note_pos(note)[:2]
                    new_note = calc_new_note(note, new_pos)
                    update_new_note(notes, n, new_note)

    return notes


def sanity_check_notes(notes: list, timings):
    [notes_r, notes_l, notes_b] = split_notes_rl(notes)
    # test = unpslit_notes(notes_r, notes_l, notes_b)

    # notes_r = correct_cut_dir(notes_r, timings)
    # notes_l = correct_cut_dir(notes_l, timings)

    print("Right notes:", end=' ')
    notes_r = correct_notes(notes_r, timings)
    print("Left notes: ", end=' ')
    notes_l = correct_notes(notes_l, timings)

    time_diffs = np.concatenate((np.ones(1), np.diff(timings)), axis=0)

    # shift notes in cut direction
    notes_l = shift_blocks_up_down(notes_l, time_diffs)
    notes_r = shift_blocks_up_down(notes_r, time_diffs)
    # print("Right notes:", end=' ')
    # notes_r = correct_notes(notes_r, timings)
    # print("Left notes: ", end=' ')
    # notes_l = correct_notes(notes_l, timings)

    # emphasize some beats randomly
    notes_l = emphasize_beats(notes_l, time_diffs)
    notes_r = emphasize_beats(notes_r, time_diffs)

    # check static position for next and last note for left and right together
    notes_r, notes_l, notes_b = correct_notes_all(notes_r, notes_l, notes_b, time_diffs)

    # rebuild notes
    new_notes = unpslit_notes(notes_r, notes_l, notes_b)
    return new_notes


def calc_note_pos(n, add_cut=True, inv=None):
    position = []
    for i in range(int(len(n) / 4)):
        pos = n[0 + i * 4:2 + i * 4]
        if pos not in position:
            position.append(pos)

        if add_cut:
            def add_position(n, pos, inv):
                cut_x, cut_y = get_cut_dir_xy(n[3 + i * 4])
                if cut_x == cut_y == 0:
                    return None
                else:
                    # cut_pos = [pos[0] - int(cut_x), pos[1] - int(cut_y)]
                    # cut_pos.append([pos[0] + int(cut_x), pos[1] + int(cut_y)])
                    if not inv:
                        cut_pos = [pos[0] - int(cut_x), pos[1] - int(cut_y)]
                    else:
                        cut_pos = [pos[0] + int(cut_x), pos[1] + int(cut_y)]
                    if 0 <= cut_pos[0] < 4:  # x axis
                        if 0 <= cut_pos[1] < 3:  # y axis
                            return cut_pos
                return None

            if inv is None:
                cut_pos = add_position(n, pos, True)
                if cut_pos is not None and cut_pos not in position:
                    position.append(cut_pos)
                cut_pos = add_position(n, pos, False)
                if cut_pos is not None and cut_pos not in position:
                    position.append(cut_pos)
            else:
                cut_pos = add_position(n, pos, inv)
                if cut_pos is not None and cut_pos not in position:
                    position.append(cut_pos)

    return position


def cut_dir_values(notes):
    cut_values = []
    for idx in range(int(len(notes) / 4)):
        cut_values.append(notes[idx * 4 + 3])
    return cut_values


def offset_notes(notes_l, idx, offset, rm_counter):
    for i in range(int(len(notes_l[idx]) / 4)):
        if i > 1:
            print("")
        pos_before = np.asarray(notes_l[idx][i*4:i*4+2])
        new_pos = list(pos_before - [offset, 0])
        if 0 <= new_pos[0] < 4 and 0 <= new_pos[1] < 3:
            notes_l[idx][i*4:i*4+2] = new_pos
        else:
            rm_counter += 1
        return notes_l, rm_counter


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

        pos_r = calc_note_pos(nr)
        pos_l = calc_note_pos(nl)
        pos_b = calc_note_pos(nb, add_cut=False)
        # cut_r = cut_dir_values(nr)

        # calculate next beat for notes
        if idx < len(notes_r) - 1:
            pos_r_next = calc_note_pos(notes_r[idx + 1])
            pos_l_next = calc_note_pos(notes_l[idx + 1])
        else:
            pos_r_next = []
            pos_l_next = []

        # check bombs
        if len(pos_b) > 0:
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

        # check left notes
        if len(pos_l) > 0:
            pos_r_check = pos_r_last.copy()
            pos_r_check.extend(pos_r)
            pos_r_check.extend(pos_r_next)
            for pl in pos_l:
                if pl in pos_r_check:
                    # try to re-arrange left note
                    offset = [2, 3, 1, -1, -2]
                    for offs in offset:
                        pl_opt = list(np.asarray(pl) - [offs, 0])
                        if 0 <= pl_opt[0] < 4 and 0 <= pl_opt[1] < 3:
                            if pl_opt not in pos_r_check:
                                notes_l, rm_counter = offset_notes(notes_l, idx, offs, rm_counter)
                                pos_l = calc_note_pos(notes_l[idx])
                                break
                    # if second_run:
                    for pl_test in pos_l:
                        if pl_test in pos_r_check:
                            # remove left note(s)
                            rm_counter += len(pos_l)
                            notes_l[idx] = []
                            break
                    break

        pos_r_last = pos_r
        pos_l_last = pos_l
        # cut_r_last = cut_r
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
                cut_dir = notes[idx][3]
                for n in range(int(len(notes[idx]) / 4) - 1):
                    # check if cut direction is same
                    if notes[idx][(n+1)*4 + 3] != cut_dir:
                        notes[idx][(n+1)*4 + 3] = 8
                    n *= 4
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

    dist = config.reaction_time     # reaction time
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


################
# Postprocessing
################
def fill_map_times(map_times):
    diff = np.diff(map_times)
    new_map_times = []
    for idx in range(len(diff)):
        if config.add_beat_low_bound < diff[idx] < config.add_beat_hi_bound:
            if np.random.random() < config.add_beat_fact:
                beat_time = (map_times[idx] + map_times[idx+1]) / 2
                new_map_times.append(beat_time)
    if len(new_map_times) > 0:
        map_times = np.hstack((map_times, new_map_times))
        map_times = np.sort(map_times)
    return map_times


def shift_blocks_up_down(notes: list, time_diffs: np.array):
    for idx in range(len(notes)):
        if len(notes[idx]) > 2:
            note_pos = calc_note_pos(notes[idx], add_cut=False)
            cut_x, cut_y = get_cut_dir_xy(notes[idx][3])

            new_pos = []
            new_pos2 = []
            # if cut_y == -1:    # up
            for pos in note_pos:
                new_pos.append([int(pos[0]-cut_x), int(pos[1]-cut_y)])
                new_pos2.append([int(pos[0]-2*cut_x), int(pos[1]-2*cut_y)])

            # check all new positions:
            valid = check_note_pos_valid(new_pos)
            valid2 = check_note_pos_valid(new_pos2)     # if true always shift
            if valid:
                if valid2 or np.random.random() < config.shift_beats_fact:
                    for ipos in range(len(new_pos)):
                        notes[idx][0+4*ipos] = new_pos[ipos][0]
                        notes[idx][1+4*ipos] = new_pos[ipos][1]
    return notes


def check_note_pos_valid(positions: list) -> bool:
    for pos in positions:
        if not 0 <= pos[0] <= 3:    # line index
            return False
        if not 0 <= pos[1] <= 2:    # line layer
            return False
    return True


if __name__ == '__main__':
    notes = np.load(paths.temp_path + 'notes.npy', allow_pickle=True)
    timings = np.load(paths.temp_path + 'timings.npy', allow_pickle=True)
    notes = list(notes)

    sanity_check_notes(notes, timings)
