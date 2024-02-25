"""
This script contains all analytic functions
to improve the note generation output
"""

import numpy as np
import aubio
import librosa
# import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from tools.config import config, paths
from tools.utils.numpy_shorts import get_factor_from_max_speed, add_onset_half_times


def sanity_check_notes(notes: list, timings: list):
    # last sanity check for notes,
    # result is written to map

    [notes_r, notes_l, notes_b] = split_notes_rl(notes)

    notes_r = set_first_note_dir(notes_r)
    notes_l = set_first_note_dir(notes_l)

    # notes_r = correct_cut_dir(notes_r, timings)
    # notes_l = correct_cut_dir(notes_l, timings)

    # print("Right notes:", end=' ')
    notes_r = correct_notes(notes_r, timings)
    # print("Left notes: ", end=' ')
    notes_l = correct_notes(notes_l, timings)

    time_diffs = np.concatenate((np.ones(1), np.diff(timings)), axis=0)

    if config.add_waveform_pattern_flag > 0:
        # shift consecutive blocks into waveform
        notes_l, notes_r = apply_waveform_pattern(notes_l, notes_r)

    # shift notes in cut direction
    notes_l = shift_blocks_up_down(notes_l, time_diffs)
    notes_r = shift_blocks_up_down(notes_r, time_diffs)
    # shift notes left and right for better flow
    notes_l, notes_r = shift_blocks_left_right(notes_l, notes_r, time_diffs)

    if config.add_waveform_pattern_flag > 1:
        # shift consecutive blocks into waveform
        notes_l, notes_r = apply_waveform_pattern(notes_l, notes_r)

    # check static position for next and last note for left and right together
    notes_r, notes_l, notes_b = correct_notes_all(notes_r, notes_l, notes_b, time_diffs)

    # shift notes away from the middle
    notes_r, notes_l, notes_b = shift_blocks_middle(notes_r, notes_l, notes_b)

    # # (TODO: add bombs for long pause to focus on next note direction)
    # notes_b, timings_b = add_pause_bombs(notes_r, notes_l, notes_b, timings, pitch_algo, pitch_times)
    # (TODO: remove blocking bombs)

    # turn notes leading into correct direction
    notes_r, dot_idx_r = turn_notes_single(notes_r)
    notes_l, dot_idx_l = turn_notes_single(notes_l)

    if config.add_breaks_flag:
        # from tools.utils.load_and_save import save_pkl
        notes_l = add_breaks(notes_l, timings)
        notes_r = add_breaks(notes_r, timings)

    if config.emphasize_beats_flag:
        # emphasize some beats randomly
        notes_l = emphasize_beats(notes_l, time_diffs, notes_r)
        notes_r = emphasize_beats(notes_r, time_diffs, notes_l)

    if config.allow_dot_notes:
        notes_l = apply_dots(notes_l, dot_idx_l)
        notes_r = apply_dots(notes_r, dot_idx_r)
        if config.add_dot_notes > 0:
            notes_l = add_dots(notes_l, time_diffs.copy())
            notes_r = add_dots(notes_r, time_diffs.copy())

    # rebuild notes
    new_notes = unpslit_notes(notes_r, notes_l, notes_b)
    return new_notes


def sanity_check_beat(beat):
    beat = beat.reshape(len(beat))
    # beat_counts = beat.sum()
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
    # print(f"Got {beat.sum()} beats after sanity check beat"
    #       f" (removed {beat_counts - beat.sum()})")

    return beat


def sanity_check_timing2(name, timings):
    # TODO: add threshold end implementation
    samplerate_music = 44100
    factor = config.thresh_onbeat / config.thresh_onbeat_orig
    pre_max = 1 if int(5 * factor) <= 0 else int(5 * factor)
    post_max = 1 if int(5 * factor) <= 0 else int(5 * factor)
    pre_avg = 1 if int(7 * factor) <= 0 else int(7 * factor)
    post_avg = 1 if int(8 * factor) <= 0 else int(8 * factor)

    delta = config.thresh_onbeat
    wait = int(8 * factor)
    max_time_diff = 1.0

    file = paths.songs_pred + name + ".egg"
    # Load the audio file
    y, sr = librosa.load(file, sr=samplerate_music)

    # Compute the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

    # Detect onsets using amplitude thresholding
    try:
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512,
                                            backtrack=False, pre_max=pre_max, post_max=post_max,
                                            pre_avg=pre_avg, post_avg=post_avg, delta=delta, wait=wait)
    except Exception as e:
        print(f"Error: {type(e).__name__}")
        print(f"Error message: {e}")
        print(f"Error in onset detection for song: {file}\nPlease remove song and restart.")
        exit()

    # Convert frame indices to time (in seconds)
    onsets_sec = librosa.frames_to_time(onsets, sr=sr, hop_length=512)

    # TODO: make two loops, first using original timings, second with added half steps
    onsets_sec_temp = np.copy(onsets_sec)
    iterations = int(config.max_speed / 9)
    if iterations < 1:
        iterations = 1
    elif iterations > 5:
        iterations = 5
    for iteration in range(iterations):
        del_times = []  # only keep delete field from last iteration
        if iteration > 0:
            min_time = 0.1 / iteration
            max_time = 1 * iteration
            onsets_sec_temp = add_onset_half_times(onsets_sec_temp, min_time, max_time)
        tim_old = 0
        for idx, tim in enumerate(timings):
            diff_ar = np.abs(onsets_sec_temp - tim)
            if np.min(diff_ar) < max_time_diff and tim_old != tim:
                tim_old = tim
                # set new time
                timings[idx] = onsets_sec_temp[np.argmin(diff_ar)]
            else:
                # delete old time
                del_times.append(idx)
                # timings[idx] = 0

    for del_t in del_times:
        timings[del_t] = 0
    return timings


# def sanity_check_timing(name, timings, song_duration):
#     samplerate_music = 44100
#
#     #####################################
#     # import song to analyze volume peaks
#     #####################################
#     file = paths.songs_pred + name + ".egg"
#
#     # analyze song pitches
#     total_read = 0
#     onset_list = []
#     tempo_list = []
#     notes_list = []
#     # samples_list = []
#     src = aubio.source(file, channels=1, samplerate=samplerate_music)
#     aubio_tempo = aubio.tempo(samplerate=samplerate_music)
#     aubio_onset = aubio.onset(samplerate=samplerate_music)
#     aubio_notes = aubio.notes(samplerate=samplerate_music)
#     while True:
#         samples, read = src()
#         onset = aubio_onset(samples)
#         tempo = aubio_tempo(samples)
#         notes = aubio_notes(samples)
#         # samples_list.extend(samples)
#         onset_list.extend(onset)
#         tempo_list.extend(tempo)
#         notes_list.extend(notes)
#         total_read += read
#         if read < src.hop_size:
#             break
#
#     # calc volume peaks
#     pitches = np.asarray(notes_list)
#     # len(pitch_list) * 512 / samplerate_music = time in seconds
#     # plt.plot(pitches)
#     # plt.show()
#
#     last_pitch = 0
#     threshold = np.quantile(pitches[pitches > 0], config.thresh_pitch) * config.thresh_pitch
#     # if threshold < 100:
#     #     if np.mean(pitches) > threshold:
#     #         threshold = np.mean(pitches)
#
#     threshold_end = config.threshold_end * threshold
#     idx_end = int(len(pitches) / 30)
#     idx_end_list = list(range(idx_end))
#     idx_end_list.extend(list(range(len(pitches) - idx_end, len(pitches))))
#     beat_flag = False
#     beat_pos = np.zeros_like(pitches)
#     for idx in range(len(pitches)):
#         if idx in idx_end_list:
#             cur_thresh = threshold_end
#         else:
#             cur_thresh = threshold
#         if pitches[idx] > last_pitch and pitches[idx] > cur_thresh:
#             beat_flag = True
#         else:
#             if beat_flag:
#                 beat_pos[idx - 1] = 1
#                 beat_flag = False
#         last_pitch = pitches[idx]
#
#     # plt.plot(beat_pos)
#     # plt.show()
#
#     allowed_timings = beat_pos * np.arange(0, len(beat_pos), 1)
#     allowed_timings *= 512 / samplerate_music
#     allowed_timings = allowed_timings[allowed_timings > 0]
#
#     # match timing from beat generator
#     max_time_diff = 1.0
#     early_start = 0
#     last_beat = 1
#     for i in range(len(timings)):
#         diff = np.abs(allowed_timings - timings[i] - early_start)
#         min_diff = diff.min()
#         if min_diff < max_time_diff:
#             cur_beat = allowed_timings[np.argmin(diff)]
#             if last_beat < cur_beat < song_duration:
#                 timings[i] = cur_beat
#                 last_beat = cur_beat
#             else:
#                 timings[i] = 0
#         else:
#             timings[i] = 0
#
#     return timings, allowed_timings


def improve_timings(new_notes, timings, pitch_input, pitch_times):
    # Improve timings after all notes have been set.
    # Use the pitch detection to find the perfect timing between notes
    # (currently 0.035s accuracy)
    mc_factor = config.improve_timings_mcfactor
    max_change = config.improve_timings_mcchange * mc_factor
    activation_time = np.float64(config.improve_timings_act_time / mc_factor)
    activation_time_index = np.where(pitch_times >= activation_time)[0][0]

    def check_for_notes(new_notes, idx):
        if idx > len(new_notes) - 2 or idx < 1:
            return 1
        if len(new_notes[idx]) > 0:
            return 1
        return 0

    def get_surrounding_beats(timings, idx, new_notes):
        if idx == 0:
            last_beat = 0.0
            idx_iterator = idx + 1
            while not check_for_notes(new_notes, idx_iterator):
                idx_iterator += 1
            next_beat = timings[idx_iterator]
        elif idx >= len(timings) - 1:
            idx_iterator = idx - 1
            while not check_for_notes(new_notes, idx_iterator):
                idx_iterator -= 1
            last_beat = timings[idx_iterator]
            next_beat = timings[idx] + max_change
        else:
            idx_iterator = idx - 1
            while not check_for_notes(new_notes, idx_iterator):
                idx_iterator -= 1
            last_beat = timings[idx_iterator]
            # if idx_iterator == 0:
            #     last_beat -= max_change
            idx_iterator = idx + 1
            while not check_for_notes(new_notes, idx_iterator):
                idx_iterator += 1
            next_beat = timings[idx_iterator]
            # if idx_iterator == len(timings):
            #     next_beat += max_change
        return last_beat, next_beat

    for idx in range(len(timings)):
        if check_for_notes(new_notes, idx):
            cur_beat = timings[idx]
            pre_beat, post_beat = get_surrounding_beats(timings, idx, new_notes)
            t_diff_pre = np.min([cur_beat - pre_beat, max_change]) / mc_factor
            t_diff_post = np.min([post_beat - cur_beat, max_change]) / mc_factor

            pre_beat = cur_beat - t_diff_pre
            post_beat = cur_beat + t_diff_post

            # cur_idx_pitch = np.argmin(np.abs(pitch_times - np.float64(cur_beat)))
            pre_idx_pitch = np.argmin(np.abs(pitch_times - np.float64(pre_beat)))
            post_idx_pitch = np.argmin(np.abs(pitch_times - np.float64(post_beat)))

            if post_idx_pitch - pre_idx_pitch > activation_time_index:
                if np.max(pitch_input[pre_idx_pitch:post_idx_pitch + 1]) > 0:
                    new_idx = pre_idx_pitch + np.argmax(pitch_input[pre_idx_pitch:post_idx_pitch + 1])
                    new_timing = pitch_times[new_idx]
                    timings[idx] = new_timing

    return timings


def apply_dots(notes_single, dots_idx):
    for idx in dots_idx:
        if len(notes_single[idx]) == 4:
            notes_single[idx][3] = 8
        elif len(notes_single[idx]) > 4:
            pass  # do not apply for multiple notes
        else:
            pass  # note removed, no change needed
    return notes_single


def add_dots(notes_single, time_diffs):
    # Set timings without notes to 100
    time_diffs[[len(notes_single[idx]) != 4 for idx in range(len(notes_single))]] = 100
    # Get indices for x percent fastest notes
    idx_fast = np.argsort(time_diffs)
    idx_fast = idx_fast[:int(config.add_dot_notes * sum(time_diffs < 100) / 100)]
    for idx in idx_fast:
        notes_single[idx][3] = 8
    return notes_single


def add_breaks(notes_single, timings):
    break_counter = 0
    start_window = config.decr_speed_range
    # remove timing without notes
    idx_diffs = [len(notes_single[idx]) >= 4 for idx in range(len(notes_single))]
    real_timings = timings[idx_diffs]
    real_diffs = np.hstack([1, np.diff(real_timings)])
    real_timings = real_timings[start_window:-start_window]
    real_diffs = real_diffs[start_window:-start_window]
    if len(real_timings) > 50:
        # apply window filter to diffs
        real_diffs_filt = savgol_filter(real_diffs, window_length=31, polyorder=4)
        # if False:
        #     plt.figure()
        #     plt.plot(real_diffs)
        #     plt.plot(real_diffs_filt)
        #     plt.show()

        thresh_diffs = np.quantile(real_diffs_filt, 0.37)
        strong_counter = 0
        # strong_reset = 0
        strong_reset_threshold = 2
        pattern_length = 15
        for idx, diff in enumerate(real_diffs_filt):
            if diff < thresh_diffs:
                # strong pattern
                strong_counter += 1
                # if strong_reset >= strong_reset_threshold:
                # strong_reset = 0
            else:
                if strong_counter > pattern_length:
                    if idx < len(real_diffs_filt) - 2:
                        if real_diffs_filt[idx + 1] >= thresh_diffs and real_diffs_filt[idx + 2] >= thresh_diffs:
                            # remove next two notes

                            # if strong_reset >= strong_reset_threshold:
                            #     if strong_counter > pattern_length:
                            # add break
                            cur_idx = int(np.argwhere(timings == real_timings[idx])[0])
                            # remove this note
                            notes_single[cur_idx] = []
                            # remove next note
                            for next_idx, note_flag in enumerate(idx_diffs[cur_idx:]):
                                if next_idx != 0 and note_flag:
                                    notes_single[cur_idx + next_idx] = []
                                    break
                            # reset counters
                            # strong_reset = 0
                            strong_counter = 0
                            break_counter += 1
                    # else:
                    #     # reset pattern
                    #     strong_counter = 0
                # else:
                #     strong_reset += 1
    # print(f"Add {break_counter} breaks.")
    return notes_single


def emphasize_beats(notes, timings, notes_second):
    emphasize_beats_3 = config.emphasize_beats_3 + config.emphasize_beats_3_fact * config.max_speed
    emphasize_beats_2 = config.emphasize_beats_2 + config.emphasize_beats_2_fact * config.max_speed
    start_end_idx = 4
    emphasize_beats_wait = np.quantile(timings, config.emphasize_beats_quantile) + 0.05

    def calc_new_note(note, new_pos):
        new_note = note * len(new_pos)
        for i in range(len(new_pos)):
            new_note[i * 4:i * 4 + 2] = new_pos[i]
        return new_note

    def update_new_note(notes, n, new_note, notes_second):
        if len(new_note) > 4:
            # get note positions on both sides
            notes_second_pos = []
            for i in range(0, len(notes_second[n]), 4):
                notes_second_pos.append(notes_second[n][i + 0:i + 2])
            if len(notes_second_pos) > 0:
                # only check if other side has notes too
                notes_pos = []
                for i in range(0, len(new_note), 4):
                    notes_pos.append(new_note[i + 0:i + 2])
                # sanity check for overlapping notes
                for new_pos_i in range(len(notes_pos) - 1, -1, -1):
                    if notes_pos[new_pos_i] in notes_second_pos:
                        # remove particular note
                        new_note.pop(new_pos_i * 4)
                        new_note.pop(new_pos_i * 4)
                        new_note.pop(new_pos_i * 4)
                        new_note.pop(new_pos_i * 4)

            notes[n] = new_note
        return notes

    for n in range(start_end_idx, len(notes)):
        if timings[n:n + 1].max() >= emphasize_beats_wait:  # TODO: maybe increase timing window to before+after
            note = notes[n]
            if len(note) > 0:
                rd = np.random.random()
                if rd > 1 - emphasize_beats_3:
                    new_pos = calc_note_pos(note)
                    new_note = calc_new_note(note, new_pos)
                    if len(new_pos) < 3:
                        new_pos = calc_note_pos(new_note)[:3]
                        new_note = calc_new_note(note, new_pos)
                    notes = update_new_note(notes, n, new_note, notes_second)
                elif rd > 1 - emphasize_beats_3 - emphasize_beats_2:
                    new_pos = calc_note_pos(note)[:2]
                    for ip in range(len(new_pos) - 1, 0, -1):
                        if new_pos[ip] in [[1, 1], [2, 1]]:
                            new_pos.pop(ip)
                    # only check first entry if enough notes are available
                    if len(new_pos) > 1:
                        if new_pos[0] in [[1, 1], [2, 1]]:
                            new_pos.pop(0)
                    new_note = calc_new_note(note, new_pos)
                    notes = update_new_note(notes, n, new_note, notes_second)

    return notes


def set_first_note_dir(notes_x):
    for idx, first_note in enumerate(notes_x):
        if len(first_note) > 0:
            break

    if not config.allow_double_first_notes:
        if len(first_note) > 4:
            # only keep first entry
            first_note = first_note[0:4]
            notes_x[idx] = first_note

    if first_note[3] == 8 or config.check_all_first_notes:
        # dot note is not allowed as first note
        layer_pos = first_note[1]
        if layer_pos >= config.first_note_layer_threshold:
            # change direction to upwards
            new_dir = 0
        else:
            # change direction to downwards
            new_dir = 1
        notes_x[idx][3] = new_dir

    return notes_x


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
        pos_before = np.asarray(notes_l[idx][i * 4:i * 4 + 2])
        new_pos = list(pos_before - [offset, 0])
        if 0 <= new_pos[0] < 4 and 0 <= new_pos[1] < 3:
            notes_l[idx][i * 4:i * 4 + 2] = new_pos
        else:
            rm_counter += 1
        return notes_l, rm_counter


def correct_notes_all(notes_r, notes_l, notes_b, time_diff):
    pos_r_last = []
    pos_l_last = []
    # pos_b_last = []
    # last_bomb_idx = 0
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
    # print(f"Static sanity check removed {rm_counter} notes.")

    return notes_r, notes_l, notes_b


def shift_blocks_middle(notes_r, notes_l, notes_b):
    # Shift blocks up or down to prevent blocking view
    counter = 0
    for idx in range(len(notes_r)):
        nb = notes_b[idx]
        nr = notes_r[idx]
        nl = notes_l[idx]

        pos_r = calc_note_pos(nr, add_cut=False)
        pos_l = calc_note_pos(nl, add_cut=False)
        pos_b = calc_note_pos(nb, add_cut=False)

        pos_all = pos_r.copy()
        pos_all.extend(pos_l)
        pos_all.extend(pos_b)

        for ir in range(len(pos_r)):
            if len(pos_r) <= 1:
                if pos_r[ir] in [[1, 1], [2, 1]]:
                    new_pos = calc_note_pos(nr, add_cut=True, inv=False)[-1]
                    if new_pos not in pos_all:
                        # change note down or up
                        notes_r[idx][ir * 4:ir * 4 + 2] = [new_pos[0], new_pos[1]]
                        counter += 1
                    else:
                        new_pos[0] = pos_r[ir][0]
                        if new_pos not in pos_all:
                            # change note down or up
                            notes_r[idx][ir * 4:ir * 4 + 2] = [new_pos[0], new_pos[1]]
                            counter += 1

        for il in range(len(pos_l)):
            if len(pos_l) <= 1:
                if pos_l[il] in [[1, 1], [2, 1]]:
                    new_pos = calc_note_pos(nl, add_cut=True, inv=False)[-1]
                    if new_pos not in pos_all:
                        # change note down or up
                        notes_l[idx][il * 4:il * 4 + 2] = [new_pos[0], new_pos[1]]
                        counter += 1
                    else:
                        new_pos[0] = pos_l[il][0]
                        if new_pos not in pos_all:
                            # change note down or up
                            notes_l[idx][il * 4:il * 4 + 2] = [new_pos[0], new_pos[1]]
                            counter += 1

        for ib in range(len(pos_b)):
            if pos_b[ib] in [[1, 1], [2, 1]]:
                new_pos = [pos_b[ib][0], 0]
                if new_pos not in pos_all:
                    # change note down or up
                    notes_b[idx][ib * 4:ib * 4 + 2] = new_pos
                    counter += 1

    # print(f"Shifted {counter} blocks away from the middle.")

    return notes_r, notes_l, notes_b


# def add_pause_bombs(notes_r, notes_l, notes_b, timings, pitch_algo, pitch_times):
#     # add bombs between notes to improve flow
#
#     # calculate time difference between notes
#     time_diffs = np.concatenate((np.ones(1), np.diff(timings)), axis=0)
#     note_r_mask = np.asarray([1 if len(x) > 0 else 0 for x in notes_r])
#     note_l_mask = np.asarray([1 if len(x) > 0 else 0 for x in notes_l])
#     time_diff_r = time_diffs * note_r_mask
#     time_diff_l = time_diffs * note_l_mask
#
#     # walk through notes times
#     new_bomb_times = []
#     new_bomb_pos = []
#
#     def add_bomb_pos(up=True, right=True, n=1):
#         if up:
#             n_up = 2
#         else:
#             n_up = 0
#         if right:
#             n_right = [2, 3]
#         else:
#             n_left = [0, 1]
#         # [2, 2, 3, 8, 3, 2, 3, 8]
#         bomb_ar = n * [[n_right[0], n_up, 3, 8, n_right[1], n_up, 3, 8]]
#         return bomb_ar
#
#     def get_time_pos(time_window, pitch_algo, pitch_times):
#         # get the time indices
#         pitch_times = pitch_times[0]
#         t1_all = pitch_times - time_window[0]
#         t1_all = np.argmax(t1_all >= config.t_diff_bomb_react)
#         t2_all = pitch_times - time_window[1]
#         t2_all = np.argmin(t2_all <= -config.t_diff_bomb_react) - 1
#         t1_beat = pitch_algo - time_window[0]
#         t1_beat = np.argmax(t1_beat >= config.t_diff_bomb_react)
#         t2_beat = pitch_algo - time_window[1]
#         t2_beat = np.argmin(t2_beat <= -config.t_diff_bomb_react) - 1
#
#         t_pos_all = pitch_times[t1_all:t2_all]
#         if t1_beat < t2_beat:
#             t_pos_beat = pitch_algo[t1_beat:t2_beat]
#         else:
#             t_pos_beat = []
#         return t_pos_all, t_pos_beat
#
#     # check right notes
#     for idx, t_diff in enumerate(time_diff_r):
#         if idx == 0:
#             continue
#         # (TODO: not finished)
#         if t_diff > config.t_diff_bomb:
#             # add bombs
#             cur_notes = notes_r[idx]
#             # get all positions of the current notes
#             positions = calc_note_pos(cur_notes, add_cut=False)
#             # get (first) cut_dir
#             cut_x, cut_y = get_cut_dir_xy(cur_notes[3])
#
#             # get time window
#             time_window = [timings[idx - 1], timings[idx]]
#             time_pos_all, time_pos_beat = get_time_pos(time_window, pitch_algo, pitch_times)
#
#             if cut_y == 1:
#                 # next note is downwards -> bombs down
#                 new_bomb_times.extend(time_pos_all)
#                 new_bomb_pos.extend(add_bomb_pos(up=False, right=True, n=len(time_pos_all)))
#             elif cut_y == -1:
#                 # next note is upwards -> bombs up
#                 new_bomb_times.extend(time_pos_all)
#                 new_bomb_pos.extend(add_bomb_pos(up=True, right=True, n=len(time_pos_all)))
#             else:
#                 pass
#
#     # check left notes
#     for idx, t_diff in enumerate(time_diff_l):
#         if idx == 0:
#             continue
#         if t_diff > config.t_diff_bomb:
#             # add bombs
#             cur_notes = notes_l[idx]
#             # get all positions of the current notes
#             positions = calc_note_pos(cur_notes, add_cut=False)
#             # get (first) cut_dir
#             cut_x, cut_y = get_cut_dir_xy(cur_notes[3])
#
#             # get time window
#             time_window = [timings[idx - 1], timings[idx]]
#             time_pos_all, time_pos_beat = get_time_pos(time_window, pitch_algo, pitch_times)
#
#             if cut_y == 1:
#                 # next note is downwards -> bombs down
#                 new_bomb_times.extend(time_pos_all)
#                 new_bomb_pos.extend(add_bomb_pos(up=False, right=False, n=len(time_pos_all)))
#             elif cut_y == -1:
#                 # next note is upwards -> bombs up
#                 new_bomb_times.extend(time_pos_all)
#                 new_bomb_pos.extend(add_bomb_pos(up=True, right=False, n=len(time_pos_all)))
#             else:
#                 pass
#
#     # add new bombs to notes framework
#     new_times = list(timings)
#     new_times.extend(new_bomb_times)
#     return notes_b


def turn_notes_single(notes_single):
    dot_notes_rem = []

    def calc_diff_from_list(cd_old, cd_new):
        diff_score = abs(cd_old[0] - cd_new[0]) + abs(cd_old[1] - cd_new[1])
        return diff_score

    def get_move_dir_xy(notes, notes_old):
        dirx = -1 * (notes[0] - notes_old[0])
        diry = -1 * (notes[1] - notes_old[1])
        move_grid_threshold = 1
        if abs(dirx) >= abs(diry) + move_grid_threshold:
            diry = 0
        elif abs(diry) >= abs(dirx) + move_grid_threshold:
            dirx = 0
        if abs(dirx) > 1:
            dirx = np.sign(dirx)
        if abs(diry) > 1:
            diry = np.sign(diry)
        return dirx, diry

    if config.flow_model_flag:
        # empty_note_last = False
        notes_old = None
        for idx, notes in enumerate(notes_single):
            if len(notes) == 0:
                continue  # skip empty notes
            if notes_old is None:
                notes_old = notes
                continue
            if len(notes) > 4:
                # skip multi notes
                notes_old = None
                continue
            dirx, diry = get_move_dir_xy(notes, notes_old)
            if notes[3] == 8:
                if config.allow_dot_notes:
                    dot_notes_rem.append(idx)  # remember to redo this
                #     if not empty_note_last:
                #         new_cut_dir = reverse_get_cut_dir(dirx, diry)
                #         notes[3] = new_cut_dir
                #         notes_single[idx] = notes
                #         empty_note_last = True
                #     else:
                #         notes_old = None
                #         continue
                # else:
                new_cut_dir = reverse_get_cut_dir(dirx, diry)
                notes[3] = new_cut_dir
                notes_single[idx] = notes
            # else:  # last note has direction
            #     empty_note_last = False

            # check if new flow direction suits to (inverse last) cut direction
            cd_old = get_cut_dir_xy(notes_old[3])
            cd_old = (cd_old[0] * -1, cd_old[1] * -1)
            diff_score = calc_diff_from_list(cd_old, [dirx, diry])
            if diff_score == 0:
                pass
            elif diff_score == 1:
                # use new cut direction
                new_cut_dir = reverse_get_cut_dir(dirx, diry)
                notes[3] = new_cut_dir
                notes_single[idx] = notes
            elif diff_score == 2:
                if dirx == cd_old[0] or diry == cd_old[1]:
                    new_dirx = int(np.round(0.5 * (dirx + cd_old[0]), 0))
                    new_diry = int(np.round(0.5 * (diry + cd_old[1]), 0))
                    new_cut_dir = reverse_get_cut_dir(new_dirx, new_diry)
                    notes[3] = new_cut_dir
                    notes_single[idx] = notes
                elif config.allow_dot_notes:
                    dot_notes_rem.append(idx)
                    # notes[3] = 8
                    # notes_single[idx] = notes
                else:
                    pass
            elif diff_score == 3:
                pass
            else:
                pass

            # update old notes
            notes_old = notes

    # Sanity check cut directions
    notes_old = None
    for idx, notes in enumerate(notes_single):
        if len(notes) == 0:
            continue  # skip empty notes
        if notes_old is None:
            notes_old = notes
            continue
        cd_new_x, cd_new_y = list(get_cut_dir_xy(notes[3]))
        cd_old_x, cd_old_y = list(get_cut_dir_xy(notes_old[3]))
        if cd_new_x == cd_new_y == 0 or cd_old_x == cd_old_y == 0:
            if config.allow_dot_notes:
                notes[3] = reverse_cut_dir_xy(notes_old[3])
                notes = notes[:4]  # make it single notes if necessary
                notes_single[idx] = notes
                dot_notes_rem.append(idx)
                cd_new_x, cd_new_y = list(get_cut_dir_xy(notes[3]))
                # notes_old = notes
                # continue  # skip notes without direction
            else:
                notes_single[idx] = []
                continue
        # inverse old cut dir
        cd_old_x *= -1
        cd_old_y *= -1

        df_score = calc_diff_from_list([cd_old_x, cd_old_y], [cd_new_x, cd_new_y])
        if df_score >= 3:
            if config.allow_mismatch_flag:
                notes[3] = reverse_get_cut_dir(0, 0)
                notes_single[idx][3] = notes[3]
            else:
                notes_single[idx] = []
                continue  # do not update old notes
        elif df_score >= 2:
            # only ones
            if len(notes) == 4:
                if abs(cd_old_x) == abs(cd_new_x) == abs(cd_old_y) == abs(cd_new_y) == 1:
                    if cd_old_x != cd_new_x:
                        notes[3] = reverse_get_cut_dir(0, cd_new_y)
                    else:
                        notes[3] = reverse_get_cut_dir(cd_new_x, 0)
                # each one is zero
                else:
                    if cd_new_x == 0:
                        notes[3] = reverse_get_cut_dir(cd_old_x, cd_new_y)
                    else:
                        notes[3] = reverse_get_cut_dir(cd_new_x, cd_old_y)
            else:
                pass  # ignore multi notes    # (TODO: change multi notes)
            # update notes_single
            notes_single[idx][3] = notes[3]

        # update old notes
        notes_old = notes

    return notes_single, dot_notes_rem


def correct_notes(notes, timings):
    # calculate movement speed and remove too fast notes
    nl_last = None
    last_time = 0
    rm_counter = 0

    # reduce note difficulty at start and end of song
    se_idx = config.decr_speed_range  # start_end_index
    # compensate quick start behavior
    se_idx_start = se_idx + int(0.3 * config.quick_start * config.lstm_len + 1.0 * config.lstm_len)
    decrease_range = list(range(se_idx_start))
    decrease_range.extend(list(range(len(notes) - se_idx, len(notes))))
    # decrease_val = config.decr_speed_val
    decrease_val = np.ones(len(notes))
    decrease_val[:se_idx_start] = np.linspace(config.decr_speed_val, 1, se_idx_start)
    decrease_val[-se_idx:] = np.linspace(1, config.decr_speed_val, se_idx)

    for idx in range(len(notes)):
        if len(notes[idx]) == 0:
            continue
        # elif len(notes[idx]) == 4:
        elif len(notes[idx]) >= 4:
            # check cut direction movement (of first element in each time step)
            notes[idx] = check_note_movement(nl_last, notes[idx])

            # # notes[idx] = optimize_note_movement(nl_last, notes[idx])
            # notes[idx] = check_border_notes(notes, timings, idx)

            # calculate movement speed (of first element)
            new_time = timings[idx]
            speed = calc_note_speed(nl_last, notes[idx], new_time - last_time)

            # remove too fast elements
            if idx in decrease_range:
                mx_speed = config.max_speed * decrease_val[idx]
            else:
                mx_speed = config.max_speed

            factor = get_factor_from_max_speed(mx_speed, 0.5, 1)
            mx_speed *= factor

            if speed > mx_speed:
                # remove notes at this point
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
                    if notes[idx][(n + 1) * 4 + 3] != cut_dir:
                        notes[idx][(n + 1) * 4 + 3] = 8
                    n *= 4
                    speed1 = calc_note_speed(notes[idx][n:n + 4],
                                             notes[idx][n + 4:n + 8],
                                             time_diff=0.08, cdf=1.1, react=False)
                    speed2 = calc_note_speed(notes[idx][n + 4:n + 8],
                                             notes[idx][n:n + 4],
                                             time_diff=0.08, cdf=1.1, react=False)
                    speed = np.min([speed1, speed2])
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

    # print(f"Sanity check note speed removed {rm_counter} elements")
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

    # cut_x_last, cut_y_last = get_cut_dir_xy(notes_last[3])
    # cut_x_new, cut_y_new = get_cut_dir_xy(notes_new[3])
    # dist_x = int(np.abs(cut_x_last - cut_x_new))
    # dist_y = int(np.abs(cut_y_last - cut_y_new))
    #
    # if dist_x != 2 and dist_y != 2:
    #     if dist_x == dist_y == 1:
    #         return notes_new

    # change cut direction
    new_cut = reverse_cut_dir_xy(notes_last[3])
    notes_new[3] = new_cut

    return notes_new


def calc_note_speed(notes_last, notes_new, time_diff,
                    cdf=config.cdf, cdf_lr=config.cdf_lr, react=True):
    if notes_last is None:
        return 0

    # reaction time
    if react:
        dist = config.reaction_time + config.reaction_time_fact * config.max_speed
    else:
        dist = 0

    cut_x_last, cut_y_last = get_cut_dir_xy(notes_last[3])
    cut_x_new, cut_y_new = get_cut_dir_xy(notes_new[3])
    # x direction
    dist += np.abs((notes_last[0] - cdf * cut_x_last) -
                   (notes_new[0] + cdf * cut_x_new))
    dist *= cdf_lr
    # y direction
    dist += np.abs((notes_last[1] - cdf * cut_y_last) -
                   (notes_new[1] + cdf * cut_y_new))
    if time_diff > 0:
        speed = dist / time_diff  # TODO: check why time_diff can get 0
    else:
        speed = 1e9

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
    if config.verbose_level > 2:
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


def reverse_get_cut_dir(mov_x, mov_y):
    cut_dir_x = np.asarray([[1, 0, -1]] * 3).flatten()
    cut_dir_y = np.asarray([[-1] * 3, [0] * 3, [1] * 3]).flatten()
    cut_dir_order = np.asarray([[4, 0, 5], [2, 8, 3], [6, 1, 7]]).flatten()
    cut_dir = cut_dir_order[(cut_dir_x == mov_x) & (cut_dir_y == mov_y)][0]
    return cut_dir


################
# Postprocessing
################
def remove_silent_times(map_times, silent_times):
    threshold_timing = 0.03
    for silent in silent_times:
        while min(abs(map_times - silent)) < threshold_timing:
            # remove this value
            index = np.argmin(abs(map_times - silent))
            map_times = np.delete(map_times, index)
    return map_times


# def fill_map_times(map_times):
#     se_thresh = int(len(map_times) / 22)  # don't apply filling for first and last 4% of song
#     diff = np.diff(map_times)
#     new_map_times = []
#     for idx in range(se_thresh, len(diff) - se_thresh):
#         if config.add_beat_low_bound < diff[idx] < config.add_beat_hi_bound:
#             if np.random.random() < config.add_beat_fact:
#                 beat_time = (map_times[idx] + map_times[idx + 1]) / 2
#                 new_map_times.append(beat_time)
#     if len(new_map_times) > 0:
#         map_times = np.hstack((map_times, new_map_times))
#         map_times = np.sort(map_times)
#     return map_times


def fill_map_times_scale(map_times, scale_index=5):
    # scale lower and upper bounds for fill algorithm (index 0-10)
    se_thresh = int(len(map_times) * 0.05)  # don't apply filling for first and last 5% of song
    diff = np.diff(map_times)
    new_map_times = []

    mb = config.add_beat_max_bounds
    low_bound_matrix = np.linspace(mb[1], mb[0], config.map_filler_iters)
    high_bound_matrix = np.linspace(mb[2], mb[3], config.map_filler_iters)

    low_bound = low_bound_matrix[scale_index]
    high_bound = high_bound_matrix[scale_index]
    if low_bound < 0.1:
        low_bound = 0.1

    for idx in range(se_thresh, len(diff) - se_thresh):
        if low_bound < diff[idx] < high_bound:
            # if np.random.random() < config.add_beat_fact:
            beat_time = (map_times[idx] + map_times[idx + 1]) / 2
            new_map_times.append(beat_time)
    if len(new_map_times) > 0:
        map_times = np.hstack((map_times, new_map_times))
        map_times = np.sort(map_times)
    return map_times


def add_lstm_prerun(map_times):
    # add cutoff map times to compensate for lstm reshape
    qs = config.quick_start
    if qs > 0:
        new_map_times = np.linspace(0, qs, int(config.lstm_len * qs))
        map_times = np.hstack((new_map_times, map_times))
        # map_times = np.sort(map_times)
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
                new_pos.append([int(pos[0] - cut_x), int(pos[1] - cut_y)])
                new_pos2.append([int(pos[0] - 2 * cut_x), int(pos[1] - 2 * cut_y)])

            # check all new positions:
            valid = check_note_pos_valid(new_pos)
            valid2 = check_note_pos_valid(new_pos2)  # if true always shift
            if valid:
                if valid2 or np.random.random() < config.shift_beats_fact:
                    for ipos in range(len(new_pos)):
                        notes[idx][0 + 4 * ipos] = new_pos[ipos][0]
                        notes[idx][1 + 4 * ipos] = new_pos[ipos][1]
    return notes


# def shift_blocks_left_right(notes: list, left_note: bool, time_diffs: np.array):
#     last_note_pos = [[-1, -1]]
#     # if left_note:
#     #     shift = [0, 0]
#     # else:
#     #     shift = [0, 0]
#
#     for idx in range(len(notes)):
#         if len(notes[idx]) > 2:
#             note_pos = calc_note_pos(notes[idx], add_cut=False)
#             # cut_x, cut_y = get_cut_dir_xy(notes[idx][3])
#             new_pos = []
#             new_pos2 = []
#             for pos in note_pos:
#                 if pos in last_note_pos:
#                     if left_note:
#                         new_pos.append([pos[0]-1, pos[1]])
#                         new_pos2.append([pos[0]+1, pos[1]])
#                     else:
#                         new_pos.append([pos[0]+1, pos[1]])
#                         new_pos2.append([pos[0]-1, pos[1]])
#             if len(new_pos) > 0:
#                 valid = check_note_pos_valid(new_pos)
#                 valid2 = check_note_pos_valid(new_pos2)
#                 if valid:
#                     for ipos in range(len(new_pos)):
#                         notes[idx][0+4*ipos] = new_pos[ipos][0]
#                         notes[idx][1+4*ipos] = new_pos[ipos][1]
#                 elif valid2:
#                     for ipos in range(len(new_pos2)):
#                         notes[idx][0+4*ipos] = new_pos2[ipos][0]
#                         notes[idx][1+4*ipos] = new_pos2[ipos][1]
#
#             last_note_pos = note_pos
#     return notes


def apply_waveform_pattern(notes_l, notes_r):
    def random_pattern_distribution(len_notes):
        n_counts = len(config.waveform_pattern)
        rand_dist = []
        while len(rand_dist) <= len_notes:
            new_rand = np.random.randint(0, n_counts)
            rand_dist.extend([new_rand] * config.waveform_pattern_length)
        return rand_dist

    def check_up_down_direction(notes_idx):
        if notes_idx[3] in config.waveform_apply_dir:
            return True
        return False

    def update_idx_up_down(idx_up_down, idx_up_down_temp):
        if len(idx_up_down_temp) > 0:
            idx_up_down.append(idx_up_down_temp.copy())
            idx_up_down_temp = []
        return idx_up_down, idx_up_down_temp

    ####################################
    # Get all potential waveform indices
    ####################################
    def get_potential_indices(notes_x):
        idx_up_down_x = []
        idx_up_down_x_temp = []
        for idx_x in range(len(notes_x)):
            if len(notes_x[idx_x]) > 0:
                if check_up_down_direction(notes_x[idx_x]):
                    # potential index for wave pattern found
                    idx_up_down_x_temp.append(idx_x)
                else:
                    idx_up_down_x, idx_up_down_x_temp = update_idx_up_down(idx_up_down_x,
                                                                           idx_up_down_x_temp)
        idx_up_down_x, idx_up_down_x_temp = update_idx_up_down(idx_up_down_x, idx_up_down_x_temp)
        return idx_up_down_x

    idx_up_down_l = get_potential_indices(notes_l)

    # idx_up_down_r = get_potential_indices(notes_r)

    #############################
    # Get all LineIndex positions
    #############################

    def get_all_line_index_pos(notes_x):
        last_pos = []
        for idx_x in range(len(notes_x)):
            note_pos = []
            if len(notes_x[idx_x]) > 2:
                note_pos = calc_note_pos(notes_x[idx_x], add_cut=False)
                note_pos = [pos[0] for pos in note_pos]
            last_pos.append(note_pos.copy())
        return last_pos

    last_pos_l = get_all_line_index_pos(notes_l)

    # last_pos_r = get_all_line_index_pos(notes_r)

    ###############################
    # Apply waveform where possible
    ###############################
    def find_last_pos(last_pos: list, idx_pos: int) -> (list, bool):
        if idx < 1:
            return last_pos[idx_pos], True
        last_pos_before = last_pos[:idx_pos]
        last_pos_before.reverse()
        for pos in last_pos_before:
            if len(pos) > 0:
                return pos, False
        return last_pos[idx_pos], True

    def find_index_pos(last_pos: list, idx_pos: int, start: bool) -> (list, int):
        default_count = 9999
        if start:
            if idx < 1:
                return [], default_count
            last_pos_before = last_pos[:idx_pos]
            last_pos_before.reverse()
            for i, pos in enumerate(last_pos_before):
                if len(pos) > 0:
                    return pos, i
            return [], default_count
        else:
            last_pos_after = last_pos[idx_pos:]
            for i, pos in enumerate(last_pos_after):
                if len(pos) > 0:
                    return pos, i
            return [], default_count

    def find_next_start_pos(pos_last: int, pattern: list):
        # get start index
        if pos_last in pattern:
            wave_start = pattern.index(pos_last)
        else:
            if pos_last - 1 in pattern:
                wave_start = pattern.index(pos_last - 1)
            elif pos_last + 1 in pattern:
                wave_start = pattern.index(pos_last + 1)
            elif pos_last - 2 in pattern:
                wave_start = pattern.index(pos_last - 2)
            elif pos_last + 2 in pattern:
                wave_start = pattern.index(pos_last + 2)
            else:
                wave_start = 0
        return wave_start

    def calc_waveform_selection(note_pos_last: int, waveform_pattern: list,
                                start_index: int):
        # find first start position (only for left side)
        if start_index == 0:
            start_index = find_next_start_pos(note_pos_last, waveform_pattern)

        # get new note position (x)
        start_index += 1
        # repeat wave pattern, cut start_index
        waveform_pattern_rep = waveform_pattern * 3
        if start_index > len(waveform_pattern):
            start_index = start_index % len(waveform_pattern)

        return waveform_pattern_rep[start_index], start_index

    # def calc_waveform_selection(note_pos_last: int, waveform_pattern: list,
    #                             start_index: int, first_flag=False):
    #     start_index += 1
    #     if first_flag:
    #         return note_pos_last, start_index
    #     # repeat wave pattern
    #     waveform_pattern_rep = waveform_pattern * 3
    #     if start_index > len(waveform_pattern):
    #         start_index = start_index % len(waveform_pattern)
    #     waveform_pattern_rep = waveform_pattern_rep[start_index:]
    #
    #     # get start index
    #     if note_pos_last in waveform_pattern:
    #         wave_start = waveform_pattern_rep.index(note_pos_last)
    #     else:
    #         if note_pos_last - 1 in waveform_pattern:
    #             wave_start = waveform_pattern_rep.index(note_pos_last - 1)
    #         elif note_pos_last + 1 in waveform_pattern:
    #             wave_start = waveform_pattern_rep.index(note_pos_last + 1)
    #         elif note_pos_last - 2 in waveform_pattern:
    #             wave_start = waveform_pattern_rep.index(note_pos_last - 2)
    #         elif note_pos_last + 2 in waveform_pattern:
    #             wave_start = waveform_pattern_rep.index(note_pos_last + 2)
    #         elif note_pos_last - 3 in waveform_pattern:
    #             wave_start = waveform_pattern_rep.index(note_pos_last - 3)
    #         elif note_pos_last + 3 in waveform_pattern:
    #             wave_start = waveform_pattern_rep.index(note_pos_last + 3)
    #     # get new note position (x)
    #     wave_start += 1
    #     return waveform_pattern_rep[wave_start], start_index

    def check_waveform_selection(new_pos, forbidden_pos):
        if new_pos in forbidden_pos:
            return None
        return new_pos

    def apply_waveform_selection(new_pos, notes_x, idx_x):
        number_notes = len(notes_x[idx_x]) / 4
        if number_notes < 1.5:
            notes_x[idx_x][0] = new_pos
        else:
            for i in range(len(notes_x[idx_x])):
                if i % 4 == 0:
                    notes_x[idx_x][i] = new_pos
        return notes_x

    rd_pat_list = random_pattern_distribution(len(notes_l))

    start_idx = 0
    start_idx_list = []
    last_pat = []
    for idx_list in idx_up_down_l:
        if len(idx_list) > config.waveform_threshold:
            for idx in idx_list:
                pos_l_before, first_flag = find_last_pos(last_pos_l, idx)
                pos_l_before = pos_l_before[0]
                cur_pattern = config.waveform_pattern[rd_pat_list[idx]]
                if last_pat != cur_pattern:
                    start_idx = 0
                    last_pat = cur_pattern.copy()
                new_pos_l, start_idx = calc_waveform_selection(pos_l_before, cur_pattern, start_idx)
                # new_pos_l = check_waveform_selection(new_pos_l, [])
                if new_pos_l is not None:
                    notes_l = apply_waveform_selection(new_pos_l, notes_l, idx)
                # save start index and position
                start_idx_list.append([idx, start_idx])

    ########################
    # repeat for right notes
    ########################
    def calc_forbidden_pos(last_pos_x, idx_x):
        f_pos = []
        # before
        f_pos_start, near_count_start = find_index_pos(last_pos_x, idx_x, start=True)
        # after
        f_pos_end, near_count_end = find_index_pos(last_pos_x, idx_x, start=False)
        if near_count_start > near_count_end:
            return f_pos_end
        else:
            return f_pos_start

    def calc_pos_from_left(wave_index_array: np.ndarray, cur_idx):
        array_index = np.argmin(abs(wave_index_array[:, 0] - cur_idx))
        array_value = wave_index_array[array_index, 1]
        return array_value

    # idx_up_down_l = get_potential_indices(notes_l)
    idx_up_down_r = get_potential_indices(notes_r)
    last_pos_l = get_all_line_index_pos(notes_l)
    last_pos_r = get_all_line_index_pos(notes_r)

    start_idx = 0
    start_idx_list = np.asarray(start_idx_list)
    for idx_list in idx_up_down_r:
        if len(idx_list) > config.waveform_threshold:
            for idx in idx_list:
                pos_r_before, first_flag = find_last_pos(last_pos_r, idx)
                pos_r_before = pos_r_before[0]
                cur_pattern = config.waveform_pattern[rd_pat_list[idx]]
                # calculate next position from left side
                start_idx = calc_pos_from_left(start_idx_list, idx)
                new_pos_r, start_idx = calc_waveform_selection(pos_r_before, cur_pattern, start_idx)
                # calculate forbidden positions from left notes
                pos_l = calc_forbidden_pos(last_pos_l, idx)
                # continue
                new_pos_r = check_waveform_selection(new_pos_r, pos_l)
                if new_pos_r is not None:
                    notes_r = apply_waveform_selection(new_pos_r, notes_r, idx)

    return notes_l, notes_r


def shift_blocks_left_right(notes_l: list, notes_r: list, time_diffs: np.array):
    last_note_pos_l = [[-1]]
    last_note_pos_r = [[-1]]

    def new_pos_helper(note_pos, last_note_pos_l, last_note_pos_r, left_note):
        new_pos = []
        new_pos2 = []
        for pos in note_pos:
            if pos in last_note_pos_l or pos in last_note_pos_r:
                if left_note:
                    # new_pos.append([pos[0]-1, pos[1]])
                    # new_pos2.append([pos[0]+1, pos[1]])
                    new_pos.append([pos[0] - 1])
                    new_pos2.append([pos[0] + 1])
                else:
                    new_pos.append([pos[0] + 1])
                    new_pos2.append([pos[0] - 1])
        return new_pos, new_pos2

    def new_note_helper(notes, idx, new_pos, new_pos2):
        if len(new_pos) > 0:
            valid = check_note_pos_valid(new_pos, xy='x')
            valid2 = check_note_pos_valid(new_pos2, xy='x')
            if valid:
                for ipos in range(len(new_pos)):
                    notes[idx][0 + 4 * ipos] = new_pos[ipos][0]
            elif valid2:
                for ipos in range(len(new_pos2)):
                    notes[idx][0 + 4 * ipos] = new_pos2[ipos][0]
        return notes

    for idx in range(len(notes_l)):
        last_note_l_temp = last_note_pos_l
        if len(notes_l[idx]) > 2:
            note_pos = calc_note_pos(notes_l[idx], add_cut=False)
            note_pos = [[pos[0]] for pos in note_pos]
            new_pos, new_pos2 = new_pos_helper(note_pos, last_note_pos_l, last_note_pos_r, True)
            notes_l = new_note_helper(notes_l, idx, new_pos, new_pos2)
            if len(new_pos) == 0:
                last_note_pos_l = note_pos
            else:  # recalculate
                note_pos = calc_note_pos(notes_l[idx], add_cut=False)
                last_note_pos_l = [[pos[0]] for pos in note_pos]
                last_note_l_temp.extend(last_note_pos_l)
        if len(notes_r[idx]) > 2:
            note_pos = calc_note_pos(notes_r[idx], add_cut=False)
            note_pos = [[pos[0]] for pos in note_pos]
            new_pos, new_pos2 = new_pos_helper(note_pos, last_note_l_temp, last_note_pos_r, False)
            notes_r = new_note_helper(notes_r, idx, new_pos, new_pos2)
            if len(new_pos) == 0:
                last_note_pos_r = note_pos
            else:  # recalculate
                note_pos = calc_note_pos(notes_r[idx], add_cut=False)
                last_note_pos_r = [[pos[0]] for pos in note_pos]

    return notes_l, notes_r


def check_note_pos_valid(positions: list, xy=None) -> bool:
    if xy is None:
        for pos in positions:
            if not 0 <= pos[0] <= 3:  # line index
                return False
            if not 0 <= pos[1] <= 2:  # line layer
                return False
    elif xy == 'x':
        for pos in positions:
            if not 0 <= pos[0] <= 3:  # line index
                return False
    elif xy == 'y':
        for pos in positions:
            if not 0 <= pos[1] <= 2:  # line index
                return False
    else:
        print("Error, unknown position specified to check valid note")
        return False
    return True


if __name__ == '__main__':
    notes = np.load(paths.temp_path + 'notes.npy', allow_pickle=True)
    timings = np.load(paths.temp_path + 'timings.npy', allow_pickle=True)
    notes = list(notes)

    sanity_check_notes(notes, timings, None)
