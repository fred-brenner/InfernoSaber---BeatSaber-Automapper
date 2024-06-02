import pickle
import numpy as np
import copy

from tools.config import paths, config


def load_raw_beat_data(name_ar):
    map_dict_events = []
    map_dict_notes = []
    map_dict_obstacles = []
    for song_name in name_ar:
        # import beat data
        # Load notes, events, obstacles, all already divided by bpm from info file! (in real sec)
        map_dict_events.append(np.load(paths.dict_all_path + song_name + "_events.dat", allow_pickle=True))
        map_dict_notes.append(np.load(paths.dict_all_path + song_name + "_notes.dat", allow_pickle=True))
        map_dict_obstacles.append(np.load(paths.dict_all_path + song_name + "_obstacles.dat", allow_pickle=True))

    return map_dict_notes, map_dict_events, map_dict_obstacles


def sort_beats_by_time(song_ar: list):
    new_song_ar = []
    new_time_ar = []
    min_time = config.min_time_diff

    for map_ar in song_ar:
        # setup temp variables
        last_time = -1
        new_map_ar = []
        timeline = []

        for idx in range(map_ar.shape[1]):
            cur_map = map_ar[:, idx]
            # run sanity check
            if run_notes_sanity_check(cur_map):
                cur_time = cur_map[0]
                cur_map = cur_map[1:].astype('int16')
                if cur_time > last_time + min_time:
                    # append new beat
                    new_map_ar.append(cur_map)
                    last_time = cur_time
                    timeline.append(cur_time)
                else:  # time unchanged
                    # check if position changed
                    if pos_changed(new_map_ar[-1], cur_map):
                        # add notes to last beat
                        new_map_ar[-1] = np.vstack((new_map_ar[-1], cur_map))
        new_song_ar.append(new_map_ar)
        new_time_ar.append(timeline)

    return new_song_ar, new_time_ar


def remove_duplicate_notes(notes_ar_all):
    for idx_naa, notes_ar in enumerate(notes_ar_all):

        # get all notes with multiple simultaneous entries
        mask = [True if len(n_idx.shape) > 1 else False for n_idx in notes_ar]
        mask = np.arange(1, len(mask) + 1) * mask
        mask = np.asarray(mask[mask != 0], dtype=int) - 1
        # iterate through duplicate notes and use only first occurrence
        for idx in mask:
            # changed_notes = False
            notes = notes_ar[idx]
            # split left and right notes
            nl = []
            nr = []
            # TODO: what to do about boms?
            for note in notes:
                if note[2] == 0:
                    nl.append(note)
                elif note[2] == 1:
                    nr.append(note)
            if len(nl) > 0:
                # remove secondary left notes
                # if len(nl) > 1:
                #     changed_notes = True
                nl = get_first_note_lr(nl)
            if len(nr) > 0:
                # remove secondary right notes
                # if len(nr) > 1:
                #     changed_notes = True
                nr = get_first_note_lr(nr)

            # if changed_notes:
            # stack notes back together
            if len(nl) > 1:
                if len(nr) > 1:
                    nl = np.stack([nl, nr])
            elif len(nr) > 1:
                nl = nr
            else:
                nl = np.array([], dtype=int)
            notes_ar[idx] = nl

            # sanity check that only one note per side passes through
            # if len(notes_ar[idx]) > 2 and len(notes_ar[idx].shape) > 1:
            #     print("Error")

        notes_ar_all[idx_naa] = notes_ar

    return notes_ar_all


def get_pos_and_dir_from_notes(pos_ar):
    dir_ar = copy.deepcopy(pos_ar)
    note_ar = copy.deepcopy(pos_ar)
    for idx0, notes_song in enumerate(dir_ar):
        for idx1 in range(len(notes_song)):
            if len(notes_song[idx1]) < 1:
                # skip empty notes
                continue
            if len(notes_song[idx1].shape) == 1:
                pos_ar[idx0][idx1] = notes_song[idx1][:3]
                note_ar[idx0][idx1] = notes_song[idx1][:4]
                dir_ar[idx0][idx1] = notes_song[idx1][3:4]
            elif len(notes_song[idx1].shape) == 2:
                pos_ar[idx0][idx1] = notes_song[idx1][:, :3]
                note_ar[idx0][idx1] = notes_song[idx1][:, :4].reshape(-1)
                dir_ar[idx0][idx1] = notes_song[idx1][:, 3]
            else:
                print("Error in beat_data_helper: forbidden notes length")
    return pos_ar, dir_ar, note_ar


def get_first_note_lr(notes_lr):
    if len(notes_lr) == 1:
        return notes_lr[0]
    elif len(notes_lr) < 1:
        return notes_lr

    notes_ar = np.asarray(notes_lr)
    directions, dir_counts = np.unique(notes_ar[:, 3], return_counts=True)
    directions = directions[dir_counts.argsort()][::-1]
    for direction in directions:
        if direction != 8:
            break

    # build note
    posx = notes_lr[0][0]
    posy = notes_lr[0][1]
    # TODO: intelligent choice of position based on direction
    single_note = [posx, posy, notes_lr[0][2], direction]
    return np.asarray(single_note, dtype=int)


def run_notes_sanity_check(beat_matrix, verbose=True):
    # check data types for notes
    if beat_matrix[0] < 0:
        if verbose:
            print("beat_sanity_check: Found time below zero, skipping.")
        return False
    if beat_matrix[1] not in [0, 1, 2, 3]:
        if verbose:
            print("beat_sanity_check: Found wrong _lineIndex, skipping.")
        return False
    if beat_matrix[2] not in [0, 1, 2]:
        if verbose:
            print("beat_sanity_check: Found wrong _lineLayer, skipping.")
        return False
    if beat_matrix[3] not in [0, 1, 3]:
        if verbose:
            print("beat_sanity_check: Found wrong _type, skipping.")
        return False
    if beat_matrix[4] not in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        if verbose:
            print("beat_sanity_check: Found wrong _cutDir, skipping.")
        return False
    # check completed
    return True


def pos_changed(last_map, cur_map):
    if len(last_map.shape) > 1:
        last_map = last_map[-1]
    # check lineindex or linelayer changed
    if last_map[1 - 1] != cur_map[1 - 1] or last_map[2 - 1] != cur_map[2 - 1]:
        return True
    # else
    return False


def cluster_notes_in_classes(notes_ar):
    # notes_flattened = [item for sublist in notes_ar for item in sublist]
    # create classify dictionary
    class_key = []
    idx = -1
    # get class ID for each unique value
    new_song_ar = []
    for song in notes_ar:
        new_class_ar = []
        for beat in song:
            # if idx == 345:
            #     print("")
            beat = encode_beat_ar(beat)
            if beat not in class_key:
                # unknown pattern
                class_key.append(beat)
                idx += 1
            # known pattern
            key_idx = class_key.index(beat)
            new_class_ar.append(key_idx)
            # if key_idx != idx:
            #     print("")
        new_song_ar.append(new_class_ar)

    # save classify dictionary
    with open(paths.notes_classify_dict_file, "wb") as dict_file:
        pickle.dump(class_key, dict_file)

    return new_song_ar


def encode_beat_ar(beat):
    # Remove all double notes
    if config.remove_double_notes:
        if len(beat.shape) > 1:
            new_beat = []
            notes_done = []
            for idx in range(len(beat)):
                cur_note = beat[idx, 2]
                if cur_note not in notes_done:
                    new_beat.append(beat[idx])
                    notes_done.append(cur_note)
            beat = np.asarray(new_beat)

    beat = list(beat.reshape(-1))
    beat_f = ""
    for el in beat:
        beat_f += f"{el}"
    return beat_f
