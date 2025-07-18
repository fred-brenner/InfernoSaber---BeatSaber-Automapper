import numpy as np
import pickle
import json
import aubio
import os
import shutil

from map_creation.note_postprocessing import remove_double_notes
from map_creation.sanity_check import sanity_check_notes, improve_timings
from map_creation.gen_obstacles import calculate_obstacles
# from map_creation.gen_sliders import calculate_sliders
from map_creation.artificial_mod import gimme_more_notes
from tools.config import config, paths


def create_map_depr(y_class_num, timings, events, name, bpm, pitch_input, pitch_times):
    # load notes classify keys
    with open(paths.notes_classify_dict_file, 'rb') as f:
        class_keys = pickle.load(f)

    notes = decode_beats(y_class_num, class_keys)

    ################################################################
    def write_map(notes, timings, events, name, bpm, bs_diff,
                  pitch_input, pitch_times):
        # Sanity check timings for first notes
        time_last = 1.0
        for idx in range(10):
            # check if timings are in line
            if time_last > timings[idx]:
                rm_idx = idx - 1 if idx > 0 else idx
                # remove index
                timings = np.delete(timings, rm_idx)
                notes.pop(rm_idx)
                events = np.delete(events, rm_idx, axis=0)
            time_last = timings[idx]

        if config.single_notes_only_flag:
            notes = remove_double_notes(notes)
        # run all beat and note sanity checks
        notes = sanity_check_notes(notes, timings)
        # timings = improve_timings(notes, timings, pitch_input, pitch_times)

        if config.add_obstacle_flag:
            obstacles = calculate_obstacles(notes, timings)

        # compensate bps
        timings = timings * bpm / 60
        assert (len(timings) == len(notes))

        ###########
        # write map
        ###########
        # create output folder
        new_map_folder = f"{paths.new_map_path}/1234_{config.max_speed_orig:.1f}_{name}/"
        os.makedirs(new_map_folder, exist_ok=True)

        # write difficulty data
        file = f'{new_map_folder}{bs_diff}.dat'
        events_json = events_to_json(events, timings)
        notes_json = notes_to_json(notes, timings)
        if config.add_obstacle_flag:
            obstacles_json = obstacles_to_json(obstacles, bpm)
        else:
            obstacles_json = ""
        complete_map = get_map_string(notes=notes_json, events=events_json,
                                      obstacles=obstacles_json)
        with open(file, 'w') as f:
            f.write(complete_map)

        # write info data
        file = new_map_folder + 'info.dat'
        complete_info_map = get_info_map_string(name, bpm, bs_diff)
        with open(file, 'w') as f:
            f.write(complete_info_map)
        return new_map_folder

    ################################################################

    bs_diff = config.general_diff
    new_map_folder = write_map(notes.copy(), timings.copy(), events.copy(), name,
                               bpm, bs_diff, pitch_input, pitch_times)
    if config.create_expert_flag:
        bs_diff = 'Expert'
        config.max_speed *= config.expert_fact
        new_map_folder = write_map(notes, timings, events, name, bpm, bs_diff,
                                   pitch_input, pitch_times)
        # reset max speed
        config.max_speed = config.max_speed_orig
    # copy supplementary files to folder
    src = f"{paths.new_map_path}cover.jpg"
    shutil.copy(src, new_map_folder)
    src = f"{paths.songs_pred}{name}.egg"
    shutil.copy(src, new_map_folder)
    if config.verbose_level > 1:
        print(f"Finished song: {name}")


def decode_beats(y_class_num, class_keys):
    notes = []
    for idx in range(len(y_class_num)):
        y = int(y_class_num[idx])
        encoded = class_keys[y]
        notes.append(decode_class_keys(encoded))
    if config.gimme_more_notes_flag:
        notes = gimme_more_notes(notes)
    return notes


def decode_class_keys(encoded):
    # encoding:
    # beat = list(beat.reshape(-1))
    # beat_f = ""
    # for el in beat:
    #   beat_f += f"{el}"
    # return beat_f
    decoded = []
    for enc in encoded:
        decoded.append(int(enc))
    return decoded


def notes_to_json(notes, timings):
    note_json = ""
    for idx in range(len(notes)):
        for n in range(int(len(notes[idx]) / 4)):
            note_json += '{'
            note_json += f'"_time":{timings[idx]:.5f},' \
                         f'"_lineIndex":{notes[idx][0 + 4 * n]:.0f},' \
                         f'"_lineLayer":{notes[idx][1 + 4 * n]:.0f},' \
                         f'"_type":{notes[idx][2 + 4 * n]:.0f},' \
                         f'"_cutDirection":{notes[idx][3 + 4 * n]:.0f}'
            note_json += '},'

            # "_notes":[{"_time":4.116666793823242,
            # "_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},

    # remove last comma
    note_json = note_json[:-1]

    return note_json


def events_to_json(notes, timings):
    timings = timings[:len(notes)]
    if len(timings) != len(notes):
        notes = notes[:len(timings)]
    assert (len(timings) == len(notes))

    if len(timings) == 0:
        return f""

    note_json = ""
    for idx in range(len(notes)):
        note_json += '{'
        note_json += f'"_time":{timings[idx]:.4f},' \
                     f'"_type":{notes[idx][0]:.0f},' \
                     f'"_value":{notes[idx][1]:.0f}'
        note_json += '},'
        # "_events":[{"_time":4.1,"_type":3,"_value":1},

    # remove last comma
    note_json = note_json[:-1]

    return note_json


def obstacles_to_json(obstacles, bpm):
    obstacles = np.asarray(obstacles)
    obstacles[:, 0] = obstacles[:, 0] * bpm / 60
    obstacles[:, 3] = obstacles[:, 3] * bpm / 60
    note_json = ""
    for idx in range(len(obstacles)):
        note_json += '{'
        note_json += f'"_time":{obstacles[idx][0]:.4f},' \
                     f'"_lineIndex":{obstacles[idx][1]:.0f},' \
                     f'"_type":{obstacles[idx][2]:.0f},' \
                     f'"_duration":{obstacles[idx][3]:.3f},' \
                     f'"_width":{obstacles[idx][4]:.0f}'
        note_json += '},'
        # _obstacles":[{"_time":64.39733123779297,"_lineIndex":0,
        #               "_type":0,"_duration":6.5,"_width":1}

    # remove last comma
    note_json = note_json[:-1]

    return note_json


def get_map_string(events='', notes='', obstacles=''):
    map_string = '{"_version":"2.0.0","_BPMChanges":[],'
    map_string += f'"_events":[{events}],'
    map_string += f'"_notes":[{notes}],'
    map_string += f'"_obstacles":[{obstacles}],'
    map_string += f'"_bookmarks":[]'
    map_string += '}'
    return map_string


# {"_version":"2.0.0","_BPMChanges":[{"_BPM":103,"_time":54.616668701171875,"_beatsPerBar":4,"_metronomeOffset":4},{"_BPM":110,"_time":56.657623291015625,"_beatsPerBar":4,"_metronomeOffset":4},{"_BPM":108,"_time":60.60308837890625,"_beatsPerBar":4,"_metronomeOffset":4},{"_BPM":105,"_time":64.6216049194336,"_beatsPerBar":4,"_metronomeOffset":4},{"_BPM":0,"_time":71.61666870117188,"_beatsPerBar":4,"_metronomeOffset":4}],"_events":[{"_time":3.616666793823242,"_type":13,"_value":8},{"_time":3.616666793823242,"_type":1,"_value":1},{"_time":3.616666793823242,"_type":12,"_value":8},{"_time":4.116666793823242,"_type":3,"_value":3},{"_time":4.116666793823242,"_type":8,"_value":0},{"_time":4.116666793823242,"_type":2,"_value":2},{"_time":4.116666793823242,"_type":0,"_value":3},{"_time":4.116666793823242,"_type":1,"_value":5},{"_time":4.366666793823242,"_type":1,"_value":2},{"_time":4.616666793823242,"_type":4,"_value":6},{"_time":4.616666793823242,"_type":8,"_value":0},{"_time":4.866666793823242,"_type":1,"_value":6},{"_time":4.866666793823242,"_type":2,"_value":7},{"_time":5.366666793823242,"_type":0,"_value":7},{"_time":5.366666793823242,"_type":1,"_value":2},{"_time":5.491666793823242,"_type":4,"_value":0},{"_time":5.616666793823242,"_type":4,"_value":2},{"_time":5.616666793823242,"_type":8,"_value":0},{"_time":6.116666793823242,"_type":1,"_value":6},{"_time":6.366666793823242,"_type":1,"_value":2},{"_time":6.366666793823242,"_type":2,"_value":7},{"_time":6.491666793823242,"_type":4,"_value":0},{"_time":6.616666793823242,"_type":3,"_value":3},{"_time":6.616666793823242,"_type":4,"_value":6},{"_time":6.616666793823242,"_type":8,"_value":0},{"_time":6.866666793823242,"_type":0,"_value":3},{"_time":6.866666793823242,"_type":1,"_value":6},{"_time":6.866666793823242,"_type":0,"_value":3},{"_time":7.366666793823242,"_type":3,"_value":7},{"_time":7.366666793823242,"_type":1,"_value":2},{"_time":7.491666793823242,"_type":4,"_value":0},{"_time":7.616666793823242,"_type":4,"_value":2},{"_time":7.616666793823242,"_type":8,"_value":0},{"_time":7.616666793823242,"_type":1,"_value":6},{"_time":8.116666793823242,"_type":2,"_value":7},{"_time":8.116666793823242,"_type":1,"_value":2},{"_time":8.366666793823242,"_type":1,"_value":6},{"_time":8.366666793823242,"_type":2,"_value":7},{"_time":8.491666793823242,"_type":4,"_value":0},{"_time":8.616666793823242,"_type":8,"_value":0},{"_time":8.616666793823242,"_type":4,"_value":6},{"_time":8.616666793823242,"_type":1,"_value":2},{"_time":9.116666793823242,"_type":0,"_value":3},{"_time":9.116666793823242,"_type":1,"_value":6},{"_time":9.491666793823242,"_type":4,"_value":0},{"_time":9.616666793823242,"_type":1,"_value":2},{"_time":9.616666793823242,"_type":4,"_value":2},{"_time":9.616666793823242,"_type":3,"_value":3},{"_time":9.616666793823242,"_type":8,"_value":0},{"_time":9.741666793823242,"_type":4,"_value":3},{"_time":9.866666793823242,"_type":0,"_value":7},{"_time":9.866666793823242,"_type":1,"_value":6},{"_time":10.616666793823242,"_type":1,"_value":2},{"_time":10.616666793823242,"_type":2,"_value":7},{"_time":11.491666793823242,"_type":1,"_value":6},{"_time":11.616666793823242,"_type":1,"_value":7},{"_time":11.700000762939453,"_type":3,"_value":3},{"_time":11.783333778381348,"_type":0,"_value":3},{"_time":12.491666793823242,"_type":4,"_value":0},{"_time":12.616665840148926,"_type":3,"_value":3},{"_time":12.616666793823242,"_type":1,"_value":2},{"_time":12.616666793823242,"_type":8,"_value":0},{"_time":12.616666793823242,"_type":4,"_value":6},{"_time":13.491666793823242,"_type":4,"_value":0},{"_time":13.61666488647461,"_type":2,"_value":7},{"_time":13.616666793823242,"_type":4,"_value":2},{"_time":13.616666793823242,"_type":8,"_value":0},{"_time":13.616666793823242,"_type":1,"_value":6},{"_time":14.116662979125977,"_type":3,"_value":3},{"_time":14.116662979125977,"_type":0,"_value":7},{"_time":14.116666793823242,"_type":1,"_value":2},{"_time":14.491666793823242,"_type":4,"_value":0},{"_time":14.61666488647461,"_type":3,"_value":3},{"_time":14.616666793823242,"_type":8,"_value":0},{"_time":14.616666793823242,"_type":4,"_value":6},{"_time":14.616666793823242,"_type":1,"_value":6},{"_time":14.741666793823242,"_type":1,"_value":7},{"_time":14.866666793823242,"_type":2,"_value":7},{"_time":15.491666793823242,"_type":4,"_value":0},{"_time":15.616666793823242,"_type":8,"_value":0},{"_time":15.616666793823242,"_type":0,"_value":7},{"_time":15.616666793823242,"_type":4,"_value":2},{"_time":16.116666793823242,"_type":3,"_value":3},{"_time":16.491666793823242,"_type":4,"_value":0},{"_time":16.616666793823242,"_type":1,"_value":2},{"_time":16.616666793823242,"_type":8,"_value":0},{"_time":16.616666793823242,"_type":4,"_value":6},{"_time":16.616666793823242,"_type":2,"_value":7},{"_time":17.491666793823242,"_type":4,"_value":0},{"_time":17.616666793823242,"_type":4,"_value":2},{"_time":17.616666793823242,"_type":1,"_value":6},{"_time":17.616666793823242,"_type":8,"_value":0},{"_time":17.616666793823242,"_type":0,"_value":7},{"_time":18.116666793823242,"_type":1,"_value":2},{"_time":18.116666793823242,"_type":0,"_value":7},{"_time":18.491666793823242,"_type":4,"_value":0},{"_time":18.616666793823242,"_type":4,"_value":6},{"_time":18.616666793823242,"_type":8,"_value":0},{"_time":18.700000762939453,"_type":3,"_value":3},{"_time":18.741666793823242,"_type":1,"_value":6},{"_time":18.78333282470703,"_type":0,"_value":3},{"_time":18.866666793823242,"_type":1,"_value":7},{"_time":18.866666793823242,"_type":2,"_value":7},{"_time":19.491666793823242,"_type":4,"_value":0},{"_time":19.616666793823242,"_type":4,"_value":2},{"_time":19.616666793823242,"_type":8,"_value":0},{"_time":19.616666793823242,"_type":3,"_value":3},{"_time":20.491666793823242,"_type":4,"_value":0},{"_time":20.616666793823242,"_type":8,"_value":0},{"_time":20.616666793823242,"_type":2,"_value":7},{"_time":20.616666793823242,"_type":4,"_value":6},{"_time":20.616666793823242,"_type":1,"_value":2},{"_time":21.491666793823242,"_type":4,"_value":0},{"_time":21.616666793823242,"_type":8,"_value":0},{"_time":21.616666793823242,"_type":1,"_value":6},{"_time":21.616666793823242,"_type":0,"_value":7},{"_time":21.616666793823242,"_type":3,"_value":3},{"_time":21.616666793823242,"_type":4,"_value":2},{"_time":22.116666793823242,"_type":2,"_value":7},{"_time":22.116666793823242,"_type":1,"_value":2},{"_time":22.491666793823242,"_type":4,"_value":0},{"_time":22.616666793823242,"_type":4,"_value":6},{"_time":22.616666793823242,"_type":2,"_value":3},{"_time":22.616666793823242,"_type":8,"_value":0},{"_time":22.700000762939453,"_type":0,"_value":3},{"_time":22.741666793823242,"_type":1,"_value":6},{"_time":22.741666793823242,"_type":4,"_value":7},{"_time":22.78333282470703,"_type":3,"_value":3},{"_time":22.866666793823242,"_type":0,"_value":7},{"_time":22.866666793823242,"_type":1,"_value":7},{"_time":24.116666793823242,"_type":1,"_value":2},{"_time":24.116666793823242,"_type":2,"_value":7},{"_time":24.616666793823242,"_type":3,"_value":7},{"_time":24.616666793823242,"_type":3,"_value":3},{"_time":24.616666793823242,"_type":1,"_value":6},{"_time":25.116666793823242,"_type":1,"_value":2},{"_time":25.116666793823242,"_type":0,"_value":7},{"_time":25.200000762939453,"_type":2,"_value":3},{"_time":25.28333282470703,"_type":2,"_value":3},{"_time":25.616666793823242,"_type":3,"_value":3},{"_time":25.616666793823242,"_type":1,"_value":6},{"_time":26.116666793823242,"_type":1,"_value":2},{"_time":26.116666793823242,"_type":0,"_value":3},{"_time":26.866666793823242,"_type":2,"_value":7},{"_time":26.866666793823242,"_type":1,"_value":6},{"_time":26.866666793823242,"_type":0,"_value":3},{"_time":26.991666793823242,"_type":1,"_value":7},{"_time":28.491666793823242,"_type":4,"_value":0},{"_time":28.616666793823242,"_type":1,"_value":2},{"_time":28.616666793823242,"_type":8,"_value":0},{"_time":28.616666793823242,"_type":0,"_value":7},{"_time":28.616666793823242,"_type":4,"_value":6},{"_time":29.491666793823242,"_type":4,"_value":0},{"_time":29.616666793823242,"_type":1,"_value":6},{"_time":29.616666793823242,"_type":4,"_value":2},{"_time":29.616666793823242,"_type":8,"_value":0},{"_time":29.616666793823242,"_type":0,"_value":7},{"_time":30.116666793823242,"_type":0,"_value":3},{"_time":30.116666793823242,"_type":1,"_value":2},{"_time":30.491666793823242,"_type":4,"_value":0},{"_time":30.616666793823242,"_type":8,"_value":0},{"_time":30.616666793823242,"_type":4,"_value":6},{"_time":30.700000762939453,"_type":3,"_value":3},{"_time":30.783334732055664,"_type":0,"_value":3},{"_time":30.866666793823242,"_type":2,"_value":7},{"_time":30.866666793823242,"_type":1,"_value":6},{"_time":30.991666793823242,"_type":1,"_value":7},{"_time":31.366666793823242,"_type":3,"_value":3},{"_time":31.491666793823242,"_type":4,"_value":0},{"_time":31.616666793823242,"_type":8,"_value":0},{"_time":31.616666793823242,"_type":4,"_value":2},{"_time":31.866666793823242,"_type":0,"_value":7},{"_time":32.116668701171875,"_type":3,"_value":3},{"_time":32.491668701171875,"_type":4,"_value":0},{"_time":32.616668701171875,"_type":8,"_value":0},{"_time":32.616668701171875,"_type":4,"_value":6},{"_time":32.616668701171875,"_type":1,"_value":2},{"_time":32.616668701171875,"_type":2,"_value":7},{"_time":32.616668701171875,"_type":3,"_value":7},{"_time":33.366668701171875,"_type":1,"_value":6},{"_time":33.366668701171875,"_type":0,"_value":7},{"_time":33.491668701171875,"_type":4,"_value":0},{"_time":33.616668701171875,"_type":8,"_value":0},{"_time":33.616668701171875,"_type":4,"_value":2},{"_time":33.616668701171875,"_type":0,"_value":7},{"_time":34.116668701171875,"_type":1,"_value":2},{"_time":34.116668701171875,"_type":3,"_value":3},{"_time":34.491668701171875,"_type":4,"_value":0},{"_time":34.616668701171875,"_type":8,"_value":0},{"_time":34.616668701171875,"_type":4,"_value":6},{"_time":34.616668701171875,"_type":0,"_value":7},{"_time":34.70000076293945,"_type":2,"_value":3},{"_time":34.741668701171875,"_type":1,"_value":6},{"_time":34.78333282470703,"_type":2,"_value":3},{"_time":34.866668701171875,"_type":1,"_value":7},{"_time":35.366668701171875,"_type":3,"_value":7},{"_time":35.491668701171875,"_type":4,"_value":0},{"_time":35.616668701171875,"_type":8,"_value":0},{"_time":35.616668701171875,"_type":4,"_value":2},{"_time":35.616668701171875,"_type":0,"_value":3},{"_time":35.866668701171875,"_type":3,"_value":3},{"_time":36.491668701171875,"_type":4,"_value":0},{"_time":36.616668701171875,"_type":8,"_value":0},{"_time":36.616668701171875,"_type":1,"_value":2},{"_time":36.616668701171875,"_type":2,"_value":7},{"_time":36.616668701171875,"_type":4,"_value":6},{"_time":37.491668701171875,"_type":4,"_value":0},{"_time":37.616668701171875,"_type":2,"_value":7},{"_time":37.616668701171875,"_type":4,"_value":2},{"_time":37.616668701171875,"_type":1,"_value":6},{"_time":37.616668701171875,"_type":3,"_value":3},{"_time":37.616668701171875,"_type":8,"_value":0},{"_time":38.116668701171875,"_type":1,"_value":2},{"_time":38.116668701171875,"_type":2,"_value":7},{"_time":38.491668701171875,"_type":4,"_value":0},{"_time":38.616668701171875,"_type":8,"_value":0},{"_time":38.616668701171875,"_type":4,"_value":6},{"_time":38.616668701171875,"_type":3,"_value":3},{"_time":38.741668701171875,"_type":1,"_value":6},{"_time":38.866668701171875,"_type":1,"_value":7},{"_time":38.866668701171875,"_type":2,"_value":3},{"_time":39.491668701171875,"_type":4,"_value":0},{"_time":39.616668701171875,"_type":0,"_value":7},{"_time":39.616668701171875,"_type":8,"_value":0},{"_time":39.616668701171875,"_type":4,"_value":2},{"_time":40.116668701171875,"_type":3,"_value":3},{"_time":40.116668701171875,"_type":1,"_value":2},{"_time":40.116668701171875,"_type":0,"_value":3},{"_time":40.491668701171875,"_type":4,"_value":0},{"_time":40.616668701171875,"_type":2,"_value":7},{"_time":40.616668701171875,"_type":1,"_value":6},{"_time":40.616668701171875,"_type":8,"_value":0},{"_time":40.616668701171875,"_type":4,"_value":6},{"_time":41.116668701171875,"_type":1,"_value":2},{"_time":41.116668701171875,"_type":0,"_value":3},{"_time":41.116668701171875,"_type":3,"_value":3},{"_time":41.491668701171875,"_type":4,"_value":0},{"_time":41.616668701171875,"_type":1,"_value":6},{"_time":41.616668701171875,"_type":3,"_value":7},{"_time":41.616668701171875,"_type":0,"_value":7},{"_time":41.616668701171875,"_type":4,"_value":2},{"_time":41.616668701171875,"_type":8,"_value":0},{"_time":42.491668701171875,"_type":4,"_value":0},{"_time":42.616668701171875,"_type":8,"_value":0},{"_time":42.616668701171875,"_type":1,"_value":2},{"_time":42.616668701171875,"_type":2,"_value":7},{"_time":42.616668701171875,"_type":4,"_value":6},{"_time":43.491668701171875,"_type":4,"_value":0},{"_time":43.616668701171875,"_type":2,"_value":7},{"_time":43.616668701171875,"_type":4,"_value":2},{"_time":43.616668701171875,"_type":1,"_value":6},{"_time":43.616668701171875,"_type":8,"_value":0},{"_time":44.116668701171875,"_type":0,"_value":3},{"_time":44.116668701171875,"_type":3,"_value":3},{"_time":44.116668701171875,"_type":1,"_value":2},{"_time":44.241668701171875,"_type":1,"_value":3},{"_time":44.491668701171875,"_type":4,"_value":0},{"_time":44.616668701171875,"_type":8,"_value":0},{"_time":44.616668701171875,"_type":4,"_value":6},{"_time":44.866668701171875,"_type":3,"_value":7},{"_time":45.116668701171875,"_type":0,"_value":7},{"_time":45.491668701171875,"_type":4,"_value":0},{"_time":45.616668701171875,"_type":8,"_value":0},{"_time":45.616668701171875,"_type":4,"_value":6},{"_time":46.366668701171875,"_type":2,"_value":7},{"_time":46.491668701171875,"_type":4,"_value":0},{"_time":46.616668701171875,"_type":4,"_value":6},{"_time":46.616668701171875,"_type":8,"_value":0},{"_time":46.616668701171875,"_type":3,"_value":3},{"_time":46.90833282470703,"_type":0,"_value":3},{"_time":47.36668395996094,"_type":3,"_value":7},{"_time":47.491668701171875,"_type":4,"_value":0},{"_time":47.616668701171875,"_type":8,"_value":0},{"_time":47.616668701171875,"_type":4,"_value":6},{"_time":47.61668395996094,"_type":0,"_value":7},{"_time":47.866676330566406,"_type":2,"_value":7},{"_time":48.116676330566406,"_type":3,"_value":3},{"_time":48.491668701171875,"_type":4,"_value":0},{"_time":48.616668701171875,"_type":1,"_value":6},{"_time":48.616668701171875,"_type":8,"_value":0},{"_time":48.616668701171875,"_type":4,"_value":2},{"_time":49.116668701171875,"_type":1,"_value":2},{"_time":49.116668701171875,"_type":2,"_value":7},{"_time":49.366668701171875,"_type":3,"_value":3},{"_time":49.366668701171875,"_type":1,"_value":6},{"_time":49.491668701171875,"_type":4,"_value":0},{"_time":49.616668701171875,"_type":4,"_value":6},{"_time":49.616668701171875,"_type":8,"_value":0},{"_time":49.866668701171875,"_type":2,"_value":3},{"_time":49.94999694824219,"_type":0,"_value":3},{"_time":49.991668701171875,"_type":1,"_value":6},{"_time":50.03333282470703,"_type":3,"_value":3},{"_time":50.491668701171875,"_type":4,"_value":0},{"_time":50.616668701171875,"_type":4,"_value":2},{"_time":50.616668701171875,"_type":1,"_value":2},{"_time":50.616668701171875,"_type":8,"_value":0},{"_time":51.116668701171875,"_type":1,"_value":6},{"_time":51.116668701171875,"_type":2,"_value":7},{"_time":51.366668701171875,"_type":1,"_value":2},{"_time":51.366668701171875,"_type":0,"_value":7},{"_time":51.491668701171875,"_type":4,"_value":0},{"_time":51.616668701171875,"_type":4,"_value":6},{"_time":51.616668701171875,"_type":8,"_value":0},{"_time":51.866668701171875,"_type":1,"_value":6},{"_time":52.366668701171875,"_type":3,"_value":3},{"_time":52.491668701171875,"_type":4,"_value":0},{"_time":52.616668701171875,"_type":8,"_value":0},{"_time":52.616668701171875,"_type":1,"_value":2},{"_time":52.616668701171875,"_type":4,"_value":2},{"_time":53.116668701171875,"_type":2,"_value":7},{"_time":53.116668701171875,"_type":1,"_value":6},{"_time":53.366668701171875,"_type":1,"_value":2},{"_time":53.366668701171875,"_type":0,"_value":7},{"_time":53.491668701171875,"_type":4,"_value":0},{"_time":53.616668701171875,"_type":4,"_value":6},{"_time":53.616668701171875,"_type":2,"_value":3},{"_time":53.616668701171875,"_type":8,"_value":0},{"_time":53.69999694824219,"_type":0,"_value":3},{"_time":53.78333282470703,"_type":3,"_value":3},{"_time":53.866668701171875,"_type":1,"_value":6},{"_time":53.991668701171875,"_type":1,"_value":7},{"_time":54.491668701171875,"_type":4,"_value":0},{"_time":54.616668701171875,"_type":8,"_value":0},{"_time":54.616668701171875,"_type":3,"_value":3},{"_time":54.616668701171875,"_type":4,"_value":2},{"_time":55.116668701171875,"_type":0,"_value":7},{"_time":55.366668701171875,"_type":3,"_value":7},{"_time":55.53839111328125,"_type":4,"_value":0},{"_time":55.616668701171875,"_type":8,"_value":0},{"_time":55.67005920410156,"_type":0,"_value":3},{"_time":55.670066833496094,"_type":4,"_value":6},{"_time":55.93341064453125,"_type":3,"_value":3},{"_time":56.460113525390625,"_type":4,"_value":0},{"_time":56.59178924560547,"_type":1,"_value":2},{"_time":56.59178924560547,"_type":4,"_value":2},{"_time":56.59178924560547,"_type":8,"_value":0},{"_time":56.59178924560547,"_type":2,"_value":7},{"_time":57.15080261230469,"_type":0,"_value":7},{"_time":57.15080261230469,"_type":1,"_value":6},{"_time":57.39739227294922,"_type":1,"_value":2},{"_time":57.52069091796875,"_type":4,"_value":0},{"_time":57.59178924560547,"_type":8,"_value":0},{"_time":57.64398193359375,"_type":3,"_value":3},{"_time":57.64398193359375,"_type":4,"_value":6},{"_time":57.85945129394531,"_type":0,"_value":7},{"_time":57.89057922363281,"_type":1,"_value":6},{"_time":58.507049560546875,"_type":4,"_value":0},{"_time":58.59178924560547,"_type":8,"_value":0},{"_time":58.630348205566406,"_type":4,"_value":2},{"_time":58.630348205566406,"_type":1,"_value":2},{"_time":58.63035583496094,"_type":0,"_value":3},{"_time":58.63035583496094,"_type":3,"_value":3},{"_time":59.12352752685547,"_type":1,"_value":6},{"_time":59.12353515625,"_type":3,"_value":7},{"_time":59.493408203125,"_type":4,"_value":0},{"_time":59.59178924560547,"_type":8,"_value":0},{"_time":59.61670684814453,"_type":4,"_value":6},{"_time":59.61671447753906,"_type":1,"_value":2},{"_time":59.61671447753906,"_type":2,"_value":7},{"_time":59.863304138183594,"_type":1,"_value":6},{"_time":59.863311767578125,"_type":2,"_value":7},{"_time":60.109901428222656,"_type":3,"_value":3},{"_time":60.47978210449219,"_type":4,"_value":0},{"_time":60.60307312011719,"_type":4,"_value":2},{"_time":60.60308837890625,"_type":1,"_value":2},{"_time":60.60308837890625,"_type":8,"_value":0},{"_time":61.096275329589844,"_type":2,"_value":7},{"_time":61.10539245605469,"_type":1,"_value":6},{"_time":61.466163635253906,"_type":3,"_value":3},{"_time":61.48213195800781,"_type":1,"_value":2},{"_time":61.482139587402344,"_type":4,"_value":0},{"_time":61.60308837890625,"_type":8,"_value":0},{"_time":61.60771179199219,"_type":4,"_value":6},{"_time":61.85887145996094,"_type":0,"_value":3},{"_time":61.85887145996094,"_type":1,"_value":6},{"_time":61.98444366455078,"_type":1,"_value":7},{"_time":62.48677062988281,"_type":4,"_value":0},{"_time":62.60308837890625,"_type":8,"_value":0},{"_time":62.612335205078125,"_type":3,"_value":7},{"_time":62.612342834472656,"_type":4,"_value":2},{"_time":63.114654541015625,"_type":2,"_value":7},{"_time":63.365806579589844,"_type":0,"_value":7},{"_time":63.49139404296875,"_type":4,"_value":0},{"_time":63.60308837890625,"_type":8,"_value":0},{"_time":63.616973876953125,"_type":4,"_value":6},{"_time":64.49602508544922,"_type":4,"_value":0},{"_time":64.62159729003906,"_type":0,"_value":3},{"_time":64.62159729003906,"_type":3,"_value":3},{"_time":64.62159729003906,"_type":1,"_value":2},{"_time":64.6216049194336,"_type":8,"_value":0},{"_time":64.6216049194336,"_type":4,"_value":2},{"_time":64.99833679199219,"_type":1,"_value":6},{"_time":65.37507629394531,"_type":1,"_value":2},{"_time":65.378662109375,"_type":2,"_value":7},{"_time":65.378662109375,"_type":2,"_value":7},{"_time":65.52577209472656,"_type":4,"_value":0},{"_time":65.6216049194336,"_type":8,"_value":0},{"_time":65.65493774414062,"_type":4,"_value":6},{"_time":65.91326904296875,"_type":1,"_value":6},{"_time":65.91326904296875,"_type":3,"_value":3},{"_time":65.91328430175781,"_type":0,"_value":3},{"_time":66.55909729003906,"_type":1,"_value":2},{"_time":66.5591049194336,"_type":4,"_value":0},{"_time":66.55911254882812,"_type":3,"_value":7},{"_time":66.55911254882812,"_type":0,"_value":7},{"_time":66.68827056884766,"_type":4,"_value":2},{"_time":66.68829345703125,"_type":8,"_value":0},{"_time":67.07576751708984,"_type":1,"_value":6},{"_time":67.42021942138672,"_type":2,"_value":3},{"_time":67.42021942138672,"_type":2,"_value":3},{"_time":67.46327209472656,"_type":1,"_value":2},{"_time":67.59243774414062,"_type":4,"_value":0},{"_time":67.72160339355469,"_type":4,"_value":6},{"_time":67.72162628173828,"_type":8,"_value":0},{"_time":67.85076904296875,"_type":4,"_value":7},{"_time":68.1091079711914,"_type":1,"_value":6},{"_time":68.23827362060547,"_type":1,"_value":7},{"_time":69.0993881225586,"_type":0,"_value":7},{"_time":69.0993881225586,"_type":2,"_value":7},{"_time":69.14244079589844,"_type":1,"_value":2},{"_time":69.78826141357422,"_type":3,"_value":3},{"_time":69.78827667236328,"_type":1,"_value":6},{"_time":69.91744232177734,"_type":1,"_value":7},{"_time":70.04658508300781,"_type":0,"_value":7},{"_time":70.5833511352539,"_type":0,"_value":3},{"_time":71.0799331665039,"_type":3,"_value":3},{"_time":71.59660339355469,"_type":2,"_value":7},{"_time":72.59857940673828,"_type":1,"_value":2},{"_time":72.59859466552734,"_type":4,"_value":2},{"_time":72.5986099243164,"_type":0,"_value":7},{"_time":73.08951568603516,"_type":1,"_value":6},{"_time":73.08953857421875,"_type":4,"_value":6},{"_time":73.08953857421875,"_type":3,"_value":3},{"_time":73.33499145507812,"_type":1,"_value":2},{"_time":73.33501434326172,"_type":2,"_value":7},{"_time":73.33501434326172,"_type":0,"_value":7},{"_time":73.33501434326172,"_type":4,"_value":2},{"_time":73.90779113769531,"_type":3,"_value":3},{"_time":73.94869232177734,"_type":1,"_value":6},{"_time":73.94869995117188,"_type":4,"_value":6},{"_time":74.56237030029297,"_type":4,"_value":2},{"_time":74.56237030029297,"_type":1,"_value":2},{"_time":74.56239318847656,"_type":3,"_value":7},{"_time":74.56239318847656,"_type":0,"_value":7},{"_time":75.05331420898438,"_type":1,"_value":6},{"_time":75.0533218383789,"_type":4,"_value":6},{"_time":75.0533676147461,"_type":3,"_value":3},{"_time":75.0533676147461,"_type":2,"_value":7},{"_time":75.54427337646484,"_type":4,"_value":2},{"_time":75.54428100585938,"_type":1,"_value":2},{"_time":75.54431915283203,"_type":3,"_value":3},{"_time":76.03521728515625,"_type":1,"_value":6},{"_time":76.03522491455078,"_type":4,"_value":6},{"_time":76.03524780273438,"_type":0,"_value":3},{"_time":76.03524780273438,"_type":2,"_value":7},{"_time":76.157958984375,"_type":4,"_value":7},{"_time":76.157958984375,"_type":1,"_value":7},{"_time":77.13986206054688,"_type":1,"_value":2},{"_time":77.1398696899414,"_type":4,"_value":2},{"_time":77.75357055664062,"_type":0,"_value":7},{"_time":77.83537292480469,"_type":2,"_value":3},{"_time":77.87629699707031,"_type":1,"_value":6},{"_time":77.87630462646484,"_type":4,"_value":6},{"_time":77.91719818115234,"_type":2,"_value":3},{"_time":78.2445068359375,"_type":1,"_value":2},{"_time":78.24451446533203,"_type":8,"_value":0},{"_time":78.24451446533203,"_type":4,"_value":2},{"_time":78.2445297241211,"_type":0,"_value":2},{"_time":78.2445297241211,"_type":3,"_value":6},{"_time":78.2445297241211,"_type":2,"_value":6},{"_time":78.36724090576172,"_type":0,"_value":0},{"_time":78.36724090576172,"_type":2,"_value":0},{"_time":78.36724090576172,"_type":3,"_value":0},{"_time":78.36724853515625,"_type":4,"_value":0},{"_time":78.36724853515625,"_type":8,"_value":0},{"_time":78.36724853515625,"_type":1,"_value":0},{"_time":78.489990234375,"_type":8,"_value":0}],"_notes":[{"_time":4.116666793823242,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":4.116666793823242,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":4.866666793823242,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":4.866666793823242,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":5.366666793823242,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":5.616666793823242,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":6.116666793823242,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":6.366666793823242,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":6.616666793823242,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":6.866666793823242,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":6.866666793823242,"_lineIndex":1,"_lineLayer":1,"_type":0,"_cutDirection":1},{"_time":7.366666793823242,"_lineIndex":2,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":7.616666793823242,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":7.866666793823242,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":8.116666793823242,"_lineIndex":0,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":8.366666793823242,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":8.616666793823242,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":9.116666793823242,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":9.616666793823242,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":9.866666793823242,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":10.616666793823242,"_lineIndex":0,"_lineLayer":0,"_type":0,"_cutDirection":6},{"_time":10.616666793823242,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":11.616666793823242,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":6},{"_time":11.700000762939453,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":8},{"_time":11.783333778381348,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":8},{"_time":12.61666488647461,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":3},{"_time":12.616665840148926,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":3},{"_time":13.61666488647461,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":14.116662979125977,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":14.116662979125977,"_lineIndex":2,"_lineLayer":2,"_type":0,"_cutDirection":3},{"_time":14.61666488647461,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":5},{"_time":14.69999885559082,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":8},{"_time":14.783331871032715,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":8},{"_time":14.866666793823242,"_lineIndex":0,"_lineLayer":0,"_type":0,"_cutDirection":6},{"_time":15.616666793823242,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":16.116666793823242,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":16.616666793823242,"_lineIndex":0,"_lineLayer":0,"_type":0,"_cutDirection":6},{"_time":16.616666793823242,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":17.616666793823242,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":18.116666793823242,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":18.116666793823242,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":18.616666793823242,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":6},{"_time":18.700000762939453,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":8},{"_time":18.78333282470703,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":8},{"_time":18.866666793823242,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":6},{"_time":19.616666793823242,"_lineIndex":2,"_lineLayer":2,"_type":0,"_cutDirection":3},{"_time":20.616666793823242,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":2},{"_time":20.616666793823242,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":3},{"_time":21.616666793823242,"_lineIndex":1,"_lineLayer":2,"_type":1,"_cutDirection":2},{"_time":21.616666793823242,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":22.116666793823242,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":22.116666793823242,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":3},{"_time":22.616666793823242,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":7},{"_time":22.700000762939453,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":8},{"_time":22.78333282470703,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":8},{"_time":22.866666793823242,"_lineIndex":1,"_lineLayer":2,"_type":1,"_cutDirection":2},{"_time":24.116666793823242,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":24.116666793823242,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":7},{"_time":24.616666793823242,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":24.616666793823242,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":25.116666793823242,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":4},{"_time":25.200000762939453,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":8},{"_time":25.28333282470703,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":8},{"_time":25.616666793823242,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":26.116666793823242,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":26.116666793823242,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":26.866666793823242,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":26.866666793823242,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":28.116666793823242,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":28.616666793823242,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":28.616666793823242,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":29.616666793823242,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":30.116666793823242,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":30.116666793823242,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":5},{"_time":30.616666793823242,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":6},{"_time":30.700000762939453,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":8},{"_time":30.783334732055664,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":8},{"_time":30.866666793823242,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":31.366666793823242,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":31.616666793823242,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":31.866666793823242,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":32.116668701171875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":32.616668701171875,"_lineIndex":0,"_lineLayer":0,"_type":0,"_cutDirection":6},{"_time":32.616668701171875,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":33.366668701171875,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":33.616668701171875,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":34.116668701171875,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":34.116668701171875,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":34.616668701171875,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":4},{"_time":34.70000076293945,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":8},{"_time":34.78333282470703,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":8},{"_time":34.866668701171875,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":35.366668701171875,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":35.616668701171875,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":35.866668701171875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":36.616668701171875,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":36.616668701171875,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":37.616668701171875,"_lineIndex":0,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":37.616668701171875,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":38.116668701171875,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":38.616668701171875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":5},{"_time":38.70000457763672,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":8},{"_time":38.78333282470703,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":8},{"_time":38.866668701171875,"_lineIndex":0,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":39.616668701171875,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":40.116668701171875,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":40.116668701171875,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":40.616668701171875,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":40.616668701171875,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":41.116668701171875,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":41.116668701171875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":41.616668701171875,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":41.616668701171875,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":42.616668701171875,"_lineIndex":0,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":42.616668701171875,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":43.616668701171875,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":43.616668701171875,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":44.116668701171875,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":44.116668701171875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":44.866668701171875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":0},{"_time":45.116668701171875,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":0},{"_time":45.866668701171875,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":46.366668701171875,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":46.616668701171875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":46.90833282470703,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":47.36668395996094,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":47.61668395996094,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":47.866676330566406,"_lineIndex":0,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":48.116676330566406,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":48.616668701171875,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":49.116668701171875,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":49.366668701171875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":49.866668701171875,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":7},{"_time":49.94999694824219,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":8},{"_time":50.03333282470703,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":8},{"_time":50.616668701171875,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":0},{"_time":50.616668701171875,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":51.116668701171875,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":51.366668701171875,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":51.866668701171875,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":52.366668701171875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":5},{"_time":52.44999694824219,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":8},{"_time":52.53333282470703,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":8},{"_time":53.116668701171875,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":53.366668701171875,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":53.616668701171875,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":7},{"_time":53.69999694824219,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":8},{"_time":53.78333282470703,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":8},{"_time":53.866668701171875,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":54.616668701171875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":55.116668701171875,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":55.366668701171875,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":55.67005920410156,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":55.93341064453125,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":56.59178924560547,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":56.59178924560547,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":57.15080261230469,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":57.39739227294922,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":57.64398193359375,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":5},{"_time":57.72618103027344,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":8},{"_time":57.808380126953125,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":8},{"_time":57.85945129394531,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":58.63035583496094,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":58.63035583496094,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":59.12353515625,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":59.61671447753906,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":59.863311767578125,"_lineIndex":0,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":60.109901428222656,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":60.60309600830078,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":61.096275329589844,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":61.466163635253906,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":61.85887145996094,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":62.612335205078125,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":63.114654541015625,"_lineIndex":0,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":63.365806579589844,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":63.868133544921875,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":64.62159729003906,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":64.62159729003906,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":65.00910186767578,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":0},{"_time":65.00910949707031,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":65.378662109375,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":0},{"_time":65.378662109375,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":65.91326904296875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":65.91328430175781,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":66.55911254882812,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":66.55911254882812,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":67.07577514648438,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":67.07577514648438,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":1},{"_time":67.42021942138672,"_lineIndex":0,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":67.42021942138672,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":1},{"_time":68.32438659667969,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":0},{"_time":68.41048431396484,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":8},{"_time":68.49658966064453,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":8},{"_time":69.0993881225586,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":6},{"_time":69.0993881225586,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":69.78826141357422,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":5},{"_time":69.87437438964844,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":8},{"_time":69.96047973632812,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":8},{"_time":70.04658508300781,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":70.5833511352539,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":71.0799331665039,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":71.59660339355469,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":72.13333892822266,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":72.5986099243164,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":72.5986099243164,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":73.08953857421875,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":5},{"_time":73.08953857421875,"_lineIndex":3,"_lineLayer":1,"_type":1,"_cutDirection":5},{"_time":73.33501434326172,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":73.33501434326172,"_lineIndex":1,"_lineLayer":1,"_type":0,"_cutDirection":4},{"_time":73.90779113769531,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":1},{"_time":73.90779113769531,"_lineIndex":3,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":74.56239318847656,"_lineIndex":1,"_lineLayer":2,"_type":0,"_cutDirection":0},{"_time":74.56239318847656,"_lineIndex":2,"_lineLayer":2,"_type":1,"_cutDirection":0},{"_time":75.0533676147461,"_lineIndex":0,"_lineLayer":0,"_type":1,"_cutDirection":6},{"_time":75.0533676147461,"_lineIndex":2,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":75.54431915283203,"_lineIndex":2,"_lineLayer":1,"_type":1,"_cutDirection":5},{"_time":75.54431915283203,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":76.03524780273438,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":4},{"_time":76.03524780273438,"_lineIndex":1,"_lineLayer":0,"_type":1,"_cutDirection":1},{"_time":77.13988494873047,"_lineIndex":3,"_lineLayer":0,"_type":0,"_cutDirection":7},{"_time":77.13988494873047,"_lineIndex":3,"_lineLayer":2,"_type":1,"_cutDirection":5},{"_time":77.75357055664062,"_lineIndex":1,"_lineLayer":0,"_type":0,"_cutDirection":4},{"_time":77.83537292480469,"_lineIndex":0,"_lineLayer":1,"_type":0,"_cutDirection":8},{"_time":77.91719818115234,"_lineIndex":0,"_lineLayer":2,"_type":0,"_cutDirection":8},{"_time":78.24451446533203,"_lineIndex":2,"_lineLayer":1,"_type":1,"_cutDirection":1},{"_time":78.24452209472656,"_lineIndex":2,"_lineLayer":0,"_type":1,"_cutDirection":1}],"_obstacles":[],"_bookmarks":[]}


def get_info_map_string(name, bpm, bs_diff):
    jump_speed = [np.round(config.jump_speed + config.jump_speed_fact * config.max_speed, 1)]
    if bs_diff == 'Expert':
        diff_plus = config.max_speed / config.expert_fact
        diff_list = ['Expert', 'ExpertPlus']
        exp_jump_speed = np.round(config.jump_speed + config.jump_speed_fact * diff_plus, 1)
        jump_speed = [exp_jump_speed, exp_jump_speed * config.jump_speed_expert_factor]
    else:
        diff_plus = config.max_speed
        diff_list = ['ExpertPlus']

    for i in range(len(jump_speed)):
        if jump_speed[i] > (config.max_njs - 2.5 * i):
            jump_speed[i] = config.max_njs - 2.5 * i
        jump_speed[i] += config.jump_speed_offset
    jump_speed.reverse()  # Set in order Expert (low), ExpertPlus (high)

    if bs_diff == 'Expert':
        jsb_offset = config.jsb_offset.copy()
        jsb_offset[1] -= diff_plus * config.jsb_offset_factor
        jsb_offset[0] -= 2 / 3 * (diff_plus * config.jsb_offset_factor)

        # sanity check too fast jump_speed
        jsb_offset[0] = max([jsb_offset[0], config.jsb_offset_min[0]])
        jsb_offset[1] = max([jsb_offset[1], config.jsb_offset_min[1]])

    else:
        jsb_offset = [config.jsb_offset[1] - diff_plus * config.jsb_offset_factor]
        # sanity check too fast jump_speed (expert+ only)
        jsb_offset[0] = max([jsb_offset[0], config.jsb_offset_min[1]])

    for i in range(len(jump_speed)):
        if jump_speed[i] < 15.5:
            jump_speed[i] = 15.5
            jsb_offset[i] += 0.2

    def calculate_jump_distance(njs, c_bpm, offset):
        half_jump = 4.0
        num = 60.0 / c_bpm
        # Ensure njs is not too low
        if njs <= 0.01:
            njs = 10.0
        # Adjust half_jump to meet the condition
        while njs * num * half_jump > 17.999:
            half_jump /= 2
        half_jump += offset
        if half_jump < 0.25:
            half_jump = 0.25
        jump_distance = njs * num * half_jump * 2
        return jump_distance

    if bpm != 100:
        for i in range(len(jump_speed)):
            last_abs = 99999
            last_offset = 0
            jump_distance_orig = calculate_jump_distance(jump_speed[i], 100, jsb_offset[i])
            for _ in range(20):
                jump_distance_new = calculate_jump_distance(jump_speed[i], bpm, jsb_offset[i])
                if jump_distance_new > jump_distance_orig:
                    jsb_offset[i] -= 0.1
                    last_offset = -0.1
                elif jump_distance_new < jump_distance_orig:
                    jsb_offset[i] += 0.1
                    last_offset = 0.1
                abs_diff = abs(jump_distance_new - jump_distance_orig)
                if abs_diff < 0.5:
                    # good enough
                    break
                if abs_diff > last_abs:
                    # if it gets worse, go back
                    jsb_offset[i] -= last_offset
                    break
                last_abs = abs_diff

    info_string = '{\n'
    info_string += '"_version": "2.0.0",\n'
    info_string += f'"_songName": "{name}",\n'
    info_string += f'"_songSubName": "Diff_{diff_plus / 4:.1f}",\n'
    info_string += '"_songAuthorName": "unknown",\n'
    info_string += '"_levelAuthorName": "InfernoSaber",\n'
    info_string += f'"_beatsPerMinute": {bpm},\n'
    info_string += '"_songTimeOffset": 0,\n'
    info_string += '"_shuffle": 0,\n'
    info_string += '"_shufflePeriod": 0.5,\n'
    info_string += '"_previewStartTime": 10,\n'
    info_string += '"_previewDuration": 20,\n'
    info_string += f'"_songFilename": "{name}.egg",\n'
    info_string += '"_coverImageFilename": "cover.jpg",\n'
    info_string += '"_environmentName": "DefaultEnvironment",\n'
    info_string += '"_allDirectionsEnvironmentName": "GlassDesertEnvironment",\n'
    info_string += '"_difficultyBeatmapSets": ['
    info_string += '{\n'
    info_string += '"_beatmapCharacteristicName": "Standard",\n'
    info_string += '"_difficultyBeatmaps": [\n'

    for i, diff in enumerate(diff_list):
        info_string += '{\n'
        info_string += f'"_difficulty": "{diff}",\n'
        if diff == 'Expert':
            info_string += '"_difficultyRank": 7,\n'
        else:
            info_string += '"_difficultyRank": 9,\n'
        info_string += f'"_beatmapFilename": "{diff}.dat",\n'
        info_string += f'"_noteJumpMovementSpeed": {jump_speed[i]},\n'
        info_string += f'"_noteJumpStartBeatOffset": {jsb_offset[i]:.2f}\n'
        if i + 1 < len(diff_list):
            info_string += '},\n'
        else:
            info_string += '}\n'

    info_string += ']}],\n'
    info_string += ('"_customData": {"_editors": {"_lastEditedBy": '
                    '"InfernoSaber", "InfernoSaber": {"version": "'
                    f"{config.InfernoSaber_version}"
                    '"}}}\n')
    info_string += '}\n'

    return info_string
