######################################################
# Process data from maps to dictionary (maps_dict_all)
# Cast metadata to events, notes and obstacles
######################################################

import os
import numpy as np
# from progressbar import ProgressBar

from tools.config import paths
from tools.utils.index_find_str import return_find_str
from tools.fail_list.black_list import append_fail, delete_fails


def check_map_valid(check_type, i_s):
    valid = True
    if check_type == "notes":
        # time = i_s[0]
        # if min(time) < 0:
        #     print("Time below zero...")
        #     valid = False
        index = i_s[1]
        if max(index) > 3 or min(index) < 0:
            print("Wrong note index...")
            valid = False
        layer = i_s[2]
        if max(layer) > 2 or min(layer) < 0:
            print("Wrong note layer...")
            valid = False
        type = i_s[3]
        if max(type) > 3 or min(type) < 0:
            print("Wrong note type...")
            valid = False
        direct = i_s[4]
        if max(direct) > 8 or min(direct) < 0:
            print("Wrong note type...")
            valid = False

    elif check_type == "obstacles":
        # index = i_s[1]
        # if max(index) > 3 or min(index) < 0:
        #     print("Wrong obstacles index...")
        #     valid = False
        type = i_s[2]
        if max(type) > 1 or min(type) < 0:
            print("Wrong obstacles type...")
            valid = False
        # width = i_s[4]
        # if max(width) > 2 or min(width) < 1:
        #     print("Wrong obstacles width...")
        #     valid = False

    # if not valid:
    #     input("Enter")
    return valid


def map_to_dict_all():
    # delete old map data
    print("\nDelete old dictionary data")
    for de in os.listdir(paths.dict_all_path):
        # input("Delete {}?" .format(de))
        os.remove(paths.dict_all_path + de)

    print("\nStart casting to dictionary")
    num_cur = 0
    num_all = len(os.listdir(paths.copy_path_map)) + 1
    # bar = ProgressBar(max_value=num_all)
    title_list = os.listdir(paths.copy_path_map)
    title_list.reverse()
    for title in title_list:
        print(title)
        num_cur += 1
        # bar.update(num_cur)
        if title.endswith("_info.dat"):
            with open(paths.copy_path_map + title, 'r') as f:
                for line in f:
                    if "beatsPerMinute" in line:
                        bpm, _ = return_find_str(0, line, '"_beatsPerMinute"', False)
                        bpm = float(bpm)
                        break

        elif title.endswith(".dat"):
            # print(title)
            i = 0
            # time = ['time']
            # direct = ['direction']
            # index = ['index']
            # layer = ['layer']
            # type = ['type']
            time = list()
            direct = list()
            index = list()
            layer = list()
            type = list()
            # time_idx = 0
            with open(paths.copy_path_map + title, 'r') as f:
                for line in f:
                    i += 1
                    if i > 1:
                        # only one line allowed!
                        print("Fail by " + title + ". Probably modded mapping")
                        append_fail(title[:-4])
                        break

                    # search keyword notes
                    time_idx = line.find('_notes')
                    while time_idx != -1:
                        # check if not empty
                        if line.find('_time', time_idx) == -1:
                            append_fail(title[:-4])
                            print("#!No notes found in " + title)
                            break
                        # time index
                        value, time_idx = return_find_str(time_idx, line, '_time', False)
                        time.append(value)

                        # line index
                        value, time_idx = return_find_str(time_idx, line, '_lineIndex', False)
                        index.append(value)

                        # line layer
                        value, time_idx = return_find_str(time_idx, line, '_lineLayer', False)
                        layer.append(value)

                        # type
                        value, time_idx = return_find_str(time_idx, line, '_type', False)
                        type.append(value)

                        # cut direction
                        value, time_idx = return_find_str(time_idx, line, '_cutDirection', False)
                        direct.append(value)

                        # print(time_idx)
                        if line[time_idx + 1] == ']':
                            np_notes = np.asarray([time, index, layer, type, direct], dtype="float")
                            np_notes[0] = np_notes[0] * 60 / bpm
                            # check correct mapping
                            if not check_map_valid("notes", np_notes):
                                print("Information: Found unknown note pattern in: {} || Skipping.".format(title[:-4]))
                                append_fail(title[:-4])
                                break
                            # save dictionary
                            np_notes.dump(paths.dict_all_path + title[:-4] + "_notes.dat")
                            break
                    #######################################################################################
                    # search keyword events
                    time_events = list()
                    type_events = list()
                    value_events = list()
                    time_idx = line.find('_events')
                    # {"_time":1.312,"_type":8,"_value":0}
                    while time_idx != -1:
                        # check if not empty
                        check_empty = line.find('[', time_idx)
                        if line[check_empty + 1] == ']':
                            # print('#!No events found in ' + title)
                            np_events = np.zeros(0)
                            np_events.dump(paths.dict_all_path + title[:-4] + "_events.dat")
                            break
                        # time index
                        value, time_idx = return_find_str(time_idx, line, '_time', False)
                        time_events.append(value)

                        # type index
                        value, time_idx = return_find_str(time_idx, line, '_type', False)
                        type_events.append(value)

                        # value index
                        value, time_idx = return_find_str(time_idx, line, '_value', False)
                        value_events.append(value)

                        # print(time_idx)
                        if line[time_idx + 1] == ']':
                            np_events = np.asarray([time_events, type_events, value_events], dtype="float")
                            np_events[0] = np_events[0] * 60 / bpm
                            # check correct mapping
                            # if check_map_valid("events", np_events) == False:
                            #     print("Information: Found unknown pattern in: {} Skipping." .format(title[.-4]))
                            #     append_fail(title[:-4])
                            #     break
                            # save dictionary
                            np_events.dump(paths.dict_all_path + title[:-4] + "_events.dat")
                            break

                    #######################################################################################
                    # search keyword obstacles
                    time_obs = list()
                    line_obs = list()
                    type_obs = list()
                    duration_obs = list()
                    width_obs = list()
                    time_idx = line.find('_obstacles')
                    # [{"_time":64.33,"_lineIndex":0,"_type":0,"_duration":6.5,"_width":1},
                    while time_idx != -1:
                        # check if not empty
                        check_empty = line.find('[', time_idx)
                        if line[check_empty + 1] == ']':
                            # print('#!No obstacles found in ' + title)
                            np_obstacles = np.zeros(0)
                            np_obstacles.dump(paths.dict_all_path + title[:-4] + "_obstacles.dat")
                            break
                        # time index
                        value, time_idx = return_find_str(time_idx, line, '_time', False)
                        time_obs.append(value)

                        # line index
                        value, time_idx = return_find_str(time_idx, line, '_lineIndex', False)
                        line_obs.append(value)

                        # type index
                        value, time_idx = return_find_str(time_idx, line, '_type', False)
                        type_obs.append(value)

                        # duration index
                        value, time_idx = return_find_str(time_idx, line, '_duration', False)
                        duration_obs.append(value)

                        # width index
                        value, time_idx = return_find_str(time_idx, line, '_width', False)
                        width_obs.append(value)

                        # print(time_idx)
                        if line[time_idx + 1] == ']':
                            np_obstacles = np.asarray([time_obs, line_obs, type_obs, duration_obs, width_obs],
                                                      dtype="float")
                            np_obstacles[0] = np_obstacles[0] * 60 / bpm
                            # check correct mapping
                            if not check_map_valid("obstacles", np_obstacles):
                                print("Information: Found unknown obstacles in: {} || Skipping.".format(title[:-4]))
                                append_fail(title[:-4])
                                break
                            # save dictionary
                            np_obstacles.dump(paths.dict_all_path + title[:-4] + "_obstacles.dat")
                            break

    print("Finished casting to dictionary")

    # Delete all unfinished songs
    delete_fails()
