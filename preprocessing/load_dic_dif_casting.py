####################
#Load map dictionary
####################

import numpy as np


def load_dic_dif_casting(paths):

    print("Load map dictionary and difficulty casting")
    map_dict_events = []
    map_names = []
    map_dict_notes = []
    map_dict_obstacles = []
    diff_ar = []
    # diff = []
    # dict_all_list = os.listdir(dict_all_path)
    name_name, name_idx = load_names_np(paths.pic_path)

    #Get unique names
    map_names = []
    map_names_count = []
    for count_maps in name_name:
        if len(map_names) == 0 or count_maps != map_names[-1]:
            map_names.append(count_maps)
            map_names_count.append(1)
        else:
            map_names_count[-1] = map_names_count[-1]+1
    map_names = np.asarray(map_names)
    map_names_count = np.asarray(map_names_count)

    ###############################################
    # Test 
    test = len(np.unique(np.asarray(name_name)))
    if test != len(map_names):
        print("Error when matching map names")
        exit()
    if sum(map_names_count) != len(name_name):
        print("Error when counting map names")
        exit()

    #######################################################
    # Load all notes, events and obstacles in correct order
    # Cast difficulties for each map segment
    #######################################################
    map_names_count_index = 0
    map_idx = 0
    print("Loading maps input data")
    for dict_name in map_names:
        # Load notes, events, obstacles, all already divided by bpm from info file! (in real sec)
        map_dict_events.append(np.load(paths.dict_all_path + dict_name + "_events.dat"), allow_pickle=True)
        map_dict_notes.append(np.load(paths.dict_all_path + dict_name + "_notes.dat"), allow_pickle=True)
        map_dict_obstacles.append(np.load(paths.dict_all_path + dict_name + "_obstacles.dat"), allow_pickle=True)

        # Test notes available in song
        if map_dict_notes[-1].shape[0] == 0:
            print("Could not find notes in " + str(dict_name) + " Exit!")
            exit()
        
        # Cast difficulty for map names in pictures
        # return_bps = False
        for names_i in name_name:
            if names_i == dict_name:
                # #time index fitting to window size (starts with 1)
                # diff_find_idx = name_idx[map_idx] - 1
                # #append difficulty from window size
                # diff_ar = diff_find(diff_ar, map_dict_notes[-1], config.window, diff_find_idx, return_bps)
                bps_ar = np.load(paths.diff_path + "diff_ar.npy")
                names_ar = np.load(paths.diff_path + "name_ar.npy")
                diff_ar.append(bps_to_diff(bps_ar, names_ar, names_i))
                map_idx += 1

        # Test difficulty array length for every dict_name
        map_names_count_index += 1
        if len(diff_ar) != np.sum(map_names_count[:map_names_count_index]):
            print("Error: difficulty casting at " + dict_name)

    return map_names, map_names_count, name_name, name_idx, map_dict_notes, map_dict_events, map_dict_obstacles, diff_ar