##########################################
# config file for all paths used in project
##########################################
# edit directory paths (C:/...) for each PC
# !!! Only use "/" and not "\"
# !!! Always end with "/"
##########################################

# /mnt/c/Users/frede/Desktop/BS_Automapper/InfernoSaber---BeatSaber-Automapper

import os
from tools.config import config


################################# (change this for your pc)
# setup folder for input data
dir_path = "C:/Users/frede/Desktop/BS_Automapper/Data/"

bs_song_path = "E:/SteamLibrary/steamapps/common/Beat Saber/Beat Saber_Data/CustomLevels/"
bs_input_path = "C:/Users/frede/Desktop/BS_Automapper/Data/training/custom_bs_input/"

############################# (no need to change)
# main workspace path
main_path = os.path.abspath(os.getcwd())
max_tries = 3
for i in range(0, max_tries):
    if not os.path.isfile(main_path + '/main.py'):
        # not found, search root folder
        main_path = os.path.dirname(main_path)
    else:
        # found main folder
        break

# linux wsl2
if main_path.startswith('/mnt/'):
    dir_path = os.path.dirname(main_path) + '/Data/'
    bs_input_path = os.path.join('/mnt/', bs_input_path.replace('C:', 'c'))
    bs_song_path = os.path.join('/mnt/', bs_song_path.replace('E:', 'e'))
    # import sys
    # sys.path.append(main_path + '/tools/')

# try Google Drive path
if not os.path.isdir(dir_path):
    dir_path = "/content/drive/My Drive/InfernoSaber/Data/"
    main_path = "/content/drive/My Drive/InfernoSaber/Code/"


if not os.path.isfile(main_path + '/main.py'):
    print(f"dir_path={dir_path}")
    print(f"main_path={main_path}")
    print("Could not find root directory. Exit")
    exit()
main_path += '/'

########################
# input folder structure
model_path = dir_path + "model/"
if config.use_mapper_selection == '' or config.use_mapper_selection is None:
    model_path += "general_new/"
else:
    model_path += f"{config.use_mapper_selection.lower()}/"
pred_path = dir_path + "prediction/"
train_path = dir_path + "training/"
temp_path = dir_path + "temp/"

############################
# input subfolder structure
copy_path_song = train_path + "songs_egg/"
copy_path_map = train_path + "maps/"

dict_all_path = train_path + "maps_dict_all/"
# pic_path = train_path + "songs_pic/"

songs_pred = pred_path + "songs_predict/"
# pic_path_pred = pred_path + "songs_pic_predict/"

# pred_path = pred_path + "np_pred/"
# pred_input_path = pred_path + "input/"
new_map_path = pred_path + "new_map/"

fail_path = train_path + "fail_list/"
diff_path = train_path + "songs_diff/"
song_data = train_path + "song_data/"

# class_maps = train_path + "classify_maps/"
ml_input_path = train_path + "ml_input/"

diff_ar_file = diff_path + "diff_ar.npy"
name_ar_file = diff_path + "name_ar.npy"

ml_input_beat_file = ml_input_path + "beat_ar.npy"
ml_input_song_file = ml_input_path + "song_ar.npy"

black_list_file = fail_path + "black_list.txt"

notes_classify_dict_file = f"{model_path}notes_class_dict.pkl"
# beats_classify_encoder_file = pred_path + f"onehot_encoder_beats_{config.min_bps_limit}-{config.max_bps_limit}.pkl"
beats_classify_encoder_file = model_path + "onehot_encoder_beats.pkl"
events_classify_encoder_file = model_path + "onehot_encoder_events.pkl"
############################

# check runtime environment
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
