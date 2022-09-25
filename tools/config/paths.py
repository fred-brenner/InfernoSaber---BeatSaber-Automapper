##########################################
# config file for all paths used in project
##########################################
# edit directory paths (C:/...) for each PC
# !!! Only use "/" and not "\"
# !!! Always end with "/"
##########################################

import os

################################# (change this for your pc)
# setup folder for input data
dir_path = "C:/Users/frede/Desktop/BS_Automapper/Data/"

bs_song_path = "E:/SteamLibrary/steamapps/common/Beat Saber/Beat Saber_Data/CustomLevels/"

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

if not os.path.isfile(main_path + '/main.py'):
    print("Could not find root directory. Exit")
    exit()
main_path += '/'

########################
# input folder structure
model_path = dir_path + "model/"
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
pred_input_path = pred_path + "input/"
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

# class_beat_list_file = class_maps + "beat_list.pkl"
# class_notes_dict_file = class_maps + "notes_dict_list.pkl"
# class_output_ML_file = class_maps + "output_ML.pkl"
# class_time_list_file = class_maps + "time_list.pkl"
# class_song_input_file = class_maps + "song_input_list.pkl"
# class_song_title_file = class_maps + "song_title_list.pkl"
# class_title_file = class_maps + "title_list.pkl"
# class_map_diff_file = class_maps + "map_diff.pkl"

pred_song_input_file = pred_input_path + "song_input.pkl"
pred_beat_input_file = pred_input_path + "beat_input.pkl"
pred_ml_output_file = pred_input_path + "ml_output.pkl"
pred_title_file = pred_input_path + "title_list.pkl"

black_list_file = fail_path + "black_list.txt"

model_autoenc_music_file = model_path + "autoenc_music"
model_automap_file = model_path + "automapper"
notes_classify_dict_file = pred_path + "notes_class_dict.pkl"
beats_classify_encoder_file = pred_path + "onehot_encoder.pkl"
############################
