import os
import shutil

from tools.config import paths, config
from tools.config.check_folder_structure import check_folder_structure


def set_app_paths(input_dir):
    print(f"Checking directory setup for: {input_dir}")
    update_file_paths(input_dir)
    check_folder_structure()


def update_file_paths(input_dir):
    if not input_dir.endswith('/'):
        input_dir += '/'
    paths.dir_path = input_dir

    ########################
    # input folder structure
    paths.model_path = paths.dir_path + "model/"
    if config.use_mapper_selection == '' or config.use_mapper_selection is None:
        paths.model_path += "general_new/"
    else:
        paths.model_path += f"{config.use_mapper_selection.lower()}/"
    paths.pred_path = paths.dir_path + "prediction/"
    paths.train_path = paths.dir_path + "training/"
    paths.temp_path = paths.dir_path + "temp/"

    ############################
    # input subfolder structure
    paths.copy_path_song = paths.train_path + "songs_egg/"
    paths.copy_path_map = paths.train_path + "maps/"

    paths.dict_all_path = paths.train_path + "maps_dict_all/"

    paths.songs_pred = paths.pred_path + "songs_predict/"

    paths.new_map_path = paths.pred_path + "new_map/"
    if not os.path.isfile(paths.new_map_path + "cover.jpg"):
        src = f"{paths.main_path}app_helper/cover.jpg"
        dst = f"{paths.new_map_path}cover.jpg"
        shutil.copy(src, dst)
    paths.fail_path = paths.train_path + "fail_list/"
    paths.diff_path = paths.train_path + "songs_diff/"
    paths.song_data = paths.train_path + "song_data/"

    paths.ml_input_path = paths.train_path + "ml_input/"

    paths.diff_ar_file = paths.diff_path + "diff_ar.npy"
    paths.name_ar_file = paths.diff_path + "name_ar.npy"

    paths.ml_input_beat_file = paths.ml_input_path + "beat_ar.npy"
    paths.ml_input_song_file = paths.ml_input_path + "song_ar.npy"

    paths.black_list_file = paths.fail_path + "black_list.txt"

    paths.notes_classify_dict_file = paths.model_path + "notes_class_dict.pkl"
    paths.beats_classify_encoder_file = paths.model_path + "onehot_encoder_beats.pkl"
    paths.events_classify_encoder_file = paths.model_path + "onehot_encoder_events.pkl"
