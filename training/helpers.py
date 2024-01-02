import numpy as np
import tensorflow as tf
import glob
import os
from keras.models import load_model

from tools.utils.load_and_save import load_npy
from tools.config import paths, config
from tools.config.mapper_selection import return_mapper_list, get_full_model_path
from preprocessing.map_info_processing import get_maps_from_mapper


def ai_encode_song(song):
    # Load pretrained model
    encoder_path = get_full_model_path(config.enc_version)
    encoder = load_model(encoder_path)
    # apply autoencoder to input
    in_song_l = encoder.predict(song, verbose=0)
    return in_song_l


def filter_by_bps(min_limit=None, max_limit=None):
    if config.use_bpm_selection:
        print("Importing maps by BPM")
        # return songs in difficulty range
        diff_ar = load_npy(paths.diff_ar_file)
        name_ar = load_npy(paths.name_ar_file)

        if min_limit is not None:
            selection = diff_ar > min_limit
            name_ar = name_ar[selection]
            diff_ar = diff_ar[selection]
        if max_limit is not None:
            selection = diff_ar < max_limit
            name_ar = name_ar[selection]
            diff_ar = diff_ar[selection]
    else:
        print(f"Importing maps by mapper: {config.use_mapper_selection}")
        mapper_name = return_mapper_list(config.use_mapper_selection)
        name_ar = get_maps_from_mapper(mapper_name)
        diff_ar = np.ones_like(name_ar, dtype='float')*config.min_bps_limit

    return list(name_ar), list(diff_ar)


def test_gpu_tf():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        return True
    else:
        print('Warning: No GPU found')
        input('Continue with CPU?')
        return True
        # tf.test.is_built_with_cuda()
        # print(tf.config.list_physical_devices())
    return False


def load_keras_model(save_model_name, lr=None):
    model = None
    # print("Load keras model from disk")
    if save_model_name == "old":
        keras_models = glob.glob(paths.model_path + "*.h5")
        latest_file = max(keras_models, key=os.path.getctime)
    else:
        if not save_model_name.startswith(paths.model_path):
            latest_file = paths.model_path + save_model_name
        else:
            latest_file = save_model_name
        if not latest_file.endswith('.h5'):
            latest_file += '.h5'

    if os.path.isfile(latest_file):
        model = load_model(latest_file)
        latest_file = os.path.basename(latest_file)
        if config.verbose_level > 4:
            print("Keras model loaded: " + latest_file)
    else:
        print(f"Could not find model on disk: {latest_file}")
        print("Creating new model...")
        return None, save_model_name

    # # print(K.get_value(model.optimizer.lr))
    # if lr is not None:
    #     K.set_value(model.optimizer.lr, lr)
    #     print("Set learning rate to: " + str(K.get_value(model.optimizer.lr)))
    return model, latest_file


def categorical_to_class(cat_ar):
    cat_num = np.argmax(cat_ar, axis=-1)
    # cat_num = np.asarray(cat_num)
    return cat_num


def calc_class_weight(np_ar):
    classes, counts = np.unique(np_ar, return_counts=True)

    counts = counts.min() / counts
    # counts -= 0.1
    class_weight = {}
    for i, cls in enumerate(classes):
        class_weight[cls] = counts[i]

    print(f"Weight matrix: {class_weight}")

    return class_weight
