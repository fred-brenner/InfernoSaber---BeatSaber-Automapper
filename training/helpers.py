import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
from keras.models import load_model

from tools.utils.load_and_save import load_npy
from tools.config import paths, config


def ai_encode_song(song):
    # Load pretrained model
    encoder_path = paths.model_path + config.enc_version
    encoder = load_model(encoder_path)
    # apply autoencoder to input
    in_song_l = encoder.predict(song)
    return in_song_l


def filter_by_bps(min_limit=None, max_limit=None):
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

    return list(name_ar), list(diff_ar)


def plot_autoenc_results(img_in, img_repr, img_out, n_samples, scale_repr=True, save=False):
    bneck_reduction = len(img_repr.flatten()) / len(img_in.flatten()) * 100
    print(f"Bottleneck shape: {img_repr.shape}. Reduction to {bneck_reduction:.1f}%")
    print("Plot original images vs. reconstruction")
    fig, axes = plt.subplots(nrows=3, ncols=n_samples, figsize=(12, 8))
    fig.suptitle(f"Reduction to {bneck_reduction:.1f}%")
    if scale_repr:
        img_repr -= img_repr.min()
        img_repr /= img_repr.max()

    # plot original image
    for idx in np.arange(n_samples):
        fig.add_subplot(3, n_samples, idx + 1)
        plt.imshow(np.transpose(img_in[idx], (0, 1, 2)), cmap='hot')
        # plt.imshow(np.transpose(img_in[idx], (1, 2, 0)), cmap='hot')

    # plot bottleneck distribution
    if len(img_repr.shape) < 4:
        square_bottleneck = int(img_repr.shape[1]/4) if int(img_repr.shape[1]/4) > 0 else 1
        img_repr = img_repr.reshape((img_repr.shape[0]), 1, -1, square_bottleneck)
    for idx in np.arange(n_samples):
        fig.add_subplot(3, n_samples, idx + n_samples + 1)
        plt.imshow(np.transpose(img_repr[idx], (1, 2, 0)), cmap='hot')

    # plot output image
    for idx in np.arange(n_samples):
        fig.add_subplot(3, n_samples, idx + 2*n_samples + 1)
        plt.imshow(np.transpose(img_out[idx], (0, 1, 2)), cmap='hot')

    plt.axis('off')
    if save:
        save_path = f"{paths.model_path}bneck{config.bottleneck_len}_encoder_decoder_example.png"
        fig.savefig(save_path)

    plt.show()


def run_plot_autoenc(enc_model, auto_model, ds_test, save=False):
    # Plot first batch of test images
    output = auto_model.predict(ds_test)
    repr_out = enc_model.predict(ds_test)

    plot_autoenc_results(ds_test, repr_out, output, len(ds_test), save=save)


def test_gpu_tf():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        return True
    else:
        print('No GPU found')
    return False


def load_keras_model(save_model_name, lr=None):
    model = None
    print("Load keras model from disk")
    if save_model_name == "old":
        keras_models = glob.glob(paths.model_path + "*.h5")
        latest_file = max(keras_models, key=os.path.getctime)
    else:
        latest_file = paths.model_path + save_model_name
        if not latest_file.endswith('.h5'):
            latest_file += '.h5'

    if os.path.isfile(latest_file):
        model = load_model(latest_file)
        latest_file = os.path.basename(latest_file)
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
