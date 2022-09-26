import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
from keras.models import load_model
from keras import backend as K

from tools.utils.load_and_save import load_npy
from tools.config import paths, config


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


def plot_autoenc_results(img_in, img_repr, img_out, n_samples, scale_repr=True):
    bneck_reduction = len(img_repr.flatten()) / len(img_in.flatten()) * 100
    print(f"Bottleneck shape: {img_repr.shape}. Reduction to {bneck_reduction:.1f}%")
    print("Plot original images vs. reconstruction")
    fig, axes = plt.subplots(nrows=3, ncols=n_samples, figsize=(12, 8))

    if scale_repr:
        img_repr -= img_repr.min()
        img_repr /= img_repr.max()

    # plot original image
    for idx in np.arange(n_samples):
        fig.add_subplot(3, n_samples, idx + 1)
        plt.imshow(np.transpose(img_in[idx], (1, 2, 0)), cmap='hot')

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
        plt.imshow(np.transpose(img_out[idx], (1, 2, 0)), cmap='hot')

    plt.axis('off')
    plt.show()


def run_plot_autoenc(model, device, data_loader, n_samples):
    # Plot first batch of test images
    model.eval()
    dataiter = iter(data_loader)
    images = dataiter.next()
    images = images.to(device)

    # Sample outputs
    output = model(images)
    repr_out = model.encoder(images)
    images = images.cpu().numpy()
    output = output.cpu().detach().numpy()
    repr_out = repr_out.cpu().detach().numpy()

    plot_autoenc_results(images, repr_out, output, n_samples)


def calculate_loss_score(model, device, data_loader, criterion):
    model.eval()
    loss = 0.0
    for images in data_loader:
        images = images.to(device)
        output = model(images)
        loss_each = criterion(output, images)
        loss += loss_each.item() * images.size(0)
    loss = loss / len(data_loader)
    return loss


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
        keras_models = glob.glob(paths.keras_path_sec + "*.h5")
        latest_file = max(keras_models, key=os.path.getctime)
    else:
        latest_file = paths.keras_path_sec + save_model_name
        if not latest_file.endswith('.h5'):
            latest_file += '.h5'
            save_model_name = latest_file
            model = load_model(latest_file)
            latest_file = os.path.basename(latest_file)
            print("Keras model loaded: " + latest_file)

        if not os.path.isfile(latest_file):
            print(f"Could not find model on disk: {latest_file}")
            print("Creating new model...")

    # # print(K.get_value(model.optimizer.lr))
    # if lr is not None:
    #     K.set_value(model.optimizer.lr, lr)
    #     print("Set learning rate to: " + str(K.get_value(model.optimizer.lr)))
    return model, save_model_name
