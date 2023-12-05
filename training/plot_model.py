import matplotlib.pyplot as plt
import numpy as np

from tools.config import paths, config


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
    output = auto_model.predict(ds_test, verbose=0)
    repr_out = enc_model.predict(ds_test, verbose=0)

    plot_autoenc_results(ds_test, repr_out, output, len(ds_test), save=save)
