import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.pytorch_models import ConvAutoencoder
from preprocessing.music_processing import run_music_preprocessing
from tools.config import config, paths


def get_song_repr(name_ar):

    # Setup configuration
    #####################
    np.random.seed(config.random_seed)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = paths.model_autoenc_music_file + '_saved_model.pth'

    model = ConvAutoencoder()
    model.load_state_dict(torch.load(save_path))
    model = model.to(device)
    model.eval()

    # Data Preprocessing
    ####################

    # load song input
    song_ar = run_music_preprocessing(name_ar, save_file=False, song_combined=True)

    # reshape image into 3D tensor
    song_ar = song_ar.reshape((song_ar.shape[0], 1, song_ar.shape[1], song_ar.shape[2]))

    # set up data loader
    data_loader = DataLoader(song_ar, batch_size=512)

    # Run Music Encoder
    encoded = None
    for images in data_loader:
        images = images.to(device)
        output = model.encoder(images)

        output = output.cpu().detach().numpy()
        if encoded is None:
            encoded = output
        else:
            encoded = np.vstack((encoded, output))

    return encoded
