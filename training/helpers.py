import numpy as np
import torch
from tools.utils.load_and_save import load_npy
from tools.config import paths, config
import matplotlib.pyplot as plt


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


def check_cuda_device():
    if not torch.cuda.is_available():
        print("No Cuda device detected. Continue?")
        input("Enter")
    return None


def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))
