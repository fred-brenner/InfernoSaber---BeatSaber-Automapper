"""
Import and Export of numpy arrays into npy files
"""

import numpy as np
import tools.config.config as config


def load_npy(data_path: str) -> np.array:
    ar = np.load(data_path, allow_pickle=True)
    return ar


def save_npy(ar: np.array, data_path: str):
    ar.dump(data_path)
