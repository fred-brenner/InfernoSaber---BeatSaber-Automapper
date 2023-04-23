"""
Import and Export of numpy arrays into npy files
"""

import numpy as np
import pickle
import tools.config.config as config


def load_npy(data_path: str) -> np.array:
    ar = np.load(data_path, allow_pickle=True)
    return ar


def save_npy(ar: np.array, data_path: str):
    ar.dump(data_path)


def load_pkl(data_path: str) -> list:
    with open(data_path + 'debug_variables.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl(data: list, data_path: str):
    with open(data_path + 'debug_variables.pkl', 'wb') as f:
        pickle.dump(data, f)
