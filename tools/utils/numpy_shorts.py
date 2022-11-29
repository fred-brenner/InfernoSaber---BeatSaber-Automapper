import numpy as np


def np_append(old, new, axis=0):
    if old is None:
        out = new
    else:
        out = np.concatenate((old, new), axis=axis)
    return out


def minmax_3d(ar: np.array) -> np.array:
    ar -= ar.min()
    ar /= ar.max()
    return ar
