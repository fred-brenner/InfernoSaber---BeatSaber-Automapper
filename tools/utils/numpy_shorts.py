import numpy as np
from random import shuffle


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


def reduce_number_of_songs(name_ar, hard_limit=50):
    if len(name_ar) > hard_limit:
        shuffle(name_ar)
        print(f"Info: Loading reduced song number into generator to not overload the RAM (from {len(name_ar)})")
        name_ar = name_ar[:hard_limit]
    print(f"Importing {len(name_ar)} songs")
    return name_ar
