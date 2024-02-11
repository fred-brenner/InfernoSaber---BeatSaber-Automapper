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


def get_factor_from_max_speed(max_speed, lb=0.5, ub=1.5):
    max_speed = max_speed / 4
    ls = 0
    us = 10

    if max_speed <= ls:
        return lb
    elif max_speed >= us:
        return ub
    else:
        factor = lb + (ub - lb) * (max_speed / us)
        return factor


def add_onset_half_times(times, min_time=0.1, max_time=1.5):
    diff = times[1:] - times[:-1]
    new_times = list(times)
    for idx, d in enumerate(diff):
        if min_time <= d <= max_time:
            new_times.append(times[idx] + (d / 2))

    new_times = np.asarray(new_times)
    new_times = np.sort(new_times)
    return new_times
