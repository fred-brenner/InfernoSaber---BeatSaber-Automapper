import numpy as np


def align_beats_on_bpm(timings_in_s, bpm):
    """
    Aligns the timings to the nearest beat of the given bpm.
    :param timings_in_s: list of timings in seconds
    :param bpm: beats per minute
    :return: list of timings in seconds aligned to the nearest beat of the given bpm
    """
    average_bpm = (timings_in_s[-1] - timings_in_s[0]) / len(timings_in_s) * 60
    timings_bs = timings_in_s * bpm / 60

    # try to first match all elements on integer numbers
    timings_bs_new = np.round(timings_bs)
    timing_helper = timings_bs_new[1:] - timings_bs_new[:-1]
    already_done = False
    for idx in range(len(timing_helper)):
        if timing_helper[idx] != 0 and already_done:
            already_done = False
        if timing_helper[idx] == 0 and not already_done:
            already_done = True
            # found a clash, need to check first how many elements
            n_clash = 1
            for j in range(idx + 1, len(timing_helper)):
                if timing_helper[j] != 0:
                    break
                n_clash += 1
            # in case the last element is a zero, the previous range would be skipped
            if idx + 2 >= len(timings_bs_new):
                j = len(timing_helper) - 1
            # define start and end time to put clash in between
            start_time = timings_bs_new[idx]
            if (j + 2) >= len(timings_bs_new):
                # use the last element as end time
                end_time = timings_bs_new[-1] + 1
            else:
                end_time = timings_bs_new[j + 1]

            # define new entries
            new_entries = np.linspace(start_time, end_time, n_clash + 2)
            for _j in range(n_clash):
                timings_bs_new[idx + _j + 1] = new_entries[_j + 1]

    # timings_aligned = np.asarray(timings_bs_new, dtype=float) * 60 / bpm
    timings_aligned = timings_bs_new * 60 / bpm
    return timings_aligned