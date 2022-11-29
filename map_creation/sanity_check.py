import numpy as np
from tools.config import config


def sanity_check_beat(beat):
    beat = beat.reshape(len(beat))
    beat_counts = beat.sum()
    min_count = config.beat_spacing * config.min_beat_time
    min_count = int(np.round(min_count, 0))

    found_last_beat = -10
    for idx in range(len(beat)):
        if beat[idx] == 1:
            if found_last_beat < idx - min_count:
                found_last_beat = idx
            else:
                # found too many beats
                beat[idx] = 0

    # print result
    print(f"Got {beat.sum()} beats after sanity check"
          f" (removed {beat_counts-beat.sum()})")

    return beat
