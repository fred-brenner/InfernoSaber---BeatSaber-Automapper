import numpy as np
from random import randint

from tools.config import config


def get_integer_note_color(sideL: bool):
    if sideL:
        side_integer = 0
    else:
        side_integer = 1
    return side_integer


def get_side_time_gaps(sideL, notes, timings):
    side_integer = get_integer_note_color(sideL)
    real_time_gap = []
    real_time_gap_indices = []
    b_flag = False
    t0 = -1
    for i in range(1, len(timings)):
        # check for notes
        if len(notes[i]) > 0:
            # skip if bomb is found
            if 3 in notes[i][2::4]:
                b_flag = True
                continue
            if b_flag:
                t0 = i
                b_flag = False
                continue
            # check note color
            if side_integer in notes[i][2::4]:
                if t0 < 0:
                    t0 = i
                    continue
                # found it!
                t1 = i
                real_time_gap.append(timings[t1] - timings[t0])
                real_time_gap_indices.append([t0, t1])
                t0 = i
    return real_time_gap, real_time_gap_indices


def get_position_of_note(notes, ix, side_integer):
    for n in range(int(len(notes[ix]) / 4)):
        if side_integer == notes[ix][2 + n*4]:
            # found it!
            x = notes[ix][0 + n*4]
            y = notes[ix][1 + n*4]
            d = notes[ix][3 + n*4]
            return x, y, d

    print("Error: Could not find note to attach arc.")
    exit()


def get_side_sliders(sideL, notes, timings, tg, tg_index):
    side_integer = get_integer_note_color(sideL)
    sliders = []
    for idx, time_gap in enumerate(tg):
        if config.slider_time_gap[0] < time_gap < config.slider_time_gap[1]:
            # get corresponding note position
            i0 = tg_index[idx][0]
            i1 = tg_index[idx][1]
            x0, y0, d0 = get_position_of_note(notes, i0, side_integer)
            x1, y1, d1 = get_position_of_note(notes, i1, side_integer)
            sliders.append([timings[i0], side_integer, x0, y0, d0, config.slider_radius_multiplier,
                            timings[i1], x1, y1, d1, config.slider_radius_multiplier, 0])
    return sliders


def calculate_sliders(notes, timings):
    sliders_combined = []
    # get time gaps for left side
    tg, tg_index = get_side_time_gaps(True, notes, timings)
    slidersL = get_side_sliders(True, notes, timings, tg, tg_index)
    tg, tg_index = get_side_time_gaps(False, notes, timings)
    slidersR = get_side_sliders(False, notes, timings, tg, tg_index)
    sliders_combined.extend(slidersL)
    sliders_combined.extend(slidersR)
    print(f"Generated {len(sliders_combined)} arc sliders.")
    return sliders_combined
