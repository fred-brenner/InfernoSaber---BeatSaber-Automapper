import numpy as np
from random import randint, random

from tools.config import config
from map_creation.sanity_check import calc_note_speed


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
    for i in range(0, len(timings)):
        # check for notes
        if len(notes[i]) > 0:
            # skip if bomb is found
            if 3 in notes[i][2::4]:
                b_flag = True
                continue
            elif b_flag:
                # t0 = -1
                b_flag = False
                continue
            # check note color
            if side_integer in notes[i][2::4]:
                if t0 < 0:
                    t0 = i
                    real_time_gap.append(timings[t0])
                    real_time_gap_indices.append([-1, i])
                    continue
                # found it!
                t1 = i
                real_time_gap.append(timings[t1] - timings[t0])
                real_time_gap_indices.append([t0, t1])
                t0 = i
                # if real_time_gap_indices[-1] == [286, 290]:
                #     print("")
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
        if idx == 0:
            if config.slider_turbo_start:
                i0 = tg_index[idx][0]
                i1 = tg_index[idx][1]
                if i0 < 0:
                    x1, y1, d1 = get_position_of_note(notes, i1, side_integer)
                    # d0 = 8
                    d0 = d1
                    anchor_mode = 1
                    if side_integer == 0:
                        if d1 in [4, 0, 5]:     # left side up
                            anchor_mode = 2
                    else:
                        if d1 in [6, 1, 7]:     # right side down
                            anchor_mode = 2
                    sliders.append([0.3*timings[i1], side_integer, x1, y1, d0, config.slider_radius_multiplier,
                                    timings[i1], x1, y1, d1, config.slider_radius_multiplier, anchor_mode])
            continue

        if config.slider_time_gap[0] < time_gap < config.slider_time_gap[1]:
            # get corresponding note position
            i0 = tg_index[idx][0]
            i1 = tg_index[idx][1]
            x0, y0, d0 = get_position_of_note(notes, i0, side_integer)
            x1, y1, d1 = get_position_of_note(notes, i1, side_integer)

            if config.slider_movement_minimum > 0:
                # delete some random sliders based on movement distance
                nl_last = [x0, y0, 0, d0]
                nl_new = [x1, y1, 0, d1]
                speed = calc_note_speed(nl_last, nl_new, 1,
                                        cdf_lr=1.5)
                if speed < config.slider_movement_minimum:
                    continue
            if 0 <= config.slider_probability < 1:
                if random() <= config.slider_probability:
                    continue

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
