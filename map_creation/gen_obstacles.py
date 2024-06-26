import numpy as np
from random import randint

from tools.config import config


def add_obstacle(obstacles: list, position: int, first_time, last_time):
    # check for multi occurrences of obstacles
    # obs_counter = 0
    # obs_break_counter = 4
    # for obs_last in obstacles[::-1]:
    #     obs_last_time = obs_last[0] + obs_last[3]
    #     if first_time <= obs_last_time:
    #         obs_counter += 1
    #     if obs_counter >= config.obstacle_max_count:
    #         break
    #     else:
    #         obs_break_counter -= 1
    #         if obs_break_counter <= 0:
    #             break
    # if obs_counter < config.obstacle_max_count:

    # Switch sport and obstacle mode
    if config.sporty_obstacles:
        config.obstacle_allowed_types = config.sport_obstacle_allowed_types
        config.obstacle_positions = config.sport_obstacle_positions
    else:
        config.obstacle_allowed_types = config.norm_obstacle_allowed_types
        config.obstacle_positions = config.norm_obstacle_positions

    # Add new obstacle
    rand_type = randint(0, len(config.obstacle_allowed_types) - 1)
    o_type = config.obstacle_allowed_types[rand_type]
    o_height = randint(1, config.max_obstacle_height)
    first_time += config.obstacle_time_gap[0]
    last_time -= config.obstacle_time_gap[1]

    duration = last_time - first_time
    # _obstacles":[{"_time":64.39733123779297,"_lineIndex":0,
    #               "_type":0,"_duration":6.5,"_width":1}
    cur_obstacle = [first_time, position, o_type, duration, config.obstacle_width, o_height]
    obstacles[position].append(cur_obstacle)
    return obstacles


def check_obstacle_times(first_time, last_time):
    time_diff = last_time - first_time
    if time_diff <= config.obstacle_min_duration + sum(config.obstacle_time_gap):
        return False
    else:
        return True


def combine_obstacles(obstacles_all, times_empty):
    def found_obstacle(obst_temp, first_time, last_time, position, width=config.obstacle_width):
        if width > 2:
            o_type = 2  # only allow ceiling type for crouch walls
            o_height = 3
        else:
            rand_type = randint(0, len(config.obstacle_allowed_types) - 1)
            o_type = config.obstacle_allowed_types[rand_type]
            o_height = randint(1, config.max_obstacle_height)

        for t_empty in times_empty:
            if first_time >= t_empty:
                break
            else:
                if t_empty < last_time:
                    dur_temp = round(t_empty - first_time - 0.1, 1)
                    cur_obstacle = [first_time, position, o_type, dur_temp, width, o_height]
                    obst_temp.append(cur_obstacle)
                    first_time = t_empty
        dur_temp = round(last_time - first_time, 1)
        # if dur_temp < 0:
        #     print("Error. Encountered negative duration for obstacles! Exclude.")
        if dur_temp > 0:
            cur_obstacle = [first_time, position, o_type, dur_temp, width, o_height]
            obst_temp.append(cur_obstacle)
        return obst_temp

    def check_saved_times(first, last, lt_save_first, lt_save_last, diff_min):
        if len(lt_save_last) == 0 or not config.sporty_obstacles:
            return first, last
        first_last = [first, last]
        for i_save, lt_save in enumerate([lt_save_last, lt_save_first]):
            diff_array = np.asarray(lt_save) - first_last[i_save]
            # only check negative values as positive are not possible for sporty mode
            if i_save == 1:
                diff_array *= -1
            diff_array = diff_array[diff_array < 0]
            if len(diff_array) > 0:
                diff_real = round(abs(max(diff_array)), 1)
                if diff_real < diff_min:
                    if i_save == 0:
                        first += round(diff_min - diff_real, 1)
                    if i_save == 1:
                        last -= round(diff_min - diff_real, 1)
        return first, last

    step_size = 0.1
    obstacles = []
    time_list = [[], [], [], []]
    for idx in range(len(obstacles_all)):
        for obst in obstacles_all[idx]:
            start_time = round(obst[0], 1)
            duration = round(obst[3], 1)
            new_times = np.round(np.arange(start_time, start_time + duration + step_size, step_size), 1)
            time_list[idx].extend(list(new_times))
    common_val1 = list(set(time_list[0]).intersection(time_list[1]))
    common_val2 = list(set(time_list[2]).intersection(time_list[3]))
    common_val1.sort()
    common_val2.sort()

    if config.sporty_obstacles:
        common_val3 = list(set(common_val1).intersection(common_val2))
        common_val3.sort()
    else:
        common_val3 = []

    last_time_saves = []
    first_time_saves = []
    diff_gap_min = round(max(config.obstacle_time_gap), 1)
    for idx, common_val in enumerate([common_val1, common_val2]):
        t_first = -1
        t_last = 0
        for t in common_val:
            if t in common_val3:
                pass    # do later for both sides active
            elif t_first == -1:
                t_first = t
                t_last = t
            elif t > t_last + 2 * step_size:
                rnd_pos = randint(0, len(config.obstacle_positions[idx]) - 1)
                t_first, t_last = check_saved_times(t_first, t_last, first_time_saves,
                                                    last_time_saves, diff_gap_min)
                obstacles = found_obstacle(obstacles, t_first, t_last,
                                           config.obstacle_positions[idx][rnd_pos])
                first_time_saves.append(t_first)
                last_time_saves.append(t_last)
                t_first = -1
            else:
                t_last = t
        if t_first > 0:
            if t_last - t_first >= config.obstacle_min_duration:
                rnd_pos = randint(0, len(config.obstacle_positions[idx]) - 1)
                t_first, t_last = check_saved_times(t_first, t_last, first_time_saves,
                                                    last_time_saves, diff_gap_min)
                obstacles = found_obstacle(obstacles, t_first, t_last,
                                           config.obstacle_positions[idx][rnd_pos])
                first_time_saves.append(t_first)
                last_time_saves.append(t_last)

    # Obstacles for both sides on
    t_first = -1
    t_last = 0
    diff_gap_min = round(max(config.obstacle_time_gap), 1)
    for t in common_val3:
        if t_first == -1:
            t_first = t
            t_last = t
        elif t > t_last + 2 * step_size:
            obstacle_pos = []
            for idx in range(2):
                rnd_pos = randint(0, len(config.obstacle_positions[idx]) - 1)
                obstacle_pos.append(config.obstacle_positions[idx][rnd_pos])
            if t_first < 10:
                # do not allow crouching at the beginning of song
                if obstacle_pos[0] == 1 and obstacle_pos[1] == 2:
                    obstacle_pos[1] = 3
            if obstacle_pos[0] == 1 and obstacle_pos[1] == 2:
                # crouch obstacle
                # t_first, t_last = check_saved_times(t_first, t_last, first_time_saves,
                #                                     last_time_saves, diff_gap_min)
                t_first += 0.1  # allow more time for crouching
                t_last -= 0.1
                obstacles = found_obstacle(obstacles, t_first, t_last, 0,
                                           config.obstacle_crouch_width)
            else:
                t_first, t_last = check_saved_times(t_first, t_last, first_time_saves,
                                                    last_time_saves, diff_gap_min)
                obstacles = found_obstacle(obstacles, t_first, t_last, obstacle_pos[0])
                obstacles = found_obstacle(obstacles, t_first, t_last, obstacle_pos[1])
            t_first = -1
        else:
            t_last = t

    return obstacles


def calculate_obstacles(notes, timings):    # TODO: change config to obstacle_mode with 0 None and up to3 for full
    obstacles_all = [[], [], [], []]
    rows_last = [2, 2, 2, 2]
    times_empty = [0]
    for idx in range(len(notes)):
        if len(notes[idx]) == 0:
            times_empty.append(timings[idx])
        else:
            cur_rows = notes[idx][::4]
            for n_row in cur_rows:
                # if n_row == 0 or n_row == 3:
                if check_obstacle_times(rows_last[n_row], timings[idx]):
                    obstacles_all = add_obstacle(obstacles_all, n_row, rows_last[n_row],
                                                 timings[idx])
                rows_last[n_row] = timings[idx]

    obstacles = combine_obstacles(obstacles_all, times_empty)

    # sort by timings
    obstacles = np.asarray(obstacles)
    obstacles = obstacles[np.argsort(obstacles[:, 0])]

    return obstacles
