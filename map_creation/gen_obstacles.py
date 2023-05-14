import numpy as np

from tools.config import config

time_gap = config.obstacle_time_gap * (1 - config.max_speed / 80)


def add_obstacle(obstacles: list, position: int,  first_time, last_time, times_empty):
    # TODO: check function!
    # check for multi occurrences of obstacles
    obs_counter = 0
    obs_break_counter = 4
    for obs_last in obstacles[::-1]:
        obs_last_time = obs_last[0] + obs_last[3]
        if first_time <= obs_last_time:
            obs_counter += 1
        if obs_counter >= config.obstacle_max_count:
            break
        else:
            obs_break_counter -= 1
            if obs_break_counter <= 0:
                break

    if obs_counter < config.obstacle_max_count:
        # Add new obstacle
        o_type = config.obstacle_allowed_types[1]
        first_time += time_gap
        last_time -= time_gap
        for t_empty in times_empty[::-1]:
            if first_time >= t_empty:
                break
            else:
                if t_empty < last_time:
                    duration = t_empty - first_time - 0.1
                    cur_obstacle = [first_time, position, o_type, duration, config.obstacle_width]
                    obstacles.append(cur_obstacle)
                    first_time = t_empty
        duration = last_time - first_time
        # _obstacles":[{"_time":64.39733123779297,"_lineIndex":0,
        #               "_type":0,"_duration":6.5,"_width":1}
        cur_obstacle = [first_time, position, o_type, duration, config.obstacle_width]
        obstacles.append(cur_obstacle)
    return obstacles


def check_obstacle_times(first_time, last_time):
    # TODO: further check 0,3 if obstacle is in middle (1,2)
    time_diff = last_time - first_time
    if time_diff <= config.obstacle_min_duration + time_gap * 2:
        return False
    else:
        return True


def calculate_obstacles(notes, timings):
    # TODO: improve obstacles

    obstacles = []
    rows_last = [1, 1, 1, 1]
    times_0 = []
    times_1 = []
    times_2 = []
    times_3 = []
    times_empty = [0]
    for idx in range(len(notes)):
        if len(notes[idx]) == 0:
            times_empty.append(timings[idx])
        else:
            cur_rows = notes[idx][::4]
            for n_row in cur_rows:
                if n_row == 0 or n_row == 3:
                    if check_obstacle_times(rows_last[n_row], timings[idx]):
                        obstacles = add_obstacle(obstacles, n_row, rows_last[n_row],
                                                 timings[idx], times_empty)
                        rows_last[n_row] = timings[idx]
                # if n_row == 0:
                #     times_0.append([rows_last[n_row], timings[idx]])
                #     if check_obstacle_times(rows_last[n_row], timings[idx]):
                #         obstacles = add_obstacle(obstacles, n_row, rows_last[n_row],
                #                                  timings[idx])
                #     rows_last[n_row] = timings[idx]
                # elif n_row == 1:
                #     times_1.append([rows_last[n_row], timings[idx]])
                #     if check_obstacle_times(rows_last[n_row], timings[idx]):
                #         obstacles = add_obstacle(obstacles, n_row, rows_last[n_row],
                #                                  timings[idx])
                #     rows_last[n_row] = timings[idx]
                # elif n_row == 2:
                #     times_2.append([rows_last[n_row], timings[idx]])
                #     if check_obstacle_times(rows_last[n_row], timings[idx]):
                #         obstacles = add_obstacle(obstacles, n_row, rows_last[n_row],
                #                                  timings[idx])
                #     rows_last[n_row] = timings[idx]
                # elif n_row == 3:
                #     times_3.append([rows_last[n_row], timings[idx]])
                #     if check_obstacle_times(rows_last[n_row], timings[idx]):
                #         obstacles = add_obstacle(obstacles, n_row, rows_last[n_row],
                #                                  timings[idx])
                #     rows_last[n_row] = timings[idx]

    return obstacles
