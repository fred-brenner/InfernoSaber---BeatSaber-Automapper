import math
import os

from tools.config import paths, config


bs_folder_name = "Beat Saber_Data/CustomLevels"
bs_folder_name2 = "SharedMaps/CustomLevels"

ESTIMATED_STARTUP_SECONDS = 20
ESTIMATED_SECONDS_PER_RUN = 15


def format_time_estimate(total_seconds):
    """Format seconds as 'N s', 'N.N min' or 'N.N h' with at most one decimal."""
    if total_seconds >= 3600:
        value, unit = total_seconds / 3600, "h"
    elif total_seconds >= 60:
        value, unit = total_seconds / 60, "min"
    else:
        value, unit = total_seconds, "s"
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{text} {unit}"


def check_int_input(inp, start=1, end=10):
    if isinstance(inp, int):
        if start <= inp <= end:
            return True
    return False


def check_float_input(inp, start=0.0, end=10.0):
    if isinstance(inp, float):
        if start <= inp <= end:
            return True
    return False


def check_str_input(inp, min_len=5):
    if isinstance(inp, str):
        if len(inp) >= min_len:
            return True
    return False


def get_summary(diff1=config.difficulty_1, diff2 = config.difficulty_2, diff3 = config.difficulty_3,
                diff4 = config.difficulty_4, diff5 = config.difficulty_5) -> str:
    log = []
    if not isinstance(diff1, float) and not isinstance(diff1, int):
        try:
            diff1 = diff1.value
            diff2 = diff2.value
            diff3 = diff3.value
            diff4 = diff4.value
            diff5 = diff5.value
        except:
            print("Error: Could not convert difficulty values to numbers.")

    # log number of songs found
    try:
        files = os.listdir(paths.songs_pred)
        song_list = []
        for file_name in files:
            ending = file_name.split('.')[-1]
            if ending in ['mp3', 'mp4', 'm4a', 'wav', 'aac', 'flv', 'wma', 'ogg', 'egg']:
                song_list.append(file_name)
        if len(song_list) > 0:
            log.append(f"Info: Found {len(song_list)} song(s).")
        else:
            log.append("Error: Found 0 songs. Please go to first tab or manually copy them to the input folder.")
    except OSError:
        log.append("Error: Data folder not found.")
        return "\n".join(log)

    # check export functionality
    filename = paths.bs_song_path
    # filename = filename.replace('\\\\', '/').replace('\\', '/')
    if os.path.isdir(filename) and bs_folder_name in filename:
        log.append("Info: Beat Saber folder found. Maps will be exported by default")
    elif os.path.isdir(filename) and bs_folder_name2 in filename:
        log.append("Info: Beat Saber folder found. Maps will be exported by default")
    else:
        log.append("Info: Beat Saber folder not found. Link it in the first tab to automatically export maps.")

    # check difficulty rating
    diff_count = 0
    diff_count_values = []
    if not isinstance(diff1, float) and not isinstance(diff1, int):
        log.append("Error: Difficulty 1 is not set. If not required, set it to 0")
    else:
        if diff1 > 0:
            diff_count += 1
            diff_count_values.append(diff1)
    if not isinstance(diff2, float) and not isinstance(diff2, int):
        log.append("Error: Difficulty 2 is not set. If not required, set it to 0")
    else:
        if diff2 > 0:
            diff_count += 1
            diff_count_values.append(diff2)
    if not isinstance(diff3, float) and not isinstance(diff3, int):
        log.append("Error: Difficulty 3 is not set. If not required, set it to 0")
    else:
        if diff3 > 0:
            diff_count += 1
            diff_count_values.append(diff3)
    if not isinstance(diff4, float) and not isinstance(diff4, int):
        log.append("Error: Difficulty 4 is not set. If not required, set it to 0")
    else:
        if diff4 > 0:
            diff_count += 1
            diff_count_values.append(diff4)
    if not isinstance(diff5, float) and not isinstance(diff5, int):
        log.append("Error: Difficulty 5 is not set. If not required, set it to 0")
    else:
        if diff5 > 0:
            diff_count += 1
            diff_count_values.append(diff5)
    diff_count_values = [str(diff_float) for diff_float in diff_count_values]
    total_runs = diff_count * len(song_list)
    log.append(f"Info: Generating {diff_count} difficulties for each song: [{', '.join(diff_count_values)}]. "
               f"Total runs: {total_runs}")
    if total_runs > 0:
        workers = max(int(config.num_workers), 1)
        est_seconds = ESTIMATED_STARTUP_SECONDS + math.ceil(
            ESTIMATED_SECONDS_PER_RUN * total_runs / workers)
        log.append(f"Info: Estimated time: ~{format_time_estimate(est_seconds)} "
                   f"({ESTIMATED_SECONDS_PER_RUN} s per run + {ESTIMATED_STARTUP_SECONDS} s startup, "
                   f"{workers} worker(s))")

    return "\n".join(log)
