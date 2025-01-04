import os

from tools.config import paths


bs_folder_name = "Beat Saber/Beat Saber_Data/CustomLevels"


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


def get_summary(diff1, diff2, diff3, diff4, diff5) -> str:
    log = []

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
    else:
        log.append("Info: Beat Saber folder not found. Link it in the first tab to automatically export maps.")

    # check difficulty rating
    diff_count = 0
    diff_count_values = []
    if not isinstance(diff1, int):
        log.append("Error: Difficulty 1 is not set. If not required, set it to 0")
    else:
        if diff1 > 0:
            diff_count += 1
            diff_count_values.append(diff1)
    if not isinstance(diff2, int):
        log.append("Error: Difficulty 2 is not set. If not required, set it to 0")
    else:
        if diff2 > 0:
            diff_count += 1
            diff_count_values.append(diff2)
    if not isinstance(diff3, int):
        log.append("Error: Difficulty 3 is not set. If not required, set it to 0")
    else:
        if diff3 > 0:
            diff_count += 1
            diff_count_values.append(diff3)
    if not isinstance(diff4, int):
        log.append("Error: Difficulty 4 is not set. If not required, set it to 0")
    else:
        if diff4 > 0:
            diff_count += 1
            diff_count_values.append(diff4)
    if not isinstance(diff5, int):
        log.append("Error: Difficulty 5 is not set. If not required, set it to 0")
    else:
        if diff5 > 0:
            diff_count += 1
            diff_count_values.append(diff5)
    log.append(f"Info: Generating {diff_count} difficulties for each song: [{', '.join(diff_count_values)}]")

    return "\n".join(log)
