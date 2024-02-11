import json
import os

from tools.config import config, paths


def read_json_content_file(file_path: str, filename="") -> list[str]:
    if filename != "":
        file_path = os.path.join(file_path, filename)
    try:
        with open(file_path) as f:
            dat_content = json.load(f)
    except Exception as e:
        print(f"Could not read file: {file_path}. Check file manually. Exit")
        print(f"Error: {e.args}")
        exit()
    return dat_content


def get_difficulty_file_names(info_file_path: str) -> dict:
    # import dat file
    dat_content = read_json_content_file(info_file_path)

    beatmap_set_dict = dat_content['_difficultyBeatmapSets']
    i = -1
    beatmap_dict = beatmap_set_dict[i]
    while not beatmap_dict['_beatmapCharacteristicName'] == 'Standard':
        i -= 1
        if abs(i) > len(beatmap_set_dict):
            print(f"Error: Could not find Standard beatmap key in {info_file_path}")
            exit()
        beatmap_dict = beatmap_set_dict[i]

    beatmap_dict = beatmap_dict['_difficultyBeatmaps']
    diff_file_names = {}
    for diff_dict in beatmap_dict:
        diff_file_names[diff_dict['_difficulty']] = diff_dict['_beatmapFilename']
    return diff_file_names

    # expert_plus_dict = beatmap_dict[-1]
    # expert_plus_name = expert_plus_dict['_beatmapFilename']
    # return expert_plus_name


def check_info_name(bs_song_path):
    for root, dirs, files in os.walk(bs_song_path):
        for file in files:
            if file.lower().endswith("info.dat"):
                if not file == "info.dat":
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(root, "info.dat")
                    os.rename(src_path, dst_path)


def check_beatmap_name(bs_song_path):
    # expected_name = "ExpertPlus.dat"
    for root, dirs, files in os.walk(bs_song_path):

        if len(files) <= 2:
            continue
        diff_file_names = get_difficulty_file_names(f"{root}/info.dat")
        for key, file_name in diff_file_names.items():
            src_path = os.path.join(root, file_name)
            if not os.path.isfile(src_path):
                # print(f"Error: Missing map file: {src_path}. Skipping")
                continue
            dst_path = os.path.join(root, f"{key}.dat")
            os.rename(src_path, dst_path)

        # exp_plus_name = get_difficulty_file_names(f"{root}/info.dat")
        # else:
        #     continue
        # if exp_plus_name != expected_name:
        #     src_path = os.path.join(root, exp_plus_name)
        #     if os.path.isfile(src_path):
        #         dst_path = os.path.join(root, expected_name)
        #         os.rename(src_path, dst_path)
        #     # else: already renamed


def check_info_content(bs_song_path):
    expected_name = "info.dat"
    for folders in os.listdir(bs_song_path):
        info_file = os.path.join(bs_song_path, folders, expected_name)
        if os.path.isfile(info_file):
            # pretty print / overwrite info data to allow line search
            dat_content = read_json_content_file(info_file)
            with open(info_file, "w") as f:
                json.dump(dat_content, f, indent=9)


def clean_songs():
    bs_song_path = paths.bs_input_path
    print("Warning: This script is not fully tested and might break some song files.\n"
          "Do not use on your original beat saber folder!")
    print(f"Cleanup folder: {bs_song_path}")
    input("Continue with Enter")

    check_info_name(bs_song_path)

    check_beatmap_name(bs_song_path)

    check_info_content(bs_song_path)


if __name__ == "__main__":
    clean_songs()
