import os
import json

from tools.config import config, paths


def get_mapper_name(name_ar):
    if not isinstance(name_ar, list):
        name_ar = [name_ar]

    folder = paths.copy_path_map
    for name in name_ar:
        name = f"{name}_info.dat"
        if name not in os.listdir(folder):
            print(f"Error: Could not find file: {name}")
            exit()

        with open(folder + name, 'r') as f:
            info_dict = json.load(f)
        level_author = info_dict['_levelAuthorName']

        return level_author


def get_maps_from_mapper(mapper_name_list, ignore_capital=True, full_match=False):
    music_list = []
    folder = paths.copy_path_map
    for file in os.listdir(folder):
        if file.endswith("_info.dat"):
            song_name = file[:-9]
            level_author = get_mapper_name(song_name)
            if isinstance(mapper_name_list, str):
                mapper_name_list = [mapper_name_list]

            for mapper_name in mapper_name_list:
                if ignore_capital:
                    if full_match:
                        if level_author.lower() == mapper_name.lower():
                            music_list.append(song_name)
                    else:
                        words1 = set(level_author.lower().split())
                        words2 = set(mapper_name.lower().split())
                        if words2.issubset(words1):
                            music_list.append(song_name)
                else:
                    if level_author == mapper_name:
                        music_list.append(song_name)
    print(f"Found {len(music_list)} songs from mapper {mapper_name_list}.")
    return music_list


if __name__ == '__main__':
    mapper_name_list = []
    folder = paths.copy_path_map
    for file in os.listdir(folder):
        if file.endswith("_info.dat"):
            song_name = file[:-9]
            level_author = get_mapper_name(song_name)
            # print(level_author)
            mapper_name_list.append(level_author)
    from collections import Counter

    string_counts = Counter(mapper_name_list)
    sorted_strings = sorted(string_counts.items(), key=lambda x: x[1])
    for string, count in sorted_strings:
        print(f"{string}: {count} times")
    # mapper_name = get_mapper_name('#ThatPOWER')
    # print(mapper_name)

    mapper_name = 'Skyler'
    map_list = get_maps_from_mapper(mapper_name)
    print(map_list)
