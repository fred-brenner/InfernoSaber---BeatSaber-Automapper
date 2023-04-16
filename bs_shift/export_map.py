import shutil
import os
from pydub import AudioSegment, effects

from tools.config import config, paths


def shutil_copy_maps(song_name):
    if not os.path.isdir(paths.bs_song_path):
        print("Warning: Beatsaber folder not found, automatic export disabled.")
        return 0

    src = f'{paths.new_map_path}1234_{song_name}'
    dst = f'{paths.bs_song_path}1234_{song_name}'
    shutil.copytree(src=src, dst=dst, dirs_exist_ok=True)


def check_music_files(files, dir_path):
    song_list = []
    for file_name in files:
        ending = file_name.split('.')[-1]
        if ending in ['mp3', 'mp4', 'm4a', 'wav', 'aac', 'flv', 'wma']:   # (TODO: allow more music formats)
            # convert music to ogg format
            output_file = f"{file_name[:-4]}.egg"
            convert_music_file(dir_path + file_name, dir_path + output_file)
            os.remove(dir_path + file_name)
            song_list.append(output_file)
        elif ending in ['ogg']:
            source = dir_path + file_name
            destination = dir_path + file_name.replace('.ogg', '.egg')
            shutil.move(source, destination)
            song_list.append(file_name.replace('.ogg', '.egg'))
        elif ending in ['egg']:
            song_list.append(file_name)
        else:
            print(f"Warning: Can not read {file_name} as music file.")
            pass

    # Normalize the volume for each song in advance
    if config.normalize_song_flag:
        for song_name in song_list:
            audio = AudioSegment.from_file(dir_path + song_name, format="ogg")
            if audio.max_dBFS > 0.0:
                # normalize if volume is below max, else skip
                normalized_song = effects.normalize(audio, headroom=-1)
                normalized_song.export(dir_path + song_name, format="ogg")
                print(f"Normalized volume of song: {song_name}")
    return song_list


def convert_music_file(file_name, output_file):
    # Load the mp3 file
    format = file_name.split('.')[-1]
    audio = AudioSegment.from_file(file_name, format=format)

    # Export the audio as ogg file
    audio.export(output_file, format="ogg")
