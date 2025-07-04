import shutil
import os
from pydub import AudioSegment, effects

from tools.config import config, paths


def shutil_copy_maps(song_name, index="1234_"):
    if not os.path.isdir(paths.bs_song_path):
        print("Warning: Beatsaber folder not found, automatic export disabled.")
        return False

    src = f'{paths.new_map_path}{index}{song_name}'
    dst = f'{paths.bs_song_path}{index}{song_name}'
    shutil.copytree(src=src, dst=dst, dirs_exist_ok=True)
    return True


def check_music_files(files, dir_path):
    song_list = []
    for file_name in files:
        ending = file_name.split('.')[-1]
        # if ending in ['mp3', 'mp4', 'm4a', 'wav', 'aac', 'flv', 'wma']:   # (TODO: test more music formats)
        if ending in ['mp3', 'mp4', 'm4a']:   # (TODO: test more music formats)
        # if ending in ['NOT_WORKING!']:
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

    # Check the file name for unsupported characters
    for idx, song_name in enumerate(song_list):
        new_name = song_name.replace(' &', ',')
        new_name = new_name.replace('&', ',')
        new_name = new_name.replace('{', '(')
        new_name = new_name.replace('}', ')')
        while new_name[:-4].endswith(' '):
            new_name = f"{new_name[:-5]}.egg"
        if new_name != song_name:
            shutil.move(dir_path + song_name, dir_path + new_name)
            song_list[idx] = new_name

    # Normalize the volume for each song in advance
    if config.normalize_song_flag:
        print("Running volume check for input songs...")
        for song_name in song_list:
            audio = AudioSegment.from_file(dir_path + song_name, format="ogg")
            if config.increase_volume_flag:
                rms = audio.rms/1e9
                # print(f"Audio rms: {rms:.2f} x1e9")
            else:
                rms = 10
            if audio.max_dBFS > 0.0 or rms < config.audio_rms_goal:
                # normalize if volume is below max, else skip
                headroom = -1 * (0.42 + (config.audio_rms_goal - rms) * 16)
                normalized_song = effects.normalize(audio, headroom=headroom)
                error_flag = 0
                while normalized_song.rms / 1e9 < config.audio_rms_goal:
                    error_flag += 1
                    headroom -= 0.5
                    # headroom = -1 * (0.42 + (config.audio_rms_goal - rms) * 16)
                    normalized_song = effects.normalize(audio, headroom=headroom)
                    if error_flag > 10:
                        print(f"Warning: Maximum iterations for normalizing song exceeded: {song_name}. Continue")
                        break

                normalized_song.export(dir_path + song_name, format="ogg")
                print(f"Normalized volume of song: {song_name} with new RMS: {normalized_song.rms/1e9:.2f}")
    return song_list


def convert_music_file(file_name, output_file):
    # Load the mp3 file
    m_format = file_name.split('.')[-1]
    if m_format == 'egg':
        m_format = 'ogg'
    audio = AudioSegment.from_file(file_name, format=m_format)

    output_file_format = output_file.split('.')[-1]

    # Export the audio as ogg file
    if output_file_format == 'ogg' or output_file_format == 'egg':
        audio.export(output_file, format="ogg")
    elif output_file_format == 'mp3':
        audio.export(output_file, format="mp3")
    else:
        print(f"Error in music converter: Format unknown: {m_format}")
        exit()
