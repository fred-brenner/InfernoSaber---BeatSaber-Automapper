import time
import requests  # to check for updates opn GitHub

import gradio as gr
from tkinter import Tk, filedialog
import os
import shutil

import queue
import threading
import subprocess

from app_helper.check_input import get_summary
from app_helper.set_app_paths import set_app_paths
from app_helper.update_dir_path import update_dir_path
# from main import main
from main_multi import main_multi_par
from tools.config import paths, config
from tools.config.mapper_selection import update_model_file_paths
from tools.utils.huggingface import model_download

data_folder_name = 'Data'
bs_folder_name = "Beat Saber_Data/CustomLevels"
bs_folder_name2 = "SharedMaps/CustomLevels"

update_check_response = None


def check_for_updates():
    try:
        global update_check_response
        if update_check_response is None:
            repo_owner = "fred-brenner"
            # repo_name = "InfernoSaber---BeatSaber-Automapper"
            repo_name = "InfernoSaber-App"
            github_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"

            response = requests.get(github_url)
            latest_version = response.json()[0]['tag_name']
            current_version = f"v{config.InfernoSaber_version}"
            if latest_version != current_version:
                update_check_response = (
                    f"Version <{latest_version}> is released! Your version is <{config.InfernoSaber_version}>. "
                    f"Please update with the Pinokio Update function on the left.")
                return (f"Version <{latest_version}> is released! Your version is <{config.InfernoSaber_version}>. "
                        f"Please update with the Pinokio Update function on the left.")
            update_check_response = f"You are up to date ({latest_version})"
            return f"You are up to date ({latest_version})"

        else:
            return update_check_response
    except:
        print("Could not reach GitHub for update check.")
        try:
            print(f"Response: {response.json()[0]}")
        except:
            pass
    update_check_response = "Could not reach GitHub for update check."
    return "Could not reach GitHub for update check."


# call update check once to mitigate rate limiting
update_status = check_for_updates()


# update_status = "Currently not available. Please wait for app5 release in Discord."


# # Function to handle folder selection using tkinter
# def on_browse(data_type):
#     root = Tk()
#     root.attributes("-topmost", True)
#     root.withdraw()
#     if data_type == "Files":
#         filenames = filedialog.askopenfilenames()
#         if len(filenames) > 0:
#             root.destroy()
#             return str(filenames)
#         else:
#             filename = "Files not selected"
#             root.destroy()
#             return str(filename)
#
#     elif data_type == "Folder":
#         filename = filedialog.askdirectory()
#         if filename:
#             if os.path.isdir(filename):
#                 if not os.path.basename(filename) == data_folder_name:
#                     filename = os.path.join(filename, data_folder_name)
#                     if not os.path.isdir(filename):
#                         os.mkdir(filename)
#                 root.destroy()
#                 return str(filename)
#             else:
#                 filename = "Folder not available"
#                 root.destroy()
#                 return str(filename)
#         else:
#             filename = "Folder not selected"
#             root.destroy()
#             return str(filename)


def on_browse_input_path():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    filename = filedialog.askdirectory()
    filename = filename.replace('\\\\', '/').replace('\\', '/')
    if filename:
        if os.path.isdir(filename):
            if len(os.listdir(filename)) > 5:
                return str(filename), 'Error: Please select an empty folder to set up InfernoSaber'
            if not os.path.basename(filename) == data_folder_name:
                # filename = os.path.join(filename, data_folder_name)
                filename += f"/{data_folder_name}"
                if not os.path.isdir(filename):
                    os.mkdir(filename)
            if not filename.endswith('/'):
                filename += '/'
            root.destroy()
            set_app_paths(filename)
            return str(filename), 'Finished folder setup'
        else:
            filename = "Folder not available"
            root.destroy()
            return str(filename), 'not set'
    else:
        filename = "Folder not selected"
        root.destroy()
        return str(filename), 'not set'


def on_browse_bs_path():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    filename = filedialog.askdirectory()
    filename = filename.replace('\\\\', '/').replace('\\', '/')
    if filename:
        if os.path.isdir(filename):
            # copy in path anyway as soon as it is valid
            paths.bs_song_path = filename

            for bs_folders in [bs_folder_name, bs_folder_name2]:
                found = False
                bs_folders = bs_folders.split('/')
                for i_bs, bs_folder in enumerate(bs_folders):
                    # check for root folder
                    if i_bs == 0 and bs_folder not in filename:
                        break
                    # search custom level folder
                    if i_bs > 0 and bs_folder not in filename:
                        # filename = os.path.join(filename, bs_folder)
                        filename += f"/{bs_folder}"
                    found = True
                if found:
                    break
            if not found:
                paths.bs_song_path = ""
                print("Error: Could not find root BS folder.")
                return str(filename), 'BS root folder not found'

            if not filename.endswith('/'):
                filename += '/'
            if not os.path.isdir(filename):
                print("Error: Input path not valid")
                paths.bs_song_path = ""
                return str(filename), 'Could not find custom maps folder in Beat Saber'

            root.destroy()
            paths.bs_song_path = filename
            update_dir_path('tools/config/paths.py', 'bs_song_path', filename)
            print(f"Set BS export path to: {filename}")
            return str(filename), 'Found BS folder'
        else:
            filename = "Folder not available"
            paths.bs_song_path = ""
            root.destroy()
            return str(filename), 'Folder does not exist'
    else:
        filename = "Folder not selected"
        paths.bs_song_path = ""
        root.destroy()
        return str(filename), 'not set'


# Function to set the folder after selection
def set_input_folder(folder_path):
    if folder_path == "No folder selected." or not folder_path:
        return "Please select a valid folder."
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return f"Input folder set to: {folder_path}"


# Function to handle file upload
def upload_files(files):
    if not files or len(files) == 0:
        return "No files selected."
    music_folder_name = paths.songs_pred
    print(f"Copying to folder: {music_folder_name}")
    if not os.path.exists(music_folder_name):
        return "Error: Folder not found."

    for file in files:
        file_name = os.path.basename(file.name)
        destination = os.path.join(music_folder_name, file_name)
        shutil.copyfile(file, destination)
    return f"{len(files)} file(s) successfully imported to {music_folder_name}"


def open_folder_music(folder_path):
    if os.path.isdir(folder_path):
        music_folder_name = paths.songs_pred
        if os.path.isdir(music_folder_name):
            os.startfile(music_folder_name)
        else:
            print(f"Error: Could not find folder: {music_folder_name}")
    else:
        print("Not set up yet")
    return


def open_folder_maps(folder_path):
    if os.path.isdir(folder_path):
        maps_folder_name = paths.new_map_path
        if os.path.isdir(maps_folder_name):
            os.startfile(maps_folder_name)
        else:
            print(f"Error: Could not find folder: {maps_folder_name}")
    else:
        print("Not set up yet")
    return


# Run main.py with live output
def run_process(num_workers, use_model, diff1, diff2, diff3, diff4, diff5):
    # Check if all inputs are valid
    summary_log = get_summary(diff1, diff2, diff3, diff4, diff5)
    if "Error: " in summary_log:
        yield "Error: Inputs not set up yet. See Summary on top."
        time.sleep(1)
        return

    # Read the difficulty settings
    diff = []
    if diff1 > 0:
        diff.append(diff1)
    if diff2 > 0:
        diff.append(diff2)
    if diff3 > 0:
        diff.append(diff3)
    if diff4 > 0:
        diff.append(diff4)
    if diff5 > 0:
        diff.append(diff5)
    if len(diff) == 0:
        yield "Error: Inputs not set up yet. See Summary on top."
        time.sleep(1)
        return

    config.use_mapper_selection = use_model
    update_dir_path('tools/config/config.py', 'use_mapper_selection', use_model)
    progress_log = []  # List to store logs
    log_queue = queue.Queue()  # Thread-safe queue for logs

    # Download InfernoSaber AI Model
    update_model_file_paths(check_model_exists=False)
    # Download AI Model from huggingface
    progress_log.append(f"Loading AI Model: {use_model}")
    yield "\n".join(progress_log)
    try:
        update_model_file_paths()
    except:
        progress_log.append("Downloading Model from HuggingFace (~1GB, see second tab)")
        yield "\n".join(progress_log)
    model_download(use_model)
    progress_log.append("Model Found")
    yield "\n".join(progress_log)

    def logger_callback(message):
        log_queue.put(message)

    # Read logs to determine BS export status
    export_results_to_bs = False
    if "Info: Beat Saber folder found." in summary_log:
        export_results_to_bs = True
        print("Activated automatic export to Beat Saber")
        progress_log.append("Activated automatic export to Beat Saber")
        yield "\n".join(progress_log)

    # Run main() in a separate thread and pass the logger callback
    thread = threading.Thread(
        target=main_multi_par,
        kwargs={
            "n_workers": num_workers,
            "diff_list": diff,
            "logger_callback": logger_callback,
            "export_results_to_bs": export_results_to_bs,
        }
    )
    thread.start()

    # Collect logs in real-time
    while thread.is_alive() or not log_queue.empty():
        while not log_queue.empty():
            log_message = log_queue.get()
            progress_log.append(log_message)
            yield "\n".join(progress_log)
        time.sleep(0.2)  # Prevent busy-waiting

    # Final message after completion
    progress_log.append("Process finished!")
    yield "\n".join(progress_log)
    return


def update_num_workers(workers):
    config.num_workers = workers
    update_dir_path('tools/config/config.py', 'num_workers', workers)
    return


# Function to update RAM requirement
def update_ram(workers):
    return f"{workers * 2} GB free RAM, {workers} physical CPU cores"


def update_auto_cut_done(move_flag):
    config.auto_move_song_afterwards = move_flag
    update_dir_path('tools/config/config.py', 'auto_move_song_afterwards', move_flag)
    return


def set_version_selector(version_value):
    config.bs_mapping_version = version_value
    update_dir_path('tools/config/config.py', 'bs_mapping_version', version_value)
    return


def set_difficulty_1(value):
    config.difficulty_1 = value
    update_dir_path('tools/config/config.py', 'difficulty_1', value)


def set_difficulty_2(value):
    config.difficulty_2 = value
    update_dir_path('tools/config/config.py', 'difficulty_2', value)


def set_difficulty_3(value):
    config.difficulty_3 = value
    update_dir_path('tools/config/config.py', 'difficulty_3', value)


def set_difficulty_4(value):
    config.difficulty_4 = value
    update_dir_path('tools/config/config.py', 'difficulty_4', value)


def set_difficulty_5(value):
    config.difficulty_5 = value
    update_dir_path('tools/config/config.py', 'difficulty_5', value)


def set_single_mode(single_mode_value):
    if single_mode_value:
        config.emphasize_beats_flag = False
        config.single_notes_only_flag = True
    else:
        config.emphasize_beats_flag = True
        config.single_notes_only_flag = False
    update_dir_path('tools/config/config.py', 'emphasize_beats_flag', config.emphasize_beats_flag)
    update_dir_path('tools/config/config.py', 'single_notes_only_flag', config.single_notes_only_flag)
    return config.emphasize_beats_flag


def set_more_notes_prob(more_notes_prob_value):
    config.gimme_more_notes_prob = more_notes_prob_value / 100
    update_dir_path('tools/config/config.py', 'gimme_more_notes_prob', config.gimme_more_notes_prob)
    return


def set_emphasize_mode(emphasize_mode_value):
    if emphasize_mode_value:
        config.emphasize_beats_flag = True
    else:
        config.emphasize_beats_flag = False
    update_dir_path('tools/config/config.py', 'emphasize_beats_flag', config.emphasize_beats_flag)
    return


def set_dot_notes(dot_notes_value):
    config.allow_dot_notes = dot_notes_value
    update_dir_path('tools/config/config.py', 'allow_dot_notes', dot_notes_value)
    return


def set_add_obstacles(add_obstacles_value):
    config.add_obstacle_flag = add_obstacles_value
    update_dir_path('tools/config/config.py', 'add_obstacle_flag', add_obstacles_value)
    return


def set_add_obstacles_sporty(add_sporty_obstacles_value):
    config.sporty_obstacles = add_sporty_obstacles_value
    update_dir_path('tools/config/config.py', 'sporty_obstacles', add_sporty_obstacles_value)
    return


def set_js_offset(js_offset_value):
    config.jump_speed_offset = js_offset_value
    update_dir_path('tools/config/config.py', 'jump_speed_offset', config.jump_speed_offset)
    config.jump_speed_offset_orig = js_offset_value
    # update_dir_path('tools/config/config.py', 'jump_speed_offset_orig', config.jump_speed_offset_orig)
    return


def set_intensity(intensity_value):
    config.add_beat_intensity = intensity_value
    update_dir_path('tools/config/config.py', 'add_beat_intensity', intensity_value)
    config.add_beat_intensity_orig = intensity_value
    update_dir_path('tools/config/config.py', 'add_beat_intensity_orig', intensity_value)
    return


def set_silence_threshold(silence_threshold_value):
    orig_value = 0.2
    config.silence_threshold = orig_value * (silence_threshold_value / 100)
    update_dir_path('tools/config/config.py', 'silence_threshold', config.silence_threshold)
    config.silence_threshold_orig = orig_value * (silence_threshold_value / 100)
    update_dir_path('tools/config/config.py', 'silence_threshold_orig', config.silence_threshold_orig)
    update_dir_path('tools/config/config.py', 'silence_threshold_percentage', silence_threshold_value)
    return


# def set_quick_start(quick_start_value):
#     config.quick_start = quick_start_value
#     update_dir_path('tools/config/config.py', 'quick_start', quick_start_value)
#     return


def set_random_behaviour(random_behaviour_value):
    config.random_note_map_factor = random_behaviour_value
    update_dir_path('tools/config/config.py', 'random_note_map_factor', random_behaviour_value)
    return


def set_add_arcs(add_arcs_value):
    config.add_slider_flag = add_arcs_value
    update_dir_path('tools/config/config.py', 'add_slider_flag', add_arcs_value)
    return


def set_arc_time(arc_start_time_value, arc_end_time_value):
    config.slider_time_gap[0] = arc_start_time_value
    config.slider_time_gap[1] = arc_end_time_value
    update_dir_path('tools/config/config.py', 'slider_time_gap',
                    f'[{arc_start_time_value}, {arc_end_time_value}]')
    return


def set_arc_probability(arc_probability_value):
    config.slider_probability = (arc_probability_value / 100)
    update_dir_path('tools/config/config.py', 'slider_probability', config.slider_probability)
    return


def set_arc_movement_min(arc_movement_min_value):
    config.slider_movement_minimum = arc_movement_min_value
    update_dir_path('tools/config/config.py', 'slider_movement_minimum', arc_movement_min_value)
    return


def set_bpm_overwrite(bpm_overwrite_flag):
    if bpm_overwrite_flag:
        config.use_fixed_bpm = 100
    else:
        config.use_fixed_bpm = 0
    update_dir_path('tools/config/config.py', 'use_fixed_bpm', config.use_fixed_bpm)
    return


def song_counting():
    music_folder_name = paths.songs_pred
    # print(f"Copying to folder: {music_folder_name}")
    if not os.path.exists(music_folder_name):
        return "Folder not set up yet."

    if os.path.isdir(music_folder_name):
        files = os.listdir(music_folder_name)
        song_counter = 0
        for f in files:
            if any(f.endswith(ext) for ext in ['.egg', '.ogg', '.mp3', '.m4a', '.mp4']):
                song_counter += 1
        return f"{song_counter} song(s) in folder"
    else:
        return "Folder not set up yet."


def import_urls(urls):
    if not urls.strip():
        return "No URLs found."
    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    print(f"{len(url_list)} URLs found: " + ", ".join(url_list))

    music_folder_name = paths.songs_pred
    if not os.path.exists(music_folder_name):
        return "Error: Input folder not set up."

    results = []
    for url in url_list:
        try:
            cmd = [
                "yt-dlp",
                "-x",
                "--audio-format", "mp3",
                "-o", os.path.join(music_folder_name, "%(title)s.%(ext)s"),
                url
            ]
            subprocess.run(cmd, check=True)
            results.append(f"Downloaded: {url}")
        except Exception as e:
            results.append(f"Error for {url}: {e}")

    return "\n".join(results)


# Gradio App Setup Section
with gr.Blocks() as demo:
    ################################
    # TAB 1: Setup
    ################################
    # Setup Tab
    with gr.Tab("Setup"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Setup")
                gr.Markdown(f"""Version: {config.InfernoSaber_version} | InfernoSaber is free and OpenSource.
                If you encounter problems, please check the Discord channel.
                [GitHub Repo](https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper/tree/main_app): View the code
                [Discord Channel](https://discord.com/invite/cdV6HhpufY): Questions, suggestions, and improvements are welcome

                Info: All your changes will be applied immediately and saved for the next runs. Only updates will reset the settings.
                """)
            with gr.Column():
                gr.Markdown("### Check for updates")
                version_info = gr.Textbox(label="Version Info", interactive=False,
                                          value=update_status)

        # Two-column layout
        with gr.Row():
            # Left Column
            with gr.Column():
                # Folder Selection
                gr.Markdown("**Folder Selection**")
                gr.Markdown("Please create a new folder (once) for the InfernoSaber input to "
                            "make sure you do not lose any music/data.")
                input_path = gr.Textbox(label='InfernoSaber Input Folder', interactive=False, value=paths.dir_path)
                image_browse_btn = gr.Button('Browse', min_width=1)
                path_status = gr.Textbox(label='Folder Status', value='not set', interactive=False)
                image_browse_btn.click(on_browse_input_path, inputs=[], outputs=[input_path, path_status])

                # Optional BS Link
                gr.Markdown("**Optional: BeatSaber Link**")
                gr.Markdown("Select your BeatSaber folder to automatically export generated songs.")
                bs_path = gr.Textbox(label='BeatSaber Folder', interactive=False, value=paths.bs_song_path)
                image_browse_btn_2 = gr.Button('Browse', min_width=1)
                bs_path_status = gr.Textbox(label='Folder Status', placeholder='(optional)', interactive=False)
                image_browse_btn_2.click(on_browse_bs_path, inputs=[], outputs=[bs_path, bs_path_status])

            # Right Column
            with gr.Column():
                # Music Import
                gr.Markdown("**Music Import: Currently only ogg/egg/mp3/m4a files supported!**")
                # TODO: fix/test for wma, wav, etc.
                music_loader = gr.File(
                    label='Select Music Files',
                    file_types=['.ogg', '.egg', '.mp3', '.m4a', '.mp4'],
                    file_count='multiple'
                )
                file_status = gr.Textbox(label='File Import Status', placeholder='(optional)', interactive=False)
                upload_button = gr.Button('Copy Files to Input Folder')
                upload_button.click(upload_files, inputs=[music_loader], outputs=[file_status])

                # Add button to open the folder
                gr.Markdown("... Or copy your songs to this folder:")
                open_folder_button = gr.Button('Open Input Folder')
                open_folder_button.click(open_folder_music, inputs=[input_path], outputs=[])

                # Show status text box of how many songs are in th folder
                song_count = gr.Textbox(label='Song Count', interactive=False,
                                        value=song_counting, every=3)
                # upload_button.click(song_counting, inputs=[], outputs=[song_count])

                # YT download
                gr.Markdown("**Music download (YT_DLP)**")
                gr.Markdown("Add one or more YouTube URLs to download music files. Using [yt-dlp](https://github.com/yt-dlp/yt-dlp), use at your own risk!")
                url_input = gr.Textbox(label='URL(s)', lines=3, placeholder='https://...')
                url_status = gr.Textbox(label='URL Import Status', interactive=False)
                url_import_btn = gr.Button('Start import')
                url_import_btn.click(
                    fn=import_urls,
                    inputs=[url_input],
                    outputs=[url_status]
                )

    ################################
    # TAB 2: Parameters
    ################################

    # Second Tab for "Specify Parameters"
    with gr.Tab("Specify Parameters"):
        gr.Markdown("## Specify Parameters")
        gr.Markdown("All changes are automatically applied and saved for the next runs. There is no save button.")
        with gr.Row():
            with gr.Column():
                # Dropdown Menu for Model Selection
                gr.Markdown("### Model Selection")
                model_selector = gr.Dropdown(
                    choices=["fav_15", "pp3_15", "easy_15", "expert_15"],
                    label="Select Model", value=config.use_mapper_selection, interactive=True,
                    info="Select the model to be used for song generation. "
                         "The model will be downloaded from [HuggingFace](https://huggingface.co/BierHerr/InfernoSaber)"
                         " if required at runtime."
                )

            with gr.Column():
                # Dropdown Menu for Version Selection
                gr.Markdown("Select Beat Saber mapping verison: v3-default, v2-without arcs")
                version_selector = gr.Dropdown(
                    choices=["v3", "v2"],
                    label="Select BS Mapping Output Version", value=config.bs_mapping_version, interactive=True,
                    info="Both currently produce the same notes. v3 includes arcs, v2 does not."
                )
                version_selector.input(set_version_selector, inputs=[version_selector], outputs=[])

        # Difficulty Selection
        gr.Markdown("### Difficulty Selection")
        gr.Markdown("""Specify the difficulties you want to use for the song generation.
            You can input **up to 5** difficulties. Set any unused difficulties to **0** to reduce computation time.\n
            Each difficulty value must be between 0.01 and 1000 (or 0 to deactivate).
            The value is leaned on the notes-per-second / star rating (~1 Easy to ~6 Expert+)."""
                    )
        # Inputs for difficulties
        with gr.Row():
            difficulty_1 = gr.Number(
                label="Difficulty 1",
                value=config.difficulty_1, precision=2, interactive=True,
                step=0.25, minimum=0, maximum=1000,
                info="First difficulty setting. Must be set above 0."
            )
            difficulty_1.input(set_difficulty_1, inputs=[difficulty_1], outputs=[])

            difficulty_2 = gr.Number(
                label="Difficulty 2 (optional)",
                value=config.difficulty_2, precision=2, interactive=True,
                step=0.25, minimum=0, maximum=1000,
            )
            difficulty_2.input(set_difficulty_2, inputs=[difficulty_2], outputs=[])

            difficulty_3 = gr.Number(
                label="Difficulty 3 (optional)",
                value=config.difficulty_3, precision=2, interactive=True,
                step=0.25, minimum=0, maximum=1000,
            )
            difficulty_3.input(set_difficulty_3, inputs=[difficulty_3], outputs=[])

            difficulty_4 = gr.Number(
                label="Difficulty 4 (optional)",
                value=config.difficulty_4, precision=2, interactive=True,
                step=0.25, minimum=0, maximum=1000,
            )
            difficulty_4.input(set_difficulty_4, inputs=[difficulty_4], outputs=[])

            difficulty_5 = gr.Number(
                label="Difficulty 5 (optional)",
                value=config.difficulty_5, precision=2, interactive=True,
                step=0.25, minimum=0, maximum=1000,
            )
            difficulty_5.input(set_difficulty_5, inputs=[difficulty_5], outputs=[])

        with gr.Row():
            gr.Markdown("""### Further
                        ### Modifications""")
            # set gimme_more_notes_prob
            more_notes_prob = gr.Number(label="Left/Right Notes Filling (%)", value=config.gimme_more_notes_prob * 100,
                                        precision=0, interactive=True, step=5, minimum=0, maximum=100,
                                        info="Increase to always have notes on both sides.")
            more_notes_prob.input(set_more_notes_prob, inputs=[more_notes_prob], outputs=[])
            # emphasize beats on/off
            emphasize_beats = gr.Checkbox(label="Emphasize Beats", value=config.emphasize_beats_flag,
                                          info="Emphasize beats with double/triple notes.")
            emphasize_beats.input(set_emphasize_mode, inputs=[emphasize_beats], outputs=[])
            # single mode on/off
            single_mode = gr.Checkbox(label="Single Mode", value=config.single_notes_only_flag,
                                      info="Generate maximum one note per beat.")
            single_mode.input(set_single_mode, inputs=[single_mode], outputs=[emphasize_beats])
            # dot notes on/off
            dot_notes = gr.Checkbox(label="Allow Dot Notes", value=config.allow_dot_notes,
                                    info="Allow (rare) notes without arrow direction.")
            dot_notes.input(set_dot_notes, inputs=[dot_notes], outputs=[])
            # add obstacles on/off
            add_obstacles = gr.Checkbox(label="Add Obstacles", value=config.add_obstacle_flag,
                                        info="Add obstacles on the left/right with little movement.")
            add_obstacles.input(set_add_obstacles, inputs=[add_obstacles], outputs=[])
        with gr.Row():
            # add sporty obstacles on/off
            add_sporty_obstacles = gr.Checkbox(label="Add Sporty Obstacles", value=config.sporty_obstacles,
                                               info="Add obstacles with movement and squats.")
            add_sporty_obstacles.input(set_add_obstacles_sporty, inputs=[add_sporty_obstacles], outputs=[])
            # set js offset
            js_offset = gr.Number(label="Jump Speed Offset", value=config.jump_speed_offset, precision=1,
                                  interactive=True, step=0.5, minimum=-5, maximum=5,
                                  info="In-/decrease the (jump) speed of the notes.")
            js_offset.input(set_js_offset, inputs=[js_offset], outputs=[])
            # set intensity
            intensity = gr.Number(label="Beat Intensity (%)", value=config.add_beat_intensity, precision=0,
                                  interactive=True, step=5, minimum=70, maximum=125,
                                  info="Acts as multiplier to the difficulty target. "
                                       "High values will enforce more notes in calm sections.")
            intensity.input(set_intensity, inputs=[intensity], outputs=[])
            # set silence threshold
            silence_threshold = gr.Number(label="Silence Threshold (%)", value=config.silence_threshold_percentage,
                                          precision=2, interactive=True, step=10, minimum=50, maximum=400,
                                          info="In-/decrease the silence threshold. "
                                               "High values will enforce less notes in calm sections.")
            silence_threshold.input(set_silence_threshold, inputs=[silence_threshold], outputs=[])
            # # set quick start value
            # quick_start = gr.Number(label="Quick Start", value=config.quick_start, precision=1,
            #                         interactive=True, step=0.5, minimum=0, maximum=3)
            # quick_start.input(set_quick_start, inputs=[quick_start], outputs=[])
            # set random behaviour value
            random_behaviour = gr.Number(label="Random Behaviour", value=config.random_note_map_factor, precision=1,
                                         interactive=True, step=0.05, minimum=0, maximum=0.6,
                                         info="Increase to produce different outcomes on each iteration. "
                                              "0 deactivates the feature.")
            random_behaviour.input(set_random_behaviour, inputs=[random_behaviour], outputs=[])

        with gr.Row():
            # Set arc stats
            gr.Markdown("""### Arc Modifications
                        ### (v3 only)""")

            # add Arcs (v3 only) on/off
            add_arcs = gr.Checkbox(label="Add Arcs (v3 only)", value=config.add_slider_flag,
                                   info="Enable or disable arcs between notes.")
            add_arcs.input(set_add_arcs, inputs=[add_arcs], outputs=[])
            # slider_start_time
            arc_start_time = gr.Number(label="Arc Start Time", value=config.slider_time_gap[0], precision=1,
                                       interactive=True, step=0.25, minimum=0, maximum=2,
                                       info="Minimum time in seconds to activate arcs.")
            # arc_start_time.input(set_arc_time, inputs=[arc_start_time, arc_end_time], outputs=[])
            # slider_end_time
            arc_end_time = gr.Number(label="Arc End Time", value=config.slider_time_gap[1], precision=0,
                                     interactive=True, step=1, minimum=3, maximum=20,
                                     info="Maximum time in seconds to activate arcs.")
            arc_start_time.input(set_arc_time, inputs=[arc_start_time, arc_end_time], outputs=[])
            arc_end_time.input(set_arc_time, inputs=[arc_start_time, arc_end_time], outputs=[])
            # slider_probability
            arc_probability = gr.Number(label="Arc Probability (%)", value=config.slider_probability * 100, precision=0,
                                        interactive=True, step=10, minimum=10, maximum=100,
                                        info="Probability to add arcs, if other conditions are met.")
            arc_probability.input(set_arc_probability, inputs=[arc_probability], outputs=[])
            # slider_movement_min
            arc_movement_min = gr.Number(label="Arc Movement Min", value=config.slider_movement_minimum, precision=1,
                                         interactive=True, step=0.5, minimum=0, maximum=5,
                                         info="Note distance required to allow arcs.")
            arc_movement_min.input(set_arc_movement_min, inputs=[arc_movement_min], outputs=[])
            # add bpm overwrite
            bpm_overwrite = gr.Checkbox(label="BPM Overwrite", value=config.use_fixed_bpm > 0,
                                        info="Default False (uses real song bpm instead of fixed value). "
                                             "Activate only in case of note jump speed bugs in BS at high difficulties.")
            bpm_overwrite.input(set_bpm_overwrite, inputs=[bpm_overwrite], outputs=[])

    ################################
    # TAB 3: Runtime
    ################################

    # Third Tab for "Run"
    with gr.Tab("Run"):
        gr.Markdown("## Run")

        with gr.Row():
            with gr.Column():
                # Empty Textbox
                gr.Markdown("### Summary")
                log_output = gr.Textbox(
                    label="Information",
                    lines=3,
                    interactive=False,
                    value=get_summary,
                    inputs=[difficulty_1, difficulty_2, difficulty_3, difficulty_4, difficulty_5],
                    every=4,
                )

                # CPU Workers Selection and RAM Requirement Display
                gr.Markdown("### CPU Workers")
                num_workers = gr.Slider(
                    label="Number of CPU Workers",
                    minimum=1,
                    maximum=16,
                    step=1,
                    value=config.num_workers,
                    interactive=True,
                    info="Memory estimation depends on the model. Start low and check your Task Manager."
                )
                ram_required = gr.Textbox(
                    label="Estimated RAM Requirement",
                    value=f"{config.num_workers * 2} GB free RAM, {config.num_workers} physical CPU cores",
                    interactive=False,
                )

                # Update RAM display when number of workers changes
                num_workers.change(update_ram, inputs=[num_workers], outputs=[ram_required])
                num_workers.change(update_num_workers, inputs=[num_workers], outputs=[])

            with gr.Column():
                # Checkbox to select automatic cut to "done" folder
                gr.Markdown("### Move finished songs")
                auto_cut_done = gr.Checkbox(label="Automatically move songs to 'y_done' folder",
                                            value=config.auto_move_song_afterwards,
                                            info="Songs in 'y_done' will not be processed again, "
                                                 "until moved back to the input folder.")
                auto_cut_done.input(update_auto_cut_done, inputs=[auto_cut_done], outputs=[])

        # Run Button
        gr.Markdown("### Start Processing")
        run_button = gr.Button("Run")

        # Textbox for Progress/ETA
        progress_eta = gr.Textbox(
            label="Progress: After run plus max. 1 minute, it should display an ETA. "
                  "Else check the server logs and try another song",
            placeholder="""Processing progress or ETA will appear here.
            After latest 1 minute, it should display an ETA like that:
            Loading AI Model: fav_15
            Model Found
            Starting multi map generator with 10 workers.
            Checking and Normalizing Song Files...
            Found 3 songs. Iterating...
            ### ETA: 0.3 minutes. Time per song: 9 s ###
            ### ETA: 0.2 minutes. Time per song: 10 s ###
            ### ETA: 0.0 minutes. Time per song: 10 s ###
            Process finished!
            """,
            lines=5,
            interactive=False,
        )

        # Link Run button to the process function
        run_button.click(
            run_process,
            inputs=[num_workers, model_selector, difficulty_1, difficulty_2, difficulty_3, difficulty_4,
                    difficulty_5],
            outputs=[progress_eta],
            queue=True,
        )

        # Add button to open the folder
        with gr.Row():
            with gr.Column():
                gr.Markdown("... Maps will be generated here:")
                open_folder_button2 = gr.Button('Open Output Folder')
                open_folder_button2.click(open_folder_maps, inputs=[input_path], outputs=[])
            with gr.Column():
                gr.Markdown("... or view the maps here:")
                gr.Markdown("[ArcViewer](https://allpoland.github.io/ArcViewer/)")

# Launch the app
if __name__ == "__main__":
    demo.launch()
