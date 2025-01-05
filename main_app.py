import time

import gradio as gr
from tkinter import Tk, filedialog
import os
import shutil

import queue
import threading

from app_helper.check_input import get_summary
from app_helper.set_app_paths import set_app_paths
# from main import main
from main_multi import main_multi_par
from tools.config import paths, config

data_folder_name = 'Data'
bs_folder_name = "Beat Saber/Beat Saber_Data/CustomLevels"


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

            bs_folders = bs_folder_name.split('/')
            for i_bs, bs_folder in enumerate(bs_folders):
                # check for root folder
                if i_bs == 0 and bs_folder not in filename:
                    print("Error: Could not find root BS folder.")
                    return str(filename), 'BS root folder not found'
                # search custom level folder
                if i_bs > 0 and bs_folder not in filename:
                    # filename = os.path.join(filename, bs_folder)
                    filename += f"/{bs_folder}"

            if not filename.endswith('/'):
                filename += '/'
            if not os.path.isdir(filename):
                print("Error: Input path not valid")
                paths.bs_song_path = ""
                return str(filename), 'Could not find custom maps folder in Beat Saber'

            root.destroy()
            paths.bs_song_path = filename
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
def upload_files(input_folder, files):
    music_folder_name = paths.songs_pred
    print(f"Copying to folder: {music_folder_name}")
    if not os.path.exists(music_folder_name):
        return "Error: Folder not found."

    for file in files:
        file_name = os.path.basename(file.name)
        destination = os.path.join(music_folder_name, file_name)
        shutil.copyfile(file, destination)
    return f"{len(files)} file(s) successfully imported to {music_folder_name}"


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
    progress_log = []  # List to store logs
    log_queue = queue.Queue()  # Thread-safe queue for logs

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


# Function to update RAM requirement
def update_ram(workers):
    return f"{workers * 2} GB free RAM, {workers} physical CPU cores"


def update_auto_cut_done(move_flag):
    config.auto_move_song_afterwards = move_flag
    return


def set_version_selector(version_value):
    config.bs_mapping_version = version_value
    return


def set_single_mode(single_mode_value):
    if single_mode:
        config.emphasize_beats_flag = False
        config.single_notes_only_flag = True
    else:
        config.emphasize_beats_flag = True
        config.single_notes_only_flag = False
    return


def set_dot_notes(dot_notes_value):
    config.allow_dot_notes = dot_notes_value
    return


def set_add_obstacles(add_obstacles_value):
    config.add_obstacle_flag = add_obstacles_value
    return


def set_add_obstacles_sporty(add_sporty_obstacles_value):
    config.sporty_obstacles = add_sporty_obstacles_value
    return


def set_js_offset(js_offset_value):
    config.jump_speed_offset += js_offset_value
    config.jump_speed_offset_orig += js_offset_value
    return


def set_intensity(intensity_value):
    config.add_beat_intensity = intensity_value
    config.add_beat_intensity_orig = intensity_value
    return


def set_silence_threshold(silence_threshold_value):
    config.silence_threshold *= (silence_threshold_value / 100)
    config.silence_threshold_orig *= (silence_threshold_value / 100)
    return


# def set_quick_start(quick_start_value):
#     config.quick_start = quick_start_value
#     return


def set_random_behaviour(random_behaviour_value):
    config.random_note_map_factor = random_behaviour_value
    return


def set_add_arcs(add_arcs_value):
    config.add_slider_flag = add_arcs_value
    return


def set_arc_start_time(arc_start_time_value):
    config.slider_time_gap[0] = arc_start_time_value
    return


def set_arc_end_time(arc_end_time_value):
    config.slider_time_gap[1] = arc_end_time_value
    return


def set_arc_probability(arc_probability_value):
    config.slider_probability = (arc_probability_value / 100)
    return


def set_arc_movement_min(arc_movement_min_value):
    config.slider_movement_minimum = arc_movement_min_value
    return


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
                gr.Markdown(f"""Version: {config.InfernoSaber_version}  
                This app is under development and not extensively tested on different systems.  
                If you encounter problems, please check the Discord channel. It is free to use and open source.  
                [GitHub Repo](https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper/tree/main_app): View the code  
                [Discord Channel](https://discord.com/invite/cdV6HhpufY): Questions, suggestions, and improvements are welcome""")

        # Two-column layout
        with gr.Row():
            # Left Column
            with gr.Column():
                # Folder Selection
                gr.Markdown("**Folder Selection**")
                gr.Markdown("Please create a new folder (once) for the InfernoSaber input to "
                            "make sure you do not lose any music/data.")
                input_path = gr.Textbox(label='InfernoSaber Input Folder', interactive=False)
                image_browse_btn = gr.Button('Browse', min_width=1)
                path_status = gr.Textbox(label='Folder Status', value='not set', interactive=False)
                image_browse_btn.click(on_browse_input_path, inputs=[], outputs=[input_path, path_status])

                # Optional BS Link
                gr.Markdown("**Optional: BeatSaber Link**")
                gr.Markdown("Select your BeatSaber folder to automatically export generated songs.")
                bs_path = gr.Textbox(label='BeatSaber Folder', interactive=False)
                image_browse_btn_2 = gr.Button('Browse', min_width=1)
                bs_path_status = gr.Textbox(label='Folder Status', value='not set', interactive=False)
                image_browse_btn_2.click(on_browse_bs_path, inputs=[], outputs=[bs_path, bs_path_status])

            # Right Column
            with gr.Column():
                # Music Import
                gr.Markdown("**Music Import**")
                music_loader = gr.File(
                    label='Select Music Files',
                    file_types=['.mp3', '.wav', '.ogg', '.egg'],
                    file_count='multiple'
                )
                file_status = gr.Textbox(label='File Import Status', value='not set', interactive=False)
                upload_button = gr.Button('Copy Files to Input Folder')
                upload_button.click(upload_files, inputs=[input_path, music_loader], outputs=[file_status])

    ################################
    # TAB 2: Parameters
    ################################

    # Second Tab for "Specify Parameters"
    with gr.Tab("Specify Parameters"):
        gr.Markdown("## Specify Parameters")
        with gr.Row():
            with gr.Column():
                # Dropdown Menu for Model Selection
                gr.Markdown("### Model Selection")
                model_selector = gr.Dropdown(
                    choices=["fav_15", "pp3_15", "easy_15", "expert_15"],
                    label="Select Model", value="fav_15", interactive=True,
                )
                gr.Markdown("Info: Selected model will be download from "
                            "[HuggingFace Repo](https://huggingface.co/BierHerr/InfernoSaber) at runtime if required")

            with gr.Column():
                # Dropdown Menu for Version Selection
                gr.Markdown("Select Beat Saber mapping verison: v3-default, v2-without arcs")
                version_selector = gr.Dropdown(
                    choices=["v3", "v2"],
                    label="Select BS Mapping Output Version", value="v3", interactive=True,
                )
                version_selector.input(set_version_selector, inputs=[version_selector], outputs=[])

        # Difficulty Selection
        gr.Markdown("### Difficulty Selection")
        gr.Markdown("""Specify the difficulties you want to use for the song generation.  
            You can input **up to 5** difficulties. Set any unused difficulties to **0** to reduce computation time.  
            Each difficulty value must be between 0.01 and 1000 (or 0 to deactivate)."""
                    )
        # Inputs for difficulties
        with gr.Row():
            difficulty_1 = gr.Number(
                label="Difficulty 1",
                value=3, precision=2, interactive=True,
                step=0.5, minimum=0, maximum=1000,
            )
            difficulty_2 = gr.Number(
                label="Difficulty 2 (optional)",
                value=0, precision=2, interactive=True,
                step=0.5, minimum=0, maximum=1000,
            )
            difficulty_3 = gr.Number(
                label="Difficulty 3 (optional)",
                value=0, precision=2, interactive=True,
                step=0.5, minimum=0, maximum=1000,
            )
            difficulty_4 = gr.Number(
                label="Difficulty 4 (optional)",
                value=0, precision=2, interactive=True,
                step=0.5, minimum=0, maximum=1000,
            )
            difficulty_5 = gr.Number(
                label="Difficulty 5 (optional)",
                value=0, precision=2, interactive=True,
                step=0.5, minimum=0, maximum=1000,
            )

        with gr.Row():
            gr.Markdown("""### Further 
                        ### Modifications""")

            # single mode on/off
            single_mode = gr.Checkbox(label="Single Mode", value=False)
            single_mode.input(set_single_mode, inputs=[single_mode], outputs=[])
            # dot notes on/off
            dot_notes = gr.Checkbox(label="Allow Dot Notes", value=True)
            dot_notes.input(set_dot_notes, inputs=[dot_notes], outputs=[])
            # add obstacles on/off
            add_obstacles = gr.Checkbox(label="Add Obstacles", value=True)
            add_obstacles.input(set_add_obstacles, inputs=[add_obstacles], outputs=[])
            # add sporty obstacles on/off
            add_sporty_obstacles = gr.Checkbox(label="Add Sporty Obstacles", value=False)
            add_sporty_obstacles.input(set_add_obstacles_sporty, inputs=[add_sporty_obstacles], outputs=[])
            # set js offset
            js_offset = gr.Number(label="Jump Speed Offset", value=-0.4, precision=1, interactive=True, step=0.5,
                                  minimum=-5, maximum=5)
            js_offset.input(set_js_offset, inputs=[js_offset], outputs=[])
            # set intensity
            intensity = gr.Number(label="Beat Intensity (%)", value=95, precision=0, interactive=True, step=5,
                                  minimum=70, maximum=125)
            intensity.input(set_intensity, inputs=[intensity], outputs=[])
            # set silence threshold
            silence_threshold = gr.Number(label="Silence Threshold (%)", value=100, precision=2, interactive=True,
                                          step=10, minimum=50, maximum=200)
            silence_threshold.input(set_silence_threshold, inputs=[silence_threshold], outputs=[])
            # # set quick start value
            # quick_start = gr.Number(label="Quick Start", value=0, precision=1, interactive=True, step=0.5, minimum=0, maximum=3)
            # quick_start.input(set_quick_start, inputs=[quick_start], outputs=[])
            # set random behaviour value
            random_behaviour = gr.Number(label="Random Behaviour", value=0.3, precision=1, interactive=True,
                                         step=0.05, minimum=0, maximum=0.6)
            random_behaviour.input(set_random_behaviour, inputs=[random_behaviour], outputs=[])

        with gr.Row():
            # Set arc stats
            gr.Markdown("""### Arc Modifications
                        ### (v3 only)""")

            # add Arcs (v3 only) on/off
            add_arcs = gr.Checkbox(label="Add Arcs (v3 only)", value=True)
            add_arcs.input(set_add_arcs, inputs=[add_arcs], outputs=[])
            # slider_start_time
            arc_start_time = gr.Number(label="Arc Start Time", value=0.5, precision=1, interactive=True, step=0.25,
                                       minimum=0, maximum=2)
            arc_start_time.input(set_arc_start_time, inputs=[arc_start_time], outputs=[])
            # slider_end_time
            arc_end_time = gr.Number(label="Arc End Time", value=12, precision=0, interactive=True, step=1,
                                     minimum=3, maximum=20)
            arc_end_time.input(set_arc_end_time, inputs=[arc_end_time], outputs=[])
            # slider_probability
            arc_probability = gr.Number(label="Arc Probability (%)", value=80, precision=0, interactive=True,
                                        step=10, minimum=10, maximum=100)
            arc_probability.input(set_arc_probability, inputs=[arc_probability], outputs=[])
            # slider_movement_min
            arc_movement_min = gr.Number(label="Arc Movement Min", value=3, precision=1, interactive=True, step=0.5,
                                         minimum=0, maximum=5)
            arc_movement_min.input(set_arc_movement_min, inputs=[arc_movement_min], outputs=[])

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
                    every=5,
                )

                # CPU Workers Selection and RAM Requirement Display
                gr.Markdown("### CPU Workers")
                num_workers = gr.Slider(
                    label="Number of CPU Workers",
                    minimum=1,
                    maximum=16,
                    step=1,
                    value=4,
                    interactive=True,
                )
                ram_required = gr.Textbox(
                    label="Estimated RAM Requirement",
                    value="8 GB RAM, 4 physical CPU cores",  # Default: 4 workers * 5GB = 20GB
                    interactive=False,
                )

                # Update RAM display when number of workers changes
                num_workers.change(update_ram, inputs=[num_workers], outputs=[ram_required])

            with gr.Column():
                # Checkbox to select automatic cut to "done" folder
                gr.Markdown("### Move finished songs")
                auto_cut_done = gr.Checkbox(label="Automatically move songs to 'done' folder", value=False)
                auto_cut_done.input(update_auto_cut_done, inputs=[auto_cut_done], outputs=[])

        # Run Button
        gr.Markdown("### Start Processing")
        run_button = gr.Button("Run")

        # Textbox for Progress/ETA
        progress_eta = gr.Textbox(
            label="Progress/ETA",
            placeholder="Processing progress or ETA will appear here.",
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

# Launch the app
if __name__ == "__main__":
    demo.launch()
