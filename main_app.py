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
    return f"{workers * 5} GB RAM"


def update_auto_cut_done(flag):
    config.auto_move_song_afterwards = flag
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
            with gr.Column():
                gr.Markdown("Info: Selected model will be download from "
                            "[HuggingFace Repo](https://huggingface.co/BierHerr/InfernoSaber) at runtime if required")

        # Difficulty Selection
        gr.Markdown("### Difficulty Selection")
        gr.Markdown(
            """Specify the difficulties you want to use for the song generation.  
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
                    value=2,
                    interactive=True,
                )
                ram_required = gr.Textbox(
                    label="Estimated RAM Requirement (GB)",
                    value="20 GB",  # Default: 4 workers * 5GB = 20GB
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

        # image_browse_btn.click(get_summary, inputs=[], outputs=[log_output])
        # image_browse_btn_2.click(get_summary, inputs=[], outputs=[log_output])
        # upload_button.click(get_summary, inputs=[], outputs=[log_output])

# Launch the app
if __name__ == "__main__":
    demo.launch()
