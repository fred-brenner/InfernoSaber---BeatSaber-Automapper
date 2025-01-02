import gradio as gr
from tkinter import Tk, filedialog
import os
import shutil

from app_helper.set_app_paths import set_app_paths
from tools.config import paths

data_folder_name = 'Data'


# Function to handle folder selection using tkinter
def on_browse(data_type):
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    if data_type == "Files":
        filenames = filedialog.askopenfilenames()
        if len(filenames) > 0:
            root.destroy()
            return str(filenames)
        else:
            filename = "Files not selected"
            root.destroy()
            return str(filename)

    elif data_type == "Folder":
        filename = filedialog.askdirectory()
        if filename:
            if os.path.isdir(filename):
                if not os.path.basename(filename) == data_folder_name:
                    filename = os.path.join(filename, data_folder_name)
                    if not os.path.isdir(filename):
                        os.mkdir(filename)
                root.destroy()
                return str(filename)
            else:
                filename = "Folder not available"
                root.destroy()
                return str(filename)
        else:
            filename = "Folder not selected"
            root.destroy()
            return str(filename)


def on_browse_input_path():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    filename = filedialog.askdirectory()
    if filename:
        if os.path.isdir(filename):
            if not os.path.basename(filename) == data_folder_name:
                filename = os.path.join(filename, data_folder_name)
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


# Gradio App Setup Section
with gr.Blocks() as demo:
    # Setup Tab
    with gr.Tab("Setup"):
        gr.Markdown("## Setup")
        gr.Markdown("""
        **Warning:** This app is under development and not extensively tested on different systems. If you encounter any problems, please check the Discord channel. It is free to use and open source.
        [GitHub Repo](https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper/tree/main_app): View the code
        [Discord Channel](https://discord.com/invite/cdV6HhpufY): Questions, suggestions, and improvements are welcome
        """)

        # Folder Selection
        gr.Markdown("**Folder Selection**")
        gr.Markdown("Please create a new folder (once) for the InfernoSaber input to make sure "
                    "you do not lose any music/data.")
        input_path = gr.Textbox(label='Select InfernoSaber input folder', scale=5, interactive=False)
        image_browse_btn = gr.Button('Browse', min_width=1)
        path_status = gr.Textbox(label='File Import Status', value='not set', interactive=False)
        image_browse_btn.click(on_browse_input_path, inputs=[], outputs=[input_path, path_status])

        # Music Import
        gr.Markdown("**Music Import**")
        music_loader = gr.File(label='Select Files', file_types=['.mp3', '.wav', '.ogg', '.egg'], file_count='multiple')
        file_status = gr.Textbox(label='File Import Status', value='not set', interactive=False)
        upload_button = gr.Button('Copy Files to input folder')
        upload_button.click(upload_files, inputs=[input_path, music_loader], outputs=[file_status])

# Launch the app
if __name__ == "__main__":
    demo.launch()
