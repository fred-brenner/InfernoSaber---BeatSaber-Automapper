import tkinter
import customtkinter as ctk
from tkinter import filedialog, messagebox
import requests
import os
from io import BytesIO
from zipfile import ZipFile

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("InfernoSaber Automapper")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (3x?)
        self.grid_columnconfigure((0, 2, 3, 4, 5), weight=5)
        self.grid_columnconfigure((1), weight=1)

        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), weight=1)

        #####################
        # Create first pillar
        #####################
        # create frame
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, columnspan=2, rowspan=10, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.sidebar_label = ctk.CTkLabel(self.sidebar_frame, text="SETUP", font=ctk.CTkFont(size=20, weight="bold"))
        self.sidebar_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # create main folder selection
        self.header_select_home = ctk.CTkLabel(self.sidebar_frame, text="Select home folder:", anchor="w")
        self.header_select_home.grid(row=0, column=0, padx=20, pady=(10, 0))
        self.var_folder_path = ctk.StringVar(self, value="Select main folder path")
        self.entry_main_folder = ctk.CTkEntry(self, textvariable=self.var_folder_path)
        self.entry_main_folder.grid(row=1, column=0, padx=(20, 0), pady=(20, 20), sticky="nsew")
        # self.button_select_main = ctk.CTkButton(master=self, fg_color="transparent", border_width=2,
        #                                         text_color=("gray10", "#DCE4EE"))
        self.button_select_main = ctk.CTkButton(master=self, text="Select main folder",
                                                command=self.open_folder_dialog)
        self.button_select_main.grid(row=1, column=1, padx=(20, 0), pady=(20, 20), sticky="nsew")
        # create download project from git
        self.button_download_git = ctk.CTkButton(master=self, text="Download project from git",
                                                 command=self.download_git)
        self.button_download_git.grid(row=2, column=0, columnspan=2,
                                      padx=(20, 0), pady=(20, 20), sticky="nsew")
        # create UI options
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                      command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = ctk.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                              command=self.change_scaling_event)
        self.scaling_menu.grid(row=8, column=0, padx=20, pady=(10, 20))

        ###############################
        # set default values
        ###############################
        self.appearance_mode_menu.set("Dark")
        self.scaling_menu.set("100%")

    def open_folder_dialog(self):
        # Open folder selection dialog
        folder_selected = filedialog.askdirectory()

        # Update the label with the selected folder path
        self.var_folder_path.set(folder_selected)

    def check_folder_path(self, folder_path=''):
        if os.path.isdir(folder_path):
            return True
        return False

    def download_git(self, branch='app'):
        folder_path = self.var_folder_path.get()
        if not self.check_folder_path(folder_path):
            messagebox.showerror("Error", "Please specify a project folder first")
        else:
            username = 'fred-brenner'
            repository = 'InfernoSaber---BeatSaber-Automapper'
            repo_url = f"https://github.com/{username}/{repository}/archive/{branch}.zip"
            # Make a request to the GitHub API to get the repository as a zip file
            response = requests.get(repo_url)
            # Check if the request was successful
            if response.status_code == 200:
                # Extract the zip file to the selected folder
                with ZipFile(BytesIO(response.content), 'r') as zip_ref:
                    zip_ref.extractall(folder_path)
                # Display a message after successful download
                messagebox.showinfo("Download Complete", "Repository downloaded successfully!")
            else:
                # Display an error message if the request was unsuccessful
                messagebox.showerror("Error", f"Failed to download repository. Status code: {response.status_code}")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)


if __name__ == "__main__":
    app = App()
    app.mainloop()
