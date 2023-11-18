import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog


from app_helpers.file_downloader import download_model
# from app_helpers.file_handling import select_folder
from app_helpers.run_python import run_inferno_saber


def select_folder():
    folder_path = filedialog.askdirectory()
    folder_var.set(folder_path)


# Set up the GUI
window = ctk.CTk()
window.title("InfernoSaber")

# Category 1: Setup
setup_frame = ctk.CTkLabel(window, text="Setup")
setup_frame.pack(padx=10, pady=10)

folder_var = ctk.StringVar()
folder_label = ctk.CTkLabel(setup_frame, textvariable=folder_var)
folder_label.pack()

folder_button = ctk.CTkButton(setup_frame, text="Select Folder", command=select_folder)
folder_button.pack()

model_dropdown = ctk.CTkOptionMenu(setup_frame, ctk.StringVar(), "Model 1", "Model 2", "Model 3", command=download_model)
model_dropdown.pack()

# Category 2: Specify Parameters
params_frame = ctk.LabelFrame(window, text="Specify Parameters")
params_frame.pack(padx=10, pady=10)

model_dropdown = tk.OptionMenu(params_frame, tk.StringVar(), "Model 1", "Model 2", "Model 3")
model_dropdown.pack()

difficulty_var = tk.DoubleVar()
difficulty_entry = tk.Entry(params_frame, textvariable=difficulty_var)
difficulty_entry.pack()

# Category 3: Run InfernoSaber
run_frame = tk.LabelFrame(window, text="Run InfernoSaber")
run_frame.pack(padx=10, pady=10)

folder_display = tk.Label(run_frame, textvariable=folder_var)
folder_display.pack()

model_var = tk.IntVar()
model_dropdown = tk.OptionMenu(run_frame, model_var, "1", "2", "3")
model_dropdown.pack()

run_button = tk.Button(run_frame, text="Run InfernoSaber", command=run_inferno_saber)
run_button.pack()

debug_var = tk.IntVar()
debug_check = tk.Checkbutton(run_frame, text="Debug Mode", variable=debug_var)
debug_check.pack()

# Start the GUI
window.mainloop()

# if __name__ == "__main__":
#     main_app()
