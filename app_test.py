# import gradio as gr
# import os
# from tkinter import Tk, filedialog
#
#
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
#             filename = "Files not seleceted"
#             root.destroy()
#             return str(filename)
#
#     elif data_type == "Folder":
#         filename = filedialog.askdirectory()
#         if filename:
#             if os.path.isdir(filename):
#                 root.destroy()
#                 return str(filename)
#             else:
#                 root.destroy()
#                 return str(filename)
#         else:
#             filename = "Folder not seleceted"
#             root.destroy()
#             return str(filename)
#
#
# def main():
#     with gr.Blocks() as demo:
#         data_type = gr.Radio(choices=["Files", "Folder"], value="Files", label="Offline data type")
#         input_path = gr.Textbox(label="Select Multiple videos", scale=5, interactive=False)
#         image_browse_btn = gr.Button("Browse", min_width=1)
#         image_browse_btn.click(on_browse, inputs=data_type, outputs=input_path, show_progress="hidden")
#     return demo
#
#
# demo = main()
# demo.launch(inbrowser=True)


"""
import gradio as gr


# Define your functions for each section
def setup_action(config_option):
    return f"Setup completed with option: {config_option}"


def specify_params(param1, param2):
    return f"Parameters specified: Param1={param1}, Param2={param2}"


def run_action(data):
    return f"Processing data: {data}"


# Create the Gradio app with Tabs
with gr.Blocks() as demo:
    gr.Markdown("# My Gradio App")

    # Setup Tab
    with gr.Tab("Setup"):
        gr.Markdown("## Setup")
        config_option = gr.Dropdown(["Option 1", "Option 2", "Option 3"], label="Configuration Option")
        setup_output = gr.Textbox(label="Setup Status")
        setup_button = gr.Button("Run Setup")
        setup_button.click(setup_action, inputs=[config_option], outputs=[setup_output])

    # Specify Parameters Tab
    with gr.Tab("Specify Parameters"):
        gr.Markdown("## Specify Parameters")
        param1 = gr.Textbox(label="Parameter 1")
        param2 = gr.Textbox(label="Parameter 2")
        params_output = gr.Textbox(label="Parameter Status")
        params_button = gr.Button("Set Parameters")
        params_button.click(specify_params, inputs=[param1, param2], outputs=[params_output])

    # Run Tab
    with gr.Tab("Run"):
        gr.Markdown("## Run")
        data_input = gr.Textbox(label="Input Data")
        run_output = gr.Textbox(label="Result")
        run_button = gr.Button("Run")
        run_button.click(run_action, inputs=[data_input], outputs=[run_output])

# Launch the app
if __name__ == "__main__":
    demo.launch()
"""