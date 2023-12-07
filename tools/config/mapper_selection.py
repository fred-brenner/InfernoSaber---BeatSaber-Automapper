import os

from tools.config import paths, config


def return_mapper_list(mapper_shortcut):
    if mapper_shortcut == 'curated1':
        mapper_list = ['Nuketime', 'Heisenberg', 'Ruckus',
                       'Joetastic', 'BennyDaBeast', 'Hexagonial',
                       'Ryger', 'Skyler Wallace', 'Uninstaller',
                       'Teuflum', 'GreatYazer', 'puds'
                                                'Moriik', 'Ab', 'DE125', 'Skeelie',
                       'Psyc0pathic', 'Hexagonial', 'Electrostats',
                       'DankruptMemer', 'StyngMe', 'Rustic',
                       'Souk', 'Oddloop', 'Chroma', 'Pendulum',
                       'Excession', 'Jez', 'Kry', 't+pazolite']

    elif mapper_shortcut == 'curated2':
        mapper_list = ['Nuketime', 'Ruckus', 'Joetastic',
                       'BennyDaBeast', 'Hexagonial', 'Ryger',
                       'Skyler Wallace', 'Uninstaller', 'Teuflum',
                       'Oddloop', 'Souk', 'Pendulum', 't+pazolite']

    else:
        mapper_list = mapper_shortcut
    return mapper_list


def get_full_model_path(model_name_partial, full_path=True):
    model_folder = paths.model_path
    files = os.listdir(model_folder)
    if not (7 < len(files) < 10):  # allow some extra files, e.g. extra zip file
        raise FileNotFoundError(f"Model save files missing or corrupt. Check content of:  {model_folder}")
    for f in files:
        if f.startswith(model_name_partial):
            if full_path:
                return os.path.join(model_folder, f)
            else:
                return f

    raise FileNotFoundError(f"Could not find model {model_name_partial} in {model_folder}")


def update_model_file_paths():
    # update the file paths for the models if the folder is changed
    paths.model_path = paths.dir_path + "model/"
    if config.use_mapper_selection == '' or config.use_mapper_selection is None:
        paths.model_path += "general_new/"
    else:
        paths.model_path += f"{config.use_mapper_selection.lower()}/"
    paths.notes_classify_dict_file = os.path.join(paths.model_path, "notes_class_dict.pkl")
    paths.beats_classify_encoder_file = os.path.join(paths.model_path, "onehot_encoder_beats.pkl")
    paths.events_classify_encoder_file = os.path.join(paths.model_path, "onehot_encoder_events.pkl")

    # check that model exists on the example of event generator
    _ = get_full_model_path(config.event_gen_version)
    print(f"Using model: {config.use_mapper_selection}")
