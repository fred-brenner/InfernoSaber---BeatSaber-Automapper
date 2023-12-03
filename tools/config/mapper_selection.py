import os


from tools.config import paths

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
    if len(files) != 8:
        raise FileNotFoundError(f"Wrong number {model_folder}")
    for f in files:
        if f.startswith(model_name_partial):
            if full_path:
                return os.path.join(model_folder, f)
            else:
                return f

    raise FileNotFoundError(f"Could not find model {model_name_partial} in {model_folder}")
