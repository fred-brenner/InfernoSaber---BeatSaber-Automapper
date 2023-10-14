

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

