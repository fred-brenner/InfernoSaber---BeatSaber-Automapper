from tensorflow.keras import layers, Input
from tensorflow.keras import models
from tensorflow.keras import Model

from tools.utils.ask_parameter import ask_parameter
from tools.config import config, paths


def build_model(ml_input_song, ml_input_beat):
    q = ask_parameter("Create new model? [y or n]", ['y', 'n'])
    if q == 'y':
        # build keras model

        # input
        input_song_layer = Input(shape=1, name='song_input')

        # hidden
        hidden1 = layers.Dense(128, activation='relu', name='hidden1')(input_song_layer)
        hidden2 = layers.Dense(32, activation='relu', name='hidden2')(hidden1)
        hidden3 = layers.Dense(8, activation='relu', name='hidden3')(hidden2)

        # output
        output_out_layer = layers.Dense(1, activation='sigmoid', name='output_beat')(hidden3)

        # init
        model = Model(input_song_layer, output_out_layer, name="BS_BeatMapper")

    else:
        model = models.load_model(paths.model_path + config.model_name)

    return model
