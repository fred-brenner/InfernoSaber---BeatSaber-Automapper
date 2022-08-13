"""
Training script
"""

from tensorflow.python.client import device_lib
from sklearn.preprocessing import OneHotEncoder

from old.load_ml_data import load_ml_arrays
from old.build_model import build_model
from tools.config import config


def train():
    # check for gpu support
    cpu_gpu = device_lib.list_local_devices()
    if len(cpu_gpu) <= 1:
        print(f"Could not find GPU, devices found:\n{cpu_gpu}")
        exit()

    # import data
    song_ar, beat_ar = load_ml_arrays()

    # create model
    model = build_model(song_ar, beat_ar)
    model.summary()

    # compile model
    model.compile(optimizer='Adam', loss='BinaryCrossentropy', metrics=['accuracy', 'mse'])

    # start training
    model.fit(x=song_ar, y=beat_ar, batch_size=config.batch_size,
              epochs=10, validation_split=0.25, shuffle=True)


    # test model
    pass


if __name__ == '__main__':
    train()
