from keras.layers import Dense, Input, LSTM, Flatten, Dropout, \
    MaxPooling2D, Conv2D, BatchNormalization, SpatialDropout2D, concatenate, \
    Reshape, Conv2DTranspose, UpSampling2D, CuDNNLSTM
from tcn import TCN  # pip install keras-tcn
from keras.models import Model
import numpy as np

from tools.config import config


def create_keras_model(model_type, dim_in=[], dim_out=None):
    print("Setup keras model")
    if model_type == 'lstm1':
        # in_song (lin), in_time (rec), in_class (rec)
        input_a = Input(shape=(dim_in[0]), name='input_song_enc')
        input_b = Input(shape=(dim_in[1]), name='input_time_lstm')
        input_c = Input(shape=(dim_in[2]), name='input_class_lstm')

        lstm_b = CuDNNLSTM(32, return_sequences=True)(input_b)
        lstm_c = CuDNNLSTM(128, return_sequences=True)(input_c)

        lstm_in = concatenate([lstm_b, lstm_c])
        lstm_out = CuDNNLSTM(128, return_sequences=False)(lstm_in)

        x = concatenate([input_a, lstm_out])
        x = Dense(500, activation='relu')(x)
        x = Dropout(0.05)(x)
        x = Dense(400, activation='sigmoid')(x)

        out = Dense(dim_out, activation='softmax', name='output')(x)

        model = Model(inputs=[input_a, input_b, input_c], outputs=out)
        return model

    # autoencoder
    if model_type == 'enc1':
        input_img = Input(shape=(24, 20, 1))
        # Conv2d(1, 32, 3, padding=1) with Relu
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        # Dropout2d(0.2)
        # x = SpatialDropout2D(0.05)(x)
        # MaxPool2d(2, 2)
        x = MaxPooling2D((2, 2), padding='same')(x)

        # Conv2d(32, 16, 3, padding=1) with Relu
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        # MaxPool2d(2, 2)
        x = MaxPooling2D((2, 2), padding='same')(x)

        # Flatten(start_dim=1)
        x = Flatten('channels_last')(x)
        # Dropout(0.1)
        x = Dropout(0.05)(x)
        # Relu(in=480, out=128)
        x = Dense(128, activation='relu')(x)
        # Relu(in=128, out=config.bottleneck_len)
        x = Dense(config.bottleneck_len, activation='relu')(x)

        model = Model(input_img, x)
        return model

    if model_type == 'dec1':
        input_img = Input(shape=config.bottleneck_len)
        # Relu(in=config.bottleneck_len, out=128)
        x = Dense(128, activation='relu')(input_img)
        # Relu(in=128, out=480)
        x = Dense(640, activation='relu')(x)
        # Unflatten(1, (16, 6, 5))
        x = Reshape(target_shape=(10, 8, 8))(x)
        # input shape (batch_size, rows, cols, channels)

        # x = Conv2D(32, kernel_size=2, activation='relu')(x)
        # x = UpSampling2D((2, 2))(x)
        # x = Conv2D(1, kernel_size=2, activation='sigmoid')(x)
        # x = UpSampling2D((2, 2))(x)

        # # ConvTranspose2d(16, 32, 2, stride=2) with Relu
        x = Conv2DTranspose(32, kernel_size=5, strides=2, activation='relu')(x)
        # ConvTranspose2d(32, 1, 2, stride=2) with Sigmoid
        x = Conv2DTranspose(1, kernel_size=2, strides=1, activation='sigmoid')(x)
        # output shape (batch_size, new_rows, new_cols, filters)

        model = Model(input_img, x)
        return model


def create_music_model(model_type, dim_in, tcn_len):
    # https://github.com/giusenso/seld-tcn/blob/master/keras_model.py
    # https://github.com/philipperemy/keras-tcn
    print("Setup music model")
    # input song 264.X  (batch_size, timesteps, input_dim)
    # output beats 1.X  (beat IO x sample)

    if model_type == 'tcn':
        # input song (batch_size, timesteps, input_dim)
        input_a = Input(shape=(tcn_len, dim_in), name='input_song_img')
        # input beat (batch_size, time_steps, 1)
        input_b = Input(shape=(tcn_len, 1), name='input_beat_prop')
        input_c = Input(shape=(tcn_len, 1), name='input_onset_detection')

        a = TCN(nb_filters=128,
                kernel_size=2,
                nb_stacks=1,
                dilations=(1, 2, 4, 8, 16, 32),
                dropout_rate=0.0,
                use_skip_connections=True,
                use_batch_norm=False,
                use_weight_norm=False,
                use_layer_norm=False,
                return_state=False,
                return_sequences=False,
                activation='relu'
                )(input_a)

        b = CuDNNLSTM(8, return_sequences=False)(input_b)
        c = CuDNNLSTM(8, return_sequences=False)(input_c)

        x = concatenate([a, b, c])
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)

        out = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[input_a, input_b, input_c], outputs=out)
        return model

    # if model_type == 'lstm':
    #     # input song (batch_size, time_steps, seq_len)
    #     input_a = Input(shape=(tcn_len, dim_in), name='input_song_img')
    #     # input beat (batch_size, time_steps, 1)
    #     input_b = Input(shape=(tcn_len, 1), name='input_beat_prop')
    #
    #     a = CuDNNLSTM(256, return_sequences=True)(input_a)
    #     a = CuDNNLSTM(128, return_sequences=True)(a)
    #
    #     b = CuDNNLSTM(8, return_sequences=True)(input_b)
    #
    #     ab = concatenate([a, b])
    #     x = CuDNNLSTM(64, return_sequences=False)(ab)
    #     x = Dense(32, activation='relu')(x)
    #     x = Dense(32, activation='relu')(x)
    #
    #     out = Dense(1, activation='sigmoid')(x)
    #
    #     model = Model(inputs=[input_a, input_b], outputs=out)
    #     return model


def create_post_model(model_type, lstm_len: int, dim_out=2):
    print("Setup keras model")
    if model_type == 'lstm1':
        # in_song (lin), in_time (rec), in_class (rec)
        input_a = Input(shape=(lstm_len, 5), name='input_type_cut_timeDiff_lastNotes_lstm')
        # input_b = Input(shape=(dim_in[1]), name='input_note_cut_lstm')
        # input_c = Input(shape=(dim_in[2]), name='input_time_diff_lstm')

        lstm_a = CuDNNLSTM(256, return_sequences=True)(input_a)
        # lstm_c = CuDNNLSTM(128, return_sequences=True)(input_c)

        # lstm_in = concatenate([lstm_b, lstm_c])
        lstm_out = CuDNNLSTM(128, return_sequences=False)(lstm_a)

        # x = concatenate([input_a, lstm_out])
        x = Dense(512, activation='relu')(lstm_out)
        x = Dropout(0.05)(x)
        x = Dense(256, activation='relu')(x)

        out = Dense(dim_out, activation='relu', name='output')(x)

        model = Model(inputs=[input_a], outputs=out)
        return model
