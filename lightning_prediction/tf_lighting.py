from keras.layers import Dense, Input, LSTM, Flatten, Dropout, \
    MaxPooling2D, Conv2D, BatchNormalization, SpatialDropout2D, concatenate, \
    Reshape, Conv2DTranspose, UpSampling2D, CuDNNLSTM
from keras.models import Model
import numpy as np


def create_tf_model(model_type, dim_in, dim_out, nr=128):
    print("Setup keras model")
    if model_type == 'lstm_light':
        # in_song (lin), in_time (rec)
        input_a = Input(shape=(dim_in[0][1]), name='input_song_enc')
        input_b = Input(shape=(dim_in[1][1:]), name='input_time_lstm')

        lstm_out = CuDNNLSTM(32, return_sequences=False)(input_b)

        x = concatenate([input_a, lstm_out])
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.05)(x)
        x = Dense(256, activation='sigmoid')(x)

        out = Dense(dim_out[1], activation='softmax', name='output')(x)

        model = Model(inputs=[input_a, input_b], outputs=out)
        return model

    elif model_type == 'lstm_full':
        # in_song (lin), in_time (rec), in_class (rec)
        input_a = Input(shape=(dim_in[0][1]), name='input_song_enc')
        input_b = Input(shape=(dim_in[1][1:]), name='input_time_lstm')
        input_c = Input(shape=(dim_in[2][1:]), name='input_class_lstm')

        lstm_b = CuDNNLSTM(32, return_sequences=True)(input_b)
        lstm_c = CuDNNLSTM(256, return_sequences=True)(input_c)

        lstm_in = concatenate([lstm_b, lstm_c])
        lstm_out = CuDNNLSTM(64, return_sequences=False)(lstm_in)

        x = concatenate([input_a, lstm_out])
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.05)(x)
        x = Dense(256, activation='sigmoid')(x)

        out = Dense(dim_out[1], activation='softmax', name='output')(x)

        model = Model(inputs=[input_a, input_b, input_c], outputs=out)
        return model

    elif model_type == 'lstm_half':
        # in_song (lin), in_time (rec), in_class (rec)
        input_a = Input(shape=(dim_in[0][1:]), name='input_song_enc')
        input_b = Input(shape=(dim_in[1][1:]), name='input_time_lstm')
        input_c = Input(shape=(dim_in[2][1:]), name='input_class_lstm')

        conv = Conv2D(32, kernel_size=3, activation='relu')(input_a)
        conv = Flatten('channels_last')(conv)
        lstm_b = CuDNNLSTM(32, return_sequences=True)(input_b)
        lstm_c = CuDNNLSTM(64, return_sequences=True)(input_c)

        lstm_in = concatenate([lstm_b, lstm_c])
        lstm_out = CuDNNLSTM(64, return_sequences=False)(lstm_in)

        x = concatenate([conv, lstm_out])
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.05)(x)
        x = Dense(dim_out[1]*dim_out[2], activation='sigmoid')(x)

        out = Reshape(target_shape=(dim_out[1:]))(x)
        # out = Dense(dim_out[1:], activation='softmax', name='output')(x)

        model = Model(inputs=[input_a, input_b, input_c], outputs=out)
        return model
