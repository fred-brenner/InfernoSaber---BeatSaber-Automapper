from keras.layers import Dense, Input, LSTM, CuDNNLSTM, Flatten, Dropout,\
    MaxPooling2D, Conv2D, BatchNormalization, SpatialDropout2D, concatenate,\
    Reshape, Conv2DTranspose, UpSampling2D
from keras import optimizers
from keras.models import Model
from keras.optimizers import adam_v2
import numpy as np
from keras.callbacks import ReduceLROnPlateau

from tools.config import config


def create_keras_model(model_type, lr, ds_in=None, ds_out=None):
    if model_type == 'lstm1':
        print("Setup keras model")
        input_a = Input(shape=(ds_in[0].shape[1], ds_in[0].shape[2]),
                       name='input_lineIndex')  # [train_i_li, train_i_ll, train_i_ty, train_i_cd, diff_i]
        input_b = Input(shape=(ds_in[1].shape[1], ds_in[1].shape[2]), name='input_lineLayer')
        input_c = Input(shape=(ds_in[2].shape[1], ds_in[2].shape[2]), name='input_type')
        input_d = Input(shape=(ds_in[3].shape[1], ds_in[3].shape[2]), name='input_cutDirection')
        input_e = Input(shape=(ds_in[4].shape[1],), name='input_diff')
        input_f = Input(shape=(ds_in[5].shape[1], ds_in[5].shape[2]), name='input_song')

        lstm_a = CuDNNLSTM(16, return_sequences=True)(input_a)
        lstm_b = CuDNNLSTM(16, return_sequences=True)(input_b)
        lstm_c = CuDNNLSTM(16, return_sequences=True)(input_c)
        lstm_d = CuDNNLSTM(16, return_sequences=True)(input_d)
        lstm_f = CuDNNLSTM(32, return_sequences=True)(input_f)

        lstm_in = concatenate([lstm_a, lstm_b, lstm_c, lstm_d, lstm_f])
        lstm_out = CuDNNLSTM(64, return_sequences=False)(lstm_in)

        x = concatenate([lstm_out, input_e])
        x = Dense(64, activation='relu')(x)
        x = Dense(16, activation='relu')(x)

        out = Dense(1, activation='linear', name='output')(x)

        model = Model(inputs=[input_a, input_b, input_c, input_d, input_e, input_f], outputs=out)
        adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
        #                               mode='auto', min_delta=0.001, cooldown=5, min_lr=0.00001)
        # callbacks=[reduce_lr],
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
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
        # adam = adam_v2.Adam()
        # model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
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
