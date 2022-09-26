from keras.layers import Dense, Input, LSTM, CuDNNLSTM, Flatten, Dropout, MaxPooling2D, Conv2D, BatchNormalization, SpatialDropout2D, concatenate
from keras import optimizers
from keras.models import Model
import numpy as np
from keras.callbacks import ReduceLROnPlateau


def create_keras_model(model_type, lr, ds_in, ds_out):
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

    if model_type == 'enc1':
        print("")
        # Conv2d(1, 32, 3, padding=1) with Relu
        # Dropout2d(0.2)
        # MaxPool2d(2, 2)

        # Conv2d(32, 16, 3, padding=1) with Relu
        # MaxPool2d(2, 2)

        # Flatten(start_dim=1),
        # Dropout(0.1)
        # Relu(in=480, out=128),
        # Relu(in=128, out=config.bottleneck_len)
        return model

    if model_type == 'dec1':
        print("")
        # Relu(in=config.bottleneck_len, out=128),
        # Relu(in=128, out=480)
        # Unflatten(1, (16, 6, 5))

        # ConvTranspose2d(16, 32, 2, stride=2) with Relu
        # ConvTranspose2d(32, 1, 2, stride=2) with Sigmoid
        return model