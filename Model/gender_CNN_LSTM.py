import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import LSTM, Dense, Activation

# data
maxlen = 135
embedding_size = 300

# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
nb_epoch = 5


def define_model():
    model = Sequential()
    model.add(Convolution1D(
        input_shape=(maxlen, embedding_size),
        nb_filter= nb_filter,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1
    ))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def complile_model(model):
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    return model


def eval_model(model, x_test, y_test):
    score, acc = model.evaluate( x_test, y_test, batch_size=batch_size)
    return score, acc


def run_model(x_train, y_train, x_test, y_test):
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    model = define_model()
    model = complile_model(model)
    model = train_model(model, x_train, y_train)
    print eval_model(model, x_test, y_test)


