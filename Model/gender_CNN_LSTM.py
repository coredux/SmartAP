from Config import ConfigReader
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Convolution1D, GlobalAveragePooling1D
from keras.layers import LSTM, Dense, Activation
from keras.layers.wrappers import TimeDistributed
import numpy as np

# data
maxlen = 30
#embedding_size = int(ConfigReader.ConfigReader().get('word2vec', 'dim'))
embedding_size = int(ConfigReader.ConfigReader().get('glove', 'dim'))

# Convolution
filter_length = 4
nb_filter = 64
pool_length = 4

# LSTM
lstm_hidden_size = 70

# Dense
dense_hidden_size = 32

# Training
batch_size = 30
nb_epoch = 15


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
    model.add(LSTM(lstm_hidden_size ,return_sequences=True))
    model.add(LSTM(lstm_hidden_size))
    #model.add(TimeDistributed(Dense(dense_hidden_size, activation='sigmoid')))
    #model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.add(Dense(dense_hidden_size, activation='sigmoid'))
    model.add(Dense(1,activation='sigmoid'))
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


def verify_model():
    model = define_model()
    model.summary()


def run_model(x_train, y_train, x_test, y_test):
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    y_train = np.array(y_train)
    print('shape of training data x for running %s' % str(x_train.shape))
    print('shape of training data y for running %s' % str(y_train.shape))
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    y_test = np.array(y_test)
    print('shape of testing data x for running %s' % str(x_test.shape))
    print('shape of testing data y for running %s' % str(y_test.shape))
    male = [x for x in y_test if x == 1]
    female = [x for x in y_test if x == 0]
    model = define_model()
    model = complile_model(model)
    model = train_model(model, x_train, y_train)

    print len(male), len(female)
    print eval_model(model, x_test, y_test)


