from Config import ConfigReader
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Convolution1D, GlobalAveragePooling1D
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from Data.Util import load_on_batch
import numpy as np
import pickle, os

# data
maxlen = 150
embedding_size = int(ConfigReader.ConfigReader().get('settings', 'word_embedding_dim'))
hyper_batch_size = int(ConfigReader.ConfigReader().get('settings', 'hyper_batch_size'))

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
nb_epoch = 10


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
    #model.add(LSTM(lstm_hidden_size ,return_sequences=True))
    model.add(LSTM(lstm_hidden_size))
    model.add(Dense(dense_hidden_size, activation='sigmoid'))
    model.add(Dropout(0.25))
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


# generator should be infinite (loops through data)
def batch_generator(batch_path, typename):
    while True:
        for x, y in load_on_batch(path_to_batch_dir=batch_path, name=typename):
            yield sequence.pad_sequences(x, maxlen=maxlen), np.array(y)


def count_batch(batch_path, typename):
    n_samples = 0
    for x, y in load_on_batch(path_to_batch_dir=batch_path, name=typename):
        n_samples += len(y)
    print 'batch total size: {0}\n'.format(n_samples)
    return n_samples


def train_model_on_batch(model, batch_path, typename):
    n_samples = count_batch(batch_path, typename)
    model.fit_generator(batch_generator(batch_path, typename), samples_per_epoch=n_samples, nb_epoch=nb_epoch)
    return model


def eval_model(model, x_test, y_test):
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    return score, acc


def eval_model_on_batch(model, batch_path, typename):
    n_samples = count_batch(batch_path, typename)
    score, acc = model.evaluate_generator(batch_generator(batch_path, typename), n_samples)
    print score, acc


def verify_model():
    model = define_model()
    model.summary()


def run_model_on_batch(batch_dir):
    model = define_model()
    model = complile_model(model)
    model = train_model_on_batch(model, batch_dir, 'train')
    eval_model_on_batch(model, batch_dir, 'test')


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

