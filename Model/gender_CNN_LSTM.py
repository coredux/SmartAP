from Config import ConfigReader
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Convolution1D, GlobalAveragePooling1D, Embedding
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from Data.Util import load_on_batch
import numpy as np
import pickle, os
from Data.prepare_data import load_word_embedding_weights, load_word_indexer

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

# Label
label_male = 1
label_female = 0

model_save_path = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'model')


def define_model():
    model = Sequential()
    model.add(Embedding(
        input_shape=(None, maxlen),
        input_dim=len(load_word_indexer()) + 1,
        output_dim=embedding_size,
        weights=[load_word_embedding_weights()],
        trainable=True
    ))
    model.add(Convolution1D(
        nb_filter=nb_filter,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1
    ))
    #model.add(LSTM(lstm_hidden_size ,return_sequences=True))
    model.add(LSTM(lstm_hidden_size))
    model.add(Dense(dense_hidden_size, activation='relu'))
    model.add(Dense(1, activation='relu'))
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


# for batch training, generator should be infinite (loops through data)
def batch_generator(batch_path, typename):
    while True:
        for x, y in load_on_batch(path_to_batch_dir=batch_path, name=typename):
            pad_x = sequence.pad_sequences(x, maxlen=maxlen, dtype='int32')
            yield pad_x, np.array(y).astype('int32')


# this is for testing
def testing_batch_iterator(batch_path, typename):
    for x, y in load_on_batch(path_to_batch_dir=batch_path, name=typename):
        for i in xrange(len(y)):
            v_doc = sequence.pad_sequences(x[i], maxlen=maxlen, dtype='int32')
            yield v_doc, np.array(y[i]).astype('int32')


def count_batch(batch_path, typename):
    n_samples = 0
    for x, y in load_on_batch(path_to_batch_dir=batch_path, name=typename):
        n_samples += len(y)
    print 'batch total size: {0}\n'.format(n_samples)
    return n_samples


def __train_model_on_batch(model, batch_path, typename):
    n_samples = count_batch(batch_path, typename)
    model.fit_generator(batch_generator(batch_path, typename), samples_per_epoch=n_samples, nb_epoch=nb_epoch)
    return model


def eval_model(model, x_test, y_test):
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    return score, acc


def __eval_model_on_batch(model, batch_path, typename):
    p = []
    ans = []
    exc_cnt = 0
    for x, y in testing_batch_iterator(batch_path, typename):
        ans.append(y)
        pi = []
        try:
            pi = model.predict(x)
        except:
            exc_cnt += 1
            m = 0
            f = 0
        m = len([pij for pij in pi if pij > 0.5])
        f = len([pij for pij in pi if pij < 0.5])
        if m >= f:
            p.append(label_male)
        else:
            p.append(label_female)
    right = len([p[i] for i in xrange(len(p)) if p[i] == ans[i]])
    print 'acc: {0}'.format(float(right) / len(p))
    print exc_cnt, len(p)
    print p


def save_model(model, path):
    model.save(path)


def verify_model():
    model = define_model()
    model.summary()


def train_model_on_batch(batch_dir, save = True):
    model = define_model()
    model = complile_model(model)
    model = __train_model_on_batch(model, batch_dir, 'train')
    if save:
        if not os.path.exists(model_save_path): os.mkdir(model_save_path)
        save_model(model, os.path.join(model_save_path, 'model.h5'))
    return model


def eval_model_on_batch(batch_dir):
    if os.path.exists(os.path.join(model_save_path, 'model.h5')):
        model = load_model(os.path.join(model_save_path, 'model.h5'))
        __eval_model_on_batch(model, batch_dir, 'test')


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

