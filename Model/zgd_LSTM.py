from Config import ConfigReader
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout

# data
maxlen = 30
embedding_size = int(ConfigReader.ConfigReader().get('word2vec', 'dim'))

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
nb_epoch = 15

# Dropout
dropout_p = 0.25

def define_model():
    model = Sequential()
    model.add(LSTM(
        128,
        input_shape=(maxlen, embedding_size),
        return_sequences=False
    ))
    model.add(Dense(
        1
    ))
    model.add(Dropout(dropout_p))
    model.add(Activation('linear'))
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
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    male = [x for x in y_test if x == 1]
    female = [x for x in y_test if x == 0]
    model = define_model()
    model = complile_model(model)
    model = train_model(model, x_train, y_train)

    print len(male), len(female)
    print eval_model(model, x_test, y_test)


