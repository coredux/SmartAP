import pickle


class WordIndexer(object):
    word_index = None

    def __init__(self, path_to_index):
        with open(path_to_index, 'rb') as infile:
            keras_tk = pickle.load(infile)
        if keras_tk is not None:
            self.word_index = keras_tk.word_index

    def word_index(self):
        return self.word_index
