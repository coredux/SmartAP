import numpy as np
from Config import ConfigReader

class GloveWordVector:

    embedding_index = {}
    embedding_size = int(ConfigReader.ConfigReader().get('glove', 'dim'))

    def __init__(self, path_to_glove):
        with open(path_to_glove, 'r') as infile:
            for line in infile:
                values = line.split()
                word = values[0]
                coefs =  np.asarray(values[1:], dtype='float32')
                self.embedding_index[word] = coefs
        print('Found %s glove word vectors' % len(self.embedding_index))

    def contains_key(self,word):
        return word in self.embedding_index

    def look_up(self,word):
        if self.contains_key(word):
            return self.embedding_index[word]
        else:
            return np.zeros(self.embedding_size).astype('float32')

