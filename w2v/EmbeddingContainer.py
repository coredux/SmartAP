import gensim, numpy
from Config import ConfigReader

class EmbeddingContainer:

    model = None
    embedding_size = int(ConfigReader.ConfigReader().get('settings', 'word_embedding_dim'))

    def __init__(self, path_to_model, _binary=True,  _unicode_errors='ignore'):
        try:
            self.model = gensim.models.Word2Vec.load_word2vec_format(path_to_model, binary=_binary, unicode_errors=_unicode_errors)
        except:
            print "warning: cannot load w2v model"

    def contains_key(self, k):
        if self.model is not None:
            return k in self.model.vocab
        else:
            return False


    def look_up(self, k):
        if self.contains_key(k):
            return self.model[k]
        else:
            return numpy.zeros(self.embedding_size).astype('float32')


if __name__ == '__main__':
    w2v_file = ConfigReader.ConfigReader().get('word2vec', 'model_path')
    print w2v_file



