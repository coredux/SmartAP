import gensim

class EmbeddingContainer:

    model = None

    def __init__(self, path_to_model, _binary=True):
        try:
            self.model = gensim.models.Word2Vec.load_word2vec_format(path_to_model, binary=_binary)
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
            return None




