from nltk.stem.lancaster import LancasterStemmer

class Stemmer:

    stemmer = None

    def __init__(self):
        self.stemmer = LancasterStemmer()

    def stem(self, word):
        return self.stemmer.stem(word)

