from nltk.tokenize import TweetTokenizer

class Tokenizer:

    tokenizer = None

    def __init__(self):
        self.tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)


if __name__ == '__main__':
    sentence = 'helloworld'
    print str.join(' ', Tokenizer().tokenize(sentence))