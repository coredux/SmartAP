from Data.Util import read_lines_from_file
import ConfigParser


class ConfigReader:

    cf = ConfigParser.ConfigParser()
    config_path = '../Config/app.ini'

    def __init__(self):
        self.cf.read(self.config_path)

    def get(self, key_class, key_name):
        return self.cf.get(key_class, key_name)


if __name__ == '__main__':
    c = ConfigReader()
    print ConfigReader().get('file', 'root')
    print ConfigReader().get('word2vec', 'model_path')
