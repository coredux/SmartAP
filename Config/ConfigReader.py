from Data.Util import read_lines_from_file


class ConfigReader:

    config_dict = dict()

    def __init__(self):
        try:
            lines = read_lines_from_file('..\\Config\\app.config')
            for line in lines:
                tokens = str(line).strip().split('=')
                self.config_dict[tokens[0].strip()] = tokens[1].strip()
        except:
            raise Exception('invalid configuration')

    def get(self, key):
        if key in self.config_dict:
            return self.config_dict[key]
        else:
            raise Exception('invalid configuration key')


if __name__ == '__main__':
    c = ConfigReader()
    print ConfigReader().get('root')
