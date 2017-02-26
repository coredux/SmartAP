import os
from Util import read_lines_from_file, list_filenames_from_dir

class DocsContainer:
    docs_dir = None

    def __init__(self, docs_dir):
        if os.path.exists(docs_dir):
            self.docs_dir = docs_dir
        else:
            raise Exception('invalid dir for docs')

    def retrieve_content_in_one(self, docs_id):
        ret = []
        lines = self.retrieve_content_in_sentences(docs_id)
        #map(lambda x: ret.extend(x), map(lambda x: x.split(' '), lines))
        return str.join(' ', lines)

    def retrieve_content_in_sentences(self, docs_id):
        path_to_file = os.path.join(self.docs_dir, docs_id + ".txt")
        if os.path.exists(path_to_file):
            return read_lines_from_file(path_to_file)
        else:
            raise Exception('invalid doc id')


