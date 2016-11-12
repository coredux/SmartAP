import os
import re
from xml.sax.saxutils import unescape


def string_filter(string):
    string = unescape(string)
    rm_tag = re.compile(r'<[^>]+>',re.S)
    rm_at = re.compile(r'@', re.S)
    rp_qt = re.compile(r'&#39;|&quot;', re.S)
    rp_url = re.compile(r'https*://[^\s]*', re.S)
    return rm_at.sub('', rp_url.sub( ' URL', rp_qt.sub( '\'', rm_tag.sub('', string))))



def list_filenames_from_dir(dir):
    for name in os.listdir(dir):
        yield name


def read_content_from_file(path_to_file):
    all_text = ''
    if os.path.exists(path_to_file):
        file_object = open(path_to_file)
        try:
            all_text = file_object.read()
        finally:
            file_object.close()
    return all_text


def write_list_to(file_loc, to_write):
    with open(file_loc, 'w') as oufile:
        for line in to_write:
            oufile.write(line.encode('utf8') + '\n')

