import os, re, random
from xml.sax.saxutils import unescape


'''
remove or replace some special tokens in a string
'''
def string_filter(string):
    string = unescape(string)
    rm_tag = re.compile(r'<[^>]+>',re.S) # remove html tags
    string = rm_tag.sub('', string)
    # rm_at = re.compile(r'@', re.S)
    rp_qt = re.compile(r'&#39;|&quot;', re.S) # quote
    string = rp_qt.sub('\'', string)

    return string

'''
replace the url with the 'URL'
example: http://www.google.com ---> URL
'''
def word_purify(string):
    rp_url = re.compile(r'https*://[^\s]*', re.S)
    string = rp_url.sub('URL', string)
    return string


def list_filenames_from_dir(dir):
    for name in os.listdir(dir):
        yield name

'''
DO NOT use this function to open big files
'''
def read_lines_from_file(path_to_file):
    lines = []
    if os.path.exists(path_to_file):
        file_object = open(path_to_file)
        try:
            lines = [ x.strip() for x in file_object.readlines()]
        finally:
            file_object.close()
    return lines

'''
DO NOT use this function to open big files
'''
def read_content_from_file(path_to_file):
    all_text = ''
    if os.path.exists(path_to_file):
        file_object = open(path_to_file)
        try:
            all_text = file_object.read()
        finally:
            file_object.close()
    return all_text


'''
output the content of a list to the location
'''
def write_list_to(file_loc, to_write):
    with open(file_loc, 'w') as oufile:
        for line in to_write:
            if line is not '':
                oufile.write(line.encode('utf8') + '\n')

'''
randomly split the list into training data and testing data
'''
def randomly_choose_train_test(name_list, ratio=0.8):
    shuffled = random.sample(name_list, len(name_list))
    len_train = int(len(shuffled) * ratio)
    return shuffled[:len_train], shuffled[len_train:]

def shuffle(l):
    return random.sample(l, len(l))

