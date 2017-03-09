import os
import pickle
import Stemmer
import Tokenizer
import numpy as np
import w2v.EmbeddingContainer
import LabelContainer
from Util import list_filenames_from_dir, write_list_to, read_content_from_file,read_lines_from_file, string_filter
from Util import word_purify, randomly_choose_train_test, shuffle
from Parse import retrieve_from_xml
from Config import ConfigReader
from keras.preprocessing.text import Tokenizer as KTokenizer
from glove.WordVector import GloveWordVector

src_dir = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/dataset')
docs_dir = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/docs')
indexed_docs_dir = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/indexed_docs')
w2v_model_path = ConfigReader.ConfigReader().get('word2vec', 'model_path')
label_file_path = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/truth/n_truth.txt')

data_dir_preffix = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset')
word_index_file = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/word_index/wi')
word_embedding_weights = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/word_index/weights')
glove_model_path = ConfigReader.ConfigReader().get('glove', 'model_path')
word_embedding_dim = int(ConfigReader.ConfigReader().get('settings', 'word_embedding_dim'))


# step1: generate docs from the source data set
def __gen_docs():
    tk = Tokenizer.Tokenizer()
    stm = Stemmer.Stemmer()
    keras_tk = KTokenizer()
    for full_name in list_filenames_from_dir(src_dir):
        filename = full_name.split('.')[0]
        affix = full_name.split('.')[1]
        if affix == 'xml':
            lines = [str.join(' ', map(stm.stem, map(word_purify, tk.tokenize(string_filter(x)))))
                     for x in retrieve_from_xml(read_content_from_file(os.path.join(src_dir, full_name)))]
            write_list_to(os.path.join(docs_dir, filename + '.txt'), lines)
            keras_tk.fit_on_texts(map(lambda x: x.encode('utf8'), lines))
    with open(word_index_file, 'wb') as oufile:
        pickle.dump(keras_tk, oufile)


# step2: extracting male and female
def __classify_docs_gender():
    lc = LabelContainer.LabelContainer(label_file_path)
    male_docid = [x for x in lc.labelDict if lc.labelDict[x]['gender'] == 1]
    female_docid = [x for x in lc.labelDict if lc.labelDict[x]['gender'] == 0]
    write_list_to(os.path.join(data_dir_preffix, 'gender/male.txt'), male_docid)
    write_list_to(os.path.join(data_dir_preffix, 'gender/female.txt'), female_docid)


# step2: extracting all kinds of ages
def __classify_docs_age():
    lc = LabelContainer.LabelContainer(label_file_path)
    c1 = [x for x in lc.labelDict if lc.labelDict[x]['age'] == 0]
    c2 = [x for x in lc.labelDict if lc.labelDict[x]['age'] == 1]
    c3 = [x for x in lc.labelDict if lc.labelDict[x]['age'] == 2]
    c4 = [x for x in lc.labelDict if lc.labelDict[x]['age'] == 3]
    c5 = [x for x in lc.labelDict if lc.labelDict[x]['age'] == 4]
    write_list_to(os.path.join(data_dir_preffix, 'age/0.txt'), c1)
    write_list_to(os.path.join(data_dir_preffix, 'age/1.txt'), c2)
    write_list_to(os.path.join(data_dir_preffix, 'age/2.txt'), c3)
    write_list_to(os.path.join(data_dir_preffix, 'age/3.txt'), c4)
    write_list_to(os.path.join(data_dir_preffix, 'age/4.txt'), c5)


# step3: randomly generate the training and testing data (only doc_id)
def __gen_train_test_data():
    male = read_lines_from_file(os.path.join(data_dir_preffix, 'gender/male.txt'))
    female = read_lines_from_file(os.path.join(data_dir_preffix, 'gender/female.txt'))
    male_train, male_test = randomly_choose_train_test(male)
    female_train, female_test = randomly_choose_train_test(female)
    write_list_to(os.path.join(data_dir_preffix, 'gender/train_docsid.txt'), shuffle(male_train+female_train) )
    write_list_to(os.path.join(data_dir_preffix, 'gender/test_docsid.txt'), shuffle(male_test+female_test))

    c0 = read_lines_from_file(os.path.join(data_dir_preffix, 'age/0.txt'))
    c1 = read_lines_from_file(os.path.join(data_dir_preffix, 'age/1.txt'))
    c2 = read_lines_from_file(os.path.join(data_dir_preffix, 'age/2.txt'))
    c3 = read_lines_from_file(os.path.join(data_dir_preffix, 'age/3.txt'))
    c4 = read_lines_from_file(os.path.join(data_dir_preffix, 'age/4.txt'))
    c0_train, c0_test = randomly_choose_train_test(c0)
    c1_train, c1_test = randomly_choose_train_test(c1)
    c2_train, c2_test = randomly_choose_train_test(c2)
    c3_train, c3_test = randomly_choose_train_test(c3)
    c4_train, c4_test = randomly_choose_train_test(c4)
    age_train = shuffle(c0_train + c1_train + c2_train + c3_train + c4_train)
    age_test = shuffle(c0_test + c1_test + c2_test + c3_test + c4_test)
    write_list_to(os.path.join( data_dir_preffix, 'age/train_docsid.txt'), age_train)
    write_list_to(os.path.join( data_dir_preffix, 'age/test_docsid.txt'), age_test)


# step4: prepare word vectors using GLOVE
def __gen_GLOVE_weights():
    keras_tk = None
    with open(word_index_file, 'rb') as infile:
        keras_tk = pickle.load(infile)
    if keras_tk is None:
        print('warning: GLOVE indexes loading failed')
        return
    word_index = keras_tk.word_index
    nb_words = len(word_index) + 1
    embedding_matrix = np.zeros((nb_words, word_embedding_dim))
    glove = GloveWordVector(glove_model_path)
    for word, i in word_index.items():
        embedding_matrix[i] = glove.look_up(word)
    with open(word_embedding_weights, 'wb') as oufile:
        pickle.dump(embedding_matrix, oufile)


# step5: transfer word to index
def __gen_word_index_doc():
    keras_tk = None
    with open(word_index_file, 'rb') as infile:
        keras_tk = pickle.load(infile)
    if keras_tk is None:
        print('warning: GLOVE indexes loading failed')
        return
    for full_name in list_filenames_from_dir(docs_dir):
        filename = full_name.split('.')[0]
        affix = full_name.split('.')[1]
        if affix == 'txt':
            lines = []
            for line in read_lines_from_file(os.path.join(docs_dir, full_name)):
                lines.extend(keras_tk.texts_to_sequences([line]))
            with open(os.path.join(indexed_docs_dir, filename), 'wb') as oufile:
                pickle.dump(lines, oufile)


def load_word_indexer():
    keras_tk = None
    with open(word_index_file, 'rb') as infile:
        keras_tk = pickle.load(infile)
    if keras_tk is None:
        raise Exception('warning: GLOVE indexes loading failed')
    return keras_tk.word_index


def load_word_embedding_weights():
    w = None
    with open(word_embedding_weights, 'rb') as infile:
        w = pickle.load(infile)
    if w is None:
        raise Exception('warning: GLOVE weights loading failed')
    return w

if __name__ == '__main__':
    print ConfigReader.ConfigReader().get('file', 'root')
    #__gen_docs()
    #__classify_docs_age()
    #__classify_docs_gender()
    #__gen_train_test_data()
    #__gen_GLOVE_weights()
    __gen_word_index_doc()



