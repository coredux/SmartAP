import os
import Stemmer
import Tokenizer
import w2v.EmbeddingContainer
import LabelContainer
from Util import list_filenames_from_dir, write_list_to, read_content_from_file,read_lines_from_file, string_filter
from Util import word_purify, randomly_choose_train_test, shuffle
from Parse import retrieve_from_xml
from Config import ConfigReader

src_dir = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/dataset')
docs_dir = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/docs')
w2v_model_path = ConfigReader.ConfigReader().get('word2vec', 'model_path')
label_file_path = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/truth/n_truth.txt')

data_dir_preffix = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset')


'''
step1: generate docs from the source dataset
'''


def __gen_docs():
    tk = Tokenizer.Tokenizer()
    stm = Stemmer.Stemmer()
    embedder = w2v.EmbeddingContainer.EmbeddingContainer(w2v_model_path)
    print "begin"
    for full_name in list_filenames_from_dir(src_dir):
        filename = full_name.split('.')[0]
        affix = full_name.split('.')[1]
        if affix == 'xml':
            lines = [ str.join( ' ', filter( embedder.contains_key, map( stm.stem, map( word_purify, tk.tokenize(string_filter(x))))))
                      for x in retrieve_from_xml(read_content_from_file(os.path.join(src_dir, full_name)))]
            write_list_to(os.path.join(docs_dir, filename + '.txt'), lines)


'''
step2: extracting male and female
'''
def __classify_docs_gender():
    lc = LabelContainer.LabelContainer(label_file_path)
    male_docid = [x for x in lc.labelDict if lc.labelDict[x]['gender'] == 1]
    female_docid = [x for x in lc.labelDict if lc.labelDict[x]['gender'] == 0]
    write_list_to(os.path.join(data_dir_preffix, 'gender/male.txt'), male_docid)
    write_list_to(os.path.join(data_dir_preffix, 'gender/female.txt'), female_docid)


'''
step2: extracting all kinds of ages
'''
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


'''
step3: randomly generate the training and testing data (only doc_id)
'''
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

if __name__ == '__main__':
    print ConfigReader.ConfigReader().get('file', 'root')
    __gen_docs()
    __classify_docs_age()
    __classify_docs_gender()
    __gen_train_test_data()



