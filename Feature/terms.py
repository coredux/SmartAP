import os
import Data.DocsContainer as DC
from Config import ConfigReader
from Model.run_gender import doc_dir, train_id_file
from Data.Util import read_lines_from_file, write_list_to

gender_vocab_train_file = os.path.join(ConfigReader.ConfigReader().get('file', 'root'),
                                       'pan_dataset/gender/train_gender_vocab.txt')
female_ids_file = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/gender/female.txt')
male_ids_file = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/gender/male.txt')


def gen_vocabulary(docs):
    v = set()
    for doc in docs:
        for s in doc:
            for w in s.decode('utf8').split(' '):
                v.add(w)
    return sorted(v)


def gen_gender_vocabulary():
    dc = DC.DocsContainer(docs_dir=doc_dir)
    x_docs = map(lambda x: dc.retrieve_content_in_sentences(x), read_lines_from_file(train_id_file))
    vocab = gen_vocabulary(x_docs)
    write_list_to(gender_vocab_train_file, vocab)


def gen_gender_vocab_feature_table():
    train_gender_ids = read_lines_from_file(train_id_file)
    male_ids = read_lines_from_file(male_ids_file)
    female_ids = read_lines_from_file(female_ids_file)
    train_male_ids = [x for x in train_gender_ids if x in male_ids]
    train_female_ids = [x for x in train_gender_ids if x in female_ids]


if __name__ == '__main__':
    gen_gender_vocabulary()

