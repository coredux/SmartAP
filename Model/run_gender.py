import Data.DocsContainer as DC
import Data.LabelContainer as LC
import w2v.EmbeddingContainer as EC
import numpy as np
from Data.Util import read_lines_from_file
from Model.gender_CNN_LSTM import run_model

doc_dir = 'E:\\data\\pan_dataset\\docs'
truth_file = 'E:\\data\\pan_dataset\\truth\\n_truth.txt'
train_id_file = 'E:\\data\\pan_dataset\\gender\\train_docsid.txt'
test_id_file = 'E:\\data\\pan_dataset\\gender\\test_docsid.txt'
w2v_file = 'E:\\data\\w2vModel\\g300.bin'

def _generate_gender_x_y(docs_ids, dc, lc):
    x_sentences = map(lambda x: dc.retrieve_content_in_sentences(x), docs_ids)
    y_label_doc = map(lambda x: lc.gender_label(x), docs_ids)
    x = []
    y = []
    for i in xrange(len(x_sentences)):
        if len(x_sentences[i]) > 0:
            x.extend(x_sentences[i])
            y.extend([y_label_doc[i]] * len(x_sentences[i]))
    return x,y

def generate_gender_x_y(docs_ids, dc, lc):
    x_sentences = map(lambda x: dc.retrieve_content_in_one(x), docs_ids)
    y_label_doc = map(lambda x: lc.gender_label(x), docs_ids)
    return x_sentences, y_label_doc


if __name__ == '__main__':
    dc = DC.DocsContainer(docs_dir=doc_dir)
    lc = LC.LabelContainer(truth_file)
    ec = EC.EmbeddingContainer(w2v_file)
    print "1"
    x_train_ids = read_lines_from_file(train_id_file)
    x_train, y_train = generate_gender_x_y(x_train_ids, dc, lc)
    x_train = map(lambda x:map(lambda y: ec.look_up(y), x), x_train)
    print "2"

    x_test_ids = read_lines_from_file(test_id_file)
    x_test, y_test = generate_gender_x_y(x_test_ids, dc, lc)
    x_test = map(lambda x:map(lambda y: ec.look_up(y), x), x_test)
    print "3"

    print len(x_train)
    run_model(x_train, y_train, x_test, y_test)




