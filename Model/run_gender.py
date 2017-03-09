import os
import Data.DocsContainer as DC
import Data.LabelContainer as LC
from Data.Util import read_lines_from_file, spllit_sentences, shuffle_x_y, save_on_batch
from Model.gender_CNN_LSTM import run_model, verify_model, train_model_on_batch, eval_model_on_batch
from Config import ConfigReader

doc_dir = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/docs')
indexed_docs_dir = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/indexed_docs')
truth_file = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/truth/n_truth.txt')
train_id_file = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/gender/train_docsid.txt')
test_id_file = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/gender/test_docsid.txt')
w2v_file = ConfigReader.ConfigReader().get('word2vec', 'model_path')
glove_file = ConfigReader.ConfigReader().get('glove', 'model_path')
batch_dir = os.path.join(ConfigReader.ConfigReader().get('file', 'root'), 'pan_dataset/batch')
hyper_batch_size = int(ConfigReader.ConfigReader().get('settings', 'hyper_batch_size'))


def prepare_training(ids, dc, lc):
    x_sentences = map(lambda x: dc.retrieve_sequences(x), ids)
    y_label_doc = map(lambda x: lc.gender_label(x), ids)
    x = []
    y = []
    for i in xrange(len(x_sentences)):
        for j in xrange(len(x_sentences[i])):
            x.append(x_sentences[i][j])
            y.append(y_label_doc[i])
    return x, y


def prepare_testing(ids, dc, lc):
    x_docs = map(lambda x: dc.retrieve_sequences(x), ids)
    y_test = map(lambda x: lc.gender_label(x), ids)
    return x_docs, y_test


def prepare_batch_data():
    dc = DC.IndexedDocsContainer(docs_dir=indexed_docs_dir)
    lc = LC.LabelContainer(truth_file)
    print "resources loaded"

    x_train_ids = read_lines_from_file(train_id_file)
    x_train, y_train = prepare_training(x_train_ids, dc, lc)
    print "training data prepared"

    x_test_ids = read_lines_from_file(test_id_file)
    x_test, y_test = prepare_testing(x_test_ids, dc, lc)
    print "testing data prepared"

    if not os.path.exists(batch_dir):
        os.mkdir(batch_dir)
    save_on_batch(batch_dir, (x_train, y_train), 'train', hyper_batch_size)
    save_on_batch(batch_dir, (x_test, y_test), 'test', hyper_batch_size)
    return x_train, y_train, x_test, y_test


def run_gender_on_batch():
    train_model_on_batch(batch_dir)


def verify():
    verify_model()


if __name__ == '__main__':
    #prepare_batch_data()
    run_gender_on_batch()
    eval_model_on_batch(batch_dir)
    #verify()

