# SmartAP

A python project for Author Profiling.

## PythonPackage: Config

### ConfigReader

The ConfigReader is used for reading the __app.config__, which is the configuration file of this project.

*get(self, key)* __key__ is the key of the field. This function will return the value of the configuration field.

### app.config

This file is the configuration file. Currently it contains all of the resource paths.

## PythonPackage: Data

### DocsContainer

The DocsContainer is used as a collection to save all the documents. A document contains all the tweets that belong to the same author. Each document has a unique ID, so we can retrieve the content.

*__init__(self, docs_dir)* __docs_dir__ is the directory that contains all of the documents.

*retrive_content_in_one(self, docs_id)* __docs_id__ is the unique ID of a document. For a given id, this function returns all of the content within a string

 *retrive_content_in_sentences(self, docs_id)* __docs_id__ is the unique ID of a document. For a given id, this function returns all of the sentences within a list. 

### LabelContainer

The LabelContainer is used as a collection to save all the labels of the documents. There are 2 kinds of labels for the documents. The one is __gender__(0 denotes female and 1 denotes male), and the other is __age__(the nubmer denotes the intervals of different ages).  Each document has a unique ID, so we can retrieve the label.

*__init__(self, path_to_label_file)* __path_to_label_file__ is the address of the file that saves the labels.

*gender_label(self, doc_id)* __doc_id__ is the unique ID of a document. For a given id, this function returns the gender label of the document.

*age_label(self, doc_id)* __doc_id__ is the unique ID of a document. For a given id, this function returns the age label of the document.

*age_label_vector(label)* __label__ is the number that denotes the interval of an age. This function returns a numpy array that represents the age. As there are 5 classes in total for ages, the returned value is a 5-dimensional numpy array.

*gender_label_vector(label)* __label__ is the number that denotes the gender. This function returns a numpy array that represents the gender. As there are 2 classes in total for genders, the returned value is a 2-dimensional numpy array.

### Stemmer & Tokenizer

The Stemmer and Tokenizer is used for doing the stemming and tokenizing job. 

### Util

Please read the comments.

### Parse

The Parse.py contains the functions that can parse the XML source files to get the useful information.

*retrieve_from_xml(xml_content)* __xml_content__ is the content of an XML file. This function will yield the sentences of the given file.

### prepare_data.py

A script that generates all the data.

This file contains multiple functions. Please read the comments.

## PythonPackage: Model

### gender_CNN_LSTM

A script that defines the neural networks.

__maxlen__ The time steps of the LSTM
__embedding_size__ The dimension of the word embeddings.
__filter_length__ The extension (spatial or temporal) of each filter.
__nb_filter__ Number of convolution kernels to use (dimensionality of the output).
__pool_length__ Size of the region to which max pooling is applied
__lstm_output_size__  Dimension of the internal projections and the final output.

#### Convolutional1D

Convolution operator for filtering neighborhoods of one-dimensional inputs. When using this layer as the first layer in a model, either provide the keyword argument input_dim.

The input is the concatenated word embeddings for a document. The parameters of the CNNs are as above.

#### MaxPooling1D

Max pooling operation for temporal data.
The parameters are as above.

#### LSTM

Long-Short Term Memory unit.
The parameters are as above.

#### Dense

A regular fully connected NN layer used for classifying the final result.


##PythonPackage w2v

### EmbeddingContainer

The EmbeddingContainer is used for saving all of the word embeddings. For a given word, it returns a numpy array. Please note that if the word is not found, it will return a all-zero numpy array.

*__init__(self, path_to_model, _binary=True)* __path_to_model__ is the address of the trained word2vec model. __binary__ is the format of the file.

*contains_key(self, k)* __k__ is the word. This function judges if the word is saved in the vocab.

*look_up(self, k)* __k__ is the word. This function returns the word embedding of the given word. Please note that the returned numpy array would be all zeros if the word is not included.
