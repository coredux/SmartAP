import os, numpy

class LabelContainer:

    labelDict = None

    def __init__(self, path_to_label_file):
        if os.path.exists(path_to_label_file):
            self.labelDict = dict()
            with open(path_to_label_file, 'r') as infile:
                for line in infile:
                    info = line.strip().split('\t')
                    secondary = dict()
                    secondary["gender"] = int(info[1])
                    secondary["age"] = int(info[2])
                    self.labelDict[info[0]] = secondary
        else:
            raise Exception('label file not found')

    def gender_label(self, doc_id):
        if doc_id in self.labelDict:
            return self.labelDict[doc_id]["gender"]
        else:
            return -1

    def age_label(self, doc_id):
        if doc_id in self.labelDict:
            return self.labelDict[doc_id]["age"]

    @staticmethod
    def age_label_vector(label):
        label = int(label)
        if label == 0:
            return numpy.array([1,0]).astype('float32')
        elif label == 1:
            return numpy.array([0,1]).astype('float32')
        else:
            raise Exception('invalid label')

    @staticmethod
    def gender_label_vector(label):
        label = int(label)
        if label == 0:
            return numpy.array([1,0,0,0,0]).astype('float32')
        elif label == 1:
            return numpy.array([0,1,0,0,0]).astype('float32')
        elif label == 2:
            return numpy.array([0,0,1,0,0]).astype('float32')
        elif label == 3:
            return numpy.array([0,0,0,1,0]).astype('float32')
        elif label == 4:
            return numpy.array([0,0,0,0,1]).astype('float32')
        else:
            raise Exception('invalid label')

