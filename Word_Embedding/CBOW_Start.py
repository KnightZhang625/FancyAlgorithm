# written by Jiaxin Zhang
# 07_Nov    
# version 1.0

import os
import sys
import torch
import pickle
import argparse
from argparse import RawTextHelpFormatter
from CBOW import Model
from knight.preprocess import PreprocessAPI
from knight.Info import Info

class Data(object):
    '''
        data should be list, contains lists which consists of clean vocabulary
    '''
    def __init__(self, data_list, ngram):
        info = Info()
        self._data_list = data_list
        self._ngram = ngram
        self._buildData()
        self._WordToIx()

    def _buildData(self):
        self.data_pairs = []
        self.vocabulary = []
        for sentence in self._data_list:
            sentence = sentence.split(' ')
            self.vocabulary.extend(sentence)
            for i in range(self._ngram, len(sentence)-self._ngram):
                context_left = []
                context_right = []
                for n in range(self._ngram, 0, -1):
                    context_left.append(sentence[i-n])
                    context_right.insert(0, sentence[i+n])
                context = context_left + context_right
                target = sentence[i]
                self.data_pairs.append((context, target))
            self.vocabulary = list(set(self.vocabulary))

    def _WordToIx(self):
        self.word_to_ix = {word : i for i, word in enumerate(self.vocabulary)}
        self.ix_to_word = {i : word for i, word in enumerate(self.vocabulary)}

    def getData(self):
        return self.data_pairs, self.word_to_ix, self.ix_to_word, self.vocabulary

def cleanData(data_list):
    preprocess = PreprocessAPI(data_list)
    data = preprocess.preProcess()
    return data

def readDate(path):
    path_split = path.split('/')
    if path_split[-1].split('.')[-1] == 'pickle':
        data = readPickle(path)
    elif path_split[-1].split('.')[-1] == 'txt':
        data = readTxt(path)
    return data

def readPickle(path):
    try:
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
        return data
    except IOError as e:
        print(e)
    else:
        print('load pickle file successfully')

def readTxt(path):
    try:
        f = open(path, 'r')
        data = f.read()
        f.close()
        return data
    except IOError as e:
        print(e)
    else:
        print('load txt file successfully')
    
def Parameter():
    parser = argparse.ArgumentParser(description='\t\t\tCBOW Model\n\t\t\tprodeced by Jiaxin', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'predict'], help='train: train the model from scratch\npredict: use the pre-trained model', required=True)
    parser.add_argument('--embedding', help='determine embedding size', type=int, required=True)
    parser.add_argument('--ngram', help='choose ngram size\nif the training sentence is short, ngram=1 is the best', type=int, default=1)
    parser.add_argument('--lr', help='set learning rate', default=0.001, type=float)
    parser.add_argument('--epoch', help='iteration times', default=1, type=int)
    parser.add_argument('--path', help='select training data path', required=True)
    args = parser.parse_args()
    return {'mode':args.mode, 'embedding_dim':args.embedding, 'ngram':args.ngram, 'lr':args.lr, 'epoch':args.epoch, 'path':args.path}

if __name__ == '__main__':
    # 1. get parmeters
    parameter_dict = Parameter()

    # 2. read data and build data
    data_test = readDate(parameter_dict['path'])[:10]
    data_clean = cleanData(data_test)
    data = Data(data_clean, parameter_dict['ngram'])
    train_data, word_to_ix, ix_to_word, vocabulary = data.getData()
    # print(train_data)
    # sys.exit()
    # 3. build the model
    model = Model(vocabulary, parameter_dict['embedding_dim'], parameter_dict['ngram']*2, parameter_dict['lr'], parameter_dict['epoch'])
    model.buildData(train_data, word_to_ix)
    model.Engine_Start()

    # 4. predict
    context_idxs = torch.tensor([word_to_ix[w] for w in ['鸭汤', '好喝']], dtype=torch.long)
    predict = model._model(context_idxs)
    assert(len(vocabulary) == predict.shape[1])
    _, idx = torch.max(predict, dim=1)
    print(ix_to_word[idx.item()])