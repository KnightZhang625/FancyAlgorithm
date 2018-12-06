# Author : KnightZhang
# The code is inspired by the official CRF tutorial from sklearn
# Double Salute !

import sys
import nltk
import scipy.stats
import sklearn
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import cross_val_score
import sklearn_crfsuite
from sklearn_crfsuite import metrics

class CRF_API(object):
    def __init__(self, algorithm='lbfgs', c1='0.1', c2='0.1', max_iterations=100, all_possible_transitions=True, mode='easy', feature_fucition=None):
        '''
            mode : {easy, complex}, easy: just predict, complex: use k-fold
            feature_function : use customized feature function, see the example below
        '''
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.all_possible_transitions = all_possible_transitions
        self.mode = mode
        self.feature_funciton = feature_fucition
    
    def start(self, data_set, parameters_dict=None, cv=None):
        '''
            parameters_dict = {'c1' : scipy.stats.expon(scale=0.5), 'c2' : scipy.stats.expon(scale=0.05)}
        '''
        X_train = [self._sent2features(sent) for sent in data_set]
        y_train = [self._sent2labels(sent) for sent in data_set]
        if self.mode == 'easy':
            self.model = self._start_easy(X_train, y_train)
        elif self.mode == 'complex':
            self.model = self._start_complex(X_train, y_train, parameters_dict, cv)
        else:
            raise Exception('mode wrong')
        
    def _start_easy(self, X_train, y_train):
        crf = sklearn_crfsuite.CRF(algorithm=self.algorithm, c1=self.c1, c2=self.c2, max_iterations=self.max_iterations, all_possible_transitions=self.all_possible_transitions)
        crf.fit(X_train, y_train)
        return crf

    def _start_complex(self, X_train, y_train, parameters_dict, cv):
        '''
            should set the labels in advance
        '''
        labels = ['B-LOC', 'B-ORG', 'B-PER', 'I-PER', 'B-MISC', 'I-ORG', 'I-LOC', 'I-MISC']
        if parameters_dict is None or cv is None:
            raise Exception('please set the parameters for the complex mode')
        crf = sklearn_crfsuite.CRF(algorithm=self.algorithm, max_iterations=self.max_iterations, all_possible_transitions=self.all_possible_transitions)
        f1_score = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)
        model = RandomizedSearchCV(crf, parameters_dict, cv=cv, n_iter=50, verbose=1, n_jobs=-1, scoring=f1_score)
        model.fit(X_train, y_train)
        print(model.best_params_)
        print(model.best_score_)
        return model
        
    def predict(self, data_set):
        X = [self._sent2features(sent) for sent in data_set]
        y_pred = self.model.predict(X)
        return y_pred
    
    def score(self, data_set):
        X_test = [self._sent2features(sent) for sent in data_set]
        y_test = [self._sent2labels(sent) for sent in data_set]
        y_pred = self.model.predict(X_test)
        self.labels = list(self.model.classes_)     # do not remove directly, it will return None
        self.labels.remove('O')
        return metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=self.labels)

    def _word2feature(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias' : 1.0,
            'word_lower()' : word.lower(),
            'word[-3:]' : word[-3:],
            'word[-2:]' : word[-2:],
            'word.isupper()' : word.isupper(),
        }
        if i > 0:
            word_pre = sent[i-1][0]
            postag_pre = sent[i-1][1]
            features.update({
                '-1:word.lower()' : word_pre.lower(),
                '-1:postag' : postag_pre
            })
        else:
            features['BOS'] = True
        if i < len(sent) - 1:
            word_back = sent[i+1][0]
            postag_back = sent[i+1][1]
            features.update({
                '+1:word.lower()' : word_back.lower(),
                '+1:postag' : postag_back
            })
        else:
            features['EOS'] = True
        return features

    def _sent2features(self, sent):
        if self.feature_funciton:
            self._word2feature = self.feature_funciton
        return [self._word2feature(sent, i) for i in range(len(sent))]

    def _sent2labels(self, sent):
        return [label for token, postag, label in sent]

    def _sent2tokens(self, sent):
        return [token for token, postag, label in sent]

if __name__ == '__main__':
    '''
    Example for the customize feature function
    def function(sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias' : 1.0,
            'word_lower()' : word.lower(),
            'word[-3:]' : word[-3:],
            'word[-2:]' : word[-2:],
            'word.isupper()' : word.isupper(),
        }
        return features
    '''

    corpus = nltk.corpus.conll2002.fileids()
    train_set = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_set = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

    crf = CRF_API(mode='easy')
    crf.start(train_set, parameters_dict={'c1' : scipy.stats.expon(scale=0.5), 'c2' : scipy.stats.expon(scale=0.05)}, cv=3)
    temp = crf.score(test_set)
    print(temp)