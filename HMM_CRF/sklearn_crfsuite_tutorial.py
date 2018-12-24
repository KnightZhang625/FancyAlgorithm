# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   23/Dec/2018
# @Last Modified by:    
# @Last Modified time:  

import sys
import nltk
import numpy as np
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

# the function below specifies the features to be extracted
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    # saving the features into the dictionary
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),       # cheak whether each word start with capital word
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    # using the previous word
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]       # list contains dictionaries

def sent2labels(sent):
    return [label for token, postag, label in sent]                 # list contains true NERs

def sent2tokens(sent):
    return [token for token, postag, label in sent]                 # list contains words

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

if __name__ == '__main__':
    # 1.download the corpus first, if downloaded, comment it
    '''
    nltk.download('conll2002')
    data = nltk.corpus.conll2002.fileids()
    '''

    # 2. acquire the training data and testing data
    ## the data is a list consists of a couple of training data
    ## each data has the same format as : [('Word_1', 'PoS', 'NER'), ... , ('Word_N', 'PoS', 'NER')]
    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

    # 3. get one example feature
    '''
    feature_sample =  sent2features(train_sents[0])         # list contains multiple feature dictionaries
    print(feature_sample)
    '''

    # 4. extract all the features
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    # 5. train the model
    crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # 6. evaluate the model
    labels = list(crf.classes_)
    labels.remove('O')

    '''
    show some example testing data
    X_test = [[('entre', 'SP', '',), ('máquinas', 'NC', '')], [('minerales', 'AQ', '')]]
    y_pred = crf.predict(X_test)
    print(y_pred)
    sys.exit()
    '''
    results = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    print(results)

    '''
    from sklearn.preprocessing import MultiLabelBinarizer
    multi_label_binarizer = MultiLabelBinarizer().fit(y_test)
    y_test_bi =  multi_label_binarizer.transform(y_test)
    y_pred_bi = multi_label_binarizer.transform(y_pred)
    print(classification_report(y_pred_bi, y_test_bi))
    print(multi_label_binarizer.classes_)
    '''

    # 7. GridSearchCV
    '''
    crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
    )
    params = {'c1': np.linspace(1, 10, num=2),
              'c2': np.linspace(0.1, 1, num=2),}
    f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)
    rs = GridSearchCV(crf, params,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)

    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    '''

    # 8. display the graph
    from collections import Counter

    print("Top likely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common(20))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])
