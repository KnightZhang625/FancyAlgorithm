import sys
import nltk
import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import metrics

def word2feature(sent, i):
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

def sent2features(sent):
    return [word2feature(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

if __name__ == '__main__':
    corpus = nltk.corpus.conll2002.fileids()
    train_set = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_set = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

    X_train = [sent2features(s) for s in train_set]
    y_train = [sent2labels(s) for s in train_set]
    X_test = [sent2features(s) for s in test_set]
    y_test = [sent2labels(s) for s in test_set]
    
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    sorted_labels = sorted(labels, key=lambda name : (name[1:], name[0]))
    print(labels)
    print(sorted_labels)
    labels.remove('O')
    sys.exit()
    y_pred = crf.predict(X_test)
    print(y_pred)
    print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))