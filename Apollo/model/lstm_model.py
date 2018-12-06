# Author : KnightZhang
# Thanks to the tutorial from the Pytorch
# Double Salute

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM_Tagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, target_size):
        super(LSTM_Tagger, self).__init__()

        self.hidden_dim = hidden_dim // 2
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

        self.hidden = self.initial_hidden()
    
    def initial_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))
    
    def forward(self, X):
        embeds = self.embedding(X)
        lstm_out, lstm_hidden = self.lstm(embeds.view(len(X), 1, -1), self.hidden)
        tag_out = self.hidden2tag(lstm_out.view(len(X), -1))
        predict = F.log_softmax(tag_out, dim=1)
        return predict

class LSTM_API(object):
    def __init__(self):
        pass
    
    def make_tensor(self, seq, to_ix):
        idxs = [to_ix[n] for n in seq]
        return torch.tensor(idxs, dtype=torch.long)
    
    def build_dictionary(self, data):
        tag_to_ix = {tag : i for i, tag in enumerate(data)}
        ix_to_tag = {i : tag for i, tag in enumerate(data)}
        return tag_to_ix, ix_to_tag
    
    def buildModel(self, vocab_size, embedding_dim, hidden_dim, target_size, lr):
        model = LSTM_Tagger(vocab_size, embedding_dim, hidden_dim, target_size)
        loss_fucntion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        return {'model': model, 'loss_function': loss_fucntion, 'optimizer': optimizer}
    
    def start(self, data, parameters, epoch, word_to_ix, tag_to_ix):
        model = parameters['model']
        loss_function = parameters['loss_function']
        optimizer = parameters['optimizer']

        for epo in range(epoch):
            for sentence, tags in data:
                model.zero_grad()
                model.hidden = model.initial_hidden()
                
                X = self.make_tensor(sentence, word_to_ix)
                y_true = self.make_tensor(tags, tag_to_ix)

                predict = model(X)
                loss = loss_function(predict, y_true)
                print(loss.item())
                loss.backward()
                optimizer.step()
        return model

if __name__ == '__main__':
    training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])]

    word_all = []
    tag_all =[]

    for sent, tags in training_data:
        for word in sent:
            word_all.append(word)
        for tag in tags:
            tag_all.append(tag)

    word_all = list(set(word_all))
    tag_all = list(set(tag_all))
    
    lstm = LSTM_API()
    word_to_ix, ix_to_word = lstm.build_dictionary(word_all)
    tag_to_ix, ix_to_tag = lstm.build_dictionary(tag_all)
    
    dic = lstm.buildModel(len(word_all), 50, 30, len(tag_all), 0.1)
    model = lstm.start(training_data, dic, 100, word_to_ix, tag_to_ix)

    predict = model(lstm.make_tensor(training_data[0][0], word_to_ix))
    predict = predict.data.numpy()
    tag_index = np.argmax(predict, axis=1)
    for tag in tag_index:
        print(ix_to_tag[tag])