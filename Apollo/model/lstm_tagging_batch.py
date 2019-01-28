# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   27/Jan/2019
# @Last Modified by: Jiaxin Zhang 
# @Last Modified time:  28/Jan/2019

import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

PADDING_IDX = 0
BATCH_SIZE = 2

class Bi_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, target_size, batch_size, bidirectional=True, num_layers=1):
        super(Bi_LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        self.batch_size = batch_size
        self.bidirectional = 2 if bidirectional else 1      # use for determine the first dimension for the lstm outputs
        self.num_layers = num_layers

        # build the embedding layer
        # because of batch training, need to set the 'padding_idx' parameter
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_IDX)

        # build the lstm layer
        # if using the bi-lstm, the output for the output will be double hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=num_layers) if bidirectional \
                    else nn.LSTM(embedding_dim, hidden_dim, bidirectional=False, num_layers=num_layers)

        # build the output layer
        if bidirectional:
            self.output = nn.Linear(hidden_dim*2, target_size)
        
        # initialize the hidden layer
        self.init_hidden()
    
    def forward(self, X, X_lengths, training=True):
        '''
            X : (batch_size, seq_length, 1), 1 indicates each word is represented by one-hot-encoding
            X_lengths : the actual length for each data
        '''
        # 1. embedding the input
        embedded = self.embedding(X)       # (batch_size, seq_length, 1) ---> (batch_size, seq_length, embedding_dim)
        # 2. obtaini the batch_size, seq_length
        batch_size, seq_length, _ = embedded.size()
        if training:
            assert batch_size == self.batch_size, print('input batch size not match the set batch size of the model')
        # 3. the default input for the lstm, the batch_size should be at the second dimension
        ##   although 'batch_first' parameter could be used, still transferring the size of the embedded
        embedded = embedded.view(seq_length, batch_size, -1)    # (seq_length, batch_size, embedding_dim)
        # 4. initialize the hidden layer for lstm
        if training:
            hidden = self.init_hidden()
        else:
            hidden = self.init_hidden_test()
        # 5. pack the input
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, X_lengths)
        outputs, hidden = self.lstm(embedded, hidden)   # outputs : (seq_length, batch_size, hidden_dim)
        # 6. pad the outputs
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # 7. outputs the results
        ##   before sending the outputs into the output layer, put the batch_size to the first dimension
        outputs = outputs.view(batch_size, seq_length, -1)
        outputs = self.output(outputs)      # outputs (batch_size, seq_length, target_size)
        outputs = F.softmax(outputs, dim=2)    # dim=2 means calculating the softmax on the third dimension
        return outputs

    def calculate_loss(self, y_true, y_pred, lengths, loss_func):
        # build the mask, which saves the actual length of each data
        mask = [[1 for i in range(length)] for length in lengths]

        # initialize the y_true and the prediction
        # y_true : tensor(t1, t2, t3) 
        # y_pred_tensor : tensor((t1_1, t1_2, t1_3 ... t1_target_size), (t2_1, t2_2, t2_3 ... t2_target_size), (t3_1, t3_2, t3_3 ... t3_target_size))
        y_true_tensor = torch.tensor(y_true[0])
        y_pred_tensor = y_pred[0,]              # the item in y_pred is always tensor

        # concatenate the initial y_true and y_pred with respective left data
        for i in range(1, BATCH_SIZE):
            y_true_temp = torch.tensor(y_true[i], dtype=torch.long)
            y_true_tensor = torch.cat((y_true_tensor, y_true_temp), dim=0)
     
        for i in range(1, BATCH_SIZE):
            actual_length = sum(mask[i])
            y_pred_tensor = torch.cat((y_pred_tensor, y_pred[i, :actual_length]), dim=0)

        loss = loss_func(y_pred_tensor, y_true_tensor)
        return loss

    def init_hidden(self):
        return (torch.randn(self.bidirectional * self.num_layers, self.batch_size, self.hidden_dim),
                torch.randn(self.bidirectional * self.num_layers, self.batch_size, self.hidden_dim))

    def init_hidden_test(self):
        '''
            if predicting one, set the batch_size equal to 1,
            only used for testing
        '''
        return (torch.randn(self.bidirectional * self.num_layers, 1, self.hidden_dim),
                torch.randn(self.bidirectional * self.num_layers, 1, self.hidden_dim))

def padding(train_x, y_true):
    lengths = [len(x) for x in train_x]
    max_length = max(lengths)

    # rnn.pack_padded_sequence requires the lengths sorted by descending order
    # so that the order of train_x, y_true should match the lengths
    train_x = sorted(train_x, key=lambda x : len(x), reverse=True)
    y_true = sorted(y_true, key=lambda y : len(y), reverse=True)
    lengths = sorted(lengths, reverse=True)
    
    # build a initial x padding tensor
    x_padding = torch.zeros((len(train_x), max_length), dtype=torch.long)
    for idx, data in enumerate(train_x):
        actual_length = len(data)
        x_padding[idx, :actual_length] = torch.tensor(data, dtype=torch.long)
    return x_padding, y_true, lengths

if __name__ == '__main__':
    training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
    ('The cat barks'.split(), ['DET', 'NN', 'V']),
    ('The dog barks'.split(), ['DET', 'NN', 'V']),
    ('The boy plays the basketball'.split(), ['DET', 'NN', 'V', 'DET', 'NN']),
    ('The girl love the the boy'.split(), ['DET', 'NN', 'V', 'DET', 'DET', 'NN'])
    ]

    # build the word_idx, tag_idx
    vocab = []
    tags = []
    for word, tag in training_data:
        vocab.extend(word)
        tags.extend(tag)
    vocab = list(set(vocab))
    vocab.insert(0, 'PADDING')      # add the padding flag to the vocab
    tags = list(set(tags))
    word_idx = {word : idx for idx, word in enumerate(vocab)}
    idx_word = {v : k for k, v in word_idx.items()}
    tag_idx = {tag : idx for idx, tag in enumerate(tags)}
    idx_tag = {v : k for k, v in tag_idx.items()}

    # build the training data
    train_x = [[word_idx[w] for w in x]for x, _ in training_data]
    train_y = [[tag_idx[t] for t in y] for _, y in training_data]
    
    # build the model
    model = Bi_LSTM(len(word_idx), 4, 4, len(tag_idx), BATCH_SIZE, bidirectional=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.NLLLoss()

    # training
    train_data_size = len(training_data)
    for epoch in range(100):            
        for i in range(0, train_data_size, BATCH_SIZE):
            train_x_batch = train_x[i : i+BATCH_SIZE]
            x_padding, y_true, lengths = padding(train_x_batch, train_y[i: i+BATCH_SIZE])
            # send the data into the model
            prediction = model(x_padding, lengths)
            # calculate the loss
            loss = model.calculate_loss(y_true, prediction, lengths, loss_function)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)

    # testing
    test_data, _, lengths = padding([train_x[0]], [train_y[0]])
    predicition = model(test_data, lengths, training=False).squeeze(0)
    idx = torch.argmax(predicition, dim=1)
    print([idx_tag[i.item()] for i in idx])