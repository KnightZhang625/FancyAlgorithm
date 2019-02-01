# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   30/Jan/2019
# @Last Modified by:    
# @Last Modified time:

import os
import sys
from pathlib import Path
cur_dir = str(Path(__file__).absolute().parent.parent)
os.sys.path.insert(0, cur_dir)

import torch
import torch.nn.functional as F
import torch.nn as nn

from Utils.utils import padding_function, prepare_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PADDING_IDX = 0

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, padding_idx, num_layers=1):
        '''
            vocab_size : number of words from source language
            embedding_dim : dimension for vocab
            hidden_dim : dimension for GRU output
            num_layer : the number of layer in each GRU cell
        '''
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim // 2, bidirectional=True, num_layers=num_layers)

        self.init_hidden()
    
    def forward(self, X, X_lengths):
        embedded = self.embedding(X)
        batch_size, seq_length, _ = embedded.size()
        embedded = embedded.view(seq_length, batch_size, -1)
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, X_lengths)
        hidden = self.init_hidden()
        outputs, hidden = self.gru(embedded, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs.view(seq_length, batch_size, -1)
    
    def init_hidden(self):
        return torch.zeros(2 * self.num_layers, self.batch_size, self.hidden_dim // 2, device=device)

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PADDING_IDX)
        self.dropout = nn.Dropout(dropout_p)

        self.query_key_combine = nn.Linear(hidden_dim * 2, 1)
        self.attn_value_combine = nn.Linear(hidden_dim *2, hidden_dim)
        self.attn_concat = nn.Linear(hidden_dim, 1)
        self.attn_embedded = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, pre_hidden, encoder_outputs):
        '''
            input : (2, 1)
            pre_hidden : (1, 2, hidden_dim)
            encoder_outputs : (seq_length, batch_size, hidden_dim)
        '''
        batch_size, _ = input.size()
        seq_length = encoder_outputs.size()[0]
        embedded = self.embedding(input).view(1, batch_size, -1)    # (1, 2, hidden_dim)
        embedded_drop_out = self.dropout(embedded)

        # use concat method to calculate the attention weights
        pre_hidden_expand = pre_hidden.expand(seq_length, batch_size, -1)
        query_key = torch.cat((encoder_outputs, pre_hidden_expand), dim=2)
        attn_weights = torch.tanh(self.attn_value_combine(query_key)).view(batch_size, seq_length, -1)
        attn_weights = self.attn_concat(attn_weights).view(batch_size, 1, -1)   # (2, 1, 6)
        
        attn_value = attn_weights.bmm(encoder_outputs.view(batch_size, seq_length, -1)).view(1, batch_size, -1) # (2, 1, 4)
        attn_embedded_combine = torch.cat((attn_value, embedded_drop_out), dim=2)
        attn_embedded_combine = self.attn_embedded(attn_embedded_combine)

        output, hidden = self.gru(attn_embedded_combine, pre_hidden)
        output = self.out(output.view(batch_size, 1, -1))
        output = F.log_softmax(output, dim=2)
        return output, hidden
        
        
        

# class Train(object):
#     def __init__(self, model, train_x, train_y, batch_size, word_ix, tag_ix, device):
#        '''
#             train_x : just raw X data
#             train_y : just raw y data
#        '''
#         self.model = model
#         self.train_x = train_x
#         self.train_y = train_y
#         self.batch_size = batch_size
#         self.word_ix = word_ix
#         self.tag_ix = tag_ix
#         self.device = device

if __name__ == '__main__':
    training_data = [('my name is John'.split(), '我 的 名字 是 约翰'.split()), 
                     ('their names are bob and mike'.split(), '他们 的 名字 是 鲍勃 和 迈克'.split())]

    eng_vocab, chs_vocab = [], []
    train_x, train_y = [], []
    for data in training_data:
        eng_vocab.extend(data[0])
        chs_vocab.extend(data[1])
        train_x.append(data[0])
        train_y.append(data[1])

    eng_vocab = list(set(eng_vocab))
    eng_vocab.insert(PADDING_IDX, '*')
    chs_vocab = list(set(chs_vocab))
    chs_vocab.insert(0, 'EOS')
    chs_vocab.insert(0, 'SOS')
    eng_idx = {word : index for index, word in enumerate(eng_vocab)}
    idx_eng = {value : key for key, value in eng_idx.items()}
    chs_idx = {word : index for index, word in enumerate(chs_vocab)}
    idx_chs = {value : key for key, value in chs_idx.items()}
    
    # train_x, X_lengths = padding_function(train_x, eng_idx, device)
    
    # build the encoder model
    encoder_params = { 'vocab_size' : len(eng_vocab),
                       'embedding_dim' : 4,
                       'hidden_dim' : 4,
                       'batch_size' : 2,
                       'num_layer' : 1 }
    decoder_params = { 'vocab_size' : len(chs_idx),
                       'hidden_dim' : 4 } 
    
    encoder_model = Encoder(encoder_params['vocab_size'], encoder_params['embedding_dim'],
                    encoder_params['hidden_dim'], encoder_params['batch_size'], PADDING_IDX)
    decoder_model = Decoder(decoder_params['vocab_size'], decoder_params['hidden_dim'], )
    batch_size = encoder_params['batch_size']
    
    for i in range(0, len(train_x), encoder_params['batch_size']):
        batch_x = train_x[i: i+batch_size]
        batch_y = train_y[i: i+batch_size]
        x_padding, X_lengths = padding_function(batch_x, eng_idx, device)
        target = prepare_sequence(batch_y, chs_idx, device)
        target = sorted(target, key=lambda t : t.size()[-1], reverse=True)
        encoder_outputs = encoder_model(x_padding, X_lengths)
        
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        prev_hidden = encoder_outputs[-1, ].unsqueeze(0)
   
        for i in range(decoder_input.size()[0]):
            decoder_input[i, :] = torch.tensor(chs_idx['EOS'], dtype=torch.long, device=device)
        decoder_model(decoder_input, prev_hidden, encoder_outputs)
        
        max_length = 10
        outputs = torch.zeors
        for i in range(10):
            decoder_output, prev_hidden = decoder_model(decoder_input, prev_hidden, encoder_outputs)
            