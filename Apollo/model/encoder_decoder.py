# Author : KnightZhang
# Thanks to the tutorial from the Pytorch
# Double Salute

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_size)

        self.hidden = self.initHidden()
    
    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded.view(len(input), 1, -1))
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, target_size, temp, dropout=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.target_size =target_size
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(target_size, hidden_size)
        self.attn = nn.Linear(hidden_size*2, 1)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, temp)

    def forward(self, input, hidden, encoder_outputs):
        # try:
        #     h = torch.cat((hidden[0], hidden[1]), dim=1)
        # except Exception as e:
        #     h = hidden.view(1, -1)
        h = hidden.view(1, -1)
        hidden_expand = h.expand(encoder_outputs.size()[0], self.hidden_size).view(encoder_outputs.size()[0], 1, -1)
        hidden_encoder_combine = torch.cat((hidden_expand, encoder_outputs), 2).view(encoder_outputs.size()[0], -1)     # len(句子) * (2 * hidden_size)

        attn_weights = F.softmax(self.attn(hidden_encoder_combine), dim=1)          # len(句子) * 1

        c = torch.mm(attn_weights.t(), encoder_outputs.view(encoder_outputs.size()[0], -1))                             # 1 * hidden_size
   
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        input_t = torch.cat((embedded[0], c), 1)
        input_combine = self.attn_combine(input_t).view(1, 1, -1)

        output, hidden = self.gru(input_combine, h.view(1, 1, -1))
        output = F.log_softmax(self.out(output[0]), dim=1)
    
        return output, hidden, attn_weights
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

def make_tensor(seq, to_ix):
    idxs = [to_ix[n] for n in seq]
    return torch.tensor(idxs, dtype=torch.long)

def build_dictionary(data):
    tag_to_ix = {tag : i for i, tag in enumerate(data)}
    ix_to_tag = {i : tag for i, tag in enumerate(data)}
    return tag_to_ix, ix_to_tag

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

    # word_all.append('BOS')

    word_to_ix, ix_to_word = build_dictionary(word_all)
    tag_to_ix, ix_to_tag = build_dictionary(tag_all)

    encoder = EncoderRNN(len(word_all), 30, 50)
    encoder_optimezer = optim.SGD(encoder.parameters(), lr=0.05)
    decoder = AttnDecoderRNN(50, len(word_all), len(tag_all))
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.05)
    loss_function = nn.NLLLoss()

    for epoch in range(100):
        for sentence, tags in training_data:
            encoder_optimezer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder.initHidden()
            # decoder.initHidden()
            loss = 0
            X = make_tensor(sentence, word_to_ix)
            # y_true = make_tensor(tags, tag_to_ix)

            encoder_outputs, encoder_hidden = encoder(X)
            # print(encoder_outputs.size())
            # sys.exit()
            # decoder_input = torch.tensor(word_to_ix['BOS'])
            decoder_input = torch.tensor([0])
            decoder_hidden = encoder_hidden

            # for index, word in enumerate(sentence):
            #     encoder_out, encoder_hidden = encoder(make_tensor([word], word_to_ix))
            #     print(encoder_out.size())
            # sys.exit()    

            for index, word in enumerate(sentence):

                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
     
                y_true = torch.tensor([tag_to_ix[tags[index]]], dtype=torch.long)
                print(decoder_output.size())
                loss += loss_function(decoder_output, y_true)
                decoder_input = y_true
            # print(loss.item())

            loss.backward()

            encoder_optimezer.step()
            decoder_optimizer.step()
    
    test_data = training_data[1]
    X = make_tensor(test_data[0], word_to_ix)
    encoder_outputs, encoder_hidden = encoder(X)
    decoder_input = torch.tensor([0])
    decoder_hidden = encoder_hidden
    results = []
    for index, word in enumerate(test_data[0]):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        print(decoder_output)
        i = torch.argmax(decoder_output, dim=1)
        decoder_input = i
        results.append(decoder_output)

    for re in results:
        print(re)
        re_numpy = re.data.numpy()
        print(re_numpy)
        idx = np.argmax(re_numpy, axis=1)[-1]
        print(idx)
        print(ix_to_tag[idx])
    print(ix_to_tag)