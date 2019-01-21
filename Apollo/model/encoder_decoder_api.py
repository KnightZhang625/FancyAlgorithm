# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   14/Jan/2019
# @Last Modified by:    
# @Last Modified time:

'''
    this module is used for seq2seq task, adding the attention mechanism
    this package treat each tensor with three dimensions, including one dimenstion representing batch
    however, further mini-batch training is still studied by the author
'''

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.divice('cude' if torch.cuda.is_available() else 'cpu')       # determine each tensor variable's device parameter
SOS_TOKEN = 0                                                               # each sentence starts from 'SOS'
EOS_TOKEN = 1                                                               # each sentence ends by 'EOS'
MAX_LENGTH = 10                                                             # for decoder, max length need to be set if 'EOS' tag not occur

############################################# Encoder Module ############################################
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, bidirectional=False, num_layer=1):
        '''
            args : 
                vocab_size : the size of the whole words
            return :
                output : (1, 1, hidden_dim * 2 if bidirectional else 1)
                hidden : (num_layer * 2 if bidirectional else 1 , 1, hidden_dim)
        '''
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layer = num_layer

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, bidirectional=bidirectional, num_layer=num_layer)
    
    def forward(self, input, prev_hidden):
        '''
            transport each word of the sequence one by one,
            Step 1 : embedding
            Step 2 : send the embedded tensor to the gru module
        '''
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, prev_hidden)
        return output, hidden
    
    def init_hidden(self):
        num_directions = 2 if self.bidirectional else 1
        return torch.zeors(num_directions * self.num_layer, 1, self.hidden_dim, device=device)
#########################################################################################################

############################################# Decoder Module ############################################
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, dropout_p=0.1, max_length=MAX_LENGTH):
        '''
            args : 
                vocab_size : the size of the decoder words, vocab_size is equal to the output_size
            return :
                output : (1, vocab_size)
                hidden : (1, 1, hidden_size)
                attn_weights : (1, max_length)
        '''
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

        # used for combining the embedding and prev_hidden, which will be used for calculating the attn weights,
        # the output dimension is max_length, because the sentences are with variable length, so that we need a full attn_weights,
        # where the maximum length sentence will use all the attn_weights, the others will use part of them,
        # NOTICE : for encoder_outputs, we have variable length in the first dimenstion,
        #          so that creating a fixed length initial encoder outputs is required, see more details in the following code
        self.embedded_hidden_combine = nn.Linear(hidden_dim * 2, max_length)

        # used for creating the actual input for the gru module,
        # combine the embedded and the encoder outputs which has been processed by the attn_weights
        self.attn_embedded_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input, prev_hidden, encoder_outputs):
        # Step 1 : send the input to the embedding module
        embedded = self.embedding(input).view(1, 1, -1)
        embedded_drop_out = self.dropout(embedded)

        # Step 2 : calculate the attn_weights
        ## 2.1 : contanetate the embedded_drop_out with the prev_hidden, both of them are three dimensions
        embedded_hidden_cat = torch.cat(embedded_drop_out[0], prev_hidden[0], 1)    # (1, hidden_dim * 2)
        ## 2.2 : send the contanetated result to the NN module
        embedded_hidden = self.embedded_hidden_combine(embedded_hidden_cat)         # (1, max_length)
        ## 2.3 : calculate the attn_weights
        attn_weights = F.softmax(embedded_hidden, dim=1)

        # Step 3 : multiply the attn_weights with the encoder_outputs
        #          attn_weights : (1, max_length) after unsqueeze --->>> (1, 1, max_length)
        #          encoder_outputs : (max_length, hidden_dim) after unsqueeze --->>> (1, max_length, hidden_dim)  
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))   # (1, 1, hidden_dim)
        
        # Step 4 : combine the attn_applied with the encoder, which will be the actual input for the gru module
        attn_embedded_cat = torch.cat(attn_applied[0], embedded[0], 1)                          # (1, hidden_dim * 2)
        attn_embedded = F.relu(self.attn_embedded_combine(attn_embedded_cat)).view(1, 1, -1)    # (1, 1, hidden_dim)

        # Step 5 : send them into the gru module
        output, hidden = self.gru(attn_embedded)

        # Step 6 : output the result
        output = F.log_softmax(self.out(output[0]), dim=1)      # (1, vocab_size)
        return output, hidden, attn_weights
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)
#########################################################################################################

############################################# Training Module ############################################
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, deocoder_optimzer, 
          criterion, max_length=MAX_LENGTH, teach_forcing_ratio=0.5):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    deocoder_optimzer.zero_grad()

    input_length = input_tensor.size(0)         # for encoder loop
    target_length = target_tensor.size(0)       # for decoder loop

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)   # initialize the encoder_outputs with max_length

    loss = 0

    # 1. Encoder
    for seq_idx in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[seq_idx], encoder_hidden)
        encoder_outputs[seq_idx] = encoder_output[0, 0]
    
    # 2. Decoder
    decoder_input = torch.tensor([[EOS_TOKEN]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teach_forcing_ratio else False

    if use_teacher_forcing:
        for seq_idx in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[seq_idx])
            decoder_input = target_tensor[seq_idx]      # using teacher_forcing, transport the right word into the next time
    else:
        for seq_idx in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topI = decoder_output.topk(1)
            decoder_input = topI.squeeze().detach()                     # not sure why detach
            loss += criterion(decoder_output, target_tensor[seq_idx])
            if decoder_input.item() == EOS_TOKEN:
                '''
                    because in teacher_forcing, looping only lasts target_length time, we input each word from the right target
                    however, without teacher_forcing, we never know when the target is over, so by EOS_TOKEN
                '''
                break
    
    loss.backward()
    encoder_optimizer.step()
    deocoder_optimzer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every, learning_rate=0.01):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.NLLLoss()
    loss_total = 0

    for iter in range(1, n_iters+1):
        # input_tensor = 
        # output_tensor =

        # loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        # loss_total += loss

        # if iter % print_every == 0:
        #     print(loss_total)
##########################################################################################################
