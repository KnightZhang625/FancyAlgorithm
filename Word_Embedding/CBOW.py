# written by Jiaxin Zhang
# 07_Nov    
# version 1.0

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear_1 = nn.Linear(context_size*embedding_dim, 128)
        self.linear_2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear_1(embeds)
        out = F.relu(out)
        out = self.linear_2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

class Model(object):
    def __init__(self, vocab, embedding_dim, context_size, learning_rate, epoch):
        try:
            self._vocab = vocab
            self._embedding_dim = embedding_dim
            self._context_size= context_size
            self._learning_rate = learning_rate
            self._epoch = epoch
        except Exception as e:
            raise Exception('model parameters required')
        self._buildModel()
        self._ready = False
    
    def buildData(self, train_data, word_to_ix):
        self._train_data = train_data
        self._word_to_ix = word_to_ix
        self._ready = True
    
    def _buildModel(self):
        self._loss_function = nn.NLLLoss()
        self._model = CBOW(len(self._vocab), self._embedding_dim, self._context_size)
        self._optimizer = optim.SGD(self._model.parameters(), lr=self._learning_rate)
    
    def _start(self):
        losses = []
        for epoch in range(self._epoch):
            total_loss = 0
            for context, target in self._train_data:
                context_idxs = torch.tensor([self._word_to_ix[w] for w in context], dtype=torch.long)
                self._model.zero_grad()
                log_probs = self._model(context_idxs)
                loss = self._loss_function(log_probs, torch.tensor([self._word_to_ix[target]]))
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()
            print('The %dst iteration, the loss is %.2f'%(epoch, total_loss))
            losses.append(total_loss)
        print(losses)

    def Engine_Start(self):
        if self._ready:
            self._start()
        else:
            raise Exception('No data available')

