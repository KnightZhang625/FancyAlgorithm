import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tag_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)

        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        self.transitions.data[:, self.tag_to_ix['START']] = -100000     # data required
        self.transitions.data[self.tag_to_ix['STOP'], :] = -100000

        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        '''
            sentence : input sentence, represented by one-hot-encoding
            return : lstm_out which represents the emission probability
        '''
        # 1. initialize the hidden layer, no reason, but for the initial step of lstm
        #    initial hidden layer required
        self.hidden = self.init_hidden()
        # 2. embedding the sentence
        #    len(sentence) * embedding_dim, the second dimension is just for batch size                      
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # 3. lstm_hidden : 1 * 1 * hidden_dim  lstm_out : len(sentence) * 1 * hidden_dim
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # 4. lstm_out : len(sentence) * hidden_dim
        lstm_out = lstm_out.view(len(sentence), -1)
        # 5.  each sentence should be 1 * tag_size, which indicates emission probability
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        '''
            recall CRF loss :
            loss = log(exp(Score) / sum (exp(Scores))) = Score - log(sum(exp(Scores)))
            return : Score
        '''
        # 1. define the score as torch format
        score = torch.zeros(1)
        # 2. as tag sequence is in the torch format, we need to catenate the start with the following tags
        tags = torch.cat([torch.tensor([self.tag_to_ix['START']], dtype=torch.long), tags])
        # 3. loop each word
        for i, feat in enumerate(feats):
            # transition score : tag[i] -> tag[i+1]
            # feat[tags[i+1]] : emission score at this timestep
            score = score + self.transitions[tags[i], tags[i+1]] + feat[tags[i+1]]
        score = score + self.transitions[tags[-1], self.tag_to_ix['STOP']]
        return score
    
    def _forward_alg(self, feats):
        # 1. initialize the start timestep score, as the sequnce starts from START, so initialize with -10000
        init_alphas = torch.full((1, self.tag_size), -10000)
        # 2. the START tag is different
        init_alphas[0, self.tag_to_ix['START']] = 0
        forward_var = init_alphas

        # loop each timestep
        for feat in feats:
            alphas_t = []
            for tag in range(self.tag_size):
                emit_score = feat[tag].view(1, -1).expand(1, self.tag_size)
                trans_score = self.transitions[:, tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[:, self.tag_to_ix['STOP']]
        return self.log_sum_exp(terminal_var)
    
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
    
    def _viterbi_decode(self, feats):
        backpointers = []
        init_alphas = torch.full((1, self.tag_size), -10000)
        init_alphas[0][self.tag_to_ix['START']] = 0
        forward_var = init_alphas

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for tag in range(self.tag_size):
                next_tag_var = forward_var + self.transitions[:, tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[:, self.tag_to_ix['STOP']]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        best_path.pop()
        best_path.reverse()
        return path_score, best_path
    
    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
    
    def argmax(self, vec):
        _, idx = torch.max(vec, 1)
        return idx.item()

    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

if __name__ == '__main__':

    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    # 1. get word and index paris
    train_x = []
    train_y = []
    for x, y in training_data:
        train_x.extend(x)
        train_y.extend(y)
    train_y.extend(['START', 'STOP'])
    vocabulary = set(train_x)
    tags = set(train_y)
    word_ix = {word : index for index, word in enumerate(vocabulary)}
    ix_word = {index : word for index, word in enumerate(vocabulary)}
    tag_ix = {tag : index for index, tag in enumerate(tags)}
    ix_tag = {index : tag for index ,tag in enumerate(tags)}

    # 2. transfer the data to the tensor
    train_x = []
    train_y = []
    for x, y in training_data:
        train_x.append(prepare_sequence(x, word_ix))
        train_y.append(prepare_sequence(y, tag_ix))
    
    # 3 build the model
    model = BiLSTM_CRF(len(word_ix), tag_ix, 5, 4)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
  
    for epoch in range(300):
        for sentence, tags in training_data:
            model.zero_grad()

            sentence_in = prepare_sequence(sentence, word_ix)
            targets = prepare_sequence(tags, tag_ix)

            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()
            print(loss.item())
    
    test_seq = prepare_sequence(training_data[0][0], word_ix)
    score, tag_index = model(test_seq)
    for i in tag_index:
        print(ix_tag[i], end=' ')