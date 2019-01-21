
import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

def argmax(vec):
    '''
        find the maximum index of the vector
    '''
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    '''
     not pretty sure why log sum up for each path could be calculated by suming up log_sum_exp for each time step
    '''
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long, device=device)

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, bidirectional=False, num_layers=1):
        # store some useful variables as member variables
        super(BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.target_size = len(tag_to_ix)               # output size for tags
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1 # use for initialize the hidden states for lstm
        self.num_layers = num_layers

        # build some modules for further use
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=num_layers)
        # if bidirectional, which means the outputs(not h_t and h_c) from lstm contains both forward var and backward,
        # it should be double hidden_dim
        self.outputs = nn.Linear(hidden_dim * self.num_directions, self.target_size)

        # initialize the transitions matrix between tags
        self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size, device=device))

        # initialize the hidden states for lstm
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        # two variable which are h_t, h_c respectively,
        # lstm outputs hidden state separately, do not cocatenate the forward and backward var as output does
        # lstm outputs each layer from lstm hidden state
        return (torch.randn(self.num_directions * self.num_layers, 1, self.hidden_dim, device=device),
                torch.randn(self.num_directions * self.num_layers, 1, self.hidden_dim, device=device))
    
    def _get_lstm_emissions(self, sentence):
        '''
            this step is the first step, in order to get the emission socres,
            which actually are the outputs of the lstm from each time step(i.e. each word)
        '''
        # 1. initialize the hidden states for the initial time step
        self.hidden = self.init_hidden()
        # 2. embedded the sentence
        #   input the entire sequence, so the first dimension is the length of the sentence,
        #   the second dimension is the batch, do not consider batch training now
        #   the third dimension is embedding dim
        embedded = self.embedding(sentence).view(len(sentence), 1, -1)
        # 3. input the entire sequence to the lstm module,
        #    could input the word one by one, however, as we use bidirectional lstm,
        #    not pretty sure could be done like this   
        lstm_out, self.hidden = self.lstm(embedded, self.hidden)
        # 4. output the target
        #    lstm_out : (len(sentence), 1, -1), need eliminate the second batch dimension
        emissions = self.outputs(lstm_out.view(len(sentence), -1))
        return emissions
    
    def _forward_algorithm(self, emissions):
        # 1. build the initial score vector, initialze them by the extremely low figures
        #    as having target_size, which means having target_size scores
        init_score = torch.full((1, self.target_size), -10000, device=device)
        # 2. initialize the START_TAG as 0, cause it should have the highest score,
        #    every sentence starts from the START_TAG
        init_score[0, self.tag_to_ix[START_TAG]] = 0
        forward_score = init_score

        for emission_t in emissions:                # loop each time step
            score_t = []                            # store all the scores for each time step
            for tag in range(self.target_size):     # calculate score for each tag in current time step
                transition_score_t = self.transitions[:, tag].view(1, -1)
                emission_score = emission_t[tag].view(1, -1).expand(1, self.target_size)    # broadcast emission score for the transition score
                score_t_tag = forward_score + transition_score_t + emission_score
                score_t.append(log_sum_exp(score_t_tag))
            forward_score = torch.tensor(score_t, dtype=torch.float, device=device)         # update the forward score

        # add the last time step, i.e. the STOP_TAG
        terminal_score = log_sum_exp(forward_score + self.transitions[:, self.tag_to_ix[STOP_TAG]].view(1, -1))
        return terminal_score
    
    def _sentence_score(self, emissions, tags):
        # initialize the score as tensor type, cause it should particapate the backpropagation
        score = torch.zeros(1, device=device)
        # concatenate the start tag with the tags
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long, device=device), tags])
        for i, emission_t in enumerate(emissions):
            score = score + self.transitions[tags[i], tags[i+1]] + emission_t[tags[i+1]]
        score += self.transitions[tags[-1], self.tag_to_ix[STOP_TAG]]
        return score
    
    def _viterbi_decode(self, emissions):
        init_score = torch.full((1, self.target_size), -10000, device=device)   # same as the forward algorithm
        init_score[0, self.tag_to_ix[START_TAG]] = 0
        forward_score = init_score
        
        # create a list to store the backpoint from each time step
        backpointers = []

        for emission_t in emissions:
            back_ptr = []                       # store btr for each tag in current time step
            forward_score_t = []
            for tag in range(self.target_size):
                transition_score_t = forward_score + self.transitions[:, tag].view(1, -1)
                # find out the optimum pre_tag for cur_tag in this time step
                best_id = argmax(transition_score_t)
                # store the best pre_tag
                back_ptr.append(best_id)
                forward_score_t.append(transition_score_t[0, best_id].view(1))      # add the best score
            forward_score_t = (torch.tensor(forward_score_t, dtype=torch.float, device=device) + emission_t).view(1, -1)
            backpointers.append(back_ptr)
        
        # calculate the terminal socre
        terminal_score = forward_score_t + self.transitions[:, self.tag_to_ix[STOP_TAG]].view(1, -1)
        best_id = argmax(terminal_score)

        # back recursive
        path = [best_id]
        for back_ptr in reversed(backpointers):
            best_id = back_ptr[best_id]
            path.append(best_id)
        path.pop()       # pop the START_TAG
        path.reverse()
        return path
    
    def neg_log_likelihood(self, sentence, tags):
        emissions = self._get_lstm_emissions(sentence)
        denominator = self._forward_algorithm(emissions)
        numerator = self._sentence_score(emissions, tags)
        return denominator - numerator
    
    def forward(self, sentence):
        emissions = self._get_lstm_emissions(sentence)
        path = self._viterbi_decode(emissions)
        return path

if __name__ == '__main__':

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 2

    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    ix_to_tag = {index : tag for tag, index in tag_to_ix.items()}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            30):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long, device=device)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            print(loss.item())
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        path = model(precheck_sent)
        print([ix_to_tag[index] for index in path])