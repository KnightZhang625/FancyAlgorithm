# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   28/Feb/2019
# @Last Modified by: 
# @Last Modified time:  

import os
import sys

from pathlib import Path
cur_path = Path(__file__).absolute().parent
from Utils.utils import WordMapping, SequenceToTensor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

PADDING_IDX = 0
BATCH_SIZE = 3
device = torch.device('cpu')

class Bi_LSTM(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, target_size, batch_size, bidirectional=True):
		super(Bi_LSTM, self).__init__()
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.target_size = target_size
		self.batch_size = batch_size
		self.bidirectional = 2 if bidirectional else 1

		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_IDX)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
		self.output = nn.Linear(hidden_dim * 2, target_size)

		self.init_hidden()
	
	def forward(self, x, x_lengths, train=True):
		'''
			x : (batch_size, max_length)
		'''
		if train:
			embedded = self.embedding(x)	# (batch_size, max_length, embedding_dim)
			batch_size, seq_length, _ = embedded.size()
			embedded = embedded.view(seq_length, batch_size, -1)

			hidden = self.init_hidden()
			embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lengths)
			outputs, hidden = self.lstm(embedded, hidden)
			outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

			outputs = outputs.view(batch_size, seq_length, -1)
			outputs = self.output(outputs)
			outputs = F.log_softmax(outputs, dim=2)
			return outputs
		else:
			embedded = self.embedding(x).view(len(x), 1, -1)
			hidden = self.init_hidden_test()
			outputs, hidden = self.lstm(embedded, hidden)
			outputs = self.output(outputs)
			outputs = F.log_softmax(outputs, dim=2)
			return outputs.view(len(x), -1)
	
	def init_hidden(self):
		return (torch.zeros(self.bidirectional, self.batch_size, self.hidden_dim),
				torch.zeros(self.bidirectional, self.batch_size, self.hidden_dim))
	
	def init_hidden_test(self):
		return (torch.zeros(self.bidirectional, 1, self.hidden_dim),
				torch.zeros(self.bidirectional, 1, self.hidden_dim))
	
def calculate_loss(y_pred, y_true, lengths, loss_func):
	y_pred_no_padding = y_pred[0]	# initialize the y_pred, as the first item in y_pred is the longest one, it should not have padding
	y_true_tensor = torch.tensor(y_true[0], dtype=torch.long, device=device)

	for i in range(1, len(lengths)):
		actual_length = lengths[i]
		y_pred_no_padding = torch.cat((y_pred_no_padding, y_pred[i, :actual_length]), dim=0)
		y_true_tensor = torch.cat((y_true_tensor, torch.tensor(y_true[i], dtype=torch.long, device=device)), dim=0)

	return loss_func(y_pred_no_padding, y_true_tensor)

if __name__ == '__main__':

	training_data = [
	("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
	("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
	('The cat barks'.split(), ['DET', 'NN', 'V']),
	('The dog barks'.split(), ['DET', 'NN', 'V']),
	('The boy plays the basketball'.split(), ['DET', 'NN', 'V', 'DET', 'NN']),
	('The girl love the the boy'.split(), ['DET', 'NN', 'V', 'DET', 'DET', 'NN'])
	]

	initial_dict = {'*' : 0}

	sentences = [' '.join(data[0]) for data in training_data]
	word_idx, idx_word = WordMapping.mapping(sentences, initial_dict)

	tags = [' '.join(data[1]) for data in training_data]
	tag_idx, idx_tag = WordMapping.mapping(tags, initial_dict)

	train_x = [[word_idx[word] for word in x]for x,_ in training_data]
	train_y = [[tag_idx[tag] for tag in y] for _,y in training_data]
	
	model = Bi_LSTM(len(word_idx), 6, 3, len(tag_idx), BATCH_SIZE, True)
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	loss_func = nn.NLLLoss()

	if 'model.pt' in os.listdir():
		model.load_state_dict(torch.load('model.pt'))
		model.eval()
	else:
		for epoch in range(100):
			for i in range(0, len(train_x), BATCH_SIZE):
				train_x_padding, y_true, lengths = SequenceToTensor.transfer([train_x[i : i+BATCH_SIZE],
																			train_y[i : i+BATCH_SIZE]], word_idx, tag_idx, device,
																			padding=True, max_length=100)													   
				y_pred = model(train_x_padding, lengths)
				loss = calculate_loss(y_pred, y_true, lengths, loss_func)

				model.zero_grad()
				loss.backward()
				optimizer.step()
				print(loss.item())

	# test
	train_x_padding, _, lengths = SequenceToTensor.transfer([train_x[0:3], train_y[0:3]], word_idx, tag_idx, device, padding=True, max_length=100)
	y_pred = model(train_x_padding[0], lengths, False)

	print(training_data[0][0])
	y_pred_ = y_pred[:lengths[0]]
	prediction = [idx.item() for idx in torch.argmax(y_pred_, dim=1)]
	print([idx_tag[idx] for idx in prediction])

	# for idx, length in enumerate(lengths):
	# 	print(training_data[idx][0])
	# 	y_pred_ = y_pred[idx][:length]
	# 	prediction = [idx.item() for idx in torch.argmax(y_pred_, dim=1)]
	# 	print([idx_tag[idx] for idx in prediction])
