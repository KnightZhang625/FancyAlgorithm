# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   28/Feb/2019
# @Last Modified by: 
# @Last Modified time:  

import torch

class WordMapping(object):
	'''
		Aim: mapping the words to the index
	'''
	def __init__(self):
		pass

	@staticmethod
	def mapping(data, initial_tags=None):
		'''
			Args Format: list contains lists which consist of sentences
						 eg. [[sentence_1], [sentence_2], ... , [sentence_3]]
		'''
		word_idx = {}
		idx_word = {}
		if not (initial_tags is None):
			word_idx.update(initial_tags)
			idx_word = {idx : word for word, idx in word_idx.items()}
			n = len(initial_tags)
		else:	
			n = 0
		for sentence in data:
			word_split = sentence.split(' ')
			for word in word_split:
				if word not in word_idx.keys():
					word_idx[word] = n
					idx_word[n] =  word
					n += 1
				else:
					continue	
		return word_idx, idx_word

class SequenceToTensor(object):
	'''
		transfer sequence to tensor, with padding or not
	'''
	def __init__(self):
		pass
	
	@classmethod
	def transfer(cls, train_pair, word_idx, tag_idx, device, padding=False, max_length=-1):
		if padding and max_length != -1:		# batch training
			return cls.transfer_with_padding(train_pair, word_idx, tag_idx, device)
	
	@classmethod
	def transfer_with_padding(cls, train_pair, word_idx, tag_idx, device):
		lengths = [len(sentence) for sentence in train_pair[0]]
		max_length = max(lengths)
		
		train_x, train_y = train_pair[0], train_pair[1]
		train_x = sorted(train_x, key=lambda x : len(x), reverse=True)
		train_y = sorted(train_y, key=lambda y : len(y), reverse=True)
		lengths = sorted(lengths, reverse=True)

		train_x_padding = torch.zeros((len(train_x), max_length), dtype=torch.long, device=device)
		for idx, x in enumerate(train_x):
			length = lengths[idx]
			train_x_padding[idx, :length] = torch.tensor(x, dtype=torch.long, device=device)
		return train_x_padding, train_y, lengths