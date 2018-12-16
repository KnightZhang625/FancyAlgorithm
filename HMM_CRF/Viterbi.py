# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   16/Dec/2018
# @Last Modified by:    
# @Last Modified time:  

import sys
import numpy as np

class Node(object):
    def __init__(self, theta, pre, tag):
        self.theta = theta
        self.tag = tag
        self.pre = pre
    
    def __eq__(self, node):
        return self.theta == node.theta 

    def __lt__(self, node):
        return self.theta < node.theta
    
    def __str__(self):
        return str(self.tag)

class Dragon(object):
    '''
        this name double salutes the aircraft of SpaceX
    '''
    def __init__(self, transition_matrix, emission_matrix, sequence, tags, words):
        '''
            sequence : the observation sequence
            tags : all tags
            words : all words
        '''
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.sequence = sequence
        self.tags = tags
        self.words = words

        self.category_idx()

    def category_idx(self):
        '''
            make one-hot-encoding for words, tags, this will a very handy step
        '''
        self.tags_idxs = {tag : index for index, tag in enumerate(self.tags)}
        self.idxs_tags = {index : tag for index, tag in enumerate(self.tags)}
        
        self.words_idxs = {word : index for index, word in enumerate(self.words)}
        self.idxs_words = {index : word for index, word in enumerate(self.words)}
    
    def launch(self):
        rows = len(self.tags) - 2
        cols = len(self.words) + 2
        forward_matrix = np.array(self._initial_matrix(rows, cols))     # build forward matrix, transfer it to numpy array
        initial_theta = 1                                               # initial theta, initial tag : Start, initial word : *s*
    
        for c in range(cols):
            if c == 0:
                for r in range(rows):
                    forward_matrix[r, c] = Node(1, None, 'Start')
            elif c == (cols - 1):
                for r in range(rows):
                    theta, index = self._end_state(forward_matrix[:, c-1], self.sequence[c-1])
                    forward_matrix[r, c] = Node(theta, self.idxs_tags[index + 1], self.idxs_tags[r + 1])
            elif c == 1:
                for r in range(rows):
                    theta, index = self._initial_state(forward_matrix[:, c-1], r)
                    forward_matrix[r, c] = Node(theta, self.idxs_tags[index], self.idxs_tags[r + 1])
            else:
                for r in range(rows):
                    theta, index = self._middle_state(forward_matrix[:, c-1], r, self.sequence[c-1])
                    forward_matrix[r, c] = Node(theta, self.idxs_tags[index + 1], self.idxs_tags[r + 1])
        return forward_matrix

    def _initial_state(self, forward_array, r):
        transition_col = self.transition_matrix[:, r][0]
        cur_state = np.array([transition_col * self.emission_matrix[0, 0] * r.theta for r in forward_array])
        return np.max(cur_state), np.argmax(cur_state)
    
    def _middle_state(self, forward_array, r, word_pre):
        transition_col = self.transition_matrix[:, r][1 :]
        cur_state = np.multiply(transition_col, self.emission_matrix[:, self.words_idxs[word_pre]][1:])
        forward_array_num = np.array([r.theta for r in forward_array])
        cur_state = np.multiply(cur_state, forward_array_num)
        return np.max(cur_state), np.argmax(cur_state)
    
    def _end_state(self, forward_array, word_pre):
        transition_col = self.transition_matrix[:, -1][1 :]
        cur_state = np.multiply(transition_col, self.emission_matrix[:, self.words_idxs[word_pre]][1:])
        forward_array_num = np.array([r.theta for r in forward_array])
        cur_state = np.multiply(cur_state, forward_array_num)
        return np.max(cur_state), np.argmax(cur_state)

    def _initial_matrix(self, rows, cols):
        forward_matrix = []
        for r in range(rows):
            row_temp = []
            for c in range(cols):
                row_temp.append(None)
            forward_matrix.append(row_temp)
        return forward_matrix
    
    def decode(self, forward_matrix):
        rows = len(self.tags) - 2
        cols = len(self.words) + 2

        tags = [forward_matrix[0, -1].pre]
        cur_tag_index = self.tags_idxs[forward_matrix[0, -1].pre]

        for c in reversed(range(1, cols-1)):
            cur_tag = forward_matrix[cur_tag_index - 1, c].pre
            cur_tag_index = self.tags_idxs[cur_tag]
            tags.append(cur_tag)
        return list(reversed(tags))

if __name__ == '__main__':
    transition_matrix = np.array([[0.7, 0.3, 0], [0.2, 0.7, 0.1], [0.7, 0.2, 0.1]])
    '''
                            Next
            Current     A     B     End
            START      0.7   0.3     0
              A        0.2   0.7    0.1
              B        0.7   0.2    0.1
    '''
    emission_matrix = np.array([[1, 0, 0], [0, 0.4, 0.6], [0, 0.3, 0.7]])
    '''
                            Word
            State       *S*    x     y
            START        1     0     0
              A          0    0.4   0.6
              B          0    0.3   0.7   
    '''     
    sequence = ['*s*', 'x', 'y', 'y']
    tags = ['Start', 'A', 'B', 'End']   
    words = [ '*s*', 'x', 'y']

    dragon = Dragon(transition_matrix, emission_matrix, sequence, tags, words)
    forward_matrix =  dragon.launch()
    tags = dragon.decode(forward_matrix)
    print(tags)