# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   17/Dec/2018
# @Last Modified by:    
# @Last Modified time:  

import sys
import numpy as np
import pandas as pd

class SpaceX(object):
    '''
        Double Salute to the Elon Musk, to whom make the world better !
    '''
    class _Node(object):
        '''
            the Node object represents each item in the forward matrix
            using indicator will make the function handy
        '''
        def __init__(self, theta, pre_tag, cur_tag):
            self.theta = float(theta)               # theta stores the probability in this state
            self.pre_tag = pre_tag                  # the most likely previous tag, Node object
            self.cur_tag = str(cur_tag)             # the current tag
        
        def __eq__(self, node):
            return self.cur_tag == node.cur_tag
        
        def __lt__(self, node):
            return self.theta < node.theta
        
        def __str__(self):
            return str(self.cur_tag)

    def __init__(self, transition_matrix, emission_matrix, words, tags, sequence):
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.words = words
        self.tags = tags
        self.sequence = sequence

        self.word_idx, self.idx_word, self.tag_idx, self.idx_tag = SpaceX._word_to_idx(words, tags)

    def launch(self):
        forward_matrix = SpaceX._create_matrix(self.sequence, self.tags)
        rows = forward_matrix.shape[0]
        cols = forward_matrix.shape[1]

        for c in range(cols):               # we look through the sequence
            if c == 0:
                for r in range(rows):
                    forward_matrix[r, c] = self._Node(1, None, 'Start')
            elif c == 1:
                for r in range(rows):
                    theta, pre_tag_index = self._middle_calculate(self.transition_matrix[0, r], self.emission_matrix[0, 0], forward_matrix[:, 0])       # r indicates the col in transition matrix
                    forward_matrix[r, c] = self._Node(theta, forward_matrix[pre_tag_index ,c-1], self.idx_tag[r + 1])       # we add pre Node object, in order to trace back easily
            elif c == (cols - 1):           # the last position
                for r in range(rows):
                    pre_word = self.sequence[c - 1]
                    wordIdx = self.word_idx[pre_word]
                    theta, pre_tag_index = self._middle_calculate(self.transition_matrix[:, -1][1:], self.emission_matrix[:, wordIdx][1:], forward_matrix[:,c-1])
                    forward_matrix[r, c] = self._Node(theta, forward_matrix[pre_tag_index, c-1], 'End')
            else:
                for r in range(rows):
                    pre_word = self.sequence[c - 1]
                    wordIdx = self.word_idx[pre_word]
                    theta, pre_tag_index = self._middle_calculate(self.transition_matrix[:, r][1:], self.emission_matrix[:, wordIdx][1:], forward_matrix[:, c-1])   # [1:] means we ignore the first row belongs to the 'Start'
                    forward_matrix[r, c] = self._Node(theta, forward_matrix[pre_tag_index, c-1], self.idx_tag[r + 1])
        return forward_matrix
            
    def _middle_calculate(self, transition_array, emission_array, forward_array):
        forward_array_num = np.array([r.theta for r in forward_array])
        cur_theta = np.multiply(np.multiply(transition_array, emission_array), forward_array_num)
        return np.max(cur_theta), np.argmax(cur_theta)

    @staticmethod
    def _create_matrix(sequence, tags):
        '''
            create a matrix, the item of which will be Node object
        '''
        return  np.array([[None for c in range(len(sequence) + 1)] for r in range(len(tags) - 1)])
        '''
            len(sequence) + 1 means adding 'End'
            len(tags) - 1 means we only stores the real tags 
            except for the first and the last position
            The matrix looks like below :

                START   A   A   A   END

                START   B   B   B   END

                *s*    x   y   y  
        '''

    @staticmethod
    def _word_to_idx(words, tags):
        '''
            create word(tag) and index pairs, which will be helpful in indexing our word or tag in matrix
        '''
        word_idx = {word : index for index, word in enumerate(words)}
        idx_word = {index : word for word, index in word_idx.items()}
    
        tag_idx = {tag : index for index, tag in enumerate(tags)}
        idx_tag = {index : tag for tag, index in tag_idx.items()}

        return word_idx, idx_word, tag_idx, idx_tag
    
    @staticmethod
    def decode(forward_matrix):
        batr_cur = forward_matrix[0, -1]
        results = []
        while batr_cur is not None:
            results.append(batr_cur.cur_tag)
            batr_cur = batr_cur.pre_tag
        return list(reversed(results))

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
    tags = ['Start', 'A', 'B']              # the first item of tags must be 'Start', 
                                            # the following items should keep the order with the transition matrix's row, no 'End'
    words = ['*s*', 'x', 'y']               # the first item of words mush be '*s*'
                                            # the following item should keep the order with the emission matrix's row

    dragon = SpaceX(transition_matrix, emission_matrix, words, tags, sequence)
    forward_matrix = dragon.launch()
    print(SpaceX.decode(forward_matrix))