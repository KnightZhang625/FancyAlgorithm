# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   11/Jan/2019
# @Last Modified by:    
# @Last Modified time:

'''
    this module is used for seq2seq task, adding the attention mechanism
    this package treat each tensor with three dimensions, including one dimenstion representing batch
    however, further mini-batch training is still studied by the author
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

############################################# Encoder Module ############################################
#########################################################################################################