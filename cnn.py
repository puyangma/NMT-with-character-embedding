#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

class CNN(nn.Module):
    """ 1D Convolutional Network
    """
    def __init__(self, f, char_embed, kernel=5):
        """ 
            @param char_embed (int): char embedding size
            @param f (int): final embedding size
        """
        super(CNN, self).__init__()
        self.cnnlayer = nn.Conv1d(char_embed, f, kernel_size=kernel, stride=1)
        
    def forward(self, x_reshaped):
        
        x_conv = self.cnnlayer(x_reshaped)
        x_convout = F.relu(torch.max(x_conv, dim=2)[0])
        return x_convout


### END YOUR CODE

