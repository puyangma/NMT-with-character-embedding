#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Highway(nn.Module):
	def __init__(self, embed_size):
		super(Highway, self).__init__()
		self.embed_size = embed_size
		self.W_proj = nn.Linear(self.embed_size, self.embed_size)
		self.W_gate = nn.Linear(self.embed_size, self.embed_size)
		#using xavier initialization
		nn.init.xavier_normal_(self.W_proj.weight)
		nn.init.xavier_normal_(self.W_gate.weight)

	def forward(self, x_conv):
		proj = F.relu(self.W_proj(x_conv))
		gate = F.sigmoid(self.W_gate(x_conv))
		highway = torch.mul(gate,proj) + torch.mul((1-gate),x_conv)
		return highway
		




### END YOUR CODE 

