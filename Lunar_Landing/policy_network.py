# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:55:45 2021

@author: astonishing_wolf
"""
#input layer 128

import numpy as np
import torch.nn as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    
    def __init__(self , lr , input_dims , n_actions):
        super(PolicyNetwork , self).__init__()
        self.fc1 = nn.linear(*input_dims , 128)
        self.fc2 = nn.linear(128,128)
        self.fc3 = nn.linear(128,n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device())
        
        
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
        