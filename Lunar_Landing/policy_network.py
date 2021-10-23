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
    

class PolicyGradientAgent():
    
    def __init__(self,lr, input_dims , gamma=0.99 , n_action =4):
        self.gamma = gamma
        self.lr =lr
        self.reward_memory = []
        self.action_memory = [] 
        self.policy = PolicyNetwork(self.lr, input_dims, n_action)
        
    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        
        return action.reward()
    
    def store_reward(self,reward):
        self.reward_memory.append(reward)
        
    def learn(self):
        self.policy.optimizer.zero_grad()       
        G = np.zero_like(self.reward_memory , dtype = np.float64)
        for i in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(i,len(self.reward_memory)):
                G_sum+=self.reward_memory[k]*discount
                discount+=self.gamma
            G[i] = G_sum
        G = T.tensor(G , dtype = T.float).to(self.policy.device)
        
        loss = 0
        
        for g, logprob in zip(G,self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy.optimizer.step()
        
        self.action_memory = []
        self.reward_memory = []
    
        
        
        
        
                