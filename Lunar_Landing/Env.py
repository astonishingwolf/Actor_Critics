# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:43:46 2021

@author: astonishing wolf
"""
import gym

if __name__ =='__main__':
    env = gym.make('LunarLander-v2')
    n_games = 100
    
    for i in range(n_games):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs_,reward,done,info = env.step(action)
            score+=action
            env.render()
        print('episode', i, 'score %.if' %score)
        

