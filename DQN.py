# -*- coding: utf-8 -*-
"""


@author: Hang Yu
"""

from pickle import TRUE
import torch
import random
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import reach_env
import time
import csv
import readchar
sum_of_feedback = 1000


env = reach_env.Reaching()

agent = Agent(state_size=4, action_size=5, seed=0)
#agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

# watch an untrained agent


def dqn(n_episodes=200, max_t=1000, eps_start=0.4, eps_end=0.01, wt_start = 1, wt_end = 0.1, 
        eps_decay=300, wt_decay = 100000, model = 0, prefix =""):
    """
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    epsilon_by_frame = lambda frame_idx: eps_end + (eps_start - eps_end) * math.exp(
            -1. * frame_idx / eps_decay) # decrease epsilon
    global agent
    
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=200)  # last 200 scores
    eps = eps_start                    # initialize epsilon
    cnt = 0
    

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        filename = prefix + str(i_episode)
        temp_buffer = []
        for t in range(max_t):
            
            cnt+=1
            eps = epsilon_by_frame(cnt)
            #eps = 0.1 
            prob  = agent.act(state, eps)#fb_agent.act(state,0) #*  #* weight_by_frame(cnt)

            action = np.random.choice([i for i in range(agent.action_size)], p = prob/sum(prob))
            #action = np.random.choice(np.flatnonzero(prob == prob.max()))
            #action = 4
            
            #action  = int(readchar.readkey())
                
            buffer = env.step(action)
            print("****************start*****************")
            print(i_episode, eps,  env.abs_dis(), action, buffer[-1][2], t)
            print(prob)
            print("****************end*****************")
            score += buffer[-1][2] #

            for b in range(len(buffer)):
                agent.step(buffer[b][0], action, buffer[b][2], buffer[b][1], buffer[b][3],True,str(i_episode))
            #agent.memory.read()
            #temp_buffer.append([state, action, reward, next_state, done])

            state = env.state

            if env.done:
                break 
        
        
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = epsilon_by_frame(cnt) # decrease epsilon
        print("***************One episode done************")
        print(i_episode, np.mean(scores_window),score,eps)
        print("***************start a new one ************")
        #print('\rEpisode {}\tAverage Score: {:.2f} \tCrt Score: {:.2f} \tepsilon: {:.4f}'.format(i_episode, np.mean(scores_window),score,eps), end="")


    return scores


length = 100
times = 1
M=[0 for i in range(length)]

#reach_env.go_to_start(env.arm, start=True)
for i in range(times):
    print('\n',i,"-th Trial")
    M = np.sum([dqn(n_episodes=length,model = 1),M], axis=0)
    # S = np.sum([dqn(n_episodes=length,model = 5),S], axis=0)
    # N = np.sum([dqn(n_episodes=length,model = 10),N], axis=0)
    # R = np.sum([dqn(n_episodes=length,model = 100),R], axis=0)


x=[i+1 for i in range(length)]
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(x,M/times,color='green',label='M')
# plt.plot(x,S/times,color='red',label='S')
# plt.plot(x,N/times,color='yellow',label='N')
# plt.plot(x,R/times,color='pink',label='R')

plt.legend()


res=M/times
f = open('DQN.txt', 'w')  
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 





