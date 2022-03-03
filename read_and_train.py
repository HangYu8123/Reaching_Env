
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

agent = Agent(state_size=4, action_size=5, seed=0)
#env = reach_env.Reaching()

num_of_eps = 5
test = 5
max_t=100

for i in range(3):
    agent.memory.read(str(i) + "_hand_training")
print(agent.memory.__len__()/2)
for i in range(agent.memory.__len__()/2):    
    agent.active_learn(16)
torch.save(agent.qnetwork_local.state_dict(),  'handtrain.pth')
# for i in range(test):
#     state = env.reset()
#     score = 0

#     for t in range(max_t):

#         prob  = agent.act(state, 0)#fb_agent.act(state,0) #*  #* weight_by_frame(cnt)

#         action = np.random.choice([i for i in range(agent.action_size)], p = prob/sum(prob))
                
#         buffer = env.step(action)
#         print("****************start*****************")
#         print(i,  env.abs_dis(), action, buffer[-1][2], t)
#         print(prob)
#         print("****************end*****************")
#         score += buffer[-1][2] #

#         state = env.state

#         if env.done:
#             break 

#     print("***************One episode done************")
#     print(score)
#     print("***************start a new one ************")