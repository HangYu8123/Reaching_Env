
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

import fblib

agent = Agent(state_size=4, action_size=5, seed=0)
agent_ps = Agent(state_size=4, action_size=5, seed=0, learning_rate=0.01)
env = reach_env.Reaching()

num_of_eps = 5
test = 4
max_t=100
fb = []
fb_rp = []
scores =[]

for i in range(test/2):
    fb.appned(fblib.read_fb(str(i)+"_fb"))
for i in range(test/2):
    fb_rp.appned(fblib.read_fb(str(i)+"_fb_repeated"))


for i in range(num_of_eps/2):
    agent.memory.read_plus_fb(fb[i], filename=str(i) )

for i in range(num_of_eps/2):
    agent_ps.memory.read_plus_fb(fb[i], filename=str(i) )
for i in range(num_of_eps/2):
    agent_ps.memory.read_plus_fb( fb_rp[i], filename=str(2*i))

for i in range(agent.memory.__len__()):    
    agent.active_learn(16)
    agent_ps.active_learn(16)

for i in range(test):
    state = env.reset()
    score = 0

    for t in range(max_t):

        prob  = fblib.policy_shaping(agent.act(state, 0), agent_ps.act_ps(state,0))#fb_agent.act(state,0) #*  #* weight_by_frame(cnt)

        action = np.random.choice([i for i in range(agent.action_size)], p = prob/sum(prob))
                
        buffer = env.step(action)
        print("****************start*****************")
        print(i,  env.abs_dis(), action, buffer[-1][2], t)
        print(prob)
        print("****************end*****************")
        score += buffer[-1][2] #

        state = env.state

        if env.done:
            break 

    print("***************One episode done************")
    print(score)
    print("***************start a new one ************")
    scores.append(score)
fblib.save_score(scores, "Policy_Shapingz")