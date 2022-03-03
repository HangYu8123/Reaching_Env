import reach_env
import numpy as np
import time
from dqn_agent import Agent
import readchar
env =reach_env.Reaching()
agent = Agent(state_size=4, action_size=5, seed=0, learning_rate=0.001)
train_episodes = 3
print(env.get_obs())
for i in range(train_episodes):
    #env =reach_env.Reaching()
    state = env.reset()
    while (True):
        #time.sleep(2)
        action  = int(readchar.readkey())
        buffer = env.step(action)
        for b in range(len(buffer)):
                agent.step(buffer[b][0], action, buffer[b][2], buffer[b][1], buffer[b][3],True,str(i)+"_hand_training")
        if action  == 10:
            break 
        if env.done:
            break
        print("*********************************")
        print(buffer[-1][2])
        print("*********************************")
print(env.get_obs())



