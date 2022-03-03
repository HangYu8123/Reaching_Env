# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:47:24 2020

@author: Hang Yu
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import math

class PSAgent:
    def __init__(self, action_space, alpha = 0.5, gamma=0.8, temp = 1, epsilon = 1):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma= gamma
        self.temp = temp
        self.epsilon = epsilon
        self.feedback=pd.DataFrame(columns=[ i for i in range(self.action_space)])
    def trans(self, state):
        s = ""
        for i in range(len(state)):
            s+=str(state[i])
        return s
    def check_add(self,state):
        if self.trans(state) not in self.feedback.index:
            self.feedback.loc[self.trans(state)]=pd.Series(np.zeros(self.action_space),index=[ i for i in range(self.action_space)])
            
    def learning(self, action, feedback, state, next_state):
        self.check_add(state)
        self.check_add(next_state)
        #print(math.exp(self.feedback.loc[self.trans(state),action]))
        # self.feedback.loc[self.trans(state),action] = np.tanh(np.arctanh(self.feedback.loc[self.trans(state),action]) 
        #                       + feedback )

        self.feedback.loc[self.trans(state),action] += feedback
    def action_prob(self, state):
        self.check_add(state)
        prob = []
        if all(self.feedback.loc[self.trans(state)].to_numpy() == 0):
            return np.array([1/self.action_space for i in range(self.action_space)])
        for i in range(self.action_space):
            if self.feedback.loc[self.trans(state),i] < -50:
                self.feedback.loc[self.trans(state),i] = -50
            prob.append(math.pow(0.95,self.feedback.loc[self.trans(state),i])/
                        (math.pow(0.95,self.feedback.loc[self.trans(state),i]) + 
                         math.pow(0.05,self.feedback.loc[self.trans(state),i])) )
        return prob
    def choose_action(self, state):
        self.check_add(state)
        prob = []
        if all(self.feedback.loc[self.trans(state)].to_numpy() == 0):
            return np.random.choice([i for i in range(self.action_space)])
        for i in range(self.action_space):
            prob.append(math.pow(0.95,self.feedback.loc[self.trans(state),i])/
                        (math.pow(0.95,self.feedback.loc[self.trans(state),i]) + 
                         math.pow(0.05,self.feedback.loc[self.trans(state),i])) )
        prob = np.array(prob)
        return np.random.choice(np.flatnonzero(prob == prob.max()))

