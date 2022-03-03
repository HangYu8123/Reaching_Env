# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:53:58 2022

@author: Hang Yu
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

import Classifiers
from matplotlib.pyplot import MultipleLocator
import fblib


feedbacks = fblib.read_fb("0_fb")
feedbacks_rp = fblib.read_fb("0_fb_repeated")

#scores = fblib.read_score()



classifier = Classifiers.Classifiers()
fb_steady = classifier.STEADY(feedbacks)
fb_steady_rp = classifier.STEADY(feedbacks_rp)


plt.rcParams['figure.figsize'] = (20, 10)        
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 0.5   #设置曲线线条宽度

x = [i for i in range(len(feedbacks))]
x_ = [i for i in range(len(feedbacks_rp))]
replay_times = [i for i in range(5)] 

plt.suptitle("STEADY",fontsize=30) 
agraphic=plt.subplot(3,1,1)
agraphic.set_ylabel('feedback value',fontsize=10)   #添加轴标签
agraphic.set_xlabel('i-th feedback', fontsize=20)
agraphic.plot(x, feedbacks, linewidth=4) 
agraphic.plot(x_, feedbacks_rp, linewidth=4)
#agraphic.axhline(y=np.mean(feedbacks))
#agraphic.axhline(y=np.mean(feedbacks)) 


bgraphic=plt.subplot(3,1,2)
bgraphic.set_ylabel('feedback value',fontsize=10)   #添加轴标签
bgraphic.set_xlabel('i-th feedback', fontsize=20)
bgraphic.plot(x, fb_steady, linewidth=4) 
bgraphic.plot(x, fb_steady_rp, linewidth=4) 


plt.show()
#cgraphic=plt.subplot(3,1,3)
#cgraphic.set_ylabel('score',fontsize=10)   #添加轴标签
#cgraphic.set_xlabel('i-th replay', fontsize=20)
#cgraphic.plot(replay_times, scores, linewidth=4) 
