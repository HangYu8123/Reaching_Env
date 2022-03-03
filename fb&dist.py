import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Classifiers

from matplotlib.pyplot import MultipleLocator
from scipy.stats import skewnorm

#from dqn_agent import Agent

#agent = Agent(state_size=8, action_size=4, seed=0)
act_pos_mean = []
act_neg_mean = []

def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def gen_feedback(num):
    # if num < 500:
    #     return max( min(10,int(np.random.normal( num/50 , 1))),0)
    # else:
    #     return max( min(10,int(np.random.normal( (1000-num)/50 , 1))),0)
    # if (num + 1) % 10 == 0:
    #     return np.random.randint(0,10)
    if num%2 ==0:
        fb = max( min(10,round(np.random.normal( 7  , 1)+ round(num/400))),0) 
        act_pos_mean.append(fb)
        return fb
    else:
        fb = max( min(10,round(np.random.normal(3 ,1)+ round(num/400))),0) 
        act_neg_mean.append(fb)
        return  fb


ax=[]   #保存图1数据
ay=[]
bx=[]   #保存图2数据
by=[]

pos =[] #[i%2 for i in range(11)]
neg =[] #[i%2 for i in range(11)]

num=0   #计数
plt.ion()    # 开启一个画图的窗口进入交互模式，用于实时更新数据
# plt.rcParams['savefig.dpi'] = 200 #图片像素
# plt.rcParams['figure.dpi'] = 200 #分辨率
plt.rcParams['figure.figsize'] = (20, 10)        # 图像显示大小
plt.rcParams['font.sans-serif']=['SimHei']   #防止中文标签乱码，还有通过导入字体文件的方法
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 0.5   #设置曲线线条宽度

classifier = Classifiers.Classifiers()

while num<1000:
    plt.clf()    #清除刷新前的图表，防止数据量过大消耗内存
    plt.suptitle("STEADY",fontsize=30)             #添加总标题，并设置文字大小
    feedback = gen_feedback(num) #生成随机数画图
    #print(feedback)
	#图表1
    ax.append(num)      #追加x坐标值
    ay.append(feedback)       #追加y坐标值
    agraphic=plt.subplot(3,1,1)
    #agraphic.set_title('Participants\' feedback')      #添加子标题
    agraphic.set_xlabel('feedback value',fontsize=10)   #添加轴标签
    agraphic.set_ylabel('i-th feedback', fontsize=20)
    agraphic.plot(ax, ay,'r^')                #等于agraghic.plot(ax,ay,'g-')
    
	
    #图表2
    bx.append(num)
    
    # if classifier.steady_mean(feedback) > 0:
    #     pos.append(feedback)
    # else:
    #     neg.append(feedback)
    pos, neg = classifier.steady_mean(feedback)
    #by.append(g1*2)
    bgraphic=plt.subplot(3, 1, 2)
    #print(bgraphic.axes)
    #bgraphic.set_title('Distributions')
    bgraphic.set_xlabel('feedback value',fontsize=10)   #添加轴标签
    bgraphic.set_ylabel('frequence', fontsize=20)
    kwargs = dict(histtype='stepfilled', alpha=0.3,  bins=10)
    # if len(pos) > 0:
    #     bgraphic.hist(pos, **kwargs)
    #     #print("pos:",pos)
    # if len(neg) > 0:
    #     bgraphic.hist(neg, **kwargs)
        #print("neg:",neg)
    #bgraphic.hist(pos,bins=10)
    #bgraphic.hist(neg,bins=10)
    if len(pos)>0:
        #sns.distplot(pd.DataFrame(pos),bins = 10, ax = bgraphic.axes) #palette设置颜色
        sns.distplot(pd.DataFrame(pos),bins=10,kde=False,
                     hist_kws={"color":"tomato"},label="positive feedback")
    if len(neg)>0:
        #sns.distplot(pd.DataFrame(pos),bins = 10, ax = bgraphic.axes) #palette设置颜色
        sns.distplot(pd.DataFrame(neg),bins=10,kde=False,
                     hist_kws={"color":"lightskyblue"},label="negative feedback")
    
    
    
    cgraphic=plt.subplot(3, 1, 3)
    
    data_range = np.arange(0,10,0.1)
    
    pos_mean = np.mean(pos)
    pos_std = np.std(pos)
    pos_dis = normfun(data_range, pos_mean, pos_std)
    
    neg_mean = np.mean(neg)
    neg_std = np.std(neg)
    neg_dis = normfun(data_range, neg_mean, neg_std)
    #if len(pos) > 2:
    cgraphic.plot(data_range,pos_dis,linewidth=8)#.plot(data_range, skewnorm.pdf(data_range, *skewnorm.fit(pos)))#.plot(data_range,pos_dis)
    cgraphic.plot(data_range,neg_dis,linewidth=8)
    bgraphic.set_xlabel('feedback value',fontsize=10)   #添加轴标签
    bgraphic.set_ylabel('probability', fontsize=20)
    cgraphic.text(7, 0, "mean of positive feedback: " + str(round(pos_mean,4)), ha='center', va='bottom', fontsize=20)
    cgraphic.text(7, 0.1, "actual mean of positive feedback: " + str(round(np.mean(act_pos_mean),4)), ha='center', va='bottom', fontsize=20)
    cgraphic.text(2, 0, "mean of negtive feedback: " + str(round(neg_mean,4)), ha='center', va='bottom', fontsize=20)
    cgraphic.text(2, 0.1, "actual mean of negtive feedback: " + str(round(np.mean(act_neg_mean),4)), ha='center', va='bottom', fontsize=20)
   
    #cgraphic.text(0, 1, neg_mean, ha='center', va='bottom', fontsize=20)
    
    plt.pause(0.1)     #设置暂停时间，太快图表无法正常显示
    # if num == 100:
    #     plt.savefig('picture.png', dpi=300)  # 设置保存图片的分辨率
    #     #break
    num=num+1

plt.ioff()       # 关闭画图的窗口，即关闭交互模式
plt.show()       # 显示图片，防止闪退