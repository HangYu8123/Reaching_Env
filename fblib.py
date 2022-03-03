
import csv
import numpy as np


def save_fb(feedbacks, filename="feedback"):
    with open(filename + ".csv", 'a') as csvfile:
        kf_writer = csv.writer(csvfile, delimiter=' ',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for f in feedbacks:
            kf_writer.writerow([f])
def read_fb( filename="feedback", obs_size = 4, action_size = 1):
    feedbacks = []
    with open(filename+ ".csv", 'rb') as file:
        reader = csv.reader(file,
                            quoting = csv.QUOTE_ALL,
                            delimiter = ' ')
        for f in reader:
            feedbacks.append(int(f[0]))
    return feedbacks

def policy_shaping(p_a, p_f):
    p = []
    func = lambda x,y: x*y
    for i in range(len(p_f)):
        p_f[i] = pow(0.7, p_f[i])/(  pow(0.7, p_f[i]) + pow(0.3, p_f[i])      )
    result = map(func, p_f, p_a)
    sum_prob = sum(result)
    for i in range(len(p_a)):
        p.append(result[i].sum_prob)
    return p

def save_score(scores,filename = "scores"):
    with open(filename + ".csv", 'a') as csvfile:
        kf_writer = csv.writer(csvfile, delimiter=' ',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for s in scores:
            kf_writer.writerow([s])
def read_score( filename="score", obs_size = 4, action_size = 1):
    scores = []
    with open(filename+ ".csv", 'rb') as file:
        reader = csv.reader(file,
                            quoting = csv.QUOTE_ALL,
                            delimiter = ' ')
        for s in reader:
            scores.append(int(s[0]))
    return scores