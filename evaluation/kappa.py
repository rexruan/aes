
#to calculate kappa

from sklearn.metrics import cohen_kappa_score as kappa
import numpy as np
import sys
import os
from statistics import mean

d = {
    'IG':0,
    'G':1,
    'VG':2,
    'MVG':3
    }

def read_file(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
    return [line.strip().split(',')[1:] for line in lines[1:]]

def kappa_score(lines):
    lines = [[int(i) for i in line] for line in lines]
    m = np.asmatrix(lines)
    m = np.flip(m)
    predicts, expects = [], []
    for exp in range(len(m)):
        for pred in range(len(m)):
            for _ in range(m[exp,pred]):
                expects.append(exp)
                predicts.append(pred)
    return kappa(expects,predicts)

def main():
    path = sys.argv[1]
    kappa_scores = [0]*11
    for name in os.listdir(path):
        if 'test' in name:
            lines = read_file(path+'/'+name)
            k = kappa_score(lines)
            index = name.split('_')[-7]
            kappa_scores[int(index)] = k
    kappa_scores[0] = mean(kappa_scores[1:])
    for index, score in enumerate(kappa_scores):
        print(index,'\t',score)

main()
