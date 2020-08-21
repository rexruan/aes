#!/usr/bin/env python3
# encoding: utf-8

# data splitting for the experiment

import pickle
import random
import sys
import os

def read_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def lower_case(texts):
    for t in texts:
        for index, s in enumerate(t.sentences):
            t.sentences[index] = s.lower()
    return texts


def cross_validation(dirs, outdir, n=10, seed=0):
    '''extract data from dirs, assign to the outdir
    with n folds given random seed'''
    data = dirs2data(dirs)
    data = lower_case(data)
    random.seed(seed)
    index = list(range(len(data)))
    random.shuffle(index)
    folds = [[] for _ in range(n)]
    for i in range(1,n+1):
        folds[i-1].extend([data[i] for i in index[int(len(data)*(i-1)/n):int(len(data)*i/n)]])
    #create train, validation, test set for cross validation
    for i in range(n):
        print(type(n),type(i))
        print('There are %d folds. Loading the current fold %d' % (n, i))
        fold_numbers = set(range(n)) - {i}
        test = folds[i]
        dev = folds[fold_numbers.pop()]
        train = [t for j in fold_numbers for t in folds[j]]
        os.mkdir(outdir+'/'+str(i+1))
        for d, name in zip([train, dev, test],['train','dev','test']):
            pickle.dump(d, open(outdir+'/'+str(i+1)+'/'+name,'wb'))


def dirs2data(dirs):
    data = []
    for dir in dirs:
        for id in os.listdir(dir):
            try:
                text = read_pickle('/'.join([dir, id]))
                data.append(text)
            except Exception:
                continue

    return data



def data_split(dirs, ratio=(8,1,1), seed=None, saveAs=False): #ratio: train:dev:test set
                                                #data is a list of text instances
    if seed:
        random.seed(seed)
    else:
        random.seed(0)
    data = dirs2data(dirs)
    index = list(range(len(data)))
    random.shuffle(index)

    if len(ratio) != 3:
        print("Incorrect format of data splitting.")
        sys.exit()
        if sum(ratio) <= 0:
            print("The ratio of data splitting shall be over 0")
            sys.exit()

    train = [data[i] for i in index[:int(len(data) * ratio[0]/sum(ratio))]]
    dev = [data[i] for i in index[int(len(data) * ratio[0]/sum(ratio)) : int(len(data) * sum(ratio[:2])/sum(ratio))]]
    test = [data[i] for i in index[int(len(data) * sum(ratio[:2])/sum(ratio)):]]

    if saveAs:
        for data, name in zip([train,dev,test],['train','dev','test']):
            pickle.dump(data, open(name+'.pickle','wb'))

    return train, dev, test


def main():
    #prepare data for cross validation dataset of ann
    #cross_validation(['conll/A'],'cross_validation/ann')
    cross_validation(['conll/A','conll/R'],'cross_validation/ann_robert')



if __name__ == '__main__':
    main()


