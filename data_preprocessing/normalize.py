# feature scaling:
# 1. standardized
# 2. standardized & sigmoid
import pickle
import torch
from torch import nn
from statistics import stdev, mean

emb = '/Users/rex/Desktop/AES_rex/scripts/emb.data'

def read_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)


def max_normalize(matrix):
    cols = matrix.size()[1]
    output = torch.zeros(matrix.size())
    for col in range(cols):
        array = matrix[:,col].tolist()
        max_value = max(array)
        array = [i/max_value for i in array]
        output[:,col] = torch.FloatTensor(array).view(-1)
    return output

def tanh_normalize(matrix):
    row = len(matrix)
    normalized = []
    model = nn.Tanh()
    for col in range(len(matrix[0])):
        feature = matrix[:,col]
        normalized.append(model(feature))

def std_normalize(matrix):
    cols = matrix.size()[1]
    output = torch.zeros(matrix.size())
    for col in range(cols):
        array = matrix[:,col].tolist()
        m, s = mean(array), stdev(array)
        array = [(i-m)/s for i in array]
        output[:,col] = torch.FloatTensor(array).view(-1)
    return output

def main(filename):
    (sent2i, i2sent, sent),(text2i,i2text, text) = read_pickle(filename)
    sent = std_normalize(sent)
    text = std_normalize(text)

    pickle.dump((sent2i,i2sent,sent),(text2i,i2text,text),open(filename+'.std','wb'))


if __name__ == '__main__':
    main(emb)
