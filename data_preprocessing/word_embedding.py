#!/usr/bin/env python3
# encoding: utf-8
import pickle
import torch
from torch import nn

train_file = ''
out_file = ''

def read_pickle(t):
    with open(t,'rb') as f:
        return pickle.load(f)

def create_word_embedding(filename,num_dims,out_file):
    vocab = {'<word_unk>':0, '<word_pad>':1}
    for t in read_pickle(filename):
        for s in t.sentences:
            for w in s.split():
                if w.lower() not in vocab:
                    vocab[w.lower()] = len(vocab)

    embedding_layer = nn.Embedding(len(vocab), num_dims)
    pickle.dump((vocab, list(embedding_layer.parameters())[0].tolist()), open(out_file,'wb'))

def main():
    create_word_embedding(train_file, 10, out_file)

if __name__ == '__main__':
    main()

