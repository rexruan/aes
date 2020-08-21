#!/usr/bin/env python3
# encoding: utf-8

# This script collects functions to preprocess the data for feature extraction

import re
import torch
import torch.nn.functional as F
import pickle
from syntactic import get_syntactic_features
from morphological import get_morphological_features
from lexical import get_lexical
from general import get_general
import random
from collections import Counter
from copy import deepcopy as copy
import os

def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def median(lst):
    n = len(lst)
    if n < 1:
        return None
    if n % 2 == 1:
        return sorted(lst)[n // 2]
    else:
        return sum(sorted(lst)[n // 2 - 1:n // 2 + 1]) / 2.0

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def incsc(p,t):
    if p == 0:
        return 0
    if t == 0:
        return 1000
    return 1000 * p / t



def create_embeddings(texts, output, sent_feature_num,text_feature_num,
                      function_morph, function_syntactic, function_lexical, function_general):
    '''
    
    :param input:
    :param output:
    :param                          sent    text
    :param function_morph:          30      30 * 3
    :param function_syntactic:      11      11 * 3
    :param function_lexical:        10      10 * 3
    :param function_general:        12      12 * 3 + 2
    :param            total:        63      191
    :return:
    '''
    #texts = read_pickle(input)
    if not text_feature_num:
        text_feature_num = sent_feature_num

    sent2ind, ind2sent, sent_embeddings = {'<sent_pad>':0}, {0:'<sent_pad>'}, [[0]*sent_feature_num]
    text2ind, ind2text, text_embeddings = {'<text_pad>':0}, {0:'<text_pad>'}, [[0]*text_feature_num]
    for text in texts:
        sent_feats, text_feats = [[] for _ in range(len(text.sentences))], []
        for function in [function_morph, function_syntactic, function_lexical, function_general]:
            part_sent_feats, part_text_feats = function(text)
            text_feats.extend(part_text_feats)
            for index, part_sent_feat in enumerate(part_sent_feats):
                sent_feats[index].extend(part_sent_feat)

        for s, sent_feat in zip(text.sentences, sent_feats):
            s = str(s).lower()
            if s in sent2ind:
                continue
            else:
                sent2ind[s] = len(sent2ind)
                ind2sent[len(sent2ind)] = s
                sent_embeddings.append(sent_feat)
        text2ind[text.id] = len(text2ind)
        ind2text[len(text2ind)] = text.id
        text_embeddings.append(text_feats)

    sent_embeddings = torch.FloatTensor(sent_embeddings)
    text_embeddings = torch.FloatTensor(text_embeddings)

    pickle.dump(((sent2ind, ind2sent, sent_embeddings), (text2ind, ind2text, text_embeddings)) ,open(output, 'wb'))

def delete_space(s):
    return ''.join(s.split(' '))

def create_feature_set_for_paras(ttexts): 
    paras = []
    with open(ttexts) as f:
        ids = [id.strip() for id in f.readlines()]
    for id in ids:
        t = read_pickle('/'.join(['conll/A-copy',id]))
        p = ''
        for para in t.paragraphs:
            para = delete_space(para)
            text_para = copy(t)
            text_para.sentences = []
            while p != para:
                print('p',p,'\n')
                print('para',para,'\n')
                print(t.sentences)
                text_para.sentences.append(t.sentences[0])
                p += delete_space(str(t.sentences[0]))
                t.sentences.pop(0)
            p = ''
            paras.append(text_para)
    create_embeddings(paras, 'emb.para',63,191,get_morphological_features, get_syntactic_features,get_lexical, get_general)


def create_feature_set(dirs): # a list of directories
    texts = []
    for dir in dirs:
        ids = os.listdir(dir)
        for id in ids:
            try:
                t = read_pickle('/'.join([dir,id]))
            except Exception:
                continue
            texts.append(t)
    create_embeddings(texts, 'emb.data.new',63,191,get_morphological_features, get_syntactic_features,get_lexical, get_general)
    return

def load_embeddings(input, word=False,feature=True):
    #read text file with embeddings
    if word:
        with open(input,'rb') as f:
            word2ind, word_embeddings = pickle.load(f)
        return word2ind, word_embeddings
    if feature == 'syntactic':
        with open(input, 'rb') as f:
            (sent2ind, ind2sent,sent_embeddings), (text2ind,ind2text, text_embeddings) = pickle.load(f)
        return sent2ind, sent_embeddings, text2ind, text_embeddings

def load_grades(input):
    'load the grade file to convert grades to numeraical index'
    grade2ind = {}
    ind2grade = {}

    with open(input,'r') as f:
        for line in f:
            if line.strip() in grade2ind:
                continue
            else:
                grade2ind[line.strip()] = len(grade2ind)
                ind2grade[len(ind2grade)] = line.strip()

    return grade2ind, ind2grade



def data_loader(data,batch_size=1, shuffle=False):
    if shuffle:
        random.shuffle(data)
    dataset = []
    while data:
        dataset.append(data[:batch_size])
        data = data[batch_size:]
    return dataset

def print_grade_matrix(grade_scale, grade_dict):
    length = len(grade_scale)
    matrix = [[0] * (length) for _ in range(length)]



def get_n_lexical(text,n):
    words = []
    sentence = None
    while n:
        if sentence:
            if sentence.tokens:
                word = sentence.tokens.pop(0)
                if word.upos in ['ADJ','NOUN','VERB', 'ADV']:
                    words.append(word.norm)
                    n -= 1
            else:
                sentence = text.sentences.pop(0)
        else:
            sentence = text.sentences.pop(0)
    return set(words)







