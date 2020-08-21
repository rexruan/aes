#lower case the text

import pickle
import torch
from torch import nn
import os
import sys


def read_pickle(t):
    with open(t,'rb') as f:
        return pickle.load(f)

def lowertext(t):
    for para_index, p in enumerate(t.paragraphs):
        try:
            t.paragraphs[para_index] = str(p).lower()
        except Exception:
            print(t.id)
            print(type(p),p)
            sys.exit()
    for sent_index, s in enumerate(t.sentences):
        t.sentences[sent_index].sent = t.sentences[sent_index].sent.lower()
        for index, token in enumerate(s.tokens):
            try:
                s.tokens[index].form = token.form.lower()
                s.tokens[index].norm = token.norm.lower()
            except Exception:
                print(t.id)
                return
    return t

def lowertexts(dirs):
    for in_dir in in_dirs:
        for id in os.listdir(in_dir):
            t = lowertext(read_pickle(in_dir+'/'+id))
            pickle.dump(t,open(out_dir+'/'+id,'wb'))






