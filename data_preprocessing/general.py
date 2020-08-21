#!/usr/bin/env python3
# encoding: utf-8

# This script is to collect general features

# The list of general features:

import pickle
from util import mean, median, mean_median
import math
from statistics import stdev

def bilog(type, token):
    # type-token ratio bilogarithmic ttr: log t/log n
    try:
        return math.log(type) / math.log(token)
    except ZeroDivisionError:
        return type / token
    #return math.log(type) / math.log(token)

def root(type, token):
    # root type-token ratio ttr: t/sqrt(n)
    return type / math.sqrt(token)

def get_columns(l, index):
    return [i[index] for i in l]

def length(l): #input a list of sequences
    return [len(i) for i in l]

def over(l, base):
    if isinstance(l[0], str):
        l = [len(i) for i in l]
    return [i for i in l if i > base]

def count_features(s):

    tokens = [token.norm.lower() for token in s.tokens if token.upos != 'PUNCT']
    types = list(set([token.norm.lower() for token in s.tokens if token.upos !='PUNCT']))
    char_token_nr = sum(length(tokens))
    char_type_nr = sum(length(types))
    token_nr = len(tokens)
    type_nr = len(types)
    mean_char_token = char_token_nr / token_nr
    mean_char_type = char_type_nr / type_nr


    #fundament längd ([antal ord | ratio] före första finita verbet, flera fundament i samma mening )
    longtoken_nr = len(over(tokens, 13))
    longtype_nr = len(over(types, 13))
    misspell = len([token for token in s.tokens if token.norm.lower() != token.form.lower()])
    lix = len(tokens) + 100 * len(over(tokens, 6)) / len(tokens)
    ttr_log = bilog(type_nr, token_nr)  #type-token ratio bilogarithmic ttr: log t/log n
    ttr_root = root(type_nr, token_nr) #root type-token ratio ttr: t/sqrt(n)

    #if raw:
    #    return [tokens, types, longtoken_nr, longtype_nr, misspell]

    return [char_token_nr, char_type_nr, mean_char_token, mean_char_type, \
           token_nr, type_nr, longtoken_nr, longtype_nr, misspell, lix, ttr_log, ttr_root]  # 12 features

def get_general(t):

    '''
    The general features inlcude:
    0   char_token_nr
    1   char_type_nr
    2   mean_char_token
    3   mean_char_type
    4   token_nr
    5   type_nr
    6   longtoken_nr
    7   longtype_nr
    8   misspell
    9  lix
    10  ttr_log
    11  ttr_root'''

    feature_sentence_level = [[] for _ in range(len(t.sentences))]
    for index, s in enumerate(t.sentences):
        if set([token.upos for token in s.tokens]) == {'PUNCT'}:
            feature_sentence_level[index].extend([0]*12)
            continue
        feature_sentence_level[index].extend(count_features(s))

    tokens = [token.norm.lower() for s in t.sentences for token in s.tokens if token.upos !='PUNCT']
    types = list(set(tokens))

    text_char_token_nr      = sum(get_columns(feature_sentence_level,0))
    text_char_type_nr       = sum(length(types))
    text_token_nr           = sum(get_columns(feature_sentence_level,4))
    text_type_nr            = len(types)
    text_mean_char_token    = text_char_token_nr / text_token_nr
    text_mean_char_type     = text_char_type_nr / text_type_nr
    text_longtoken_nr       = sum(get_columns(feature_sentence_level,6))
    text_longtype_nr        = len(over(types, 13))
    text_misspell           = sum(get_columns(feature_sentence_level,8))
    text_lix                = text_token_nr / len(t.sentences) + 100 * len(over(tokens, 6)) / text_token_nr
    text_ttr_log            = bilog(text_type_nr, text_token_nr)
    text_ttr_root           = root(text_type_nr, text_token_nr)

    feature_text_level = [
        text_char_token_nr, text_char_type_nr,
        text_mean_char_token, text_mean_char_type,
        text_token_nr, text_type_nr,
        text_longtoken_nr, text_longtype_nr,
        text_misspell, text_lix, text_ttr_log, text_ttr_root
    ]

    n_general=12
    #for f in [mean_median, stdev]:
    for f in [mean, stdev]:
        for i in range(n_general):
            feature_list = get_columns(feature_sentence_level,i)
            feature_text_level.append(f(feature_list))

    #append the number of sentences and the number of paragraphs
    feature_text_level.append(len(t.sentences))
    feature_text_level.append(len(t.paragraphs))

    #maybe also the distribution of number of sentences in each paragraph
    return feature_sentence_level, feature_text_level



