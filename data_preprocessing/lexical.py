#!/usr/bin/env python3
# encoding: utf-8


# The list of lexical features:
'''
Avg KEELY log freq
A1 lemma INCSC
A2 lemma incsc
B1 lemma incsc
B2 lemma incsc
C1 lemma incsc
C2 lemma incsc
difficult word incsc
difficult noun & verb incsc
oov incsc
no lemma incsc
                No lemma incsc, the lemmas of the input sentence are checked against the SALDO lexicon,
                not via the <saldo> element, but rather based on the output of the lemmatizer integrated
                in the Sparv annotation pipeline used for preprocessing, you can find more details at these links:
                https://spraakbanken.gu.se/eng/research/infrastructure/sparv/annotations
                https://ws.spraakbanken.gu.se/ws/sparv/v2/
'''
import math
import pickle
from util import read_json
from util import mean, median, incsc, mean_median
from collections import Counter
from statistics import stdev
#########################################################################
############specific lexical feature computation#########################
#########################################################################

def lemma_incsc(sentence, cefr, kelly_dict, raw=True): # typ is which cefr level the sentence is involved, d is dict
    '''cefr is 1,2,3,4,5,6 which represents A1, A2, B1, B2, C1, C2
    this function will be applied to extra the values of the following features
    A1 lemma INCSC
    A2 lemma INCSC
    B1 lemma INCSC
    B2 lemma INCSC
    C1 lemma INCSC
    C2 lemma INCSC
    #Missing lemma form INCSC
    #Out-of Kelly-list INCSC'''

    t = len(sentence.tokens)
    sent_cefr = []
    for token in sentence.tokens:
        if token.lemma+'|'+token.upos in kelly_dict:
            sent_cefr.append(kelly_dict[token.lemma+'|'+token.upos])
        else:
            sent_cefr.append(None)
    countDict = {t:counts for t,counts in Counter(sent_cefr).most_common()}
    if raw:
        return countDict.setdefault(cefr, 0), t
    return float(countDict.setdefault(cefr,0))/t

def difficult_word(sentence, kelly_dict,noun_verb=False, raw=True):
    #difficult words refer to words above and inclunding B1 level

    t,p = len(sentence.tokens), 0
    for token in sentence.tokens:
        if token.lemma+'|'+token.upos in kelly_dict:
            if kelly_dict[token.lemma+'|'+token.upos] in "3456":
                if noun_verb:
                    if token.upos in ["NOUN","VERB"]: #Only noun and verb valid, not propn and aux
                        p += 1
                        continue
                else:
                    p += 1
    if raw:
        return p, t
    return float(p*1000)/t

def out_of_kelly(sentence, kelly_dict, raw=True):
    if not sentence:
        return 0.0
    t,p = len(sentence.tokens), 0
    for token in sentence.tokens:
        if token.lemma+"|"+token.upos not in kelly_dict:
            p += 1
    if raw:
        return p, t
    return float(p*1000)/t

def kelly_log_frequency(sentence, wpm_dict, raw=True):
    '''WPM is used to compute the avg kelly log freq
    the computation of log-freq for the tokens out of the kelly list is ignored
    the natural logarthigm is used
    '''
    if not sentence:
        return 0.0
    wpms = [wpm_dict[token.lemma+'|'+token.upos] for token in sentence.tokens if token.lemma+'|'+token.upos in wpm_dict]
    if raw:
        return wpms
    return mean([math.log(wpm) for wpm in wpms])

#########################################################################
############lexical feature computation##################################
#sentence_level##############text_level##################################
#########################################################################



def get_lexical(t):
    #L2 features(given Kelly-list based on CEFR vocabulary:
    #https://spraakbanken.gu.se/resource/kelly
    #features are [A1,A2,B1,B2,C1,C2] lemma incsc
    #difficult word incsc
    #difficult noun or verb incsc
    #out of kelly-list incsc
    #missing lemma form incsc
    #avg kelly log-frequency

    '''
    A1 lemma INCSC
    A2 lemma incsc
    B1 lemma incsc
    B2 lemma incsc
    C1 lemma incsc
    C2 lemma incsc
    difficult word incsc
    difficult noun & verb incsc
    oov incsc
    Avg KELLY log freq
    '''

    kelly_dict = read_json('sv_kelly_cefr')
    wpm_dict = read_json('sv_kelly_wpm')

    feature_sentence_level = [[] for _ in range(len(t.sentences))]
    feature_text_count = [[0,0] for _ in range(9)]
    feature_text_count.append([])
    #feature_text_wpms = []
    n_lexical = 10
    feature_text_level = [None for _ in range(n_lexical*3)]

    for index, s in enumerate(t.sentences):

        for feature_n, (p,t) in enumerate([
            lemma_incsc(s, '1', kelly_dict),
            lemma_incsc(s, '2', kelly_dict),
            lemma_incsc(s, '3', kelly_dict),
            lemma_incsc(s, '4', kelly_dict),
            lemma_incsc(s, '5', kelly_dict),
            lemma_incsc(s, '6', kelly_dict),

            difficult_word(s, kelly_dict),
            difficult_word(s, kelly_dict, noun_verb=True),
            out_of_kelly(s, kelly_dict)
        ]):
            feature_sentence_level[index].append(incsc(p,t))
            feature_text_count[feature_n][0] += p
            feature_text_count[feature_n][1] += t

        feature_sentence_level[index].append(kelly_log_frequency(s, wpm_dict,raw=False))
        feature_text_count[-1].extend(kelly_log_frequency(s, wpm_dict,raw=True))

    for feature_n, (p,t) in enumerate(feature_text_count[:-1]):
        feature_list = [feat[feature_n] for feat in feature_sentence_level]
        feature_text_level[feature_n] = incsc(p,t)
        feature_text_level[feature_n + n_lexical] = mean_median(feature_list)
        feature_text_level[feature_n + 2 * n_lexical] = stdev(feature_list)
    #insert wpms
    feature_list = [feat[n_lexical-1] for feat in feature_sentence_level]
    feature_text_level[n_lexical-1] = mean([math.log(wpm) for wpm in feature_text_count[-1]])
    feature_text_level[2*n_lexical-1] = mean_median(feature_list)
    feature_text_level[3*n_lexical-1] = stdev(feature_list)

    return feature_sentence_level, feature_text_level












