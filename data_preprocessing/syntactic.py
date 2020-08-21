#!/usr/bin/env python3
# encoding: utf-8

# This script is to collect syntactic features

# The list of syntactic features:
'''
dep_length_5+
dep_length_max
dep_length
dep_left
def_right
mod
pre
post
sub
relative
prep
'''

from collections import Counter
import time
import math
import re
from copy import deepcopy
import os
import subprocess
import pickle


from prepare_data import is_a_ud_tree
from util import mean, median, incsc, mean_median
from statistics import stdev
###########################################################
######Functions used for syntactic feature extractions#####
###########################################################







def get_path(n, heads): #n is an index in the sentence to identify the word
    path = [n]
    while n:
        path.append(heads[n])
        n = heads[n]
    return path

def dep_heads_helper(sentence, udpipe=False, heads=[]):
    if not udpipe:
        heads = [int(token.head) for token in sentence.tokens]
    is_ud = True
    if not heads:
        is_ud = False
    if not is_a_ud_tree(heads):
        #print("It is Not a standard ud tree.")
        is_ud = False
    if is_ud:
        heads.insert(0,None)
        return heads
    if udpipe:
        print(heads)
        import sys
        sys.exit()
    #use udpipe to reparse the sentence
    with open('cache.udpipe','w') as f:
        f.write('\n'.join([word.norm for word in sentence.tokens]))

    udpipe_dir = '/Users/rex/Desktop/udpipe-master/src/udpipe'
    model_dir = '/Users/rex/Desktop/AES_rex/sv_udpipe'
    with open('cache.udpipe.conll','w') as f:
        subprocess.call([udpipe_dir, '--tag', '--parse','--input', 'vertical', model_dir, 'cache.udpipe'], stdout=f)
    with open('cache.udpipe.conll','r') as f:
        lines = f.readlines()
    print(heads)
    heads = [int(line.split('\t')[6]) for line in lines if line.strip() if line[:2] != '# ']
    print(heads)
    return dep_heads_helper(None, udpipe=True, heads=heads)

def dep_paths(sentence):
    heads = dep_heads_helper(sentence)
    tokens = [token.norm for token in sentence.tokens]
    tokens.insert(0, "ROOT")
    if not heads:
        return []
    dep_paths = []
    for i in range(1,len(heads)):
        dep_paths.append(get_path(i,heads))
    for path in dep_paths:
        for i in range(len(path)):
            path[i] = tokens[path[i]]
    return dep_paths

def arc5plus(sentence):
    dep_p = dep_paths(sentence)
    if not dep_p:
        return 0
    longarcs = []
    arc5 = 6  #6 nodes, 5 arcs
    for path in dep_p:
        while arc5 < len(path)+1:
            for i in range(len(path)+1-arc5):
                longarc = path[i:i+arc5]
                if longarc not in longarcs:
                    longarcs.append(longarc)
            arc5 += 1
        arc5 = 6
    return len(longarcs)

def dep_length_per_sentence(sentence):
    heads = dep_heads_helper(sentence)
    if not heads:
        return 0.0
    dep_lengths = []
    for i in range(1,len(heads)):
        dep_lengths.append(get_path(i, heads))
    dep_length_nos = [len(path)-1 for path in dep_lengths] #nos: number of strings

    return sum(dep_length_nos)/float(len(dep_length_nos))

def avg_dep_length(text):
    dep = 0
    for sentence in text.sentences:
        dep += dep_length_per_sentence(sentence)
    if not text.sentences:
        return 0.0
    return dep/float(len(text.sentences))

def median_dep_length(text):
    med = []
    for sentence in text.sentences:
        heads = dep_heads_helper(sentence)
        if not heads:
            return []
        for i in range(1, len(heads)):
            med.append(len((get_path(i, heads))))
    return med

def dep_dir_count(sentence, dir, ratio=False): #sentence and direction
    heads = dep_heads_helper(sentence)
    if not heads:
        return 0.0, 0.0
    dir_arc = 0
    if dir == "left":
        for i in range(1, len(heads)):
            if heads[i] < i:
                dir_arc += 1
    else:
        for i in range(1, len(heads)):
            if heads[i] > i:
                dir_arc += 1
    if ratio:
        return dir_arc/float(len(heads)-1)
    return dir_arc, len(heads)-1

def mod_incsc(sentence, pre_mod=False, post_mod=False, raw=False):
    '''A nominal head does not take any core arguments but may be associated with different types of modifiers:
    An nmod is a nominal phrase modifying the head of another nominal phrase, with or without a special case marker. Treebanks may optionally use nmod:poss to distinguish non-adpositional possessives.
    An appos is a nominal phrase that follows the head of another nominal phrase and stands in a co-reference or other equivalence relation to it.
    An amod is an adjective modifying the head of a nominal phrase.
    A nummod is a numeral modifying the head of a nominal phrase.
    An acl is a clause modifying the head of a nominal phrase, with the relative clause acl:relcl as an important subtype.'''
    heads = dep_heads_helper(sentence)
    if not heads:
        return 0.0
    dep_rels = [token.deprel for token in sentence.tokens]
    pre, post = 0, 0

    for i in range(len(dep_rels)):
        if dep_rels[i] in ['nmod','appos','nummod','advmod','discourse','amod']:
            if i+1 < heads[i+1]:
                pre += 1
            elif i+1 > heads[i+1]:
                post += 1
            else:
                print('Error')
    if pre_mod:
        if raw:
            return pre, len(dep_rels)
        return 1000 / len(dep_rels) * pre
    if post_mod:
        if raw:
            return post, len(dep_rels)
        return 1000 / len(dep_rels) * post
    if raw:
        return pre+post, len(dep_rels)
    return 1000 / len(dep_rels) * (pre+post)

def find_children(heads, i): #find all children that have i as their direct or indirect head.
    children = []
    while i in heads:
        child = heads.index(i)
        children.append(child)
        heads[heads.index(i)] = None
        for c in find_children(heads, child):
            children.append(c)
    return children

def sub_incsc(sentence, raw=False):
    '''including four types:
    1 Clausal subjects(csubj)
    2 Clausal complements(objects), divided into those with obligatory control(xcomp) and those without(ccomp)
    3 Adverbial clause modifers(advcl)
    4 Adnominal clause modifiers(acl) (with relative clause as an important subtype in many languages)
    see https://universaldependencies.org/u/overview/complex-syntax.html#subordination'''
    heads = dep_heads_helper(sentence)
    if not heads:
        return 0.0
    dep_rels = [token.deprel for token in sentence.tokens]
    sub = set()

    for index in range(len(dep_rels)):
        if dep_rels[index] in ['csubj','xcomp','ccomp','advcl','acl','acl:relcl', 'acl:rel']:
            children = find_children(heads, index+1)
            for child in children:
                sub.add(child)
    if raw:
        return len(dep_rels), len(sub)
    return 1000 / len(dep_rels) * len(sub)

def relative_incsc(sentence, raw=False):
    '''A relative clause is an instance of acl,
    characterized by finiteness and usually omission of the modified noun in the embedded clause.
    Some languages use a language-particular subtype acl:relcl
    for the traditional class of relative clauses.'''
    heads = dep_heads_helper(sentence)
    if not heads:
        return 0.0
    dep_rels = [token.deprel for token in sentence.tokens]
    rel = set()
    for index in range(len(dep_rels)):
        if re.search(r"rel", dep_rels[index]):
            rel.add(index+1)
            children = find_children(heads, index+1)
            for child in children:
                rel.add(child)
    if raw:
        return len(dep_rels), len(rel)
    return 1000 / len(dep_rels) * len(rel)

def prep_incsc(sentence, raw=False):
    heads = dep_heads_helper(sentence)
    if not heads:
        return 0.0
    dep_rels = [token.deprel for token in sentence.tokens]
    pp = set()
    for index in range(len(dep_rels)):
        if dep_rels[index] == 'case':
            heads_copy = deepcopy(heads)
            pp.add(heads[index+1])
            children = find_children(heads_copy, heads[index+1])
            for child in children:
                pp.add(child)

    #To print all prepositional phrases given the following line:
    #print('pp',[sentence.tokens[i-1].norm for i in pp])
    if raw:
        return len(dep_rels), len(pp)
    return 1000 / len(dep_rels) * len(pp)

def left_dep_ratio(text, left_count=False):
    left, total = 0, 0
    left_list = []
    for sentence in text.sentences:
        l, t = dep_dir_count(sentence, "left")
        left += l
        total += t
        if left_count:
            left_list.append(l)
    if left_count:
        return left_list
    if total:
        return left/float(total)
    else:
        return 0.0

def right_dep_ratio(text, right_count=False):
    right, total = 0, 0
    right_list = []
    for sentence in text.sentences:
        r, t = dep_dir_count(sentence, "right")
        right += r
        total += t
        if right_count:
            right_list.append(r)
    if right_count:
        return right_list
    if total:
        return right/float(total)
    else:
        return 0.0





##########################################################
#########Build embeddings for syntactic features##########
##########################################################



def raw_p_t_update(l,t): # l:list, t:tuple
    l[0] = l[0] + t[0]
    l[1] = l[1] + t[1]
    return l


def get_syntactic_features(t):
    #hierachical
    #text
        #sentence
    # the length of embedding is [[(1 + 1 + 9 * 2) * len(text.sents)] for text in texts]
    #sequence of storing syn_embedding
    #
    #   feature name          function name (text)              function name (sentence)
    #1, deparc5+                                                arc5plus
    #2, max length deparc     max(median_dep_list(text)) -1     max([len(i) for i in dep_paths(sentence)]) - 1
    #3, dep_length            avg_dep_length  (median|total)    dep_length_per_sentence
    #4, right deparc ratio    right_dep_ratio (median|total)    dep_dir_count(sentence, 'right', ratio=False) dir::direction
    #5, left deparc ratio     left_dep_ratio  (median|total)    dep_dir_count(sentence, 'left', ratio=False)
    #6, modifier variation                                      mod_incsc(sentence)
    #7, pre-modifier                                            mod_incsc(sentence, pre=True)
    #8, post-modifier                                           mod_incsc(sentence, post=True)
    #9, subordinate                                             sub_incsc(sentence)
    #10, relative clause                                        relative_incsc(sentence)
    #11, pp complement                                          prep_incsc(sentence)





    feature_sentence_level = []
    text_deparc5, text_max_deparc = 0, 0
    text_total_dep_length =  0
    text_right_dep_length, text_left_dep_length = 0,0

    text_deparc5s = []
    text_max_deparcs = []
    text_dep_length_list = [] # to compute text_avg_dep_length, text_median_dep_length
    text_right_dep_list = []
    text_left_dep_list = []
    text_modifier_variation_list = []
    text_modifier_variation_raw = [0,0]
    text_pre_list = []
    text_pre_raw = [0,0]
    text_post_list = []
    text_post_raw = [0,0]
    text_sub_list = []
    text_sub_raw = [0,0]
    text_relative_list = []
    text_relative_raw = [0,0]
    text_comp_list = []
    text_comp_raw = [0,0]

    for s in t.sentences:
        '''block for syntactic feature in sentence level'''
        s_deparc5 = arc5plus(s)
        s_max_deparc = max([len(i) for i in dep_paths(s)]) - 1
        s_total_dep_length = dep_length_per_sentence(s)
        s_right_dep_ratio = dep_dir_count(s,'right',ratio=True)
        s_left_dep_ratio = dep_dir_count(s, 'left', ratio=True)
        s_modifier_variation = mod_incsc(s)
        s_pre = mod_incsc(s, pre_mod=True)
        s_post = mod_incsc(s, post_mod=True)
        s_sub = sub_incsc(s)
        s_relative = relative_incsc(s)
        s_comp = prep_incsc(s)
        feature_sentence_level.append([s_deparc5, s_max_deparc, s_total_dep_length,
                              s_right_dep_ratio, s_left_dep_ratio, s_modifier_variation,
                              s_pre, s_post, s_sub, s_relative, s_comp])

        '''modify the feature value for text level'''
        text_deparc5s.append(s_deparc5)
        text_max_deparcs.append(s_max_deparc)
        text_right_dep_length += dep_dir_count(s, 'right')[0]
        text_left_dep_length += dep_dir_count(s,'left')[0]
        text_total_dep_length += s_total_dep_length # refers to sum of averaged dep length from each sentence
        text_dep_length_list.extend([len(i)-1 for i in dep_paths(s)])
        text_right_dep_list.append(s_right_dep_ratio)
        text_left_dep_list.append(s_left_dep_ratio)
        text_modifier_variation_list.append(s_modifier_variation)
        text_modifier_variation_raw = raw_p_t_update(text_modifier_variation_raw, mod_incsc(s, raw=True))
        text_pre_list.append(s_pre)
        text_pre_raw = raw_p_t_update(text_pre_raw, mod_incsc(s, pre_mod=True, raw=True))
        text_post_list.append(s_post)
        text_post_raw = raw_p_t_update(text_post_raw, mod_incsc(s,post_mod=True, raw=True))
        text_sub_list.append(s_sub)
        text_sub_raw = raw_p_t_update(text_sub_raw, sub_incsc(s,raw=True))
        text_relative_list.append(s_relative)
        text_relative_raw = raw_p_t_update(text_relative_raw, relative_incsc(s, raw=True))
        text_comp_list.append(s_comp)
        text_comp_raw = raw_p_t_update(text_comp_raw,prep_incsc(s, raw=True))


    # add feature value based on the text level
    feature_text_level = [sum(text_deparc5s), max(text_max_deparcs), mean(text_dep_length_list),
            text_right_dep_length/(text_right_dep_length+text_left_dep_length),
            text_left_dep_length/(text_right_dep_length+text_left_dep_length),
            incsc(*text_modifier_variation_raw), incsc(*text_pre_raw),
            incsc(*text_post_raw), incsc(*text_sub_raw), incsc(*text_relative_raw),
            incsc(*text_comp_raw)]

    # add feature values to describe the distibution on sentence level: mean(mean, median), stdev
    #append the feature values for the text level
    for f in [mean_median, stdev]:
        for l in [text_deparc5s, text_max_deparcs, text_dep_length_list,
            text_right_dep_list, text_left_dep_list,text_modifier_variation_list,
            text_pre_list, text_post_list,text_sub_list, text_relative_list,
            text_comp_list]:
            feature_text_level.append(f(l))



    return feature_sentence_level, feature_text_level














