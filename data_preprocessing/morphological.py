#!/usr/bin/env python3
# encoding: utf-8

# This script is to collect morphological features

# The list of morphological features:
'''
1 	 modal verbs to verbs
2 	 particle incsc
3 	 3sg pronoun incsc
4 	 punctuation incsc
5 	 subjunction incsc
6 	 s-verb incsc
7 	 s-verbs to verbs
8 	 adjective incsc
9 	 adjective variation
10 	 adverb incsc
11 	 adverb variation
12 	 noun incsc
13 	 noun variation
14 	 verb incsc
15 	 verb variation
16 	 nominal ratio
17 	 nouns to verbs
18 	 function words incsc
19 	 lexical to non-lexical words
20 	 lexical words to all tokens
21 	 neuter gender noun incsc
22 	 con and subjunction incsc
23   past participles to verbs
24 	 past participles to verbs
25 	 present verbs to verbs
26 	 supine verbs to verbs
27 	 relative structure incsc
28 	 pronouns to nouns
29 	 pronouns to prepositions
'''

from util import mean, median, incsc, mean_median
from statistics import stdev
#########################################################################
############specific morphological feature computation###################
#########################################################################

def verbs_to_verbs(sentence,verb_type, raw=True):
    """this can be applied to   modal verbs to verbs
                                past participles to verbs
                                present verbs to verbs
                                supine verbs to verbs

    The verb type can be
                            Swedish English
                            AUX                             modal(but not vara, ha) ? bli komma
                                        MD                  modal
                            VB|PRS      VBP, VBZ            present
                            VB|PRT      VBD                 past
                            VB|SUP                          supine (only for swedish)
                            Past_part   VBN                 past participles
                            Pres_part   VBG                 present participles

    english has the following pos for verbs
    vb  VerbForm=Fin:/
    vbn Tense=Past|VerbForm=Part
    vbd Tense=Past|VerbForm=Fin
    vbg Tense=Pres|VerbForm=Part or VerbForm=Ger gerund
    vbp Tense=Pres|VerbForm=Fin
    vbz Tense=Pres|VerbForm=Fin
    """

    verb, verbs = 0,0
    if verb_type == "AUX":
        for token in sentence.tokens:
            if token.upos == "AUX": # and token.norm not in ["vara","ha"]:
                verb += 1
                verbs += 1
            elif token.upos in ["VERB","AUX"]:
                verbs += 1
    else:
        for token in sentence.tokens:
            if verb_type == "Pres_part":
                if "VerbForm=Part" in token.feats and "Tense=Pres" in token.feats:
                    verb += 1
            elif verb_type == "Past_part":
                if "VerbForm=Part" in token.feats and "Tense=Past" in token.feats:
                    verb += 1
            elif verb_type in token.xpos:
                    verb += 1
            if token.upos in ["VERB","AUX"]:
                verbs += 1
    if raw:
        return verb, verbs
    if not verbs:
        return 0.0 #needs to be modified
    return float(1000 * verb) / verbs

def pos_incsc(sentence,pos,variation=False, raw=True):
    """This function can be applied to compute for shared specific morphological
     feature in Swedish and English. These parts of speech include
     particle, punctuation, subjunction, adjective, adverb, noun, verb.
     If variation is set to be True, the base is changed from all parts of speech
     into content parts of speech: noun, verb, adverb, adjective

     """
    if not sentence:
        return 0.0
    if variation:
        t = 0
        for token in sentence.tokens:
            if token.upos in ["NOUN","ADJ","ADV","VERB"]: #propn is not included as lexical category
                t += 1
    else:
        t = len(sentence.tokens)
    p = 0
    for token in sentence.tokens:
        if token.upos == pos:
            p += 1
    if raw:
        return p, t
    if not t:
        return 0.0
    return float(1000 * p) / t

def function_tokens_incsc(sentence, lex=False, lex_to_non_lex=False, raw=True):
    if not sentence:
        return 0.0
    t = len(sentence.tokens)
    p = 0

    for token in sentence.tokens:
        if token.upos not in ["NOUN","ADJ","ADV","VERB","PROPN","INTJ"]:#a set of open class words
            p += 1 # p is count for non-lexical tokens
    if raw:
        if lex:
            return p, t
        elif lex_to_non_lex:
            return p, t-p
        else:
            return t-p, t
    if lex:
        return 1000 * (1 - float(p)/t )
    elif lex_to_non_lex and p:
        return 1000 * (float(t-p)/p)
    elif lex_to_non_lex:
        return 0.0 #(needs to be modified here)
    return float(1000 * p) / t

def con_subj_incsc(sentence, raw=True):
    if not sentence:
        return 0.0
    t = len(sentence.tokens)
    p = 0

    for token in sentence.tokens:
        if token.upos in ["CCONJ","SCONJ"]:
            p += 1
    if raw:
        return p,t
    return float(1000 * p) / t

def pos_pos_ratio(sentence, pos1, pos2, raw=True): # this is a ratio, not a incsc score
    """This can be applied for the ratio between two parts of speech
     They can be    pronouns to nouns
                    pronouns to prepositions
                    nouns to verbs
    """
    if not sentence:
        return 0.0
    p1,p2 = 0,0
    for token in sentence.tokens:
        if token.upos == pos1:
            p1 += 1
        elif token.upos == pos2:
            p2 += 1
    if raw:
        return p1, p2
    if not p1 or not p2:
        return 0.0
    return float(p1) / p2 * 1000

def thirdSG_pronoun_incsc(sentence, raw=True): #only for english
    if not sentence:
        return 0.0
    p = 0
    t = len(sentence.tokens)
    for token in sentence.tokens:
        if "Number=Sing" in token.feats and token.upos == "PRON" and token.lemma not in 'jag vi du ni':
            p += 1
    if raw:
        return p,t
    return float(1000 * p) / t

def neuter_gender_noun_incsc(sentence, raw=True): #only for swedish
    if not sentence:
        return 0.0
    p = 0
    t = len(sentence.tokens)
    for token in sentence.tokens:
        if token.upos == "NOUN" and "Gender=Neut" in token.feats:
            p += 1
    if raw:
        return p, t
    return float(1000 * p) / t

def s_verbs_incsc(sentence,verb=False, raw=True):
    """three types
    reciprocal verbs
    passive verbs
    deponent
    AUX is not considered here
    """

    if not sentence:
        return 0.0
    p = 0
    if not verb:
        t = len(sentence.tokens)
    else:
        t = 0
    for token in sentence.tokens:
        if token.upos == "VERB" and verb:
            t += 1
        if token.form[-1] == "s" and token.upos == "VERB":
            p += 1
    if raw:
        return p,t
    return float(1000 * p) / t

def relative_structure_incsc(sentence,raw=True):
    ''' (HA + HD + HP + HS) / (ALL pos tags)
    HA Interrogative/Relative Adverb)
    HD Interrogative/Relative Determiner
    HP Interrogative/Relative Pronoun
    HS Interrogative/Relative Possessive
    '''
    if not sentence:
        return 0.0
    p, t = 0, len(sentence.tokens)

    for token in sentence.tokens:
        '''Int: interrogative; Rel: Relative'''
        if "PronType=Int" in token.feats or "PronType=Rel" in token.feats:
            p += 1
    if raw:
        return p, t
    return float(1000*p) / t

def nominal_ratio(sentence,raw=True):
    '''simple: nn/vb
       full: (nn+pp+pc) / (pn+ab+vb) ''' # pc is participle
    if not sentence:
        return 0.0
    p, t = 0, 0

    for token in sentence.tokens:
        if token.xpos.split("|")[0] in ["NN","PP","PC"]:
            p += 1
        if token.xpos.split("|")[0] in ["PN","AB","VB"]:
            t += 1
    if raw:
        return p, t
    if not t:
        return 0.0
    return float(p)/t * 1000




#########################################################################
############morphological feature computation ###########################
############sentence_level########text_level#############################
#########################################################################




def get_morphological_features(t):
    #hierachical
    #text
        #sentence
    feature_sentence_level = [[] for _ in range(len(t.sentences))]
    feature_text_count = [[0,0] for _ in range(30)]
    feature_text_level = []
    #feature classifications 7 + 3 + 3 + 7 + 5 + 5 = 30 features
    '''
                                            index           category
    I. verbs to verbs
        1. modal verbs to verbs             0   aux
        2. s-verbs to verbs                 1
        3. nouns to verbs                   2
        4. past participles to verbs        3
        5. present participles to verbs     4
        5. past verbs to verbs              5
        6. present verbs to verbs           6
        7. supine verbs to verbs            7
    '''
    '''
    II. one pos to one pos
        1. pronouns to nouns                8
        2. pronouns to prepositions         9
    III. one subpos to all pos
        1. 3sg pronoun incsc                10     3 sg pronoun
        2. s-verb incsc                     11
        3. neuter gender noun incsc         12     plural, omärkerat
    '''
    '''
    IV. one pos to all pos
        1. particle incsc                   13   att
        2. punctuation incsc                14
        3. subjunction incsc                15
        4. adjective incsc                  16
        5. adverb incsc                     17
        6. noun incsc                       18
        7. verb incsc                       19
    '''
    '''
    V. multiple pos to multiple pos
        1. function words incsc             20
        2. lexical to non-lexical words     21
        3. lexical words incsc              22
        4. con and subjunction incsc        23
        5. relative structure incsc         24
    VI. lexical variation
        1. adjective variation              25
        2. adverb variation                 26
        3. noun variation                   27
        4. verb variation                   28
        5. nominal ratio                    29

    verbs: upos: AUX, VERB


    '''

    for index, s in enumerate(t.sentences):


        for feature_n, (p,t) in enumerate([
            # block 1: 8
            verbs_to_verbs(s, 'AUX'),
            s_verbs_incsc(s, verb=True),
            pos_pos_ratio(s, pos1='NOUN', pos2='VERB'),
            verbs_to_verbs(s, 'Past_part'),
            verbs_to_verbs(s, 'Pres_part'),
            verbs_to_verbs(s, 'VB|PRS'),
            verbs_to_verbs(s, 'VB|PRT'),
            verbs_to_verbs(s, 'VB|SUP'),

            #block 2: 2
            pos_pos_ratio(s, pos1='PRON', pos2='NOUN'),
            pos_pos_ratio(s, pos1='PRON',pos2='ADP'),

            #block 3: 3
            thirdSG_pronoun_incsc(s),
            s_verbs_incsc(s),
            neuter_gender_noun_incsc(s),

            #block 4: 7
            pos_incsc(s, pos='PART'),
            pos_incsc(s, pos='PUNCT'),
            pos_incsc(s, pos='SCONJ'),
            pos_incsc(s, pos='ADJ'),
            pos_incsc(s, pos='ADV'),
            pos_incsc(s, pos='NOUN'),
            pos_incsc(s, pos='VERB'),

            #block 5: 5
            function_tokens_incsc(s),
            function_tokens_incsc(s, lex_to_non_lex=True),
            function_tokens_incsc(s, lex=True),
            con_subj_incsc(s),
            relative_structure_incsc(s),

            #block 6: 5
            pos_incsc(s, pos='ADJ',variation=True),
            pos_incsc(s, pos='ADV',variation=True),
            pos_incsc(s, pos='NOUN',variation=True),
            pos_incsc(s, pos='VERB',variation=True),
            nominal_ratio(s)
        ]):
            feature_sentence_level[index].append(incsc(p,t))
            feature_text_count[feature_n][0] += p
            feature_text_count[feature_n][1] += t

    morph_n = len(feature_text_count)
    feature_text_level = [None for _ in range(morph_n*3)]
    for feature_n, (p,t) in enumerate(feature_text_count):
        feature_list = [feat[feature_n] for feat in feature_sentence_level]
        feature_text_level[feature_n] = incsc(p,t)
        feature_text_level[feature_n + morph_n] = mean_median(feature_list)
        feature_text_level[feature_n + morph_n * 2] = stdev(feature_list)

    return feature_sentence_level, feature_text_level


