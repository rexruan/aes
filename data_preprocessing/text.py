#!/usr/bin/env python3
# encoding: utf-8

# This script is to represent the text in word, sentence and paragraph level.



class Token:
    '''a class representation of a token'''
    def __init__(self, attributes):
        n, form, norm, lemma, upos, xpos, feats, head, deprel, deps, misc = attributes
        self.n = n
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc
        self.norm = norm
        self.ufeats = None
        self.path = None
        self.dep_length = None
        self.normalized = False

        self.ix = None

    def __repr__(self):
        return self.form

    def char_repr(self, char_dict):
        return [char_dict[char] for char in self.form]

class Sentence:
    '''a class representation of a sentence'''
    def __init__(self, tokens): # tokens is a list of words
        self.tokens = tokens
        self.max_word_len = max([len(token.form) for token in self.tokens])
        self.vocab = set([token.form for token in self.tokens])
        self.sent = " ".join([token.form for token in self.tokens])
        self.ix = None
        self.ud = False 

    def __repr__(self):
        return self.sent

class Paragraph:
    '''a class representation of a paragraph'''
    def __init__(self, sents):
        self.sents = sents
        self.max_word_len = max([sent.max_word_len for sent in self.sents])
        self.max_sent_len = max([len(sent.tokens) for sent in self.sents])
        self.vocab = set([form for sent in self.sents for form in list(sent.vocab)])
        self.sent_set = set([sentence.sent for sentence in self.sents])
        self.paragraph = " ".join([sentence.sent for sentence in self.sents])
        self.ix = None

    def __repr__(self):
        return self.paragraph

class Text:
    '''a class representation of a text'''
    def __init__(self, id, paragraphs):
        '''The following parameters are used to compute the blanks in lstm model'''
        self.paragraphs = paragraphs
        self.sentences = [sent for para in self.paragraphs for sent in para.sents]
        self.max_word_len = max([paragraph.max_word_len for paragraph in self.paragraphs])        #the maximum length of word in all texts
        self.max_sent_len = max([paragraph.max_sent_len for paragraph in self.paragraphs])    #the maximum length of sentence in all texts
        self.max_para_len = max([len(paragraph.sents) for paragraph in self.paragraphs])              #the maximum length of paragraph in all texts
        self.vocab = set([form for p in self.paragraphs for form in list(p.vocab)])
        self.sent_set = set([sent for p in self.paragraphs for sent in list(p.sent_set)])
        self.para_set = set([p.paragraph for p in self.paragraphs])
        #self.alphabet = set([char for word in list(self.vocab) for char in word])
        self.text= " ".join([p.paragraph for p in self.paragraphs])
        self.id = id
        self.ud = False

        ###attributes for the text
        self.sample = None
        self.grade = None
        self.time = None
        self.genre = None
        self.sex = None
        self.subject = None
        self.permission = None
        self.place = None
        self.education = None
        self.format = None

        ###attributes for the text from data of Robert
        self.grade2 = None
        self.topic1 = None
        self.topic2 = None

        #self.paragraphs = []   contains ints of lengths for each paragraph
        self.paragraph_sents = []

        self.eligible = False

        self.normalized = False
