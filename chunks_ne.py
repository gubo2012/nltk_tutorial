#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:34:49 2018

@author: gubo
"""


import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

tokenizer = PunktSentenceTokenizer(train_text)

tokenized = tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            name_ent = nltk.ne_chunk(tagged)
            
#            print(chunked)
            name_ent.draw()
    
    except Exception as e:
        print(str(e))
        
process_content()

#print(nltk.__file__)