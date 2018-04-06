#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:51:58 2018

@author: gubo
"""

from nltk.corpus import wordnet

syns = wordnet.synsets('program')
#syns = wordnet.synsets('buckeyes')

print(syns)

print(syns[0].name())

print(syns[0].lemmas()[0].name())

# definition
print(syns[0].definition())

# example
print(syns[0].examples())


synonyms = []
antonyms = []

for syn in wordnet.synsets('good'):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print('synonyms: ', set(synonyms))
print('antonyms: ', set(antonyms))