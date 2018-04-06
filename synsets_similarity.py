#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:51:58 2018

@author: gubo
"""

from nltk.corpus import wordnet

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')

print(w1.wup_similarity(w2))


w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')

print(w1.wup_similarity(w2))


w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')

print(w1.wup_similarity(w2))
