#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 21:48:16 2018

@author: gubo
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_txt = "Asian equities were mixed, while U.S. stock futures slid and the yen rose after President Donald Trump ordered his administration to consider tariffs on an additional $100 billion worth of Chinese imports, dashing investor optimism that trade tensions could ease with negotiations on the horizon."
stop_words = set(stopwords.words("english"))

words = word_tokenize(example_txt)

filtered_sentence = []
removed_words = []
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
    else:
        removed_words.append(w)
        
print('filtered_sentence', filtered_sentence)
print('removed_words', removed_words)