#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:11:31 2018

@author: gubo
"""

import nltk
import random
from nltk.corpus import movie_reviews
import pickle

documents = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids()]

random.shuffle(documents)

#print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# most frequent words
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))

#print(all_words['nice'])

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:3900]
testing_set = featuresets[3900:]

# posterior = prior occurences * likelihood / evidence

#classifier = nltk.NaiveBayesClassifier.train(training_set)

# load
classifier_f = open('naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

print('Naive Bayes Algo accuracy:', (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

# save
#save_classifier = open('naivebayes.pickle','wb')
#pickle.dump(classifier, save_classifier)
#save_classifier.close()
