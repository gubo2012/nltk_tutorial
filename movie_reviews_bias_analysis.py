#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:11:31 2018

@author: gubo
"""

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    

documents = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids()]

#random.shuffle(documents)

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

# positive data example
#training_set = featuresets[:3900]
#testing_set = featuresets[3900:]

# negative data example
training_set = featuresets[100:]
testing_set = featuresets[:100]


# posterior = prior occurences * likelihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)

# load
#classifier_f = open('naivebayes.pickle', 'rb')
#classifier = pickle.load(classifier_f)
#classifier_f.close()

print('Original Naive Bayes Algo accuracy:', (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classiflier = SklearnClassifier(MultinomialNB())
MNB_classiflier.train(training_set)
print('MNB_classiflier Naive Bayes Algo accuracy:', (nltk.classify.accuracy(MNB_classiflier, testing_set))*100)

#GaussianNB, BernoulliNB
#GaussianNB_classiflier = SklearnClassifier(GaussianNB())
#GaussianNB_classiflier.train(training_set)
#print('GaussianNB_classiflier Naive Bayes Algo accuracy:', (nltk.classify.accuracy(GaussianNB_classiflier, testing_set))*100)

BernoulliNB_classiflier = SklearnClassifier(BernoulliNB())
BernoulliNB_classiflier.train(training_set)
print('BernoulliNB_classiflier Naive Bayes Algo accuracy:', (nltk.classify.accuracy(BernoulliNB_classiflier, testing_set))*100)

#LogisticRegression, SGDClassifier
#SVC, LinearSVC, NuSVC

LogisticRegression_classiflier = SklearnClassifier(LogisticRegression())
LogisticRegression_classiflier.train(training_set)
print('LogisticRegression_classiflier Naive Bayes Algo accuracy:', (nltk.classify.accuracy(LogisticRegression_classiflier, testing_set))*100)

SGDClassifier = SklearnClassifier(SGDClassifier())
SGDClassifier.train(training_set)
print('SGDClassifier Naive Bayes Algo accuracy:', (nltk.classify.accuracy(SGDClassifier, testing_set))*100)

SVC_classiflier = SklearnClassifier(SVC())
SVC_classiflier.train(training_set)
print('SVC_classiflier Naive Bayes Algo accuracy:', (nltk.classify.accuracy(SVC_classiflier, testing_set))*100)

LinearSVC_classiflier = SklearnClassifier(LinearSVC())
LinearSVC_classiflier.train(training_set)
print('LinearSVC_classiflier Naive Bayes Algo accuracy:', (nltk.classify.accuracy(LinearSVC_classiflier, testing_set))*100)

NuSVC_classiflier = SklearnClassifier(NuSVC())
NuSVC_classiflier.train(training_set)
print('NuSVC_classiflier Naive Bayes Algo accuracy:', (nltk.classify.accuracy(NuSVC_classiflier, testing_set))*100)



voted_classifier = VoteClassifier(classifier, MNB_classiflier, LogisticRegression_classiflier, 
                                  SGDClassifier, SVC_classiflier, LinearSVC_classiflier, NuSVC_classiflier)
print('voted_classifier Naive Bayes Algo accuracy:', (nltk.classify.accuracy(voted_classifier, testing_set))*100)