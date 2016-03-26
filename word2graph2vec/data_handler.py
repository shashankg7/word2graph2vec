#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from nltk.corpus import movie_reviews
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string
import os
import sys
import re
import json
import nltk
path = '../data'


class gen_graphs(object):

    '''
    class that generates word-word, word-label and word-document graph from text
    '''

    def __init__(self):
        '''
        Initialize each graph
        '''
        # word2word graph
        self.w2w = {}
        self.w2l = {}
        self.w2d = {}
        self.all_words = {}
        self.all_labels = {}
        self.all_documents = {}
        self.nvertex = 0
        self.ndocs = 0
        self.nlabels = 0
        nltk.data.path.append(path)

    def contruct_graphs(self, args):
        '''
        Function to read text file from path and construct the corresponding
        graphs.
        '''
        documents = [(list(w.lower() for w in movie_reviews.words(fileid) if w.lower() not in string.punctuation), category, fileid)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]

        unique_count = 0
        for index in range(len(documents)):
            for word_index in range(len(documents[index][0])):
                word = documents[index][0][word_index]
                if word not in self.all_words:
                    self.all_words[word] = unique_count
                    unique_count = unique_count + 1
        self.nvertex = unique_count
        unique_count = 0
        for category in movie_reviews.categories():
            self.all_labels[category] = unique_count
            unique_count = unique_count + 1
        self.nlabels = unique_count
        unique_count = 0
        for category in movie_reviews.categories():
            for fileid in movie_reviews.fileids(category):
                self.all_documents[fileid] = unique_count
                unique_count = unique_count + 1
        self.ndocs = unique_count

        window_size = 10

        for index in range(len(documents)):
            for word_index in range(len(documents[index][0])):
                word = documents[index][0][word_index]
                if (word_index - window_size / 2) >= 0:
                    left = word_index - window_size / 2
                else:
                    left = 0
                if (word_index + window_size / 2) < len(documents[index][0]):
                    right = word_index + window_size / 2 + 1
                else:
                    right = len(documents[index][0])

                for i in xrange(left, word_index):
                    u = self.all_words[word]
                    v = self.all_words[documents[index][0][i]]
                    if (u, v) not in self.w2w and (v, u) not in self.w2w:
                        self.w2w[(u, v)] = 1
                        self.w2w[(v, u)] = 1
                    else:
                        self.w2w[(u, v)] += 1
                        self.w2w[(v, u)] += 1

                for i in xrange(word_index + 1, right):
                    u = self.all_words[word]
                    v = self.all_words[documents[index][0][i]]
                    if (u, v) not in self.w2w and (v, u) not in self.w2w:
                        self.w2w[(u, v)] = 1
                        self.w2w[(v, u)] = 1
                    else:
                        self.w2w[(u, v)] += 1
                        self.w2w[(v, u)] += 1
                v = self.all_documents[documents[index][2]]
                u = self.all_words[word]
                if (u, v) not in self.w2d:
                    self.w2d[(u, v)] = 1
                else:
                    self.w2d[(u, v)] += 1

                v = self.all_labels[documents[index][1]]
                if (u, v) not in self.w2l:
                    self.w2l[(u, v)] = 1
                else:
                    self.w2l[(u, v)] += 1

        json.dump(self.all_words, open('word_mapping.json', 'wb'))
        json.dump(self.all_labels, open('label_mapping.json', 'wb'))
        json.dump(self.all_documents, open('document_mapping.json', 'wb'))

if __name__ == "__main__":
    graph = gen_graphs()
    graph.contruct_graphs("graph")
    print graph.w2w
