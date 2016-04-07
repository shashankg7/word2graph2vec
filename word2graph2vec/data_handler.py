#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from nltk.corpus import movie_reviews
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from scipy import sparse
import string
import os
import sys
import re
import json
import nltk
import pdb
import time
import numpy as np
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
        # Store all u,v pairs
        self.w2w_inv = []
        self.w2l = {}
        self.w2d = {}
        self.all_words = {}
        self.all_labels = {}
        self.all_documents = {}
        self.nedge = 0
        self.ndocs = 0
        self.nlabels = 0
        nltk.data.path.append(path)

    def contruct_graphs(self, args):
        '''
        Function to read text file from path and construct the corresponding
        graphs.
        '''
        # documents = [(list(w.lower() for w in movie_reviews.words(fileid) if w.lower() not in string.punctuation), category, fileid)
        #              for category in movie_reviews.categories()
        #              for fileid in movie_reviews.fileids(category)]

        documents = []
        files = ['../data/train-pos.txt','../data/train-neg.txt','../data/test-pos.txt','../data/test-neg.txt']
        class_labels = ['pos','neg','pos','neg']
        document_no = 1
        index = 0
        for file_name in files:
            fp = open(file_name)
            lines = fp.readlines()
            for line in lines:
                words = line.split(" ")
                document = (words,class_labels[index],document_no)
                documents.append(document)
                document_no += 1
            index += 1

        print documents[0]

        unique_count = 0
        for index in range(len(documents)):
            for word_index in range(len(documents[index][0])):
                word = documents[index][0][word_index]
                if word not in self.all_words:
                    self.all_words[word] = unique_count
                    unique_count = unique_count + 1
        self.nvertex = unique_count
        unique_count = 0
        for c in class_labels:
            if c not in self.all_labels:
                self.all_labels[c] = unique_count
                unique_count = unique_count + 1
        self.nlabels = unique_count
        unique_count = 0
        for i in xrange(1,document_no):
            self.all_documents[i] = unique_count
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
                    if u not in self.w2w:
                        self.w2w[u] = {}
                    if v not in self.w2w[u]:
                        self.w2w[u][v] = 0
                    self.w2w[u][v] += 1
                    if v not in self.w2w:
                        self.w2w[v] = {}
                    if u not in self.w2w[v]:
                        self.w2w[v][u] = 0
                    self.w2w[v][u] += 1

                for i in xrange(word_index + 1, right):
                    u = self.all_words[word]
                    v = self.all_words[documents[index][0][i]]
                    if u not in self.w2w:
                        self.w2w[u] = {}
                    if v not in self.w2w[u]:
                        self.w2w[u][v] = 0
                    self.w2w[u][v] += 1

                    if v not in self.w2w:
                        self.w2w[v] = {}
                    if u not in self.w2w[v]:
                        self.w2w[v][u] = 0
                    self.w2w[v][u] += 1

                u = self.all_documents[documents[index][2]]
                v = self.all_words[word]
                if u not in self.w2d:
                    self.w2d[u] = {}
                if v not in self.w2d[u]:
                    self.w2d[u][v] = 0
                self.w2d[u][v] += 1

                u = self.all_labels[documents[index][1]]
                if u not in self.w2l:
                    self.w2l[u] = {}
                if v not in self.w2l[u]:
                    self.w2l[u][v] = 0
                self.w2l[u][v] += 1

        json.dump(self.all_words, open('word_mapping.json', 'wb'))
        json.dump(self.all_labels, open('label_mapping.json', 'wb'))
        json.dump(self.all_documents, open('document_mapping.json', 'wb'))
        print 'w2l', len(self.w2l.keys())
        print 'w2d', len(self.w2d.keys())
        print 'w2w', len(self.w2w.keys())

    def gen_edgeprob(self):
        '''
        returns edge probability vector (w2w graph)
        '''
        # Edge probability vector
        p = []
        v1 = []
        v2 = []
        for k in self.w2w.keys():
            for kj in self.w2w[k].keys():
                p.append(self.w2w[k][kj])
                v1.append(k)
                v2.append(kj)
                self.nedge += 1
        p = np.asarray(p, dtype=np.float64)
        p = p / float(sum(p))
        return p, v1, v2

if __name__ == "__main__":
    graph = gen_graphs()
    graph.contruct_graphs("graph")
    #t = time.time()
    #p = graph.gen_edgeprob()
    #t = time.time() - t
    #print sum(p)
    #print len(p), t
