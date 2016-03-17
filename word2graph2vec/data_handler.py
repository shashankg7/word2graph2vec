#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import os, sys, re
import numpy as np

path = '../data/file_name'


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

    def contruct_graphs(self, args):
        '''
        Function to read text file from path and construct the corresponding
        graphs.
        '''
        raise NotImplementedError
