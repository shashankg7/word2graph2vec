#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from data_handler import gen_graphs
from pte import PTE

class train_pte(object):
    '''
    class to train the pte model on the given corpus.
    '''
    def __init__(self):
        '''
        define model paramters.
        '''
        self.gen_graphs
        self.lr = 0.04
        self.window_size = 10
        self.k = 5
        self.nepochs = 2

    def train(self):
        '''
        run sgd on graph with parameters defined in constructor.
        '''
        for epoch in xrange(0, self.nepochs):

