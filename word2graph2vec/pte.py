#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from theano import tensor as T
import theano
import numpy as np
import scipy

class PTE(object):
    '''
    Defines the PTE model (cost function, parameters in theano.
    '''
    def __init__(self, inp_size, out_size, lr):
        '''
        Parameters specs:
            inp_size : no of nodes in joint graph
            out_size : dimension of node vector
            lr : learning rate.
        '''


