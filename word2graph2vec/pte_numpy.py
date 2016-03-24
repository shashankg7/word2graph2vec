#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import expit

class PTE(object):

    '''
    Defines the PTE model (cost function, parameters.
    '''

    def __init__(self, nvertex, out_dim, ndocs, nlabels, lr):
        '''
        Parameters specs:
            nvertex : no of vertices in the graph
            out_dim : node vector dimension
            ndocs : number of documents in the corpus
            nlabels : number of labels
            lr : learning rate.
        '''
        # TO-DO: Try initialization from uniform
        # Initialize model paramters
        self.W = np.asarray(np.random.rand(nvertex, out_dim) / float(out_dim),
                       dtype=np.float32)
        self.W1 = np.asarray(np.random.rand(nvertex, out_dim) / float(out_dim),
                        dtype=np.float32)
        self.D = np.asarray(np.random.rand(ndocs, out_dim) / float(out_dim),
                       dtype=np.float32)
        self.L = np.asarray(np.random.rand(nvertex, out_dim) / float(out_dim),
                       dtype=np.float32)
        self.gW = np.asarray(np.ones(nvertex, out_dim), dtype=np.float32)
        self.gW1 = np.asarray(np.ones(nvertex, out_dim), dtype=np.float32)
        self.gD = np.asarray(np.ones(nvertex, out_dim), dtype=np.float32)
        self.gL = np.asarray(np.ones(nvertex, out_dim), dtype=np.float32)

    def pretraining_ww(self, indm, indc, indr, w):
        '''
        Performs SGD update (pre-training on ww graph).
        '''
        w = W[indm, :]
        w1 = W1[indc, :]
        wR = W1[indr, :]
        cost = - 1 * np.log(expit(np.dot(w, w1))) 
        for k in range(wR.shape[0]):
            cost -= np.log(expit(- np.dot(w, wR[k, :])))
        # Compute gradients 
        return self.func_ww(indm, indc, indr)

    def pretraining_wd(self, indm, ind_doc, indr_doc):
        '''
        SGD update (pre-training on wd graph).
        '''
        w = W[indm, :]
        d = D[indc, :]
        wR = W[indr, :]
        cost = - 1 * np.log(expit(np.dot(w, d))) 
        for k in range(wR.shape[0]):
            cost -= np.log(expit(- np.dot(w, wR[k, :])))
        return self.fun_wd(self, indm, ind_doc, indr_doc)

    def finetuning(self, indm, indl, indr_wl):
        '''
        SGD update (finetuning on wl graph using w embeddings)
        '''
        return self.fun_wl(self, indm, indl, indr_wl)

    def save_model(self):
        '''
        Save embedding matrices on disk
        '''
        W = self.W.get_value() + self.W1.get_value()
        D = self.D.get_value()  # + self.W1.get_value()
        L = self.L.get_value()  # + self.W1.get_value()
        np.save('lookupW', W)
        np.save('lookupD', D)
        np.save('lookupL', L)
