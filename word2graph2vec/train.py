#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from data_handler import gen_graphs
from pte_numpy import PTE


class train_pte(object):

    '''
    class to train the pte model on the given corpus.
    '''

    def __init__(self):
        '''
        define model paramters.
        '''
        self.graphs = gen_graphs()
        self.graphs.contruct_graphs("graph")
        # Generate nnz matrices with data (i, j, w)
        self.nnz_ww = []
        self.nnz_wd = []
        self.nnz_wl = []
        self.lr = 0.04
        self.window_size = 10
        self.k = 5
        self.nepochs = 2

    def train(self):
        '''
        run training (first pre-training and than fine tuning on graph with
        parameters defined in constructor.
        '''
        # Generate nnz from graphs
        pte = PTE()
        self.nnz_ww = np.zeros((len(self.graphs.w2w), 3), dtype=np.int32)
        self.nnz_wd = np.zeros((len(self.graphs.w2d), 3), dtype=np.int32)
        self.nnz_wl = np.zeros((len(self.graphs.w2l), 3), dtype=np.int32)
        self.nnz_ww[:, 0] = map(lambda x:x[0], self.graphs.w2w.keys())
        self.nnz_ww[:, 1] = map(lambda x:x[1], self.graphs.w2w.keys())
        self.nnz_ww[:, 2] = self.graphs.w2w.values()
        self.nnz_wd[:, 0] = map(lambda x:x[0], self.graphs.w2d.keys())
        self.nnz_wd[:, 1] = map(lambda x:x[1], self.graphs.w2d.keys())
        self.nnz_wd[:, 2] = self.graphs.w2d.values()
        self.nnz_wl[:, 0] = map(lambda x:x[0], self.graphs.w2l.keys())
        self.nnz_wl[:, 1] = map(lambda x:x[1], self.graphs.w2l.keys())
        self.nnz_wl[:, 2] = self.graphs.w2l.values()

        for epoch in xrange(0, self.nepochs):
            # Pre-training
            np.random.shuffle(self.nnz_ww)
            np.random.shuffle(self.nnz_wd)
            for i in xrange(0, self.nnz_ww.shape[0]):
                indm = nnz_ww[i, 0]
                indc = nnz_ww[i, 1]
                indr = 0
                cost = pte.pretraining_ww()
            
            
if __name__ == "__main__":
    pte = train_pte()
    pte.train()
