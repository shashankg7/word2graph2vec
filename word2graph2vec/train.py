#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from data_handler import gen_graphs
from pte_theano import PTE
import logging


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
        self.ndims = 40
        self.lr = 0.04
        self.batch_size = 100
        self.window_size = 10
        self.k = 5
        self.nepochs = 10

    def train(self):
        '''
        run training (first pre-training and than fine tuning on graph with
        parameters defined in constructor.
        '''
        # setting up logger
        logger = logging.getLogger("graph2vec")
        logger.setLevel(logging.INFO)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("word2graph2vec.log")
        formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        p, v1, v2 = self.graphs.gen_edgeprob()
        logger.info("Setting up the model")
        E = self.graphs.nedge
        V = self.graphs.nvertex
        D = self.graphs.ndocs
        L = self.graphs.nlabels
        d = len(self.graphs.w2d)
        l = len(self.graphs.w2l)
        pte = PTE(V, self.ndims, self.graphs.ndocs, self.graphs.nlabels)
        pte.ww_model()
        pte.wd_model()
        pte.wl_model()
        logger.info("Training started")
        for epoch in xrange(0, self.nepochs):
            # Pre-training
            np.random.shuffle(p)
            c = 0
            try:
                # Pre-training on word 2 word model.
                for i in xrange(0, E, self.batch_size):
                    sample = np.random.choice(p.shape[0], self.batch_size, p=p)
                    c = 0
                    for j in xrange(0, sample.shape[0]):
                        indm = v1[sample[j]]
                        indc = v2[sample[j]]
                        indr = np.asarray(
                            np.random.randint(V, size=self.k), dtype=np.int32)
                        cost = pte.pretraining_ww(indm, indc, indr)
                        c += cost
                    logger.info("Cost after training one sample (batch) is %f" % c)
                # Pre-training on word-doc graph
                logger.info("Pre-training on word-word graph done")
                #for i in xrange(0, d):
                #    indw = nnz_wd[i, 0]
                #    indd = nnz_wd[i, 1]
                #    indr = np.asarray(
                #        np.random.randint(V, size=self.k), dtype=np.int32)
                #    if i % 5000 == 0:
                #        logger.info("cost is %f" % c)
                #        c = 0
                #    cost = pte.pretraining_wd(indw, indd, indr, nnz_wd[i, 2])
                #    c += cost
                # Fine-tuning on word-label graph
                #logger.info("Pre-training on word-doc done")
                #for i in xrange(0, l):
                #    indw = nnz_wl[i, 0]
                #    indl = nnz_wl[i, 1]
                #    indr = np.asarray(
                #        np.random.randint(V, size=self.k), dtype=np.int32)
                #    if i % 5000 == 0:
                #        logger.info("cost is %f" % c)
                #        c = 0
                #    cost = pte.finetuning(indw, indl, indr, nnz_wl[i, 2])
                #    c += cost
            except Exception as e:
                logger.exception("Following exception occured %s" % e)
            logger.info("Pre-training on word-label done")
        logger.info("training done, saving model")
        pte.save_model()

if __name__ == "__main__":
    pte = train_pte()
    pte.train()
