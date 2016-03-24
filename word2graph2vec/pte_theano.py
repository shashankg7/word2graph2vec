#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from theano import tensor as T
import theano
import numpy as np


class PTE(object):
    '''
    Defines the PTE model (cost function, parameters in theano.
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
        W = np.asarray(np.random.rand(nvertex, out_dim) / float(out_dim),
                       dtype=theano.config.floatX)
        W1 = np.asarray(np.random.rand(nvertex, out_dim) / float(out_dim),
                        dtype=theano.config.floatX)
        D = np.asarray(np.random.rand(ndocs, out_dim) / float(out_dim),
                       dtype=theano.config.floatX)
        L = np.asarray(np.random.rand(nvertex, out_dim) / float(out_dim),
                       dtype=theano.config.floatX)
        self.W = theano.shared(W, name='W', borrow=True)
        self.W1 = theano.shared(W1, name='W1', borrow=True)
        self.D = theano.shared(D, name='D', borrow=True)
        self.L = theano.shared(L, name='L', borrow=True)
        # Temperory variables (theano does not allows muliple updates for same
        # variable
        self.C = theano.shared(W1, name='C', borrow=True)
        self.C1 = theano.shared(W, name='C1', borrow=True)
        self.C2 = theano.shared(L, name='C2', borrow=True)
        # Indices of main node, context node, k ww samples, doc node,k wd
        # samples, label node, k wl samples respectively
        indm = T.iscalar()
        indc = T.iscalar()
        indr = T.ivector()
        ind_doc = T.iscalar()
        indr_wd = T.ivector()
        indl = T.iscalar()
        indr_wl = T.ivector()
        # extract embedding of the corresponding nodes
        w = self.W[indm, :]
        w1 = self.W1[indc, :]
        wr_ww = self.W1[indr, :]
        d = self.D[ind_doc, :]
        wr_wd = self.W1[indr_wl, :]
        l = self.L[indl, :]
        wr_wl = self.W1[indr_wl, :]
        # cost for ww graph mini batch
        cost_ww = -T.nnet.sigmoid(T.dot(w, w1))
        cost_ww -= T.sum(T.nnet.sigmoid(T.sum(w * wr_ww, axis=1)))
        # cost for wd graph mini batch
        cost_wd = -T.nnet.sigmoid(T.dot(w, d))
        cost_wd -= T.sum(T.nnet.sigmoid(T.sum(w * wr_wd, axis=1)))
        # cost for wl graph mini batch
        cost_wl = -T.nnet.sigmoid(T.dot(w, l))
        cost_wl -= T.sum(T.nnet.sigmoid(T.sum(w * wr_wl, axis=1)))
        # Gradients for each edge from 3 graphs
        grad_ww = T.grad(cost_ww, [w, w1, wr_ww])
        grad_wd = T.grad(cost_wd, [w, d, wr_wd])
        grad_wl = T.grad(cost_wl, [w, l, wr_wl])
        # update equations for each samples edges from 3 graphs
        updates1 = [
            (self.W, T.inc_subtensor(self.W[indm, :], - lr * grad_ww[0]))]
        updates2 = [
            (self.W1, T.inc_subtensor(self.W1[indc, :], - lr * grad_ww[1]))]
        updates3 = [
            (self.W1, T.inc_subtensor(self.C[indr, :], - lr * grad_ww[2]))]
        updates = updates1 + updates2 + updates3
        self.func_ww = theano.function(inputs=[indm, indc, indr], outputs=cost_ww,
                                       updates=updates)
        updates4 = [
            (self.W, T.inc_subtensor(self.W[indm, :], - lr * grad_wd[0]))]
        updates5 = [
            (self.D, T.inc_subtensor(self.D[ind_doc, :], - lr * grad_wd[1]))]
        updates6 = [
            (self.W1, T.inc_subtensor(self.W1[indr_wd, :], - lr * grad_wd[2]))]
        updates = updates4 + updates5 + updates6
        self.fun_wd = theano.function(inputs=[indm, ind_doc, indr_wd], outputs=cost_wd,
                                      updates=updates)
        updates7 = [
            (self.W, T.inc_subtensor(self.W[indm, :], - lr * grad_wl[0]))]
        updates8 = [
            (self.L, T.inc_subtensor(self.L[indl, :], - lr * grad_wl[1]))]
        updates9 = [
            (self.W1, T.inc_subtensor(self.W1[indr_wl, :], - lr * grad_wl[2]))]
        updates = updates7 + updates8 + updates9
        self.fun_wl = theano.function(inputs=[indm, indl, indr_wl], outputs=cost_wl,
                                      updtes=updates)

    def pretraining_ww(self, indm, indc, indr):
        '''
        Performs SGD update (pre-training on ww graph).
        '''
        return self.func_ww(indm, indc, indr)

    def pretraining_wd(self, indm, ind_doc, indr_doc):
        '''
        SGD update (pre-training on wd graph).
        '''
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
