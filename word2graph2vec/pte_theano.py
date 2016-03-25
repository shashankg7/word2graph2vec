#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from theano import tensor as T
import theano
import numpy as np


class PTE(object):
    '''
    Defines the PTE model (cost function, parameters in theano.
    '''
    def __init__(self, nvertex, out_dim, ndocs, nlabels, lr=0.00001):
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
        eps = 4 * np.sqrt(6.0 / float(out_dim))
        eps1 = 4 * np.sqrt(6.0 / float(out_dim))
        eps2 = 4 * np.sqrt(6.0 / float(out_dim))
        self.w = np.asarray(np.random.uniform(low=-eps, high=eps, size=(nvertex, out_dim)),
                       dtype=theano.config.floatX)
        self.w1 = np.asarray(np.random.uniform(low=-eps, high=eps, size=(nvertex, out_dim)),
                       dtype=theano.config.floatX)
        self.d = np.asarray(np.random.uniform(low=-eps1, high=eps1, size=(ndocs, out_dim)),
                       dtype=theano.config.floatX)
        self.l = np.asarray(np.random.uniform(low=-eps2, high=eps2, size=(nlabels, out_dim)),
                       dtype=theano.config.floatX)
        self.lr = lr
        # Indices of main node, context node, k ww samples, doc node,k wd
        # samples, label node, k wl samples respectively
        
    def ww_model(self):
        '''
        Performs SGD update (pre-training on ww graph).
        '''
        self.W = theano.shared(self.w, name='W', borrow=True)
        self.W1 = theano.shared(self.w1, name='W1', borrow=True)
        self.temp_W = self.W1
        grad_hist = np.ones_like(self.w, dtype=theano.config.floatX)
        grad_hist1 = np.ones_like(self.w1, dtype=theano.config.floatX)

        hist_W = theano.shared(grad_hist, name='gW', borrow=True)
        hist_W1 = theano.shared(grad_hist1, name='gW1', borrow=True)
        hist_temp = hist_W1

        indm = T.iscalar()
        indc = T.iscalar()
        indr = T.ivector()
        weight = T.iscalar()
        w = self.W[indm, :]
        w1 = self.W1[indc, :]
        wr_ww = self.W1[indr, :]
        cost_ww = T.log(T.nnet.sigmoid(T.dot(w, w1)))
        cost_ww += T.sum(T.log(T.nnet.sigmoid(T.sum(-1 * w * wr_ww, axis=1))))
        cost =  weight * cost_ww
        grad_ww = T.grad(cost, [w, w1, wr_ww])
        # Gradient clipping
       # grad_ww[0] = T.clip(grad_ww[0], -0.1, 0.1)
       # grad_ww[1] = T.clip(grad_ww[1], -0.1, 0.1)
       # grad_ww[2] =T.clip(grad_ww[2], -0.1, 0.1)

        updates1 = [(hist_W, T.inc_subtensor(hist_W[indm, :],grad_ww[0] ** 2))]
        hist_temp = T.set_subtensor(hist_temp[indc, :], hist_temp[indc, :] + grad_ww[1] ** 2)
        hist_temp = T.set_subtensor(hist_temp[indr, :], hist_temp[indr, :] + grad_ww[2] ** 2)
        updates2 = [(hist_W1, hist_temp)]

        updates3 = [
            (self.W, T.inc_subtensor(self.W[indm, :], - (self.lr / T.sqrt(hist_W[indm,:])) * grad_ww[0]))]
        self.temp_W = T.set_subtensor(self.temp_W[indc, :], self.temp_W[indc, :] - (self.lr / T.sqrt(hist_W1[indc, :]) ) * grad_ww[1])
        self.temp_W = T.set_subtensor(self.temp_W[indr, :], self.temp_W[indr, :] - (self.lr / T.sqrt(hist_W1[indr, :]) ) * grad_ww[2])
        updates4 = [(self.W1, self.temp_W)]
        updates = updates1 + updates2 + updates3 + updates4
        self.train_ww = theano.function(inputs=[indm, indc, indr, weight], outputs=cost, updates=updates)
        
    def pretraining_ww(self, indm, indc, indr, weight):
        return self.train_ww(indm, indc, indr, weight)


    def wd_model(self):
        '''
        Performs SGD update (pre-training on wd graph).
        '''
        self.D  = theano.shared(self.d, name='D', borrow=True)
        grad_hist = np.ones_like(self.d, dtype=theano.config.floatX)
        grad_hist1 = np.ones_like(self.w, dtype=theano.config.floatX)
        grad_hist2 = np.ones_like(self.w1, dtype=theano.config.floatX)
        hist_D = theano.shared(grad_hist, name='hD',  borrow=True)
        hist_W1 = theano.shared(grad_hist1, name='hD1', borrow=True)
        hist_W2 = theano.shared(grad_hist2, name='hD2',  borrow=True)
        indw = T.iscalar()
        indd = T.iscalar()
        indr = T.ivector()
        weight = T.iscalar()
        d = self.D[indd, :]
        w = self.W[indw, :]
        #####################################################################################################
        #                           TO-DO : Sample random words from W?
        #####################################################################################################
        wr_wd = self.W1[indr, :]
        cost_wd = T.log(T.nnet.sigmoid(T.dot(d, w)))
        cost_wd += T.sum(T.log((T.nnet.sigmoid(T.sum( -1.0 * d * wr_wd, axis=1)))))
        cost = weight * cost_wd
        grad_wd = T.grad(cost, [d, w, wr_wd])
        #grad_wd[0] = T.clip(grad_wd[0], -0.1, 0.1)
        #grad_wd[1] = T.clip(grad_wd[1], -0.1, 0.1)
        #grad_wd[2] =T.clip(grad_wd[2], -0.1, 0.1)
        updates1 = [(hist_D, T.inc_subtensor(hist_D[indd, :], grad_wd[0] ** 2))]
        updates2 = [(hist_W1, T.inc_subtensor(hist_W1[indw, :], grad_wd[1] ** 2))]
        updates3 = [(hist_W2, T.inc_subtensor(hist_W2[indr, :], grad_wd[2] ** 2))]

        updates4 = [
            (self.D, T.inc_subtensor(self.D[indd, :], - (self.lr / T.sqrt(hist_D[indd,:])) * grad_wd[0]))]
        updates5 = [
            (self.W, T.inc_subtensor(self.W[indw, :], - (self.lr / T.sqrt(hist_W1[indw, :])) * grad_wd[1]))]
        updates6 = [
            (self.W1, T.inc_subtensor(self.W1[indr, :], - (self.lr / T.sqrt(hist_W2[indr, :])) * grad_wd[2]))]
        updates = updates1 + updates2 + updates3 + updates4 + updates5 + updates6
        self.train_wd = theano.function(inputs=[indw, indd, indr, weight], outputs=cost, updates=updates)
        
    def pretraining_wd(self, indw, indd, indr, weight):
        '''
        SGD update (pre-training on wd graph).
        '''
        return self.train_wd(indw, indd, indr, weight)


    def wl_model(self):
        '''
        Performs SGD update (pre-training on wl graph).
        '''
        self.L  = theano.shared(self.l, borrow=True)
        grad_hist = np.ones_like(self.l)
        grad_hist1 = np.ones_like(self.w)
        grad_hist2 = np.ones_like(self.w1)
        hist_L = theano.shared(grad_hist, borrow=True)
        hist_W1 = theano.shared(grad_hist1, borrow=True)
        hist_W2 = theano.shared(grad_hist2, borrow=True)
        indw = T.iscalar()
        indl = T.iscalar()
        indr = T.ivector()
        weight = T.iscalar()
        l = self.L[indl, :]
        w = self.W[indw, :]
        #####################################################################################################
        #                           TO-DO : Sample random words from W?
        #####################################################################################################
        wr_wd = self.W1[indr, :]
        cost_wl = T.log(T.nnet.sigmoid(T.dot(l, w)))
        cost_wl += T.sum(T.log(T.nnet.sigmoid(T.sum( -1.0 * l * wr_wd, axis=1))))
        cost = weight * cost_wl
        grad_wl = T.grad(cost, [l, w, wr_wd])
        #grad_wl[0] = T.clip(grad_wl[0], -0.1, 0.1)
        #grad_wl[1] = T.clip(grad_wl[1], -0.1, 0.1)
        #grad_wl[2] =T.clip(grad_wl[2], -0.1, 0.1)


        updates1 = [(hist_L, T.inc_subtensor(hist_L[indl, :], grad_wl[0] ** 2))]
        updates2 = [(hist_W1, T.inc_subtensor(hist_W1[indw, :], grad_wl[1] ** 2))]
        updates3 = [(hist_W2, T.inc_subtensor(hist_W2[indr, :], grad_wl[2] ** 2))]

        updates4 = [
            (self.D, T.inc_subtensor(self.L[indl, :], - (self.lr / T.sqrt(hist_L[indl,:])) * grad_wl[0]))]
        updates5 = [
            (self.W, T.inc_subtensor(self.W[indw, :], - (self.lr / T.sqrt(hist_W1[indw, :])) * grad_wl[1]))]
        updates6 = [
            (self.W1, T.inc_subtensor(self.W1[indr, :], - (self.lr / T.sqrt(hist_W2[indr, :])) * grad_wl[2]))]
        updates = updates1 + updates2 + updates3 + updates4 + updates5 + updates6
        self.train_wl = theano.function(inputs=[indw, indl, indr, weight], outputs=cost, updates=updates)

    def finetuning(self, indw, indl, indr, weight):
        '''
        SGD update (finetuning on wl graph using w embeddings)
        '''
        return self.train_wl(indw, indl, indr, weight)

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
