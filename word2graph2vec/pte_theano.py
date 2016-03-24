#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from theano import tensor as T
import theano
import numpy as np


class PTE(object):
    '''
    Defines the PTE model (cost function, parameters in theano.
    '''
    def __init__(self, nvertex, out_dim, ndocs, nlabels, lr=0.01):
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
        eps = 4 * np.sqrt(6.0 / float(nvertex))
        self.w = np.asarray(np.random.uniform(low=-eps, high=eps, size=(nvertex, out_dim)),
                       dtype=theano.config.floatX)
        self.w1 = np.asarray(np.random.uniform(low=-eps, high=eps, size=(nvertex, out_dim)),
                       dtype=theano.config.floatX)
        self.d = np.asarray(np.random.rand(ndocs, out_dim) / float(ndocs),
                       dtype=theano.config.floatX)
        self.l = np.asarray(np.random.rand(nvertex, out_dim) / float(out_dim),
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
        cost_ww = T.nnet.sigmoid(T.dot(w, w1))
        cost_ww += T.sum(T.nnet.sigmoid(T.sum(w * wr_ww, axis=1)))
        cost = weight * T.log(cost_ww) 
        grad_ww = T.grad(cost, [w, w1, wr_ww])
        grad0 = T.clip(grad_ww[0], -0.1, 0.1)
        grad1 = T.clip(grad_ww[1], -0.1, 0.1)
        grad2 =T.clip(grad_ww[2], -0.1, 0.1)

        updates1 = [(hist_W, T.inc_subtensor(hist_W[indm, :],grad0 ** 2))]
        hist_temp = T.set_subtensor(hist_temp[indc, :], hist_temp[indc, :] + grad1 ** 2)
        hist_temp = T.set_subtensor(hist_temp[indr, :], hist_temp[indr, :] + grad2 ** 2)
        updates2 = [(hist_W1, hist_temp)]

        updates3 = [
            (self.W, T.inc_subtensor(self.W[indm, :], - (self.lr / T.sqrt(hist_W[indm,:])) * grad0))]
        self.temp_W = T.set_subtensor(self.temp_W[indc, :], self.temp_W[indc, :] - (self.lr / T.sqrt(hist_W1[indc, :]) ) * grad1)
        self.temp_W = T.set_subtensor(self.temp_W[indr, :], self.temp_W[indr, :] - (self.lr / T.sqrt(hist_W1[indr, :]) ) * grad2)
        updates4 = [(self.W1, self.temp_W)]
        updates = updates1 + updates2 + updates3 + updates4
        self.train_ww = theano.function(inputs=[indm, indc, indr, weight], outputs=cost, updates=updates)
        
    def pretraining_ww(self, indm, indc, indr, weight):
        return self.train_ww(indm, indc, indr, weight)


    #########################################################################################################
    #                       TO-DO : First check with ww model and than check wd model
    ########################################################################################################


    def wd_model(self):
        '''
        Performs SGD update (pre-training on wd graph).
        '''
        self.D  = theano.shared(self.d, borrow=True)
        grad_hist = np.ones_like(self.W)
        grad_hist1 = np.ones_like(self.W1)

        hist_W = theano.shared(grad_hist, borrow=True)
        hist_W1 = theano.shared(grad_hist1, borrow=True)
        hist_temp = hist_W1

        indm = T.iscalar()
        indc = T.iscalar()
        indr = T.ivector()
        w = W[indm, :]
        w1 = W1[indc, :]
        wr_ww = W1[indr, :]
        cost_ww = T.nnet.sigmoid(T.dot(w, w1))
        cost_ww += T.sum(T.nnet.sigmoid(T.sum(w * wr_ww, axis=1)))
        cost = weight * T.log(cost_ww) 
        grad_ww = T.grad(cost, [w, w1, wr_ww])

        updates1 = [(hist_W, T.inc_subtensor(hist_W[indm, :], grad_ww[0] ** 2))]
        hist_temp = T.set_subtensor(hist_temp[indc, :], hist_temp[indc, :] + grad_ww[1] ** 2)
        hist_temp = T.set_subtensor(hist_temp[indr, :], hist_temp[indr, :] + grad_ww[2] ** 2)
        updates2 = [(W1, hist_temp)]

        updates3 = [
            (self.W, T.inc_subtensor(self.W[indm, :], - (lr / T.sqrt(hist_W[ind,:])) * grad_ww[0]))]
        temp_W = T.set_subtensor(temp_W[indc, :], temp_W[indc, :] - (lr / T.sqrt(hist_W1[indc, :]) ) * grad_ww[1])
        temp_W = T.set_subtensor(temp_W[indr, :], temp_W[indr, :] - (lr / T.sqrt(hist_W1[indr, :]) ) * grad_ww[2])
        updates4 = [(W1, temp_W)]
        updates = updates1 + updates2 + updates3 + updates4
        self.train_ww = theano.function(inputs=[indm, indc, indr, weight], outputs=cost, updates=updates)
        
   
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
        #D = self.D.get_value()  # + self.W1.get_value()
        #L = self.L.get_value()  # + self.W1.get_value()
        np.save('lookupW', W)
        #np.save('lookupD', D)
        #np.save('lookupL', L)
