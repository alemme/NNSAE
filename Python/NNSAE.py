# Class Definion of the Non-Negative Sparse AutoEncoder (NNSAE)
# The class defines fields that store model parameters and implements methods of the
# NNSAE. The NNSAE uses shared weights. The class is designed to be used with
# non-negative data distributions.
#
# HowTo use this class:
# - Preparations:
#   1) create an NNSAE object by calling the constructor with your
#   specifications and call the method init
#   2) after loading the dataset, train the NNSAE by calling the method
#   train with your dataset.
#   NOTE: if you want to apply multiple training epochs call this function repeatedly
# - Apply trained NNSAE to new data:
#   1) Call method apply with the new data sample and compare with the
#   reconstruction (output argument).
#
###########################################################################
###          Copyright (c) 2016 F. A. Lemme, R. F. Reinhart   , CoR-Lab ###
###          Univertiy Bielefeld, Germany, http://cor-lab.de            ###
###########################################################################
#
# The program is free for non-commercial and academic use. Please contact the
# author if you are interested in using the software for commercial purposes.
# The software must not be modified or distributed without prior permission
# of the authors. Please acknowledge the authors in any academic publications
# that have made use of this code or part of it. Please use this BibTex for
# reference:
#
#    A. Lemme, R. F. Reinhart and J. J. Steil.
#    "Online learning and generalization of parts-based image representations
#     by Non-Negative Sparse Autoencoders". Submitted to Neural Networks,
#                              OR
#    A. Lemme, R. F. Reinhart and J. J. Steil. "Efficient online learning of
#    a non-negative sparse autoencoder". In Proc. ESANN, 2010.
#
# Please send your feedbacks or questions to:
#                           freinhar_at_cor-lab.uni-bielefeld.de
#                           alemme_at_cor-lab.uni-bielefeld.de

import math
# Make all numpy available via shorter 'num' prefix
import numpy as np
# Make all matlib functions accessible at the top level via M.func()
import numpy.matlib as M
# Make some matlib functions accessible directly at the top level via, e.g. rand(3,3)
from numpy.matlib import rand, zeros, ones


class NNSAE:

        # Constructor for a NNSAE class
        # input:
        #  - inpDim gives the Data sample dimension
        #  - hidDim specifies size of the hidden layer
        # output
        #  - net is the created Non-Negative Sparse Autoencoder

    def __init__(self, inpDim, hidDim):
        self.inpDim = inpDim  # number of input neurons (and output neurons)
        self.hidDim = hidDim  # number of hidden neurons

        self.inp = zeros((self.inpDim, 1))  # vector holding current input
        self.out = zeros((self.hidDim, 1))  # output neurons
        # neural activity before non-linearity
        self.g = zeros((self.hidDim, 1))
        self.h = zeros((self.hidDim, 1))  # hidden neuron activation
        self.a = ones((self.hidDim, 1))  # slopes of activation functions
        self.b = -3*ones((self.hidDim, 1))  # biases of activation functions
        scale = 0.025

        # shared network weights, i.e. used to compute hidden layer activations and estimated outputs
        self.W = scale * (2 * rand((self.inpDim, self.hidDim)) -
                          0.5 * ones((self.inpDim, self.hidDim))) + scale

        # learning rate for synaptic plasticity of read-out layer (RO)
        self.lrateRO = 0.01
        self.regRO = 0.0002  # numerical regularization constant
        self.decayP = 0  # decay factor for positive weights [0..1]
        self.decayN = 1  # decay factor for negative weights [0..1]

        self.lrateIP = 0.001  # learning rate for intrinsic plasticity (IP)
        self.meanIP = 0.2  # desired mean activity, a parameter of IP

    def apply(self, X):
        # Apply new data
        # Xhat = apply(net, X)
        # this function processes data sampels without learning
        #
        # input:
        #  - net is a Non-Negative Sparse Autoencoder object
        #  - X is a N x M matrix holding the data input, N is the number of samples and M the dimension of each data sample
        # output
        #  - Xhat reconstructed Input sampels N x M
        # for entire data matrix
        numSamples = X.shape[1]
        Xhat = zeros(X.shape)
        losses = zeros(numSamples)
        for i in range(numSamples):
            self.inp = X[i, :].T
            self.update()
            Xhat[i, :] = self.out.T
            losses[i] = np.mean((self.inp - self.out)**2)
        return Xhat, np.mean(losses)

    # Train the network
    # This function adapts the weight matrix W and parameters a and b
    # of the non-linear activation function (intrinsic plasticity)
    #
    # input:
    #  - X is a N x M matrix holding the data input, N is the number of samples and M the dimension of each data sample

    def train(self, X):
        numSamples = X.shape[0]
        losses = zeros(numSamples)
        # randperm(numSamples); #for randomized presentation
        p = np.random.permutation(range(numSamples))
        for i, ii in enumerate(p):
            # set input
            self.inp = X[ii, :].T  # forward propagation of activities
            self.update()

            # calculate adaptive learning rate
            lrate = self.lrateRO/(self.regRO + sum(np.power(self.h, 2)))

            # calculate error
            error = self.inp - self.out
            losses[0, i] = (error.T * error)/self.inpDim

            # update weights
            self.W = self.W + lrate[0, 0] * (error * self.h.T)

            # decay function for positive weights
            if self.decayP > 0:
                idx = where(self.W > 0)
                self.W[idx] -= self.decayP * self.W[idx]

            # decay function for negative weights
            if self.decayN == 1:
                # pure NN weights!
                self.W = np.maximum(self.W, 0)
            else:
                if self.decayN > 0:
                    idx = where(self.W < 0)
                    self.W[idx] -= self.decayN * self.W[idx]

            # intrinsic plasticity
            hones = ones((self.hidDim, 1))
            tmp = self.lrateIP * \
                (hones - (2.0 + 1.0/self.meanIP) *
                 self.h + np.power(self.h, 2)/self.meanIP)
            self.b = self.b + tmp
            self.a = self.a + self.lrateIP * hones / \
                self.a + np.multiply(self.g, tmp)

        return np.mean(losses)

    # Update network activation
    # This helper function computes the new activation pattern of the
    # hidden layer for a given input. Note that self.inp field has to be set in advance.

    def update(self):
        # excite network
        self.g = self.W.T * self.inp

        # apply activation function
        self.h = 1 / (1 + np.exp(np.multiply(-self.a, self.g) - self.b))

        # read-out
        self.out = self.W * self.h
