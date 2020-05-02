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

from torch.optim.optimizer import Optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F


class Nnsae(nn.Module):

    # Constructor for a NNSAE class
    # input:
    #  - inpDim gives the Data sample dimension
    #  - hidDim specifies size of the hidden layer
    # output
    #  - net is the created Non-Negative Sparse Autoencoder

    def __init__(self, inpDim, hidDim, batch_size=1):
        torch.autograd.set_detect_anomaly(True)
        super(Nnsae, self).__init__()
        self.inpDim = inpDim  # number of input neurons (and output neurons)
        self.hidDim = hidDim  # number of hidden neurons
        self.weights = torch.zeros(inpDim, hidDim, requires_grad=True)
        self.scale = 0.025
        self.weights.data.uniform_(0.0, 0.05)
        # self.weights.data = self.scale * (2 * torch.rand(inpDim, hidDim) -
        #                                 0.5 * torch.ones(inpDim, hidDim)) + self.scale
        self.nonlin = torch.sigmoid
        self.nonneg = lambda x: x

        self.inp = torch.zeros(self.inpDim, 1)  # vector holding current input
        self.out = torch.zeros(self.hidDim, 1)  # output neurons
        # neural activity before non-linearity
        self.h = torch.zeros(self.hidDim, batch_size)  # hidden neuron activation
        self.g = torch.zeros(self.hidDim, batch_size)  # pre hidden neuron
        self.a = torch.ones(self.hidDim, 1)
        self.b = torch.ones(self.hidDim, 1) * (-3.0)

        # learning rate for synaptic plasticity of read-out layer (RO)
        self.lrateRO = 0.01
        self.regRO = 0.0002  # numerical regularization constant
        self.decayP = 0  # decay factor for positive weights [0..1]
        self.decayN = 1  # decay factor for negative weights [0..1]

        self.lrateIP = 0.001  # learning rate for intrinsic plasticity (IP)
        self.meanIP = 0.2  # desired mean activity, a parameter of IP

    def ip(self):
        h = self.h
        tmp = self.lrateIP * (1.0 - (2.0 + 1.0/self.meanIP) * h + (h**2) / self.meanIP)
        self.b += tmp.sum(1, keepdim=True)
        a_tmp = self.lrateIP / self.a + self.g * tmp
        self.a += a_tmp.sum(1, keepdim=True)

    def bpdc(self, error):
        # calculate adaptive learning rate
        lrate = (self.lrateRO/(self.regRO + (self.h**2).sum(0, keepdim=True)))
        self.weights.data += error.mm(torch.diag(lrate) * (self.h).t())

    def forward(self, x):
        # Here the forward pass is simply a linear function
        g = self.weights.t().mm(x)
        h = self.nonlin(self.a * g + self.b)
        out = self.weights.mm(h)

        self.g[:, :] = g.detach()
        self.h[:, :] = h.detach()
        return out

    def save_state_dict(self, fileName):
        torch.save(self.state_dict(), fileName)


class BackpropagationDecoralation(Optimizer):
    # learning rate for synaptic plasticity of read-out layer (RO)
    def __init__(self, params, hidden_activations, lrateRO=0.01, regRO=0.0002):
        self.lrateRO = lrateRO
        self.regRO = regRO  # numerical regularization constant
        defaults = dict(lrateRO=self.lrateRO,
                        regRO=self.regRO,
                        hidden_activations=hidden_activations)
        super(BackpropagationDecoralation, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BackpropagationDecoralation, self).__setstate__(state)

    def step(self, closure=None):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p, h in zip(group['params'], group['hidden_activations']):
                grad = p.grad.data
                if grad is None:
                    continue

                # calculate adaptive learning rate
                lrate = (self.lrateRO/(self.regRO + (h**2).sum(0, keepdim=True)))
                d_p = -lrate * grad
                p.data.add_(d_p)

        return loss
