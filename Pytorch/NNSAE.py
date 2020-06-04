# Class Definion of the Non-Negative Sparse AutoEncoder (NNSAE)
# The class defines fields that store model parameters and implements methods of the
# NNSAE. The NNSAE uses shared weights. The class is designed to be used with
# non-negative data distributions.
#
###########################################################################
###          Copyright (c) 2012 A. Lemme, F. R. Reinhart, CoR-Lab       ###
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
#     by Non-Negative Sparse Autoencoders". Neural Networks, vol. 33, pp. 194-203, 2012
#     doi = "https://doi.org/10.1016/j.neunet.2012.05.003"#
#                                   OR
#    A. Lemme, R. F. Reinhart and J. J. Steil. "Efficient online learning of
#    a non-negative sparse autoencoder". In Proc. ESANN, 2010.


from __future__ import print_function
from torch.optim.optimizer import Optimizer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Nnsae(nn.Module):

    # Constructor for a NNSAE class
    # input:
    #  - inpDim gives the Data sample dimension
    #  - hidDim specifies size of the hidden layer
    # output
    #  - net is the created Non-Negative Sparse Autoencoder
    __constants__ = ['inpDim', 'hidDim']

    def __init__(self, inpDim, hidDim, batch_size=1):
        torch.autograd.set_detect_anomaly(True)
        super(Nnsae, self).__init__()
        self.inpDim = inpDim  # number of input neurons (and output neurons)
        self.hidDim = hidDim  # number of hidden neurons
        self.nonlin = torch.sigmoid

        self.inp = torch.zeros(self.inpDim, 1)  # vector holding current input
        self.out = torch.zeros(self.hidDim, 1)  # output neurons
        # neural activity before non-linearity
        self.h = torch.zeros(self.hidDim, batch_size)  # hidden neuron activation
        self.g = torch.zeros(self.hidDim, batch_size)  # pre hidden neuron
        self.a = Parameter(torch.ones(self.hidDim, 1))
        self.b = Parameter(torch.ones(self.hidDim, 1) * (-3.0))
        self.weights = Parameter(torch.zeros(inpDim, hidDim))
        self.scale = 0.025
        self.weights.data = self.scale * (2 * torch.rand(inpDim, hidDim) -
                                         0.5 * torch.ones(inpDim, hidDim)) + self.scale

        # learning rate for synaptic plasticity of read-out layer (RO)
        self.lrateRO = 0.01
        self.regRO = 0.0002  # numerical regularization constant

        self.lrateIP = 0.001  # learning rate for intrinsic plasticity (IP)
        self.meanIP = 0.2  # desired mean activity, a parameter of IP
        self._cuda = False

    def __setstate__(self, state):
        super(Nnsae, self).__setstate__(state)

    @property
    def cuda(self):
        return self._cuda

    def to(self, device):
        super().to(device)
        self._cuda = device.type != 'cpu'
        self.a.to(device)
        self.b.to(device)
        self.h = self.h.to(device)
        self.g = self.g.to(device)
        self.out.to(device)
        self.inp.to(device)
        self.weights.to(device)

    def ip(self):
        h = self.h
        tmp = self.lrateIP * (1.0 - (2.0 + 1.0/self.meanIP) * h + (h**2) / self.meanIP)
        self.b += tmp.sum(1, keepdim=True)
        a_tmp = self.lrateIP / self.a + self.g * tmp
        self.a += a_tmp.sum(1, keepdim=True)

    def bpdc(self, error):
        # calculate adaptive learning rate
        device = self.weights.device
        lrate = (self.lrateRO/(self.regRO + (self.h**2).sum(0, keepdim=True)))
        self.weights.data += error.mm(torch.diag(lrate).to(device) * (self.h).t())

    def fit(self, inp):
        # forward path
        out = self.forward(inp)
        # bpdc.step()
        error = inp - out
        self.bpdc(error)
        # non negative constraint
        self.weights.data[self.weights < 0] = 0
        # intrinsic plasticity
        self.ip()
        return out, error

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

    def extra_repr(self):
        s = ('({inpDim} x {hidDim})')
        s += ', Intrinsic plasticity: mean={meanIP}, leaning rate={lrateIP}'
        s += '; Synaptic plasticity: learning rate={lrateRO}, epsilon={regRO}'
        return s.format(**self.__dict__)


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
