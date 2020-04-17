# This script trains a NNSAE on the synthetic bars data set identifying
# the independent causes that explain the images. There are 2*dim causes
# in the data, where dim is the height/width of the bars images.
# Play around with the number of hidden neurons in the NNSAE (netDim parameter below)
# and explore its effects: Too many neurons result in "unused" basis images (see paper).
# Also try out different decay factor settings for alpha and beta below.
#
###########################################################################
###          Copyright (c) 2012 F. R. Reinhart, CoR-Lab                 ###
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
#     by Non-Negative Sparse Autoencoders". Submitted to Neural Networks.
#                              OR
#    A. Lemme, R. F. Reinhart and J. J. Steil. "Efficient online learning of
#    a non-negative sparse autoencoder". In Proc. ESANN, 2010.
#
# Please send your feedbacks or questions to:
#                           freinhar_at_cor-lab.uni-bielefeld.de
#                           alemme_at_cor-lab.uni-bielefeld.de

# Make all numpy available via shorter 'num' prefix
import random
from numpy.matlib import zeros
from createBarsDataSet import createBarsDataSet
from plotImagesOnGrid import plotImagesOnGrid
import torch
import NNSAE as nn

import math

# Configuration
# data parameters
numSamples = 10000  # number of images
width = 9  # image width = height


# network parameters
inpDim = width**2  # number of input/output neurons
netDim = 2 * width  # number of hidden neurons matches latent causes
# netDim = 2 * width + 2;    #number of hidden neurons exceeds latent causes by two

# alpha [0..1] is the decay rate for negative weights (alpha = 1 guarantees non-negative weights)
alpha = 1
beta = 0  # beta [0..1] is the decay rate for positive weights

numEpochs = 10  # number of sweeps through data for learning
lrateRO = 0.01  # learning rate for synaptic plasticity of the read-out layer
lrateIP = 0.001  # learning rate for intrinsic plasticity
meanIP = 0.2

# Execution
# data creation
X, xTest = createBarsDataSet(width, numSamples)
# rescale data for better numeric performance
X = 0.25 * X

# network creation
net = nn.Nnsae(inpDim, netDim)

bpdc = nn.BackpropagationDecoralation(
    [net.weights], [net.h], lrateRO=lrateRO)
loss_fkt = torch.nn.modules.MSELoss(reduction='sum')
net.lrateRO = lrateRO
net.lrateIP = lrateIP
# net.decayN = alpha
# net.decayP = beta

# training
for e in range(1, numEpochs):
    gl_loss = 0
    nums = list(range(numSamples))
    random.shuffle(nums)

    for i in nums:
        bpdc.zero_grad()
        inp = torch.from_numpy(X[i, :]).float()
        
        out = net(inp)
        loss = loss_fkt(inp, out)
        loss.backward()
        #net.bpdc(inp-out)
            
        bpdc.step()
        with torch.no_grad():
            net.ip()
            # print(loss)
        gl_loss += loss.item()
        
    print(f'epoch ({e}\{numEpochs}) loss {gl_loss/numSamples}')
################## Evaluation ###########################
# evaluation of basis images
threshold = 0.1  # parameter for analysis of weights

# sort basis images for visualization
cnt = 0
unused = []
w = net.weights.detach().t().numpy()
v = zeros((w.shape))
for i in range(netDim):
    if w[i, :].max() > threshold:  # this basis image is "used"
        v[cnt, :] = w[i, :]
        cnt = cnt + 1
    else:
        unused.append(i)


for i in range(len(unused)):
    v[cnt+i, :] = w[unused[i], :]

print('used neurons = {}/{}'.format(cnt, netDim))


################## Plotting ############################
# plotting
numCols = 5
if netDim >= 50:
    numCols = 10

plotImagesOnGrid(v, int(math.ceil(netDim/numCols)), numCols, width,
                 width, range(0, netDim), './fig/NNSAE-bars-%d-basis.png' % (netDim))
