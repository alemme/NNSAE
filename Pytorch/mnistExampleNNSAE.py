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
import os
import math
import NNSAE as nn
import torch
from torchvision import datasets, transforms
from plotImagesOnGrid import plotImagesOnGrid, plotPaperPlot
from numpy.matlib import zeros
import numpy as np
np.random.seed(1234)
use_cuda = torch.cuda.is_available()
torch.manual_seed(1234)
fileName = 'mnistExampleNNSAE.pt'
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ])), batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

numEpochs = 10  # number of sweeps through data for learning
lrateRO = 0.01  # learning rate for synaptic plasticity of the read-out layer
lrateIP = 0.0001  # learning rate for intrinsic plasticity
meanIP = 0.2
width = 28
inpDim = width**2
netDim = 200

# network creation
net = nn.Nnsae(inpDim, netDim, batch_size)

bpdc = nn.BackpropagationDecoralation(
    [net.weights], [net.h], lrateRO=lrateRO
)
loss_fkt = torch.nn.modules.MSELoss(reduction='mean')
net.lrateRO = lrateRO
net.lrateIP = lrateIP
numBatches = len(train_loader)
if not os.path.isfile(fileName):
    # training
    for e in range(1, numEpochs+1):
        gl_loss = 0
        numSamples = 0

        for i, data in enumerate(train_loader):
            # bpdc.zero_grad()
            with torch.no_grad():
                inp = data[0].view(data[0].shape[0], width**2)
                numSamples += data[0].shape[0]
                # forward path
                out = net(inp.t()).t()
                # calculate loss
                loss = loss_fkt(inp, out)
                # loss.backward()

                # bpdc.step()
                net.bpdc((inp-out).t())

                # non negative constraint
                net.weights.data[net.weights < 0] = 0
                # intrinsic plasticity
                net.ip()

                # log loss
                gl_loss += loss.item()

        # print(f'epoch ({e}\{numEpochs}) loss {gl_loss/numSamples}')
        print('epoch ({}\{}) loss {}'.format(e, numEpochs, gl_loss/numSamples))
    net.save_state_dict(fileName)
else:
    net.load_state_dict(torch.load(fileName))
################## Evaluation ###########################
# evaluation of basis images
threshold = 0.1  # parameter for analysis of weights

# sort basis images for visualization
cnt = 0
unused = []
w = net.weights.t().detach().numpy()
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
                 width, range(netDim), './fig/NNSAE-bars-%d-basis.png' % (netDim))

gl_loss = 0
numSamples = 0
net.eval()
orig = []
approx = []
activations = []
with torch.no_grad():
    num = 0
    for i, data in enumerate(test_loader):
        inp = data[0].view(data[0].shape[0], width**2)
        label = data[1]

        numSamples += data[0].shape[0]
        # forward path
        out = net(inp.t()).t()
        # calculate loss
        loss = loss_fkt(inp, out)
        label_list = label.tolist()
        while num in label_list:
            num_index = label_list.index(num)
            orig.append(inp[num_index, :])
            approx.append(out[num_index, :])
            activations.append(net.h.t()[num_index, :])
            num += 1

        gl_loss += loss.item()
    print('Evaluation loss MSE: {}'.format(gl_loss/numSamples))
    plotPaperPlot(orig, approx, activations, width, width)
