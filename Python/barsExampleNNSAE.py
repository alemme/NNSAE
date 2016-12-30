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
import numpy as np
# Make all matlib functions accessible at the top level via M.func()
import numpy.matlib as M
# Make some matlib functions accessible directly at the top level via, e.g. rand(3,3)
from numpy.matlib import rand,zeros,ones
from createBarsDataSet import createBarsDataSet
from plotImagesOnGrid import plotImagesOnGrid
import NNSAE as nn
import math

################## Configuration ##########################
## data parameters
numSamples = 10000;  #number of images
width = 9;          #image width = height


## network parameters
inpDim = width**2;           #number of input/output neurons
netDim = 2 * width;         #number of hidden neurons matches latent causes
#netDim = 2 * width + 2;    #number of hidden neurons exceeds latent causes by two

alpha = 1;          #alpha [0..1] is the decay rate for negative weights (alpha = 1 guarantees non-negative weights)
beta = 0;           #beta [0..1] is the decay rate for positive weights

#uncomment the following two lines for a symmetric decay function:
#alpha = 1e-6;          
#beta = 1e-6;

numEpochs = 10;     #number of sweeps through data for learning
lrateRO = 0.01;     #learning rate for synaptic plasticity of the read-out layer
lrateIP = 0.001;    #learning rate for intrinsic plasticity


################## Execution #############################
## data creation
X,xTest = createBarsDataSet(width, numSamples, 1, 1);
#rescale data for better numeric performance
X = 0.25 * X

## network creation
net = nn.NNSAE(inpDim, netDim	);
net.lrateRO = lrateRO;
net.lrateIP = lrateIP;
net.decayN = alpha;
net.decayP = beta;


## training
for e in range(1,numEpochs):
    print('epoch %d/ %d'%(e,numEpochs))
    net.train(X)



################## Evaluation ###########################
## evaluation of basis images
threshold = 0.1;    #parameter for analysis of weights

#sort basis images for visualization
cnt = 0;
unused =[];
w = net.W.T
v = zeros((w.shape))
for i in range (1,netDim):
    if w[i,:].max() > threshold: #this basis image is "used"
        cnt = cnt + 1
        v[cnt,:] = w[i,:]
    else:
        unused.append(i)
    

for i in range(1,len(unused)):
    v[cnt+i,:] = w[unused[i],:]

print 'used neurons = %d/%d'%(cnt,netDim)


################## Plotting ############################
## plotting
numCols = 5;
if netDim >= 50:
    numCols = 10
    
plotImagesOnGrid(v, int(math.ceil(netDim/numCols)), numCols, width, width,range(0,11),'./fig/NNSAE-bars-%d-basis.png'%(netDim))


