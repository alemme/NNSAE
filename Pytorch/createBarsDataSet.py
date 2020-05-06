import math
# Make all numpy available via shorter 'num' prefix
import numpy as np
# Make all matlib functions accessible at the top level via M.func()
import numpy.matlib as M
# Make some matlib functions accessible directly at the top level via, e.g. rand(3,3)
from numpy.matlib import rand, zeros, ones


def createBarsDataSet(dim=10, numTrain=10000, numTest=5000, nonlinear=True):
    # CREATEBARSDATASET create a set of square images stored row-wise in a matrix
    #   [Xtrain Xtest] = createBarsDataSet(dim, numTrain, numTest, nonlinear)
    #   dim - image width = height = dim
    #   numTrain - number of training images
    #   numTest - number of test images
    #   nonlinear - if this flag is true/1, pixel intensities at crossing points of bars are not added up

    imgdim = dim * dim
    X = zeros((numTrain+numTest, imgdim))

    for k in range(numTrain+numTest):
        x = zeros((dim, dim))
        for z in range(2):
            i = np.random.permutation(range(dim))
            j = np.random.permutation(range(dim))
            if nonlinear is True:
                x[i[z], :] = 1.0
                x[:, j[z]] = 1.0
            else:
                x[i[z], :] += 1.0
                x[:, j[z]] += 1.0

        if nonlinear is False:
            x = x / 4

        X[k, :] = x.reshape(1, imgdim)

    Xtrain = X[:numTrain, :]
    Xtest = X[numTrain+1:, :]

    return Xtrain, Xtest


if __name__ == "__main__":
    from plotImagesOnGrid import plotImagesOnGrid
    numCols = 5
    numSamples = 20
    width = 9
    x, xTest = createBarsDataSet(width, numSamples)
    netDim = 20
    plotImagesOnGrid(x, int(math.ceil(numSamples/numCols)), numCols, width,
                     width, range(numSamples), './fig/NNSAE-bars-basis.png')