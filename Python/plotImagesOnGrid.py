import os
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Make all numpy available via shorter 'num' prefix
import numpy as np
# Make all matlib functions accessible at the top level via M.func()
import numpy.matlib as M
# Make some matlib functions accessible directly at the top level via, e.g. rand(3,3)
from numpy.matlib import rand,zeros,ones

def plotImagesOnGrid(X, numRows, numCols, width, height, samples, fileName):
	#PLOTIMAGESONGRID plot a set of images on a grid
	#   plotImagesGrid(X, numRows, numCols, width, height, samples)
	
	#create a figure
	h = plt.figure()
	
	#begin drawing along grid
	cnt = 1
	img = []
	bgc = 1
	for i in range (0,numRows):
	    for j in range(0,numCols):
	        #determine if there is something to draw and if so prepare it
	        empty = False
	        if cnt > len(samples):
	            img = bgc * ones((height, width))
	            empty = True
	            
	        else:
	            img = X[samples[cnt-1],:]
	            
	            #normalize image (still as row)
	            minImg = img.min()
	            maxImg = img.max()
	            if minImg == maxImg:
	                img = zeros(img.shape)
	            else:
	                img = (img - minImg)/(maxImg - minImg)
	            
	    
	        #get the image
	        img = img.reshape(height, width)
	        
	        
	        #draw cell content
	        a=h.add_subplot(numRows,numCols,cnt)
	        plt.imshow(img, cmap='gray')
    
	        cnt = cnt + 1;
	
	if not os.path.exists('./fig'):
	    os.mkdir('fig')
	h.savefig(fileName)
	plt.show()
	

