-- This script trains a NNSAE on the synthetic bars data set identifying
-- the independent causes that explain the images. There are 2*dim causes 
-- in the data, where dim is the height/width of the bars images. 
-- Play around with the number of hidden neurons in the NNSAE (netDim parameter below)
-- and explore its effects: Too many neurons result in "unused" basis images (see paper). 
-- Also try out different decay factor settings for alpha and beta below.
--
---------------------------------------------------------------------------------
------          Copyright (c) 2012 F. R. Reinhart, CoR-Lab                 ------
------          Univertiy Bielefeld, Germany, http://cor-lab.de            ------
---------------------------------------------------------------------------------
--
-- The program is free for non-commercial and academic use. Please contact the
-- author if you are interested in using the software for commercial purposes.
-- The software must not be modified or distributed without prior permission
-- of the authors. Please acknowledge the authors in any academic publications
-- that have made use of this code or part of it. Please use this BibTex for
-- reference:
-- 
--    A. Lemme, R. F. Reinhart and J. J. Steil. 
--    "Online learning and generalization of parts-based image representations 
--     by Non-Negative Sparse Autoencoders". Submitted to Neural Networks.
--                              OR
--    A. Lemme, R. F. Reinhart and J. J. Steil. "Efficient online learning of
--    a non-negative sparse autoencoder". In Proc. ESANN, 2010.
--
-- Please send your feedbacks or questions to:
--                           freinhar_at_cor-lab.uni-bielefeld.de
--                           alemme_at_cor-lab.uni-bielefeld.de
local gnuplot = require 'gnuplot'
local image = require 'image'	
local NNSAE = require('NonNegativeSparseAutoEncoder')
local createBarsDataSet = require('createBarsDataSet')

local function printf(...)
 print(string.format(...))
end
------------------------------------ Configuration -----------------------------
-- data parameters
local numSamples = 10000  --number of images
local width = 9          --image width = height


---- network parameters
local inpDim = width*width           --number of input/output neurons
--local netDim = 2 * width         --number of hidden neurons matches latent causes
local netDim = 2 * width + 2;    --number of hidden neurons exceeds latent causes by two

local alpha = 1         --alpha [0..1] is the decay rate for negative weights (alpha = 1 guarantees non-negative weights)
local beta = 0          --beta [0..1] is the decay rate for positive weights

--uncomment the following two lines for a symmetric decay function:
--alpha = 1e-6;          
--beta = 1e-6;

local numEpochs = 10     --number of sweeps through data for learning
local lrateRO = 0.01     --learning rate for synaptic plasticity of the read-out layer
local lrateIP = 0.001    --learning rate for intrinsic plasticity


------------------------------------ Execution ----------------------------------------------------------
---- data creation
local X, testX = createBarsDataSet(width, numSamples, 5000, 0);
--rescale data for better numeric performance
X:mul(0.25)
testX:mul(0.25)
print(X:size())
print(testX:size())
---- network creation
net = NNSAE.new(inpDim, netDim)
net.lrateRO = lrateRO
net.lrateIP = lrateIP
net.decayN = alpha
net.decayP = beta

local save = "fig"
local logTesterr = torch.zeros(numEpochs)
local logTrainerr = torch.zeros(numEpochs)
---- training
for e=1,numEpochs do
    printf('epoch %d/%d',e, numEpochs)
    net:train(X)
    local approx = net:apply(X)
    local train_err = approx - X
    train_err = train_err:pow(2)
    train_err = train_err:sum(2)
    train_err = train_err:mean()
    logTrainerr[e] = train_err
    printf("            Training MSE: %04f",train_err)
    local tapprox = net:apply(testX)
    local test_err = tapprox - testX
    test_err = test_err:pow(2)
    test_err = test_err:sum(2)
    test_err = test_err:mean()
    logTesterr[e] = train_err
    printf("            Test     MSE: %04f",test_err)
end
gnuplot.title('Traning progress over time (proposal)')

  local xs = torch.range(1, numEpochs)
  gnuplot.pngfigure(string.format("%s/trainingProgress.png",save))
  gnuplot.title('Traning progress over time')
  gnuplot.plot(
    { 'logTrainerr', xs, logTrainerr, '-' },
    { 'logTesterr', xs, logTesterr, '-' }
  )
  gnuplot.axis({ 0, numEpochs, 0, 0.3 })
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('MSE')
  gnuplot.plotflush()

------------------------------------ Evaluation ------------------------------------------------------
---- evaluation of basis images
local threshold = 0.1    --parameter for analysis of weights

--sort basis images for visualization
local cnt = 0
unused = {}
w = net.W:t()
v = torch.zeros(w:size())
for i=1,netDim do
    if torch.max(w[{i,{}}]) > threshold then --this basis image is "used"
        cnt = cnt + 1
        v[{cnt,{}}] = w[{i,{}}]
    else
        unused[#unused+1] = i
    end
end
for i=1,#unused do
    v[{cnt+i,{}}] = w[{unused[i],{}}]
end
printf('used neurons = %d/%d',cnt, netDim)

------------------------------------ Plotting --------------------------------------------------------
function evaluation(model, data,epoch)

  local red = torch.Tensor({1,0,0})
  local green = torch.Tensor({0,1,0})
  local blue = torch.Tensor({0,0,1})
  local white = torch.Tensor({1,1,1})
  local colors = { red, green, blue, white }

  -- create detector
  local d = model
  local input = data:clone()
  local approx = d:apply(input)
  
  local file = io.open(string.format('%s/report.html',save),'w')
    file:write(string.format([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <h1>Oiginal input + Approximation</h1>
    ]],save,epoch))
    file:write'<table><tr>\n'
    for i =1,20 do
	  local img = torch.reshape(input[{i,{}}],1,width,width)*255
	  local img_app = torch.reshape(approx[{i,{}}],1,width,width)
	  
	  image.savePNG(string.format('%s/input%d.png',save, i), img)
	  image.savePNG(string.format('%s/output%d.png',save, i), img_app)
      
      file:write(string.format(
      [[<td>
		      <img src="input%d.png" alt="input" width="144" height="144" >
		      <img src="output%d.png" alt="output" width="144" height="144" >
		<td>
      ]],i,i))
    end
    file:write' </tr></table>\n'
    file:write'<h1>Basis vectors</h1><table><tr>\n'
    for i =1,net.hidDim do
	  local img = torch.reshape(net.W[{{},i}],1,width,width)
	  
	  image.savePNG(string.format('%s/weights%d.png',save, i), img)

      file:write(string.format(
      [[<td>
		      <img src="weights%d.png" alt="input" width="144" height="144" >
		<td>
      ]],i,i))
      
    end

    file:write' </tr></table>\n'
    file:write([[
    <h2>Trainingsprocess:</h2>
    <img src="trainingProgress.png" alt="trainingProgress" >
    ]])
    file:write'</body></html>'
    file:close()
end

evaluation(net,X,numEpochs)

---- plotting
--[[
numCols = 5;
if netDim >= 50
    numCols = 10;
end
plotImagesOnGrid(v, ceil(netDim/numCols), numCols, width, width);
if ~exist(['.' filesep 'fig'], 'dir')
    mkdir('fig')
end
print(['.' filesep 'fig' filesep 'NNSAE-bars-' num2str(netDim) '-basis.png'], '-dpng');
--]]	
