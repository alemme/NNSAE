require 'torch'
require 'image'
require 'MNISTdataset'
require 'optim'
local e = require 'elm'
local NNSAE = require 'NonNegativeSparseAutoEncoder'

local function printf(...)
  print(string.format(...))
end

local function mapminmax(A, mn, mx)
  local mn = mn or -1; local mx = mx or 1

  local rows, cols = A:size(1),A:size(2)
  local min_i, max_i
  for i = 1,cols do
    min_i = torch.min(A[{{},{i}}])
    max_i = torch.max(A[{{},{i}}])

    A[{{},{i}}] = 2 * (A[{{},{i}}] - min_i) / (max_i - min_i) - 1
  end
  return A
end

local function softmax(eta)
  local e = torch.exp(eta)
  local dist = e/ torch.sum(e)
  return dist
end

A = torch.Tensor({{0.1},{0.2}}):t()

print(softmax(A))
B = torch.Tensor({{-0.1},{0.2}}):t()
print(softmax(B))
C = torch.Tensor({{0.9},{-10}}):t()
print(softmax(C))
D = torch.Tensor({{0},{10}}):t()
print(softmax(D))


classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}
-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
local opt = {batchSize = 500}
----------------------------------------------------------------------
-- get/create dataset
--
if false then
  nbTrainingPatches = 60000
  nbTestingPatches = 10000
else
  nbTrainingPatches = 30000
  nbTestingPatches = 5000
  print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
--trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
--testData:normalizeGlobal(mean, std)


local function featureLearning(dataset)
local numEpochs = numEpochs or 100
local lrateRO = 0.02     --learning rate for synaptic plasticity of the read-out layer
local lrateIP = 0.001    --learning rate for intrinsic plasticity
local alpha = 1         --alpha [0..1] is the decay rate for negative weights (alpha = 1 guarantees non-negative weights)
local beta = 0          --beta [0..1] is the decay rate for positive weights
local inpDim = geometry[1]*geometry[1]
local netDim = 100    --number of hidden neurons exceeds latent causes by two

---- network creation
net = NNSAE.new(inpDim, netDim)
net.lrateRO = lrateRO
net.lrateIP = lrateIP
net.decayN = alpha
net.decayP = beta


local save = "fig"
local logTesterr = torch.zeros(numEpochs)
local logTrainerr = torch.zeros(numEpochs)
local batchSize = 100
---- training
for epoch=1,numEpochs do
    printf('epoch %d/%d',epoch, numEpochs)
    -- do one epoch
  print('<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
  for t = 1,dataset:size(),batchSize do
    -- create mini batch
    xlua.progress(t, dataset:size())
    local inputs = torch.DoubleTensor(batchSize, inpDim)
    local targets = -torch.ones(batchSize,10)
    local k = 1

    for i = t,math.min(t+batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()
      inputs[k] = input/500 --extractFeatures(input)
      targets[k][target] = 1
      k = k + 1
    end
    net:train(inputs)
    local approx = net:apply(inputs)
    local train_err = approx - inputs
    train_err = train_err:pow(2)
    train_err = train_err:sum(2)
    train_err = train_err:mean()
    logTrainerr[e] = train_err
    printf("            Training MSE: %04f",train_err)
    end
end
return net
end

print("stating featurLearning")
local nnsaeNetwork = featureLearning(trainData, testData)
torch.save('./nnsaeNetwork02.t7', nnsaeNetwork)
os.exit()
----------------------------------------------------------------------

-- training function
function train(dataset, elm)
  -- epoch tracker
  epoch = epoch or 1

  -- local vars
  local time = sys.clock()

  -- do one epoch
  print('<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  for t = 1,dataset:size(),opt.batchSize do
    -- create mini batch
    xlua.progress(t, dataset:size())
    local inputs = torch.Tensor(opt.batchSize,inpDim)
    local targets = -torch.ones(opt.batchSize,10)

    local k = 1

    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()

      inputs[k] = input

      targets[k][target] = 10
      k = k + 1
    end
    if t ==1 then
        print("Initialize with BIP")
        elm:bip(inputs:t())
        print("Finished Initializion with BIP")

    end
    elm:train(inputs:t(),targets:t())

  end
  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
  -- next epoch
  epoch = epoch + 1
end

function trainIdentity(dataset, elm)
  -- epoch tracker
  epoch = epoch or 1

  -- local vars
  local time = sys.clock()

  -- do one epoch
  print('<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  for t = 1,dataset:size(),opt.batchSize do
    -- create mini batch
    xlua.progress(t, dataset:size())
    local inputs = torch.Tensor(opt.batchSize,geometry[1]*geometry[2])
    local targets = torch.Tensor(opt.batchSize,geometry[1]*geometry[2])
    local k = 1
    
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      inputs[k] = input
      targets[k] = input
      k = k + 1
    end
    if t ==1 then
        print("Initialize with BIP")
        elm:bip(inputs:t())
        print("Finished Initializion with BIP")
    end
    elm:train(inputs:t(),targets:t())
        
  end
  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
  -- next epoch
  epoch = epoch + 1
end


-- test function
function test(dataset, elm)
  -- local vars
  local time = sys.clock()

  -- test over given dataset
  print('<trainer> on testing Set:')
  for t = 1,dataset:size(),opt.batchSize do
    -- disp progress
    xlua.progress(t, dataset:size())

    -- create mini batch
    local inputs = torch.Tensor(opt.batchSize,inpDim)
    local targets = -torch.ones(opt.batchSize,10)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()
      inputs[k] = input -- extractFeatures(input)
      targets[k][target] = target
      k = k + 1
    end

    -- test samples
    local preds = elm:forward(inputs:t())

    -- confusion:
    for i = 1,opt.batchSize do
      confusion:add(preds[i], targets[i])
    end
  end

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
end


local inpDim = geometry[1]*geometry[2]

local elm_init = OSELM(inpDim,inpDim,9900) --12000
elm_init.reg = 1e-1
trainIdentity(trainData,elm_init)

print("with new init Winp = Wout")

local elm = OSELM(inpDim,10,9900) --12000
elm.reg = 1e-1
elm.Winp:copy(elm_init.Wout:t())


train(trainData,elm)
test(testData,elm)
