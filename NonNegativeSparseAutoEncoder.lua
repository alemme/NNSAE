--[[ Class Definion of the Non-Negative Sparse AutoEncoder (NNSAE)
- The class defines fields that store model parameters and implements methods of the
- NNSAE. The NNSAE uses shared weights. The class is designed to be used with 
- non-negative data distributions.
-
- HowTo use this class:
- - Preparations:
-   1) create an NNSAE object by calling the constructor with your
-   specifications and call the method init
-   2) after loading the dataset, train the NNSAE by calling the method 
-   train with your dataset.
-   NOTE: if you want to apply multiple training epochs call this function repeatedly
- - Apply trained NNSAE to new data:
-   1) Call method apply with the new data sample and compare with the
-   reconstruction (output argument).
-
---------------------------------------------------------------------------
---          Copyright (c) 2016 F. A. Lemme, R. F. Reinhart, CoR-Lab    ---
---          Univertiy Bielefeld, Germany, http://cor-lab.de            ---
---------------------------------------------------------------------------
-
- The program is free for non-commercial and academic use. Please contact the
- author if you are interested in using the software for commercial purposes.
- The software must not be modified or distributed without prior permission
- of the authors. Please acknowledge the authors in any academic publications
- that have made use of this code or part of it. Please use this BibTex for
- reference:
- 
-    A. Lemme, R. F. Reinhart and J. J. Steil. 
-    "Online learning and generalization of parts-based image representations 
-     by Non-Negative Sparse Autoencoders". Submitted to Neural Networks,
-                              OR
-    A. Lemme, R. F. Reinhart and J. J. Steil. "Efficient online learning of
-    a non-negative sparse autoencoder". In Proc. ESANN, 2010.
-
- Please send your feedbacks or questions to:
-                           freinhar_at_cor-lab.uni-bielefeld.de
-                           alemme_at_cor-lab.uni-bielefeld.de
--]]
local NonNegativeSparseAutoEncoder = torch.class('NonNegativeSparseAutoEncoder')

-- Constructor for a NNSAE class 
-- input: 
--  - inpDim gives the Data sample dimension
--  - hidDim specifies size of the hidden layer
-- output
--  - net is the created Non-Negative Sparse Autoencoder
function NonNegativeSparseAutoEncoder:__init(inpDim,hidDim)
  self.inpDim = inpDim    --number of neurons in the input layer
  self.hidDim = hidDim    --number of neurons in the hidden layer
  self.inp = torch.zeros(self.inpDim,1)   --vector holding current input
  self.out = torch.zeros(self.inpDim,1)   --output neurons
  local scale = 0.025
  self.W = ((torch.rand(self.inpDim, self.hidDim)*2 - torch.ones(self.inpDim, self.hidDim)*0.5)+scale) *scale --weights from input layer to hidden layer
  self.g = torch.zeros(self.hidDim,1) --neural activity before non-linearity
  self.h = torch.zeros(self.hidDim,1) --hidden neuron activation
  self.a = torch.ones(self.hidDim,1)    --slope parameters of activation functions
  self.b = torch.ones(self.hidDim,1) * (-3)    --bias parameters of activation functions
  self.lrateRO = 0.01 --learning rate for synaptic plasticity of read-out layer (RO)
  self.regRO = 0.0002 --numerical regularization constant
  self.decayP = 0     --decay factor for positive weights [0..1]
  self.decayN = 1     --decay factor for negative weights [0..1]

  self.lrateIP = 0.001    --learning rate for intrinsic plasticity (IP)
  self.meanIP = 0.2       --desired mean activity, a parameter of IP
end
        
        
--- Apply new data
-- Xhat = apply(net, X)
-- this function processes data sampels without learning
--
-- input: 
--  - net is a Non-Negative Sparse Autoencoder object
--  - X is a N x M matrix holding the data input, N is the number of samples and M the dimension of each data sample
-- output
--  - Xhat reconstructed Input sampels N x M
function NonNegativeSparseAutoEncoder:apply(X)
    --for entire data matrix
    local Xhat = torch.zeros(X:size())
    for i=1,X:size(1) do
       self.inp:copy(X[{i,{}}])
       self:update()
       Xhat[{i,{}}]:copy(self.out)
    end
    
   return Xhat
end
 
-- Train the network
-- This function adapts the weight matrix W and parameters a and b
-- of the non-linear activation function (intrinsic plasticity)
--
-- input: 
--  - X is a N x M matrix holding the data input, N is the number of samples and M the dimension of each data sample
function NonNegativeSparseAutoEncoder:train(X)
    local numSamples = X:size(1)
    p = torch.randperm(numSamples) --for randomized presentation
    for i=1,numSamples do
        --set input
        self.inp:copy(X[{p[i],{}}])
        
        --forward propagation of activities
        self:update()
        
        --calculate adaptive learning rate
        local lrate = self.lrateRO/(self.regRO + torch.sum(torch.pow(self.h, 2)))

        --calculate error
        local error = self.inp - self.out        --update weights
        self.W = self.W + error * self.h:t()  * lrate

        --decay function for positive weights
        if self.decayP > 0 then
            local idx = self.W:gt(0)
            self.W[idx] = self.W[idx] - self.W[idx]* self.decayP
        end
        
        --decay function for negative weights
        local idx = self.W:lt(0)
        if self.decayN == 1 then
            --pure NN weights!
			self.W[idx]:zero()
        elseif self.decayN > 0 then
            self.W[idx] = self.W[idx] - self.W[idx]* self.decayN
        end
        
        --intrinsic plasticity
        local hones = torch.ones(self.hidDim,1)
        local tmp = self.lrateIP * (hones - (2.0 + 1.0/self.meanIP) * self.h + torch.pow(self.h, 2)/self.meanIP)

        self.b = self.b + tmp
        self.a = self.a + self.lrateIP *torch.cdiv( hones, self.a) + torch.cmul(self.g,tmp)
    end
end

-- Update network activation
-- This helper function computes the new activation pattern of the
-- hidden layer for a given input. Note that self.inp field has to be set in advance.
function NonNegativeSparseAutoEncoder:update()
    --excite network
    self.g =  self.W:t() * self.inp 

    --apply activation function
    self.h = torch.cinv(1 + torch.exp(torch.cmul(-self.a,self.g) - self.b))

    --read-out
    self.out = self.W *  self.h
end

return NonNegativeSparseAutoEncoder



