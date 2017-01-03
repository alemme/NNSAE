---CREATEBARSDATASET create a set of square images stored row-wise in a matrix
--   [Xtrain Xtest] = createBarsDataSet(dim, numTrain, numTest, nonlinear)
--   dim - image width = height = dim
--   numTrain - number of training images
--  numTest - number of test images
--   nonlinear - if this flag is true/1, pixel intensities at crossing points of bars are not added up
local function createBarsDataSet(dim, numTrain, numTest, nonlinear)

	local dim = dim or 10
	local numTrain = numTrain or 10000
	local numTest = numTest or 5000
	local nonlinear = nonlinear or True
	local imgdim = dim * dim;
	local X = torch.zeros(numTrain+numTest, imgdim)
	
	for k=1, numTrain+numTest do
	    local x = torch.zeros(dim, dim)
	    for z=1,2 do
	        i = torch.randperm(dim)
	        j = torch.randperm(dim)
	        if nonlinear > 0 then
	            x[{i[z],{}}] = 1.0
	            x[{{},j[z]}] = 1.0
	        else
	            x[{i[z],{}}] = x[{i[z],{}}] + 1.0
	            x[{{},j[z]}] = x[{{},j[z]}] + 1.0
	        end
	    end
	    if not nonlinear then
	        x = x / 4
	    end
	    X[{k,{}}] = torch.reshape(x, 1, imgdim)
	end
	
	Xtrain = X[{{1,numTrain}, {}}]
	Xtest = X[{{numTrain+1,numTrain+numTest}, {}}]
	print("Xtest:size()")
	print(Xtest:size())
	return Xtrain, Xtest
end

return createBarsDataSet