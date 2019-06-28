local M = {}

function M.setup_gpu(gpu, backend, use_cudnn)
  local dtype = 'torch.FloatTensor'
  if gpu >= 0 then
    if backend == 'cuda' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(gpu + 1)
      dtype = 'torch.CudaTensor'
      if use_cudnn then
        require 'cudnn'
        cudnn.benchmark = true
      end
    elseif backend == 'opencl' then
      require 'cltorch'
      require 'clnn'
      cltorch.setDevice(gpu + 1)
      dtype = torch.Tensor():cl():type()
      use_cudnn = false
    end
  else
    use_cudnn = false
  end
  return dtype, use_cudnn
end

function M.mask_weight_center(mask)

	local s, n = 0, 0
	for i = 1, mask:size(1) do
		local sliceSum = mask:narrow(1, i, 1):sum()
		s = s + i * sliceSum
		n = n + sliceSum
	end
	local x = s / n

	s, n = 0, 0
	for i = 1, mask:size(2) do
		local sliceSum = mask:narrow(2,i,1):sum()
		s = s + i * sliceSum
		n = n + sliceSum
	end
	local z = s / n 

	s, n = 0, 0
	for i = 1, mask:size(3) do
		local sliceSum = mask:narrow(3,i,1):sum()
		s = s + i * sliceSum
		n = n + sliceSum
	end
	local y = s / n
	
	return x, y, z
	
end

return M
