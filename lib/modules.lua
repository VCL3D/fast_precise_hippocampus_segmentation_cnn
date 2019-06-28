local utils = require 'lib.utils'

local M = {}

function M.segmentation(mri, sag_net, cor_net, ax_net, dtype, clearState)

	local mri_sag = mri:contiguous():view(mri:size(1), 1, mri:size(2), mri:size(3)):type(dtype)	
	local mask_sag = torch.FloatTensor(mri_sag:size()):type(dtype)
	for i = 1, mri_sag:size(1) do
		local slice = mri_sag:narrow(1, i, 1)
		if slice:sum() == 0 then
			mask_sag[{i, {}, {}}] = 0
		else
			mask_sag[{i, {}, {}}] = sag_net:forward(slice)
		end		
	end
	mask_sag = mask_sag:view(mask_sag:size(1), mask_sag:size(3), mask_sag:size(4)):type('torch.FloatTensor')
	if clearState then
		sag_net:clearState()
	end
		
	local mri_cor = mri:permute(3,2,1):contiguous():view(mri:size(3), 1, mri:size(2), mri:size(1)):type(dtype)	
	local mask_cor = torch.FloatTensor(mri_cor:size()):type(dtype)
	for i = 1, mri_cor:size(1) do
		local slice = mri_cor:narrow(1, i, 1)
		if slice:sum() == 0 then
			mask_cor[{i, {}, {}}] = 0
		else
			mask_cor[{i, {}, {}}] = cor_net:forward(slice)
		end		
	end
	mask_cor = mask_cor:view(mask_cor:size(1), mask_cor:size(3), mask_cor:size(4)):permute(3,2,1):type('torch.FloatTensor')
	if clearState then
		cor_net:clearState()
	end
	
	local mri_ax = mri:permute(2,3,1):contiguous():view(mri:size(2), 1, mri:size(3), mri:size(1))	
	mri_ax = image.flip(mri_ax, 3)
	mri_ax = mri_ax:type(dtype)
	local mask_ax = torch.FloatTensor(mri_ax:size()):type(dtype)
	for i = 1, mri_ax:size(1) do
		local slice = mri_ax:narrow(1, i, 1)
		if slice:sum() == 0 then
			mask_ax[{i, {}, {}}] = 0
		else
			mask_ax[{i, {}, {}}] = ax_net:forward(slice)
		end		
	end
	mask_ax = mask_ax:view(mask_ax:size(1), mask_ax:size(3), mask_ax:size(4)):permute(3,1,2):type('torch.FloatTensor')
	mask_ax = image.flip(mask_ax:contiguous(), 3)
	if clearState then
		ax_net:clearState()	
	end
	
	local mask = (mask_sag + mask_cor + mask_ax) / 3
		
	return mask:type('torch.FloatTensor')
		
end

function M.error_correction(mri, input_mask, sag_net, cor_net, ax_net, dtype, clearState)

	local mri_sag = mri:contiguous():view(mri:size(1), 1, mri:size(2), mri:size(3)):type(dtype)
	local input_mask_sag = input_mask:contiguous():view(mri:size(1), 1, mri:size(2), mri:size(3)):type(dtype)	
	local mask_sag = torch.FloatTensor(mri_sag:size()):type(dtype)
	for i = 1, mri_sag:size(1) do
		local mri_slice = mri_sag:narrow(1, i, 1)
		local input_mask_slice = input_mask_sag:narrow(1, i, 1)
		mask_sag[{i, {}, {}}] = sag_net:forward(torch.cat(mri_slice, input_mask_slice, 2))	
	end
	mask_sag = mask_sag:view(mask_sag:size(1), mask_sag:size(3), mask_sag:size(4)):type('torch.FloatTensor')
	if clearState then
		sag_net:clearState()
	end
		
	local mri_cor = mri:permute(3,2,1):contiguous():view(mri:size(3), 1, mri:size(2), mri:size(1)):type(dtype)
	local input_mask_cor = input_mask:permute(3,2,1):contiguous():view(mri:size(3), 1, mri:size(2), mri:size(1)):type(dtype)
	local mask_cor = torch.FloatTensor(mri_cor:size()):type(dtype)
	for i = 1, mri_cor:size(1) do
		local mri_slice = mri_cor:narrow(1, i, 1)
		local input_mask_slice = input_mask_cor:narrow(1, i, 1)
		mask_cor[{i, {}, {}}] = cor_net:forward(torch.cat(mri_slice, input_mask_slice, 2))	
	end
	mask_cor = mask_cor:view(mask_cor:size(1), mask_cor:size(3), mask_cor:size(4)):permute(3,2,1):type('torch.FloatTensor')
	if clearState then
		cor_net:clearState()
	end
	
	local mri_ax = mri:permute(2,3,1):contiguous():view(mri:size(2), 1, mri:size(3), mri:size(1))	
	local input_mask_ax = input_mask:permute(2,3,1):contiguous():view(mri:size(2), 1, mri:size(3), mri:size(1))
	mri_ax = image.flip(mri_ax, 3)
	input_mask_ax = image.flip(input_mask_ax, 3)
	mri_ax = mri_ax:type(dtype)
	input_mask_ax = input_mask_ax:type(dtype)
	local mask_ax = torch.FloatTensor(mri_ax:size()):type(dtype)
	for i = 1, mri_ax:size(1) do
		local mri_slice = mri_ax:narrow(1, i, 1)
		local input_mask_slice = input_mask_ax:narrow(1, i, 1)
		mask_ax[{i, {}, {}}] = ax_net:forward(torch.cat(mri_slice, input_mask_slice, 2))	
	end
	mask_ax = mask_ax:view(mask_ax:size(1), mask_ax:size(3), mask_ax:size(4)):permute(3,1,2):type('torch.FloatTensor')
	mask_ax = image.flip(mask_ax:contiguous(), 3)
	if clearState then
		ax_net:clearState()
	end
	
	local mask = (mask_sag + mask_cor + mask_ax) / 3	
		
	return mask:type('torch.FloatTensor')
		
end

function M.cropping(mri, mask, opt)

	local sag_center, cor_center, ax_center = utils.mask_weight_center(mask)
	sag_center = math.floor(sag_center + 0.5)
	cor_center = math.floor(cor_center + 0.5)
	ax_center = math.floor(ax_center + 0.5)
	
	local sag_crop_min = math.floor(sag_center - opt.sag_crop_size / 2 + 1)
	local sag_crop_max = sag_crop_min + opt.sag_crop_size - 1
	local cor_crop_min = math.floor(cor_center - opt.cor_crop_size / 2 + 1)
	local cor_crop_max = cor_crop_min + opt.cor_crop_size - 1
	local ax_crop_min = math.floor(ax_center - opt.ax_crop_size / 2 + 1)
	local ax_crop_max = ax_crop_min + opt.ax_crop_size - 1
	
	local crop_box_coords = {{sag_crop_min, sag_crop_max}, {ax_crop_min, ax_crop_max}, {cor_crop_min, cor_crop_max}}
	
	local cropped_mri = mri[crop_box_coords]
	local cropped_mask = mask[crop_box_coords]

	return cropped_mri, cropped_mask, crop_box_coords

end

return M
