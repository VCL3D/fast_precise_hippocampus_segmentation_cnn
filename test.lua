require 'torch'
require 'nn'
require 'image'
local matio = require 'matio'
local modules = require 'lib.modules'
local utils = require 'lib.utils'
local cmd = torch.CmdLine()

-- Inference options
cmd:option('-models_dir', 'models/HarP_train')
cmd:option('-input_filename', 'test_data.mat')
cmd:option('-output_filename', 'output_masks.mat')
cmd:option('-sag_crop_size', 120)
cmd:option('-cor_crop_size', 100)
cmd:option('-ax_crop_size', 100)
cmd:option('-threshold', 0.5)

-- GPU options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-clear_state', 0)

function main()

	local opt = cmd:parse(arg)

	local dtype, use_cudnn = utils.setup_gpu(opt.gpu, 'cuda', opt.use_cudnn == 1)

	print('Loading trained models')
	models = {}
	models['seg_sag_cnn'] = torch.load(opt.models_dir .. '/segmentation_sagittal.t7')
	models['seg_cor_cnn'] = torch.load(opt.models_dir .. '/segmentation_coronal.t7')
	models['seg_ax_cnn'] = torch.load(opt.models_dir .. '/segmentation_axial.t7')
	models['ec_sag_cnn'] = torch.load(opt.models_dir .. '/errorCorrection_sagittal.t7')
	models['ec_cor_cnn'] = torch.load(opt.models_dir .. '/errorCorrection_coronal.t7')
	models['ec_ax_cnn'] = torch.load(opt.models_dir .. '/errorCorrection_axial.t7')
	for _, cnn in pairs(models) do
		cnn:type(dtype)
	end
	if use_cudnn then
		for _, cnn in pairs(models) do
			cudnn.convert(cnn, cudnn)
		end
	end
	for _, cnn in pairs(models) do
		cnn:evaluate()
	end
	
	print('Loading test data')
	test_mris = matio.load(opt.input_filename).data
	
	print('Semgenting MRIs')
	local output_masks = {}
	
	for k = 1, #test_mris do
		
		print(k)
		
		local mri = test_mris[k]
		
		local seg_mask = modules.segmentation(mri, models['seg_sag_cnn'], models['seg_cor_cnn'], models['seg_ax_cnn'], dtype, opt.clear_state == 1)
		
		local cropped_mri, cropped_seg_mask, crop_box_coords = modules.cropping(mri, seg_mask, opt)
		
		local ec_mask = modules.error_correction(cropped_mri, cropped_seg_mask, models['ec_sag_cnn'], models['ec_cor_cnn'], models['ec_ax_cnn'], dtype, opt.clear_state == 1)
		
		local final_mask = torch.FloatTensor(mri:size()):zero()		
		final_mask[crop_box_coords] = ec_mask
		
		final_mask[torch.ge(final_mask, opt.threshold)] = 1
		final_mask[torch.lt(final_mask, opt.threshold)] = 0		

		output_masks['mask_' .. tostring(k)] = final_mask	
	end
	
	print('Saving masks')
	matio.save(opt.output_filename, {masks = output_masks, n = #test_mris})
	
	print('Done!')

end

main()
