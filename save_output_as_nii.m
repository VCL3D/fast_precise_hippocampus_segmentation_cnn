clear

MRI_dir = 'MRI/';

masks_filename = 'output_masks.mat';

output_dir = 'output_masks/';

%%
addpath('lib\nifti');

load(masks_filename);

input_files = dir([MRI_dir '*nii*']);

if ~exist(output_dir, 'dir')
   mkdir(output_dir)
end

for i = 1:n
   
   disp(i)

   input_nii = load_nii([MRI_dir input_files(i).name]);
    
   mask = eval(['masks.mask_' num2str(i)]);   
   mask = flip(permute(uint8(mask), [1 3 2]), 3);

   output_nii = make_nii(mask);
   output_nii.hdr.hist = input_nii.hdr.hist;  
   save_nii(output_nii, [output_dir 'output_mask_' num2str(i) '.nii.gz']);
    
end
