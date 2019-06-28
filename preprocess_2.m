clear

inputDir = 'preprocess_1_output/';
outputFilename = 'test_data.mat';

%%
addpath('lib\mominc-master');

mriFiles = dir([inputDir '*brain_corrected.mnc']);

data = cell(length(mriFiles), 1);

for i = 1:length(mriFiles)

    disp(i)
    mriFilename = mriFiles(i).name;
    maskFilename = strrep(mriFilename, 'brain_corrected', 'brain_mask');
    
    [~, mri] = minc_read([inputDir mriFilename]);   
    [~, brainMask] = minc_read([inputDir maskFilename]);
   
    brain = mri(brainMask == 1);  
    m = mean(brain);
    sd = std(brain);
    
    mri = (mri - m) / sd;
    mri(brainMask == 0) = 0;
    
    mri = single(mri);

    % At this point, the order of axes of the MRI should be: sagittal x axial x coronal
    % Use the 'permute' function if nessesary to get the desired axes order.  
    % Then, using 'flip', set the orientation of each MRI volume so that the origin
    % (voxel 1,1,1) is located to the left, top and back of the head.
    
    % For MRIs of the HarP dataset, use:
    %mri = permute(mri, [1 3 2]);
    %mri = flip(mri, 2);
    
    % For MRIs of the MICCAI dataset, use:
    mri = flip(mri, 1);
  
    data{i} = mri;    
    
end  

save(outputFilename, 'data')
