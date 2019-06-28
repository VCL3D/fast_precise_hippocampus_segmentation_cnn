INPUT_DIR=MRI 
OUTPUT_DIR=preprocess_1_output

source /opt/minc/1.9.16/minc-toolkit-config.sh 

mkdir $OUTPUT_DIR

# Uncoment to convert input MRIS from MINC to Nifti format (e.g. for HarP MRIs)
#
#echo "*** Converting MRIs to Nifti format ***"
#for file in $INPUT_DIR/*.mnc
#do
#	echo $file
#	mnc2nii $file $INPUT_DIR/$(basename $file .mnc).nii -quiet >/dev/null
#done

echo "*** Extracting brain using BET ***"
for file in $INPUT_DIR/*.nii*
do
	echo $file
	bet $file $OUTPUT_DIR/$((basename $file) | cut -f 1 -d '.')_brain -m
done

echo "*** Converting BET outputs to MINC format ***"
for file in $OUTPUT_DIR/*.nii*
do
	echo $file
	nii2mnc $OUTPUT_DIR/$(basename $file .nii.gz).nii -quiet
done

echo "*** Inhomegeneity correction using N3 ***"
for file in $OUTPUT_DIR/*[!mask].mnc
do
	echo $file
	nu_correct -quiet $file $OUTPUT_DIR/$(basename $file .mnc)_corrected.mnc >/dev/null
done

rm $OUTPUT_DIR/*.imp
rm $OUTPUT_DIR/*.nii.gz
rm $OUTPUT_DIR/*brain.mnc

echo "*** Done ***"