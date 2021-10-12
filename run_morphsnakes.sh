#!/usr/bin/env bash
# This bash script runs the script run_morphsnakes on a dataset and then smooths the output PLYs and saves the result.
# The data is 3D with potentially multiple channels but only one timepoint per file.

# Run morphosnakes on a dataset
ssfactor=4. # subsampling factor for the data (scales the output mesh relative to the pixel resolution of the data used for morphsnakes 
niter=30    # number of iterations for finding surface/level set
ofn_ply='mesh_' # output filename for the PLY, which is a mesh format
ofn_ls='msls_'	# output filename for the H5 with the (3D or 2D, depending on the data dimension) level set 
ms_scriptDir='/mnt/data/code/gut_python/'  
pre_nu=-5. 
pre_smoothing=1
smoothing=0.10
nu=0.00
post_nu=2
post_smoothing=4
lambda1=1
lambda2=1
exit_thres=0.00005
msls_exten="_prnu${pre_nu/-/n}"
msls_exten+='_prs'${pre_smoothing}
msls_exten+='_nu'${nu/'.'/'p'}
msls_exten+='_s'${smoothing/'.'/'p'}
msls_exten+='_pn'$post_nu'_ps'$post_smoothing
msls_exten+='_l'$lambda1'_l'$lambda2
mslsDir=$(pwd)'/msls_output'$msls_exten'/'
command="python ${ms_scriptDir}run_morphsnakes.py -dataset "
command+="-o $mslsDir "
command+="-i ./ "
command+="-ofn_ply $ofn_ply -rad0 10 -prenu $pre_nu -presmooth $pre_smoothing "
command+="-ofn_ls $ofn_ls -l1 $lambda1 -l2 $lambda2 -nu $nu "
command+="-smooth $smoothing -postsmooth $post_smoothing -postnu $post_nu "
command+="-n "$niter" -n0 "$niter0" -exit "$exit_thres" -prob Probabilities.h5 "
command+="-init_ls "$mslsDir"msls_apical_init.npy "
command+="-dtype h5 "
# command+="-save "
echo "$command"
$command
# example command:
# python /mnt/data/code/gut_python/run_morphsnakes.py -i Time_000000_c3.h5 -o /mnt/crunch/tolls/toll6_eve_neurotactin/msls_output_nu0p10_s1_pn4_ps4_l1_l1/ -prenu 0 -presmooth 0 -ofn_ply mesh_apical_ms_000000.ply -ofn_ls msls_apical_000000.npy -l1 1 -l2 1 -nu 0.1 -postnu 4 -channel -1 -smooth 1 -postsmooth 4 -exit 0.000001000 -dset_name inputData -rad0 10 -n 115
  


# Run one timepoint at a time
command="python ${ms_scriptDir}run_morphsnakes.py "
command+="-o $mslsDir "
command+="-ofn_ply $ofn_ply -rad0 10 -prenu $pre_nu -presmooth $pre_smoothing "
command+="-ofn_ls $ofn_ls -l1 $lambda1 -l2 $lambda2 -nu $nu "
command+="-smooth $smoothing -postsmooth $post_smoothing -postnu $post_nu "
command+="-n "$niter" -n0 "$niter0" -exit "$exit_thres" -prob Probabilities.h5 "
command+="-dtype h5 "

for (specify timepoints here)
	commandi = command+"-i "+fni
	jj=i-1
	commandi+=commandi+" -init_ls "$mslsDir"msls_apical_"jj".npy "
	echo "$commandi"
	$commandi



# Run mlx script to smooth
mlxprogram='surface_rm_resample20k_reconstruct_LS3_1p2pc_ssfactor4.mlx'
fns=$mslsDir$ofn_ply*'.ply'
for pcfile in $fns; do
    # Clean up mesh file for this timepoint using MeshLab -----------------
    outputmesh=${pcfile/$ofn_ply/'mesh_apical_0'}
    meshlabscript='./'$mlxprogram
    if [ ! -f $outputmesh ]; then
        echo $outputmesh
        command="meshlabserver -i $pcfile -o $outputmesh -s $meshlabscript -om vn"
        $command
    else
        echo "File already exists: "$outputmesh
    fi
done
