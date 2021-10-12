#!/usr/bin/env bash
# This bash script runs the script run_morphsnakes on a dataset and then smooths the output PLYs and saves the result.
# The data is 3D with potentially multiple channels but only one timepoint per file.

# Run morphosnakes on a dataset
ssfactor=1. # subsampling factor for the data (scales the output mesh relative to the pixel resolution of the data used for morphsnakes
niter=30    # number of iterations for finding surface/level set
ofn_ply='mesh_' # output filename for the PLY, which is a mesh format
ofn_ls='msls_'	# output filename for the H5 with the (3D or 2D, depending on the data dimension) level set 
ms_scriptDir='/path/to/morphsnakes_wrapper/' # path containing run_morphsnakes.py
pre_nu=-5.      # how much to dilate the intial guess level set before minimizing free energy
pre_smoothing=1 # how much to smooth the inital guess level set before minimizing free energy
smoothing=0.10  # how much to smooth per iteration during minimization of free energy
nu=0.00         # effective pressure strength, how much to dilate per iteration
post_nu=0       # after minimization has converged, dilate the result this many voxels
post_smoothing=4    # after minimization has converged, smooth the result this many times
lambda1=1       # relative weight of the attachment term1
lambda2=1       # relative weight of the attachment term2
exit_thres=0.00005  # stop iterating if our fractional change per iteration is smaller than this amount
mslsDir=$(pwd)'/msls_output/'  # output directory
inputDataFn="myData.tiff"  # input TIFF or H5 or NPY data to find level set of.
rad0=10         # radius of initial guess if no initial guess level set is supplied
initialLevelsetFn="myInitialGuess.h5" # initial guess if you want to start with an initial guess instead of default sphere

# build a command to execute the morphsnakes algorithm
command="python ${ms_scriptDir}run_morphsnakes.py "
command+="-o $mslsDir "
command+="-i "$inputDataFn
command+="-ofn_ply $ofn_ply -rad0 10 -prenu $pre_nu -presmooth $pre_smoothing "
command+="-ofn_ls $ofn_ls -l1 $lambda1 -l2 $lambda2 -nu $nu "
command+="-smooth $smoothing -postsmooth $post_smoothing -postnu $post_nu "
command+="-n "$niter" -exit "$exit_thres" "
command+="-init_ls "$mslsDir$initialLevelsetFn
command+="-dtype h5 "
command+="-save "
echo "$command"
$command
# example command:
# python /mnt/data/code/gut_python/run_morphsnakes.py -i Time_000000_c3.h5 -o /mnt/crunch/tolls/toll6_eve_neurotactin/msls_output_nu0p10_s1_pn4_ps4_l1_l1/ -prenu 0 -presmooth 0 -ofn_ply mesh_apical_ms_000000.ply -ofn_ls msls_apical_000000.npy -l1 1 -l2 1 -nu 0.1 -postnu 4 -channel -1 -smooth 1 -postsmooth 4 -exit 0.000001000 -dset_name inputData -rad0 10 -n 115
  


# Optional: Run mlx script to smooth
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
