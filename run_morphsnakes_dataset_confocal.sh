# This bash script runs the script run_morphsnakes on a dataset and then smooths the output PLYs and saves the result.

# Copy msls_apical_init.npy from previous version into new version

# CAAX parameters
# -prenu -5
# -presmooth 0
# -l1 1 
# -l2 1
# -nu 0.0
# -postnu 0
# -smooth 0.2
# -postsmooth 5 
# -channel 1 
# -permute zyxc 
# -ss 4

# Run morphosnakes on a dataset
ssfactor=2
niter=35
niter0=75
ofn_ply='mesh_ms_'
ofn_ls='msls_'
ms_scriptDir='/mnt/data/code/morphsnakes_wrapper/morphsnakes_wrapper/'
pre_nu=-10
pre_smoothing=0
lambda1=1
lambda2=1
exit_thres=0.000005
smoothing=2
nu=0
post_nu=2
post_smoothing=5
msls_exten="_prnu${pre_nu/-/n}"
msls_exten+='_prs'${pre_smoothing}
msls_exten+='_nu'${nu/'.'/'p'}
msls_exten+='_s'${smoothing/'.'/'p'}
msls_exten+='_pn'$post_nu'_ps'$post_smoothing
msls_exten+='_l'$lambda1'_l'$lambda2
mslsDir=$(pwd)'/msls_output/'
command="python ${ms_scriptDir}run_morphsnakes.py -dataset "
command+="-o $mslsDir "
command+="-i ./ -rad0 40 "
command+="-ofn_ply $ofn_ply -rad0 10 -prenu $pre_nu -presmooth $pre_smoothing "
command+="-ofn_ls $ofn_ls -l1 $lambda1 -l2 $lambda2 -nu $nu "
command+="-smooth $smoothing -postsmooth $post_smoothing -postnu $post_nu "
command+="-n "$niter" -n0 "$niter0" -exit "$exit_thres" -prob Probabilities.h5 -ss $ssfactor "
command+="-init_ls "$mslsDir"msls_initguess.h5 "
# command+=" -save "
command+="-adjust_for_MATLAB_indexing"
echo "$command"
# $command


# datDir="/path/to/my/data/";
# cd $datDir
for (( num=1; num<=1; num++ )); do     
	mslsDir="${datDir}msls_output/";   
	idx=$(printf "%03d" $(( num )));     
	prev=$(printf "%03d" $(( num-1 )))     
       if (($num>1))
       then
	# for subsequent timepoints, use the previous timepoint's h5 output
           initls=${mslsDir}msls_000${prev}.h5
	   niterThis=$niter0 
       else
	#  for the first timepoint, use the initial guess h5 (may need to iterate the first timepoint several times
           initls=${mslsDir}msls_initguess.h5
           niterThis=$niter
       fi    
	python /mnt/data/code/morphsnakes_wrapper/morphsnakes_wrapper/run_morphsnakes.py -i antpOCRLgap43_T${idx}_Probabilities.h5 -init_ls $initls -o $mslsDir -prenu $pre_nu -presmooth $pre_smoothing -ofn_ply mesh_ms_000${idx}.ply -ofn_ls msls_000${idx}.h5 -l1 $lambda1 -l2 $lambda2 -nu $nu -postnu $post_nu -smooth $smoothing -postsmooth $post_smoothing -exit $exit_thres -channel 1 -dtype h5 -ss $ssfactor -include_boundary_faces -rad0 80 -n $niterThis -save -adjust_for_MATLAB_indexing -center_guess 150,180,310 -permute cxyz;
done


## OFFSET EACH MESH BY A BIT
# datDir="/path/to/my/data/";
# cd $datDir
for (( num=2; num<=39; num++ )); do     
	mslsDir="${datDir}msls_output/";   
	idx=$(printf "%03d" $(( num )));     
	prev=$(printf "%03d" $(( num )))     

	# for subsequent timepoints, use the previous timepoint's h5 output
        initls=${mslsDir}msls_000${prev}.h5
	          
	python /mnt/data/code/morphsnakes_wrapper/morphsnakes_wrapper/run_morphsnakes.py -i antpOCRLgap43_T${idx}_Probabilities.h5 -init_ls $initls -o $mslsDir -prenu 3 -presmooth 1 -ofn_ply mesh_ms_000${idx}_offset.ply -ofn_ls msls_000${idx}_offset.h5 -l1 $lambda1 -l2 $lambda2 -nu 0 -postnu -10 -smooth 2 -postsmooth 1 -exit $exit_thres -channel 1 -dtype h5 -ss $ssfactor -include_boundary_faces -rad0 80 -n 1 -adjust_for_MATLAB_indexing -center_guess 150,180,310 -permute cxyz;
done


# Run mlx script to smooth
# mlxprogram for CAAX, LifeAct:
mlxprogram='surface_rm_resample25k_reconstruct_LS3_1p2pc_ssfactor4.mlx'
# mlxprogram for Histone data:
# mlxprogram=''
ofn_ply="mesh_ms_"
fns=$mslsDir$ofn_ply*'.ply'
for pcfile in $fns; do
    # Clean up mesh file for this timepoint using MeshLab -----------------
    outputmesh=${pcfile/$ofn_ply/'mesh_apical_stab_'}
    meshlabscript='./'$mlxprogram
    if [ ! -f $outputmesh ]; then
        echo $outputmesh
        command="meshlabserver -i $pcfile -o $outputmesh -s $meshlabscript -om vn"    
        # $command
    else
        echo "File already exists: "$outputmesh	
    fi
done
