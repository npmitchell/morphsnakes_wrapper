# This bash script runs the script run_morphsnakes on a dataset and then smooths the output PLYs and saves the result.

# Copy msls_apical_init.npy from previous version into new version

# Run morphosnakes on a dataset
ssfactor=4
niter=30
niter0=175
ofn_ply='mesh_apical_ms_'
ofn_ls='msls_apical_'
ms_scriptDir='/mnt/data/code/gut_python/'
pre_nu=-5
pre_smoothing=1
lambda1=1
lambda2=1
exit_thres=0.000005
smoothing=0.10
nu=0.00
post_nu=2
post_smoothing=4
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
command+="-n "$niter" -n0 "$niter0" -exit "$exit_thres" -prob Probabilities.h5 -save "
command+="-init_ls "$mslsDir"msls_apical_init.npy"
echo "$command"
$command

# Run mlx script to smooth
# mlxprogram for CAAX, LifeAct:
mlxprogram='surface_rm_resample20k_reconstruct_LS3_1p2pc_ssfactor4.mlx'
# mlxprogram for Histone data:
# mlxprogram=''
fns=$mslsDir$ofn_ply*'.ply'
for pcfile in $fns; do
    # Clean up mesh file for this timepoint using MeshLab -----------------
    outputmesh=${pcfile/$ofn_ply/'mesh_apical_'}
    meshlabscript='./'$mlxprogram
    if [ ! -f $outputmesh ]; then
        echo $outputmesh
        command="meshlabserver -i $pcfile -o $outputmesh -s $meshlabscript -om vn"    
        $command
    else
        echo "File already exists: "$outputmesh	
    fi
done
