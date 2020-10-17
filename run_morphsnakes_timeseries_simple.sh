
# Construct mesh for each timepoint
datDir="/mnt/crunch/gut/Mef2Gal4klarUASCAAXmChHiFP/202007151930_1p4um_0p5msexp/Time3views_25x_60s/data/deconvolved_16bit/";
cd $datDir
for (( num=0; num<=170; num++ )); do     
	mslsDir="${datDir}msls_output/";   
	idx=$(printf "%03d" $(( num )));     
	prev=$(printf "%03d" $(( num-1 )))     
        if (($num>0))
        then
	    # for subsequent timepoints, use the previous timepoint's h5 output
            initls=${mslsDir}msls_apical_stab_000${prev}.h5
        else
	    # for the first timepoint, use the initial guess h5 (may need to iterate the first timepoint several times
            initls=${mslsDir}msls_initguess.h5
        fi    
	python3 /mnt/data/code/morphsnakes_wrapper/run_morphsnakes.py -i Time_000${idx}_c1_stab_Probabilities.h5 -init_ls $initls -o         $mslsDir -prenu -4 -presmooth 0 -ofn_ply mesh_apical_ms_stab_000${idx}.ply -ofn_ls msls_apical_stab_000${idx}.h5 -l1 1 -l2 1 -nu 0.4 -postnu 1 -smooth 0.2 -postsmooth 1 -exit 0.000010 -channel 1 -dtype h5 -permute zyxc -ss 4 -include_boundary_faces -center_guess 170,75,85 -rad0 25 -n 25 -save; 
done

# Note: to ignore boundary faces, drop the -include_boundary_faces flag in the above command


############################
# Run mlx script to smooth
############################
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

