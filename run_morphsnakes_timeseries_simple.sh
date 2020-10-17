# Example with parameters set for coral project with Bez Lemma


# Construct mesh for each timepoint
datDir="/path/to/my/data/";
cd $datDir
for (( num=0; num<=170; num++ )); do     
	mslsDir="${datDir}msls_output/";   
	idx=$(printf "%03d" $(( num )));     
	prev=$(printf "%03d" $(( num-1 )))     
        if (($num>0))
        then
	    # for subsequent timepoints, use the previous timepoint's h5 output
            initls=${mslsDir}msls_000${prev}.h5
        else
	    # for the first timepoint, use the initial guess h5 (may need to iterate the first timepoint several times
            initls=${mslsDir}msls_initguess.h5
        fi    
	python3 /mnt/data/code/morphsnakes_wrapper/run_morphsnakes.py -i Time_000${idx}_c1_stab_Probabilities.h5 -init_ls $initls -o         $mslsDir -prenu 0 -presmooth 2 -ofn_ply mesh_000${idx}.ply -ofn_ls msls_000${idx}.h5 -l1 1 -l2 1 -nu 0.0 -postnu 0 -smooth 0.2 -postsmooth 1 -exit 0.00010 -channel 1 -dtype h5 -permute zyxc -ss 1 -include_boundary_faces -center_guess 170,75,85 -rad0 80 -n 100 -save; 
done

# Note: for first timepoint, may need to run the above a few times with for (( num=0; num<=0; num++ )); do...
# Note: to ignore boundary faces, drop the -include_boundary_faces flag in the above command


############################
# Run mlx script to smooth
############################
# mlxprogram for CAAX, LifeAct:
mlxprogram='surface_rm_resample10k_reconstruct_LS3_25wu.mlx'
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

