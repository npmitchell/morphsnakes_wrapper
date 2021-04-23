# Example with parameters set for CAAX excellent
# 

# Construct mesh for each timepoint
datDir="/mnt/crunch/48Ygal4UASCAAXmCherry/201902072000_excellent/Time6views_60sec_1p4um_25x_obis1p5_2/data/deconvolved_16bit/";

cd $datDir
tp=85;
for (( num=51; num<=100; num++ )); do        
	
	tpx=$(printf "%03d" $(( tp )));     
	idx=$(printf "%03d" $(( num )));     
	prev=$(printf "%03d" $(( num-1 )))     
        
	mslsDir="${datDir}msls_output_growEvolveDemo_tp${tpx}/";

	# for all iterations, use the selected timepoint's h5 output
        initls=${mslsDir}msls_grow000${prev}.h5
            
	python /mnt/data/code/morphsnakes_wrapper/morphsnakes_wrapper/run_morphsnakes.py -i Time_000${tpx}_c1_stab_Probabilities.h5 -init_ls $initls -o  $mslsDir -prenu 0 -presmooth 0 -ofn_ply mesh_grow000${idx}.ply -ofn_ls msls_grow000${idx}.h5 -l1 1 -l2 1 -nu 0.0 -postnu 0 -smooth 0.2 -postsmooth 5 -exit 0.00010 -channel 1 -dtype h5 -permute zyxc -ss 4 -include_boundary_faces -center_guess 175,75,100 -rad0 10 -n 2 ; 
done

# -postsmooth 1
# -save

# Adjustment: connect to other series by postnu=3 and prenu=-3


for (( num=0; num<=100; num++ )); do         tpx=$(printf "%03d" $(( tp )));      idx=$(printf "%03d" $(( num )));      prev=$(printf "%03d" $(( num-1 )))     ;          mslsDir="${datDir}msls_output_growEvolveDemo_tp${tpx}/";         initls=${mslsDir}msls_grow000${prev}.h5;              python /mnt/data/code/morphsnakes_wrapper/morphsnakes_wrapper/run_morphsnakes.py -i Time_000${tpx}_c1_stab_Probabilities.h5 -init_ls $initls -o  $mslsDir -prenu -3 -presmooth 0 -ofn_ply mesh_grow000${idx}.ply -ofn_ls msls_grow000${idx}.h5 -l1 1 -l2 1 -nu 0.0 -postnu 3 -smooth 0.2 -postsmooth 5 -exit 0.00010 -channel 1 -dtype h5 -permute zyxc -ss 4 -include_boundary_faces -center_guess 175,75,100 -rad0 10 -n 2 ;  done


# Note: for first timepoint, may need to run the above a few times with for (( num=0; num<=0; num++ )); do...
# Note: to ignore boundary faces, drop the -include_boundary_faces flag in the above command


############################
# Run mlx script to smooth
############################
# mlxprogram for CAAX, LifeAct:
# mlxprogram='surface_rm_resample10k_reconstruct_LS3_25wu.mlx'
mlxprogram='/mnt/data/code/meshlab_codes/laplace_surface_rm_resample30k_reconstruct_LS3_1p2pc_ssfactor4.mlx'
# mlxprogram for Histone data:
# mlxprogram=''
fns=$mslsDir$ofn_ply*'.ply'
for pcfile in $fns; do
    # Clean up mesh file for this timepoint using MeshLab -----------------
    outputmesh=${pcfile/$ofn_ply/'mesh_apical_'}
    if [ ! -f $outputmesh ]; then
        echo $outputmesh
        command="meshlabserver -i $pcfile -o $outputmesh -s $mlxprogram -om vn"    
        $command
    else
        echo "File already exists: "$outputmesh	
    fi
done

