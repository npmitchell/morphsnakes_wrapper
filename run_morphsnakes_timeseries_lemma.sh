# Morphsnakes for active foam / Bez Lemma

rootDir="/Volumes/Elements/coral_project/2020_07_11/"

####################################################
## Build middle TP = 50
####################################################
for (( num=50; num<=50; num++ ))
do
    datDir="${rootDir}Probabilities/"
    mslsDir="${rootDir}msls_output/"
    idx=$(printf "%02d" $(( num )))
    prev=$(printf "%02d" $(( num-1 )))
    ## for middle timepoint only
    prev_ls="${mslsDir}msls_initguess.h5"
    python ./run_morphsnakes.py -i \
        ${datDir}Image_T${idx}_Probabilities.h5 \
        -init_ls ${prev_ls} -o $mslsDir \
        -prenu -4 -presmooth 1 -ofn_ply mesh_ms_000${idx}.ply -ofn_ls \
        msls_000${idx}.h5 -l1 1 -l2 1 -nu 0.1 -postnu 1 -channel 1 -smooth 0.1 -postsmooth 1 \
        -exit 0.000100000 -channel 0 -dtype h5 -permute cxyz -ss 4 -include_boundary_faces -save \
        -center_guess 50,220,220 -rad0 40 -n 100
done

## Repeat the above until the surface fills the entire structure


####################################################
## Now do TP=51-end
####################################################
for (( num=51; num<=79; num++ ))
do
    datDir="${rootDir}Probabilities/"
    mslsDir="${rootDir}msls_output/"
    idx=$(printf "%02d" $(( num )))
    prev=$(printf "%02d" $(( num-1 )))
    prev_ls="${mslsDir}msls_000${prev}.h5"
    python ./run_morphsnakes.py -i \
        ${datDir}Image_T${idx}_Probabilities.h5 \
        -init_ls ${prev_ls} -o $mslsDir \
        -prenu -4 -presmooth 1 -ofn_ply mesh_ms_000${idx}.ply -ofn_ls \
        msls_000${idx}.h5 -l1 1 -l2 1 -nu 0.1 -postnu 1 -channel 1 -smooth 0.1 -postsmooth 1 \
        -exit 0.000100000 -channel 0 -dtype h5 -permute cxyz -ss 4 -include_boundary_faces -save \
        -center_guess 50,220,220 -rad0 40 -n 30
done


####################################################
## Now do early TP backwards from middle timepoint 
####################################################
for (( num=49; num>=0; num-- ))
do
    datDir="${rootDir}Probabilities/"
    mslsDir="${rootDir}msls_output/"
    idx=$(printf "%02d" $(( num )))
    prev=$(printf "%02d" $(( num+1 )))
    prev_ls="${mslsDir}msls_000${prev}.h5"
    python ./run_morphsnakes.py -i \
        ${datDir}Image_T${idx}_Probabilities.h5 \
        -init_ls ${prev_ls} -o $mslsDir \
        -prenu -4 -presmooth 1 -ofn_ply mesh_ms_000${idx}.ply -ofn_ls \
        msls_000${idx}.h5 -l1 1 -l2 1 -nu 0.1 -postnu 1 -channel 1 -smooth 0.1 -postsmooth 1 \
        -exit 0.000100000 -channel 0 -dtype h5 -permute cxyz -ss 4 -include_boundary_faces -save \
        -center_guess 50,220,220 -rad0 40 -n 30
done


####################################################
# Run mlx script to smooth
####################################################
mlxprogram='surface_rm_resample10k_reconstruct_LS3_25wu.mlx'
ofn_ply='mesh_ms_'
fns=$mslsDir$ofn_ply*'.ply'
for pcfile in $fns; do
    # Clean up mesh file for this timepoint using MeshLab 
    # For output file name, replace the ofn_ply with 'mesh_'
    outputmesh=${pcfile/$ofn_ply/'mesh_'}
    meshlabscript='./'$mlxprogram
    if [ ! -f $outputmesh ]; then
        echo $outputmesh
        command="meshlabserver -i $pcfile -o $outputmesh -s $meshlabscript -om vn"
        $command
    else
        echo "File already exists: "$outputmesh	
    fi
done


