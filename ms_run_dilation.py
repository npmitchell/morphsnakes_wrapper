from __future__ import print_function
import os
import logging
import numpy as np
from imageio import imread
import matplotlib
from matplotlib import pyplot as plt
import morphsnakes as ms
import basics.h5py_wrapper as dh5
import copy
from polyhedron.polyhedron import Polyhedron
import data_handling as dh
import basics.dataio as dio
import basics.plotting.plotting as bplt
import mcubes
from mpl_toolkits.mplot3d import Axes3D
import morphsnakes_aux_fns as msaux
import mesh
import argparse
import glob

""" Dilate morphosnakes level sets and save the dilated meshes.

Example usage:
python /mnt/data/code/gut_python/ms_run_dilation.py -savels -idtype npy -odtype h5 -o ./dilations -i ./ 


python ms_run_dilation.py -savels -idtype h5 -odtype h5 \
        -o /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/msls_nu0p10_s01_pnu05_ps04_l1p00_l1p00/ \
        -i /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/msls_nu0p10_s01_pnu05_ps04_l1p00_l1p00/dilated_ls/ \
        -ofn_ply mesh_apical_ms_ \
        -ofn_ls msls_apical_ 
"""

parser = argparse.ArgumentParser(description='Load level sets from Morphological Chan-Vese ACWE, dilate and resave.')

# Procedure options
parser.add_argument('-saveall', '--save_all', help='Save after each incremental dilation if True, rather than at end',
                    action='store_true')
parser.add_argument('-saveply', '--save_ply', help='Save the mesh from the dilated level set', action='store_true')
parser.add_argument('-savels', '--save_ls', help='Save the implicit level set after dilation', action='store_true')
parser.add_argument('-odtype', '--output_datatype', help='Datatype of the levelset to save (h5 or npy)', type=str,
                    default='h5')
parser.add_argument('-idtype', '--input_datatype', help='Datatype of the saved levelset to load (h5 or npy)', type=str,
                    default='h5')
parser.add_argument('-n', '--ndilations', help='Number of dilation steps to take', type=int, default=3)

# Parameter options
parser.add_argument('-i', '--input', help='Path to folder or file to read hdf5 probabilities' +
                                          'If dataset == True, then this is dir, else is filename.',
                    type=str, default='./')
parser.add_argument('-o', '--outputdir', help='Path to folder to which to write meshes',
                    type=str, default='./morphsnakes_output/')
parser.add_argument('-ifn_ls', '--inputfn_ls', help='Name of file in output dir to write level set as numpy array',
                    type=str, default='msls_apical_')
# Output Parameters
parser.add_argument('-ofn_ply', '--outputfn_ply', help='Name of file in output dir to write ply',
                    type=str, default='mesh_apical_ms_')
parser.add_argument('-ofn_ls', '--outputfn_ls', help='Name of file in output dir to write level set as numpy array',
                    type=str, default='none')

args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

ifn_ls_base = args.inputfn_ls
ofn_ply_base = args.outputfn_ply
ofn_ls_base = args.outputfn_ls
outputdir = dio.prepdir(args.outputdir)
inputdir = dio.prepdir(args.input)

ndilations = args.ndilations
searchstr = inputdir + ifn_ls_base + '*.' + args.input_datatype
print('searching for ' + searchstr)
fns = sorted(glob.glob(searchstr))
fns_clean = []
for fn in fns:
    if 'dilate' not in fn:
        fns_clean.append(fn)

outputdir = dio.ensure_dir(args.outputdir)
fns = fns_clean
print('fns = ', fns)
for fn in fns:
    # Load the implicit surface (the level set)
    if args.input_datatype == 'npy':
        u = np.load(fn)
    elif args.input_datatype in ['h5', 'hdf5']:
        u = dh5.h5open(fn, 'r')
        u = u['implicit_levelset']
    else:
        raise RuntimeError('Could not parse input datatype')

    # Dilate the surface
    for ii in range(ndilations):
        u = msaux.dilate(u)

        # Shall we save after this iteration?
        if args.save_all or ii == ndilations - 1:
            if args.save_ply:
                coords, triangles = mcubes.marching_cubes(u, 0.5)
                mm = mesh.Mesh()
                mm.points = coords
                mm.triangles = triangles

                # Naming
                # get extension
                exten = '.' + args.input_datatype
                tpid = fn.split(ifn_ls_base)[-1].split(exten)[0]
                outfn_ply = outputdir + args.outputfn_ply + tpid + '_dilate{0:03d}.ply'.format(ii + 1)

                print('saving ', outfn_ply)
                mm.save(outfn_ply)

            if args.save_ls:
                # Overwrite the extension if specified in the outputfn
                if args.outputfn_ls not in ['none', 'None', '']:
                    # Use custom output filename, get timepoint id tag
                    exten = '.' + args.input_datatype
                    tpid = fn.split(ifn_ls_base)[-1].split(exten)[0]
                    ofn_ls = outputdir + args.outputfn_ls + tpid
                    ofn_ls += '_dilate{0:03d}'.format(ii + 1) + '.' + args.output_datatype
                    print('ofn_ls = ', ofn_ls)
                else:
                    # Naming
                    exten = '.' + args.input_datatype
                    name = fn.split(exten)[-2].split('/')[-1]
                    print('name = ', name)
                    name += '_dilate{0:03d}'.format(ii + 1) + '.' + args.output_datatype
                    print('name = ', name)
                    ofn_ls = args.outputdir + name
                    print('name = ', ofn_ls)

                # Now save it
                if args.output_datatype == 'npy':
                    # Save ls for this timepoint as npy file
                    print('saving ', ofn_ls)
                    np.save(ofn_ls, u)
                elif args.output_datatype in ['h5', 'hdf5']:
                    # Save ls for this timepoint as an hdf5 file
                    print('saving ', ofn_ls)
                    msaux.save_ls_as_h5(ofn_ls, u)
                else:
                    raise RuntimeError('datatype for output file ls not understood')

