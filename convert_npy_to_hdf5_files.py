import numpy as np
import basics.h5py_wrapper as h5
import glob.glob
import argparse

"""Go through all .npy files in a directory and convert them to .h5 HDF5 files

Example Usage:
python convert_npy_to_hdf5_files.py -dir ../data/ -key my_dataset
"""

parser = argparse.ArgumentParser(description='Compute level set using Morphological Chan-Vese ACWE.')
# Procedure options
parser.add_argument('-dir', '--datadir', help='Directory containing .npy files', type=str, default='./')
parser.add_argument('-key', '--key', help='Name of dataset to create', type=str, default='implicit_levelset')
args = parser.parse_args()

files = glob.glob(args.datadir + '*.npy')
for file in files:
    ls = np.load(file)
    filename = file[-3:] + 'h5'
    h5.write(ls, overwrite=True, filename=filename, key=args.key)
