import sys
import glob
import h5py
import numpy as np
import tifffile
import code

#labelData = sys.argv[1]
rawData = sys.argv[1]

f = h5py.File("VCN_test1.h5", "w")

#l = sorted(glob.glob(labelData + "*.tif*"))

#img = tifffile.imread(l[0])

#xx,yy = img.shape

grp = f.create_group("volume")
#dset = grp.create_dataset("labels", shape=(len(l),yy,xx), dtype='uint32')



#for ii,each in enumerate(l):
#	print ii
#	img = tifffile.imread(each)
#	dset[ii,:,:] = img.transpose((1,0))



l = sorted(glob.glob(rawData + "*.tif*"))

img = tifffile.imread(l[0])

xx,yy = img.shape

dset = grp.create_dataset("raw", shape=(len(l),yy,xx), dtype='uint8')



for ii,each in enumerate(l):
	print ii
	img = tifffile.imread(each)
	dset[ii,:,:] = img.transpose((1,0))



f.close()


