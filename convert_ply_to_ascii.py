import subprocess
import glob
import basics.plotting.movies as bmov
import basics.plotting.colormaps as bcmaps
import basics.plotting.plotting as bplt
import argparse
import basics.dataio as dio
import mesh
import sys
from mpl_toolkits.mplot3d import axes3d

"""Convert binary PLY files to ASCII format using Turk library and make a movie of the evolving surface over time.
Assumes that ply2ascii is in your $PATH. This command is available for download in the PLY tools here:
https://www.cc.gatech.edu/projects/large_models/ 
Instructions:
At the bottom of the page linked above, click "PLY" box, then 'download PLY tools'.
Add to PATH:
PATH=/Users/npmitchell/ply/:$PATH

Also assumes a working ffmpeg executable is in the pwd.

Options
-------
makemovie : arg 
    This script creates a movie of the converted plys
fn : arg
    The wildcarded file name of the ply
dir : arg
    The directory where the PLYs are stored  
overwrite : bool
    overwrite existing ASCII PLY files

Example usage
-------------
python convert_ply_to_ascii.py -makemovie -fn mesh_apical_*.ply -movfn mesh_apical_plys -indexsz 05 \
    -dir /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4UasCAAXmCherry/48Ygal4UasCAAXmCherry_20190207200_excellent/raw_meshes/
    
python convert_ply_to_ascii.py -fn mesh_apical_0*.ply -indexsz 06 \
    -dir /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/msls_output_nu0p10_s1_pn4_ps4_l1_l1/

python convert_ply_to_ascii.py -makemovie -fn pointCloud_T*_mesh.ply -movfn mesh_apical_plys -indexsz 06 \
    -dir /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/2016021015120_objFiles/

python convert_ply_to_ascii.py -makemovie -fn mesh_apical_*.ply -movfn mesh_apical_plys -indexsz 05 \
    -dir /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/plys_depth10/
    
python convert_ply_to_ascii.py -makemovie -fn mesh_apical_*.ply -movfn mesh_apical_plys -indexsz 05 \
    -dir /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4UasCAAXmCherry/48Ygal4UasCAAXmCherry_20190207200_excellent/raw_meshes/

"""

parser = argparse.ArgumentParser(description='Convert binary PLYs ascii and optionally make a movie of the result.')
parser.add_argument('-dir', '--datadir', help='Path to networks folder containing meshes',
                    type=str, default='/Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShisRFP/2016021015120_objFiles/cleaned/')
parser.add_argument('-fn', '--filename', help='file name of the meshes',
                    type=str, default='mesh_*.ply')  # 'cleaned_pointCloud_T*_mesh.ply'
parser.add_argument('-movfn', '--moviefilename', help='file name of the movie to output',
                    type=str, default='mesh_plys')
parser.add_argument('-indexsz', '--indexsz', help='How many digits in the timestep index for each PLY',
                    type=str, default='05')
parser.add_argument('-makemovie', '--makemovie', help='Make movie of plys', action='store_true')
parser.add_argument('-overwrite', '--overwrite', help='Overwrite existing ascii plys', action='store_true')
args = parser.parse_args()

filename = args.filename
binaries = glob.glob(dio.prepdir(args.datadir) + filename)

# If makemovie is true, create imagedirectory
if args.makemovie:
    import matplotlib.pyplot as plt
    imdir = args.datadir + 'images/'
    dio.ensure_dir(imdir)

subprocess.call(['rm', 'tmp'])
print('binaries = ', binaries)

dmyi = 0
for (binaryfn, kk) in zip(binaries, range(len(binaries))):
    outputfile = glob.glob(binaryfn[:-4] + '_ascii.ply')
    yesdo = args.overwrite or not outputfile
    if yesdo and binaryfn[-10:] != '_ascii.ply':
        print('preparing to convert ' + binaryfn)
        ifn = '<' + binaryfn
        ofn = '>' + binaryfn[:-4] + '_ascii.ply'
        command_line = 'ply2ascii ' + ifn + ' ' + ofn + '\n'
        # args = ['ply2ascii', ifn, ofn]
        # print 'args = ', args
        # subprocess.call(args)
        if dmyi == 0:
            with open('tmp', 'w') as fn:
                fn.write(command_line)
        else:
            with open('tmp', 'a') as fn:
                fn.write(command_line)

        dmyi += 1

subprocess.call(['bash', 'tmp'])
subprocess.call(['rm', 'tmp'])

# Make movie of all plys
if args.makemovie:
    surffns = sorted(glob.glob(dio.prepdir(args.datadir) + filename[:-4] + '_ascii.ply'))
    cmap = bcmaps.husl_cmap(name='husl_qual', n_colors=255, h=0.01, s=0.5, l=0.3)
    for (surffn, kk) in zip(surffns, range(len(surffns))):
        # Load the mesh

        surf = mesh.Mesh(surffn)
        xx, yy, zz = -surf.points[:, 1], -surf.points[:, 2], surf.points[:, 0]

        # Create image
        imgname = imdir + binaryfn[:-4] + '.png'
        # fig, axes = bplt.initialize_2panel_centy(Wfig=360, Hfig=270, fontsize=12, wsfrac=0.5, wssfrac=0.3,
        #                                          x0frac=0., hspace=10)
        fig, axes, cax = bplt.initialize_axis_stack(2, make_cbar=False, Wfig=90, Hfig=100.2, hfrac=0.9, wfrac=1.0,
                                                    x0frac=None,
                                                    y0frac=-0.2, vspace=-45, hspace=0, fontsize=8, wcbar_frac=0.,
                                                    tspace=0, projection='3d')
        ax, ax2 = axes[0], axes[1]

        # Create dorsal view
        # fig = plt.figure()
        # ax = fig.add_subplot(2, 1, 1, projection='3d')
        ax = plt.sca(ax)
        ax = plt.gca(projection='3d')
        ax.plot_trisurf(xx, yy, zz, triangles=surf.triangles, linewidth=0.3, antialiased=True, alpha=0.9,
                        cmap=cmap)

        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        ax.set_aspect('equal')
        ax.view_init(elev=90., azim=-90)
        ax = bplt.set_axes_equal(ax)
        ax.patch.set_alpha(0.)
        ax.axis('off')

        # Create lateral view
        # ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        ax2 = plt.sca(ax2)
        ax2 = plt.gca(projection='3d')
        ax2.plot_trisurf(xx, yy, zz, triangles=surf.triangles, linewidth=0.3, antialiased=True, alpha=0.9,
                         cmap=cmap)
        # ax2.axis('off')
        # ax2.set_xlabel('x')
        # ax2.set_ylabel('y')
        # ax2.set_zlabel('z')
        ax2.view_init(elev=0., azim=-90)
        ax2.set_aspect('equal')
        ax2 = bplt.set_axes_equal(ax2)
        ax2.patch.set_alpha(0.)
        ax2.axis('off')

        # Save the image
        plt.suptitle(r'$t=$' + str(int(kk)) + ' min')
        plt.savefig(imdir + 't{0:05d}.png'.format(kk))
        plt.close('all')

    # Make the movie
    imgname = dio.prepdir(imdir) + 't'
    movname = args.datadir + args.moviefilename
    bmov.make_movie(imgname, movname, indexsz=args.indexsz, framerate=10, rm_images=True, save_into_subdir=True, imgdir=imdir)

