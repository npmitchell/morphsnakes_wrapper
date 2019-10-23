import os
import numpy as np
import run_morphsnakes as rms
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import morphsnakes as ms

"""For demonstrating the level set method, step by step output of meshes
"""


def save_iter_callback(plot_each):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Saves a mesh for the current timepoint.

    Parameters
    ----------



    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.

    """
    # Prepare the visual environment.
    counter = [-1]

    def callback(levelset, force=False):
        """

        Parameters
        ----------
        levelset : the levelset to save or pass
        force : bool
            force callback to run without returning prematurely

        Returns
        -------

        """
        counter[0] += 1
        if (counter[0] % plot_each) != 0 and not force:
            return

        if ax.collections:
            del ax.collections[0]

        coords, triangles = mcubes.marching_cubes(levelset, 0.5)
        # todo: save the mesh here

    return callback


def extract_levelset(fn, iterations=150, smoothing=0, lambda1=1, lambda2=1, nu=None, post_smoothing=1, post_nu=1,
                     channel=0, init_ls=None, exit_thres=5e-6,
                     center_guess=None, radius_guess=None, outdir=None, dset_name='exported_data',
                     plot_each=5, axis_order='xyzc',
                     clip=None, mask=None):
    """Extract the level set from a 2d or 3d image.

    Parameters
    ----------
    fn : str
        path to the image (possibly 3D) to load. Can be h5, png, or npy file
    iterations : int
        How many iterations to perform
    nu : float or None, optional
        pressure to apply periodically. If float and < 1, applies one pressure step every 1/nu iterations
    smoothing : uint, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    lambda1 : float, optional
        Weight parameter for the outer region. If `lambda1` is larger than
        `lambda2`, the outer region will contain a larger range of values than
        the inner region.
    lambda2 : float, optional
        Weight parameter for the inner region. If `lambda2` is larger than
        `lambda1`, the inner region will contain a larger range of values than
        the outer region.
    post_smoothing : int, optional
    channel : int or None
        If not None, select this channel (last index of the array) of the image to analyze.
    init_ls : int binary array of inside/outside in 3D or None
        Initial guess for the level set
    center_guess : length 3 float array
        Center of mass of the spherical guess if init_ls not supplied.
    radius_guess : float
        Radius of initial spherical guess if init_ls not supplied.
        (An argument for creating an initial guess via ms.circle_level_set() if init_ls not supplied)
    plot_each : int
        How often (in #iterations) to save a snapshot png of the morphological process.
    clip : float or None
        If not none, clip all values above this in the image on which to run morphsnakes

    Returns
    -------
    ls : L x W x H binary int array
        The output level set computed after iterations
    """
    logging.info('Running MorphACWE...')

    # Load the image.
    print('loading ' + fn)
    img = load_img(fn, channel, dset_name=dset_name, axis_order=axis_order)

    if clip is not None:
        img[img > clip] = clip

    if mask is not None:
        img *= mask

    # Initialization of the level-set.
    if init_ls is None:
        print('No initial levelset supplied, using default sphere...')
        if center_guess is None:
            center_guess = (np.shape(img)[0]*0.5, np.shape(img)[1]*0.5, np.shape(img)[2]*0.5)
        if radius_guess is None:
            radius_guess = min(np.abs(center_guess)) * 0.5

        init_ls = ms.circle_level_set(img.shape, center_guess, radius_guess)

    # Callback for visual plotting
    callback = save_iter_callback(plot_each=plot_each, outdir=outdir)

    # Morphological Chan-Vese (or ACWE)
    ls = ms.morphological_chan_vese(img, iterations=iterations,
                                    init_level_set=init_ls,
                                    smoothing=smoothing, lambda1=lambda1, lambda2=lambda2, nu=nu,
                                    post_smoothing=post_smoothing, post_nu=post_nu,
                                    iter_callback=callback, exit_thres=exit_thres)
    return ls


if __name__ == '__main__':
    import mesh
    import argparse
    import glob

    """
    Show example of finding mesh from raw image and initial guess by outputting every N steps as a (possibly 
    smoothed) mesh. 
    Note: outputdir is a directory for outputting many meshes, while input is a single input h5 file.
    Note: -n is how often to output a mesh, in iterations. -n0 is how many iterations to perform.
    

    Example usage
    python demo_morphsnakes_steps.py \
        -o /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48YGal4UASCAAXmCh/ \
        -i /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48YGal4UASCAAXmCh/Time_000001_c1_Probabilities.h5 \
        -ofn_ply mesh_visualization_ms \
        -ofn_ls msls_visualization_ -l1 1 -l2 1 -nu 2 -smooth 1 -postsmooth 1 -postnu 2 -n 1 -n0 100 -exit 5e-6


    """
    parser = argparse.ArgumentParser(description='Compute level set using Morphological Chan-Vese ACWE.')

    # Procedure options
    parser.add_argument('-dataset', '--dataset', help='Turn hdf5 dataset sequence into level sets', action='store_true')
    parser.add_argument('-sweep', '--parameter_sweep', help='Sweep over nu and smooth to compare', action='store_true')

    # Parameter options
    parser.add_argument('-i', '--input', help='Path to folder or file to read hdf5 probabilities' +
                                              'If dataset == True, then this is dir, else is filename.',
                        type=str, default='./')
    parser.add_argument('-clip', '--clip', help='Maximum intensity value for the image before evolution, if positive',
                        type=float, default=-1)
    parser.add_argument('-channel', '--channel', help='Which channel of the loaded data/image to use',
                        type=int, default=0)
    parser.add_argument('-o', '--outputdir', help='Path to folder to which to write meshes',
                        type=str, default='./morphsnakes_output/')
    parser.add_argument('-ofn_ply', '--outputfn_ply', help='Name of file in output dir to write ply',
                        type=str, default='mesh_apical_ms_')
    parser.add_argument('-ofn_ls', '--outputfn_ls', help='Name of file in output dir to write level set as numpy array',
                        type=str, default='msls_apical_')
    parser.add_argument('-init_ls', '--init_ls_fn', help='Path to numpy file to use as initial level set, if any',
                        type=str, default='empty_string')
    parser.add_argument('-prenu', '--init_ls_nu',
                        help='Number of dilation (nu > 0) or erosion (nu < 0) passes before passing init_ls to MCV',
                        type=int, default=-8)
    parser.add_argument('-presmooth', '--pre_smoothing',
                        help='Number of smoothing passes before passing init_ls to MCV', type=int, default=0)
    parser.add_argument('-l1', '--lambda1',
                        help='Weight parameter for the outer region. If `lambda1` is larger than `lambda2`, the outer '
                             'region will contain a larger range of values than the inner region',
                        type=float, default=1)
    parser.add_argument('-l2', '--lambda2',
                        help='Weight parameter for the inner region. If `lambda2` is larger than '
                             '`lambda1`, the inner region will contain a larger range of values than'
                             'the outer region.', type=float, default=2)
    parser.add_argument('-nu', '--nu',
                        help='If not None and nonzero, applies pressure to the surface. If negative,' +
                             'applies negative pressure at each iteration. int(nu) is the number of' +
                             'times to apply a dilation or erosion at each timestep', type=float, default=0.1)
    parser.add_argument('-smooth', '--smoothing', help='Number of smoothing passes per iteration',
                        type=float, default=1)
    parser.add_argument('-postnu', '--post_nu',
                        help='Number of dilation (nu > 0) or erosion (nu < 0) passes after iterations completed',
                        type=int, default=5)
    parser.add_argument('-postsmooth', '--post_smoothing', help='Number of smoothing passes after iterations completed',
                        type=int, default=5)
    parser.add_argument('-exit', '--exit_thres', help='Number of smoothing passes per iteration', type=float,
                        default=5e-6)
    parser.add_argument('-n', '--niters', help='Number of iterations per timepoint', type=int, default=26)
    parser.add_argument('-n0', '--niters0', help='Number of iterations for the first timepoint', type=int, default=76)
    parser.add_argument('-rad0', '--radius_guess',
                        help='If positive, specifies the radius of the initial implicit level set guess',
                        type=float, default=-1)
    parser.add_argument('-center_guess', '--center_guess',
                        help='If not empty_string, specifies the center of the initial level set guess.'
                             'The delimiter between each positional value is a comma',
                        type=str, default="empty_string")
    parser.add_argument('-ss', '--subsampling_factor', help='Factor to multiply the coordinates of the extracted ls',
                        type=int, default=4)

    # IO options
    parser.add_argument('-save', '--save_callback', help='Save images of ls meshes during MS', action='store_true')
    parser.add_argument('-show', '--show_callback', help='Display images of ls meshes during MS', action='store_true')
    parser.add_argument('-plot_mesh3d', '--plot_mesh3d', help='Plot the evolving 3d mesh', action='store_true')
    parser.add_argument('-dtype', '--saved_datatype', help='Filetype for output implicit level sets',
                        type=str, default='h5')
    parser.add_argument('-dset_name', '--dset_name', help='Name of dataset to load from hdf5 input file on which to run',
                        type=str, default='exported_data')
    parser.add_argument('-permute', '--permute_axes', help='Axes order of training data (xyzc, cxyz, cyxz, etc)',
                        type=str, default='xyzc')
    parser.add_argument('-invert', '--invert_probabilities', help='Axes order of training data (xyzc, cxyz, cyxz, etc)',
                        action='store_true')
    parser.add_argument('-hide_ticks', '--hide_check_axis_ticks',
                        help='Show the axis labels (numbers, ticks) for the check images',
                        action='store_true')
    parser.add_argument('-prob', '--probabilities_search_string', help='Seek this file name for probabilities.',
                        type=str, default='stab_Probabilities.h5')
    parser.add_argument('-mask', '--mask_filename', help='Seek this file name for masking the probabilities.',
                        type=str, default='empty_string')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    """Run morphological snakes on a single image to create implicit surface. Output a mesh every so often, which may
    be smoothed at each output step for visualization purposes.
    Example usage: 

    python run_morphsnakes.py \
        -o /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/morphsnakes_testing/test_out.ply \
        -ols /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/morphsnakes_testing/test_out.npy \
        -i /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/Time_000001_c1_Probabilities.h5 \
        -rootdir /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/

     python run_morphsnakes.py -i /Users/npmitchell/Dropbox/Soft_Matter/UCSB/qbio-vip8_shared/tolls/TP0_Ch3_Ill0_Ang0,45,90,135,180,225,270,315.h5
        -o /Users/npmitchell/Dropbox/Soft_Matter/UCSB/qbio-vip8_shared/tolls/msls_output_nu0p10_s1_pn4_ps4_l1_l1/
        -prenu 0 -presmooth 0 -ofn_ply mesh_apical_ms_000000.ply -ofn_ls msls_apical_000000.npy -l1 1 -nu 0.1
        -postnu -2 -channel -1 -smooth 1 -postsmooth 4 -exit 0.000001000 -dset_name inputData -rad0 30 -n 35 -save
        -dtype h5 -init_ls /Users/npmitchell/Dropbox/Soft_Matter/UCSB/qbio-vip8_shared/tolls/Time_000000_c3_levelset.h5 -l2 2 -clip 500
    """
    fn = args.input
    outputdir = os.path.join(args.outputdir, '')
    outfn_ply = outputdir + args.outputfn_ply
    if outfn_ply[-4:] != '.ply':
        outfn_ply += '.ply'
    outfn_ls = outputdir + args.outputfn_ls

    imdir = outputdir + 'morphsnakes_check/'

    if args.init_ls_fn == 'empty_string':
        load_init = False
    elif os.path.exists(args.init_ls_fn) and os.path.isfile(args.init_ls_fn):
        load_init = True
    else:
        load_init = False

    if not load_init:
        init_ls = None

        if args.radius_guess > 0:
            radius_guess = args.radius_guess
        else:
            radius_guess = None

        if not args.center_guess == 'empty_string':
            if ',' in args.center_guess:
                center_guess = tuple(float(value) for value in args.center_guess.split(','))
            else:
                center_guess = None
        else:
            center_guess = None
    else:
        # Load the initial level set
        if args.init_ls_fn[-3:] == 'npy':
            init_ls = np.load(args.init_ls_fn)
        elif args.init_ls_fn[-3:] in ['.h5', 'df5']:
            f = h5py.File(args.init_ls_fn, 'r')
            init_ls = f['implicit_levelset'][:]
            f.close()

        radius_guess = None
        center_guess = None

        # Erode the init_ls several times to avoid spilling out of ROI on next round
        for _ in range(abs(args.init_ls_nu)):
            if args.init_ls_nu > 0:
                init_ls = ndi.binary_dilation(init_ls)
            else:
                init_ls = ndi.binary_erosion(init_ls)

        for _ in range(args.pre_smoothing):
            init_ls = _curvop(init_ls)

    if args.channel < 0:
        channel = None
    else:
        channel = args.channel

    if args.clip > 0:
        clip = args.clip
    else:
        clip = None

    ls = extract_levelset_step_by_step(fn, iterations=args.niters, channel=channel, init_ls=init_ls,
                                      smoothing=args.smoothing, lambda1=args.lambda1, lambda2=args.lambda2,
                                      nu=args.nu, post_smoothing=args.post_smoothing, post_nu=args.post_nu,
                                      exit_thres=args.exit_thres, dset_name=args.dset_name,
                                      impath=imdir, plot_each=10, save_callback=args.save_callback,
                                      show_callback=args.show_callback, axis_order=args.permute_axes,
                                      comparison_mesh=None, radius_guess=radius_guess, center_guess=center_guess,
                                      plot_mesh3d=args.plot_mesh3d, clip=clip, labelcheckax=not args.hide_check_axis_ticks)
    print('Extracted level set')

    # Extract edges of level set
    coords, triangles = mcubes.marching_cubes(ls, 0.5)
    mm = mesh.Mesh()
    mm.points = coords
    mm.triangles = triangles
    print('saving ', outfn_ply)
    mm.save(outfn_ply)

    # Save ls for this timepoint as npy or hdf5 file
    if args.saved_datatype == 'npy':
        # Save ls for this timepoint as npy file
        if outfn_ls[-4:] != '.npy':
            outfn_ls += '.npy'

        print('saving ', outfn_ls)
        np.save(outfn_ls, ls)
    elif args.saved_datatype in ['h5', 'hdf5']:
        # Save ls for this timepoint as an hdf5 file
        if outfn_ls[-3:] != '.h5' and outfn_ls[-5:] != '.hdf5':
            outfn_ls += outfn_ls + '.h5'

        print('saving ', outfn_ls)
        msaux.save_ls_as_h5(outfn_ls, ls)
    else:
        print('skipping save of implicit ls...')

logging.info("Done.")