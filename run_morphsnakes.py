from __future__ import print_function
import os
import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import morphsnakes as ms
from morphsnakes import _curvop
import copy
import mcubes
import morphsnakes_aux_fns as msaux
from scipy import ndimage as ndi
import h5py

'''Extract levelset from 2d or volumetric data. This is a wrapper script for using contained in the morphsnakes module.


copyright via MIT license
NPMitchell 2019 npmitchell@kitp.ucsb.edu
'''


# in case you are running on machine without display, e.g. server
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

PATH_IMG_NODULE = 'images/mama07ORI.bmp'
PATH_IMG_STARFISH = 'images/seastar2.png'
PATH_IMG_LAKES = 'images/lakes3.jpg'
PATH_IMG_CAMERA = 'images/camera.png'
PATH_IMG_COINS = 'images/coins.png'
PATH_ARRAY_CONFOCAL = 'images/confocal.npy'


def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.

    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.

    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.

    """

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback


def visual_callback_3d(fig=None, plot_each=1, show=False, save=True, impath=None, rmimages=True,
                       comparison_mesh=None, fig2=None,
                       img=None, axis_order='yxz', alpha=0.5, compare_mesh_slices=False, sz=5,
                       thres=0.5, plot_diff=False, plot_mesh3d=False, labelcheckax=False):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 3D images.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    plot_each : positive integer
        The plot will be updated once every `plot_each` calls to the callback
        function.
    show : bool
        make the figure appear to show intermediate results
    save : bool
        save the intermediate results to disk
    impath : None or string
        path to the directory for saving images
    rmimages : bool
        Remove the images from disk after making them
    comparison_mesh : mesh or None
        A mesh to compare the intermediate results to, if desired
    fig2 : matplotlib.figure.Figure instance
        Second figure where results will be drawn. If not given, a new figure will be created if a mesh comparison is
        desired.
    plot_diff : bool
        plot the difference between current ls and previous


    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.

    """
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure(1)
    else:
        fig.clf()

    if fig2 is None:
        fig2 = plt.figure(2)
    else:
        fig2.clf()

    try:
        ax = fig.add_subplot(111, projection='3d')
    except ValueError:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')

    if img is not None:
        # Also show cross sections of the volume
        ax2 = fig2.add_subplot(131)
        ax3 = fig2.add_subplot(132)
        ax4 = fig2.add_subplot(133)
        nx = int(np.shape(img)[0] * 0.5)
        ny = int(np.shape(img)[1] * 0.5)
        nz = int(np.shape(img)[2] * 0.5)
        xslice = img[nx, :, :]
        yslice = img[:, ny, :]
        zslice = img[:, :, nz]

    if plot_diff:
        fig3 = plt.figure(3)
        fig3.clf()
        # Also show cross sections of the difference in ls
        ax5 = fig3.add_subplot(131)
        ax6 = fig3.add_subplot(132)
        ax7 = fig3.add_subplot(133)
        # Get indices if not already done (will have been done if img is not None)
        if img is None:
            nx = int(np.shape(img)[0] * 0.5)
            ny = int(np.shape(img)[1] * 0.5)
            nz = int(np.shape(img)[2] * 0.5)

    counter = [-1]
    if show or save:
        def callback(levelset, ls_prev=None, force=False):
            """

            Parameters
            ----------
            levelset
            ls_prev
            force : bool
                force callback to run without returning prematurely
            plot_mesh3d : bool

            Returns
            -------

            """
            counter[0] += 1
            if (counter[0] % plot_each) != 0 and not force:
                return

            if ax.collections:
                del ax.collections[0]

            coords, triangles = mcubes.marching_cubes(levelset, 0.5)

            ################################################################################################
            # Plot the level set mesh in 3d space
            if plot_mesh3d:
                # Plot the level set
                ax.set_aspect('equal')
                ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],  # change axes!!!!
                                triangles=triangles, alpha=alpha)

                if comparison_mesh is not None:
                    mmm = comparison_mesh
                    ax.plot_trisurf(mmm.points[:, 1], mmm.points[:, 0], mmm.points[:, 2],
                                    triangles=mm.triangles, alpha=0.3)
                    ax.set_xlabel(r'x [$\mu$m]')
                    ax.set_ylabel(r'y [$\mu$m]')
                    ax.set_zlabel(r'z [$\mu$m]')

                title = 'Morphological Chan-Vese level set'
                title += '\n' + r'$t=$' + '{0:d}'.format(counter[0])
                ax.set_title(title)

                if save:
                    if impath is not None and counter[0] == 0:
                        # ensure the directory exists
                        d = os.path.dirname(impath)
                        if not os.path.exists(d):
                            print('run_morphsnakes.py: creating dir: ', d)
                            os.makedirs(d)

                    # set axes equal
                    limits = np.array([
                        ax.get_xlim3d(),
                        ax.get_ylim3d(),
                        ax.get_zlim3d(),
                    ])
                    origin = np.mean(limits, axis=1)
                    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
                    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
                    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
                    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

                    # ax.view_init(0, 30)
                    # imfn = impath + '{0:06d}'.format(counter[0]) + '.png'
                    # print 'saving ', imfn
                    # fig.savefig(imfn)
                    ax.view_init(0, 0)
                    imfn = impath + 'viewx_{0:06d}'.format(counter[0]) + '.png'
                    print('saving ', imfn)
                    fig.savefig(imfn)
                    ax.view_init(0, 90)
                    imfn = impath + 'viewy_{0:06d}'.format(counter[0]) + '.png'
                    print('saving ', imfn)
                    fig.savefig(imfn)
                    ax.view_init(-90, 90)
                    imfn = impath + 'viewz_{0:06d}'.format(counter[0]) + '.png'
                    print('saving ', imfn)
                    fig.savefig(imfn)

                    # plt.show()
                    # plt.close('all')
                    ax.cla()
                    # raise RuntimeError
                elif show:
                    plt.pause(0.1)

            ################################################################################################
            # Show 2d slices of the 3D volume
            if img is not None:
                ax2.imshow(xslice.T)
                ax3.imshow(yslice.T)
                ax4.imshow(zslice.T)

                # Show the comparison mesh boundary
                ax2.set_aspect('equal')
                ax3.set_aspect('equal')
                ax4.set_aspect('equal')
                if compare_mesh_slices:
                    mbxpts = np.where(np.abs(mm.points[:, 0] - nx) < thres)[0]
                    mbypts = np.where(np.abs(mm.points[:, 1] - ny) < thres)[0]
                    mbzpts = np.where(np.abs(mm.points[:, 2] - nz) < thres)[0]
                    ax2.scatter(mm.points[mbxpts, 1], mm.points[mbxpts, 2], s=sz, alpha=alpha)
                    ax3.scatter(mm.points[mbypts, 0], mm.points[mbypts, 2], s=sz, alpha=alpha)
                    ax4.scatter(mm.points[mbzpts, 0], mm.points[mbzpts, 1], s=sz, alpha=alpha)

                # Show the level set
                lsxpts = np.where(np.abs(coords[:, 0] - nx) < thres)[0]
                lsypts = np.where(np.abs(coords[:, 1] - ny) < thres)[0]
                lszpts = np.where(np.abs(coords[:, 2] - nz) < thres)[0]
                ax2.scatter(coords[lsxpts, 1], coords[lsxpts, 2], s=sz, alpha=alpha)
                ax3.scatter(coords[lsypts, 0], coords[lsypts, 2], s=sz, alpha=alpha)
                ax4.scatter(coords[lszpts, 0], coords[lszpts, 1], s=sz, alpha=alpha)
                ax2.set_xlabel('y')
                ax3.set_xlabel('x')
                ax4.set_xlabel('x')
                ax2.set_ylabel('z')
                ax3.set_ylabel('z')
                ax4.set_ylabel('y')
                # Remove tick labels if labelcheckax == False
                if not labelcheckax:
                    for axx in [ax2, ax3, ax4]:
                        axx.xaxis.set_ticks([])
                        axx.yaxis.set_ticks([])

                title = 'Morphological Chan-Vese level set'
                title += '\n' + r'$t=$' + '{0:d}'.format(counter[0])
                ax3.text(0.5, 0.9, title, va='center', ha='center', transform=fig2.transFigure)

                if save:
                    print('impath = ', impath)
                    if impath is not None and counter[0] == 0:
                        # ensure the directory exists
                        d = os.path.dirname(impath)
                        if not os.path.exists(d):
                            print('run_morphsnakes.py: creating dir: ', d)
                            os.makedirs(d)

                    ax2.set_aspect('equal')
                    ax3.set_aspect('equal')
                    imfn = impath + 'slices_{0:06d}'.format(counter[0]) + '.png'
                    print('saving ', imfn)
                    fig2.savefig(imfn, dpi=250)
                    # plt.show()
                    # plt.close('all')
                    ax2.cla()
                    ax3.cla()
                    ax4.cla()
                    # raise RuntimeError
                elif show:
                    plt.pause(0.1)

            ################################################################################################
            # Plot difference in ls from previous timepoint
            if plot_diff and ls_prev is not None:
                lsxslice = levelset[nx, :, :]
                lsyslice = levelset[:, ny, :]
                lszslice = levelset[:, :, nz]
                lspx = ls_prev[nx, :, :]
                lspy = ls_prev[:, ny, :]
                lspz = ls_prev[:, :, nz]
                ax5.imshow(lsxslice - lspx)
                ax6.imshow(lsyslice - lspy)
                ax7.imshow(lszslice - lspz)

                # Show the comparison mesh boundary
                for axx in [ax5, ax6, ax7]:
                    axx.set_aspect('equal')
                    if not labelcheckax:
                        axx.xaxis.set_ticks([])
                        axx.yaxis.set_ticks([])

                # Show the level set
                ax5.set_xlabel('y')
                ax6.set_xlabel('x')
                ax7.set_xlabel('x')
                ax5.set_ylabel('z')
                ax6.set_ylabel('z')
                ax7.set_ylabel('y')

                title = r'Morphological Chan-Vese level set: $\Delta u$'
                title += '\n' + r'$t=$' + '{0:d}'.format(counter[0])
                ax5.text(0.5, 0.9, title, va='center', ha='center', transform=fig3.transFigure)

                for axx in [ax5, ax6, ax7]:
                    axx.set_aspect('equal')

                if save:
                    print('impath = ', impath)
                    if impath is not None and counter[0] == 0:
                        # ensure the directory exists
                        d = os.path.dirname(impath)
                        if not os.path.exists(d):
                            print('run_morphsnakes.py: creating dir: ', d)
                            os.makedirs(d)

                    imfn = impath + 'diff_slices_{0:06d}'.format(counter[0]) + '.png'
                    print('saving ', imfn)
                    fig3.savefig(imfn)
                    for axx in [ax5, ax6, ax7]:
                        axx.cla()

                elif show:
                    plt.pause(0.1)

    else:
        print('Images will be neither saved nor shown, so passing empty defition for callback.')

        def callback(*args, **kwargs):
            counter[0] += 1
            if (counter[0] % plot_each) != 0:
                return
            else:
                print('counter = ' + str(counter[0]), end='\r')

    return callback


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def load_img(fn, channel, dset_name='exported_data', axis_order='xyzc'):
    """Load a 2d or 3d grid of intensities from disk

    Parameters
    ----------
    fn
    channel
    dset_name

    Returns
    -------
    img : numpy float array
        intensity values in 2d or 3d grid (dataset)
    """
    print('loading ' + fn)
    if fn[-3:] == 'npy':
        if channel is not None:
            img = np.load(fn)[:, :, :, channel]

    elif fn[-2:] == 'h5':
        # filename = file_architecture.os_i(filename)
        if os.path.exists(fn):
            hfn = h5py.File(fn, 'r')
        else:
            print("File " + fn + " does not exist")
            hfn = h5py.File(fn, 'w')

        # ilastik internally swaps axes. 1: class, 2: y, 3: x 4 : z
        # so flip the axes to select channel, y, x, z
        if channel is None and len(np.shape(hfn[dset_name])) == 4:
            print('4D data but no channel specified, assuming channel is 1...')
            channel = 1

        if channel is not None:
            if len(axis_order) == 3:
                img = hfn[dset_name][:, :, :]
            elif axis_order[3] == 'c':
                axis_order = axis_order[0:3]
                img = hfn[dset_name][:, :, :, channel]
            elif axis_order[0] == 'c':
                axis_order = axis_order[1:]
                img = hfn[dset_name][channel, :, :, :]
            elif axis_order[1] == 'c':
                okax = np.array([0, 2, 3])
                axis_order = axis_order[okax]
                img = hfn[dset_name][:, channel, :, :]
            elif axis_order[2] == 'c':
                okax = np.array([0, 1, 3])
                axis_order = axis_order[okax]
                img = hfn[dset_name][:, :, channel, :]
            else:
                raise RuntimeError("Cannot parse this axis order")
        else:
            img = np.array(hfn[dset_name]) 
            img = img.astype(float)

        if axis_order == 'xyz':
            pass
        elif axis_order == 'xzy':
            img = np.swapaxes(img, 1, 2)
        elif axis_order == 'yzx':
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2)
        elif axis_order == 'yxz':
            img = np.swapaxes(img, 0, 1)
        elif axis_order == 'zyx':
            img = np.swapaxes(img, 0, 2)
        elif axis_order == 'zxy':
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
        else:
            raise RuntimeError("Did not recognize axis order:" + axis_order)

        # plt.hist(img.ravel())
        # print(np.max(img.ravel()))
        # plt.savefig('/Users/npmitchell/Desktop/test.png')
        # raise RuntimeError('Exiting now')
        img = copy.deepcopy(img)
        # file_is_open = True
        hfn.close()
    else:
        raise RuntimeError('Could not find filename for img: ' + fn)

    return img


def extract_levelset(fn, iterations=150, smoothing=0, lambda1=1, lambda2=1, nu=None, post_smoothing=1, post_nu=1,
                     channel=0, init_ls=None, show_callback=False, save_callback=False, exit_thres=5e-6,
                     center_guess=None, radius_guess=None, impath=None, dset_name='exported_data',
                     plot_each=5, comparison_mesh=None, axis_order='xyzc', plot_diff=True,
                     plot_mesh3d=False, clip=None, clip_floor=None, labelcheckax=False, mask=None):
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
    comparison_mesh : mesh.Mesh class instance or None
        If supplied, use this mesh (with attrs mesh.points and mesh.triangles) to compare to the morphsnakes output
    clip : float or None
        If not None, clip all values above this in the image on which to run morphsnakes
    clip_floor : float or None
        If not None, clip all values below this in the image on which to run morphsnakes

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

    if clip_floor is not None:
        img[img < clip_floor] = clip_floor
        img -= clip_floor

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
    callback = visual_callback_3d(show=show_callback, save=save_callback,
                                  plot_each=plot_each, impath=impath, comparison_mesh=comparison_mesh,
                                  img=img, plot_diff=plot_diff,
                                  plot_mesh3d=plot_mesh3d, labelcheckax=labelcheckax)

    # Morphological Chan-Vese (or ACWE)
    ls = ms.morphological_chan_vese(img, iterations=iterations,
                                    init_level_set=init_ls,
                                    smoothing=smoothing, lambda1=lambda1, lambda2=lambda2, nu=nu,
                                    post_smoothing=post_smoothing, post_nu=post_nu,
                                    iter_callback=callback, exit_thres=exit_thres)
    return ls


def load_data(fn, dataset_name=None):
    """Load a numpy or hdf5 file from disk

    Parameters
    ----------
    fn : str
        filename of the

    Returns
    -------
    data : numpy array
        dataset loaded from file
    """
    if args.init_ls_fn[-3:] == 'npy':
        data = np.load(args.init_ls_fn)
    elif args.init_ls_fn[-3:] in ['.h5', 'df5']:
        f = h5py.File(args.init_ls_fn, 'r')
        data = f[dataset_name][:]
        f.close()
    return data


if __name__ == '__main__':
    import mesh
    import argparse
    import glob
    """Show example of finding mesh from raw image and initial guess.
    
    Example usage
    python run_morphsnakes.py \
        -o /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/morphsnakes_testing/test_out.ply \
        -i /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/Time_000001_c1_Probabilities.h5 \
        -ofn_ply mesh_apical_ms \
        -ofn_ls msls_apical_ -l1 1 -l2 1 -nu 2 -smooth 1 -postsmooth 1 -postnu 2 -n 35 -n0 100 -exit 5e-6
        
    python /mnt/data/code/gut_python/run_morphsnakes.py \
        -dataset -o ./ -i ./ \
        -ofn_ply mesh_apical_ms_ \
        -ofn_ls msls_apical_ -l1 1 -l2 1 -nu 2 -smooth 1 -postnu 2 -postsmooth 1  -n 35 -n0 100 -exit 1e-6
        
    
    python run_morphsnakes.py \
        -dataset  -o /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/ \
        -i /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/ \
        -ofn_ply mesh_apical_ms_ \
        -ofn_ls msls_apical_ -l1 1 -l2 1 -nu 2 -smooth 1 -postnu 2 -postsmooth 1  -n 35 -n0 100 -exit 1e-6
    
    python run_morphsnakes.py -dataset \
        -o /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/ \
        -i /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/ \
        -ofn_ply mesh_apical_ms_ \
        -ofn_ls msls_apical_ -l1 1 -l2 1 -nu 2 -smooth 1 -postsmooth 5 -postnu 5 -n 26 -n0 76 -exit 5e-6
        
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
    parser.add_argument('-clip_floor', '--clip_floor',
                        help='Minimum intensity value for the image before evolution, if positive',
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
    parser.add_argument('-include_boundary_faces', '--include_boundary_faces',
                        help='Do not remove boundary faces from the mesh representation of level set',
                        action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    if args.dataset:
        """    
        python run_morphsnakes.py -dataset \
        -o /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/ \
        -i /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4-UAShistRFP/201901021550_folded_2part/ \
        -ofn_ply mesh_apical_ms_ \
        -ofn_ls msls_apical_ -l1 1 -l2 1 -nu 2 -smooth 1 -postsmooth 5 -postnu 5 -n 26 -n0 76 -exit 5e-6
        
        python run_morphsnakes.py -dataset \
        -o /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4UasCAAXmCherry/48Ygal4UasCAAXmCherry_20190207200_excellent/cells_h5/ \
        -i /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4UasCAAXmCherry/48Ygal4UasCAAXmCherry_20190207200_excellent/cells_h5/  \
        -ofn_ply mesh_apical_ms_ -rad0 10 -save -prenu -5 -presmooth 1 \
        -ofn_ls msls_apical_ -l1 1 -l2 1 -nu 0 -smooth 0.1 -postsmooth 4 -postnu 4 -n 26 -n0 126 -exit 5e-4 -prob *Probabilities_cells.h5 \
        -init_ls /Users/npmitchell/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/data/48Ygal4UasCAAXmCherry/48Ygal4UasCAAXmCherry_20190207200_excellent/cells_h5/msls_apical_init.npy
        
        """
        indir = args.input
        outdir = args.outputdir
        # Run over a dataset of hdf5 files
        todo = sorted(glob.glob(indir + '*' + args.probabilities_search_string))
        print('todo =', todo)
        for (fn, kk) in zip(todo, range(len(todo))):
            timepoint = fn.split('/')[-1].split('_c')[0].split('Time_')[-1]
            outdir_k = outdir + 'morphsnakes_check_' + timepoint + '/'

            # Ensure that the directory exists
            if args.save_callback:
                d = os.path.dirname(outdir_k)
                if not os.path.exists(d):
                    print('run_morphsnakes.py: creating dir: ', d)
                    os.makedirs(d)

            outfn_ply = outdir + args.outputfn_ply + timepoint + '.ply'
            olsfn = outdir

            # Load/define levelset if this is the first timestep
            if kk == 0:
                # Get initial guess by levelset analysis with different number of iterations
                niters = args.niters0
                # Load init_ls if path is supplied
                if args.init_ls_fn == 'empty_string':
                    init_ls = None
                    # Start with sphere in the middle of image. Obtain the radius of the sphere
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
                    # The initial level set filename was supplied. Figure out what file type it is
                    if args.init_ls_fn[-3:] == 'npy':
                        init_ls = np.load(args.init_ls_fn)
                    elif args.init_ls_fn[-3:] in ['.h5', 'df5']:
                        f = h5py.File(args.init_ls_fn, 'r')
                        init_ls = f['implicit_levelset'][:]
                        f.close()

                    # Since there is an initial set, don't use the default spherical guess
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
            else:
                niters = args.niters

            # Clip the image if the parameter clip was given as positive
            if args.clip > 0:
                clip = args.clip
            else:
                clip = None

            # Clip the image if the parameter clip_floor was given as positive
            if args.clip_floor > 0:
                clip_floor = args.clip_floor
            else:
                clip_floor = None

            # mask the data if mask filename is given
            if '.' in args.mask_filename:
                print('Since . appears in mask_filename, assuming full filename with path is given for mask file...')
                maskfn = args.mask_filename
                if os.path.exists(maskfn):
                    raise RuntimeError('Mask filename does not exist! Sought file ' + maskfn)
            elif args.mask_filename is not 'none' and args.mask_filename is not 'empty_string':
                print('Since . appears in mask_filename, assuming full filename with path is given for mask file...')
                maskfn = args.mask_filename + timepoint + args.dtype
                if os.path.exists(maskfn):
                    raise RuntimeError('Mask filename does not exist! Sought file ' + maskfn)
            else:
                maskfn = None
                mask = None

            if maskfn is not None:
                # load the mask
                mask = load_data(maskfn)

            # Perform the levelset calculation
            ls = extract_levelset(fn, iterations=niters, channel=args.channel, init_ls=init_ls,
                                  smoothing=args.smoothing, lambda1=args.lambda1, lambda2=args.lambda2,
                                  nu=args.nu, post_smoothing=args.post_smoothing, post_nu=args.post_nu,
                                  exit_thres=args.exit_thres, dset_name=args.dset_name,
                                  impath=outdir_k, plot_each=10, save_callback=args.save_callback,
                                  show_callback=args.show_callback, axis_order=args.permute_axes,
                                  plot_mesh3d=args.plot_mesh3d, mask=mask,
                                  comparison_mesh=None, radius_guess=radius_guess,
                                  center_guess=center_guess, clip=clip, clip_floor=clip_floor,
                                  labelcheckax=not args.hide_check_axis_ticks)

            # Extract edges of level set and store them in a mesh
            mm = mesh.Mesh()

            # If desired, avoid chopping the boundaries by converting all boundary pixels to zero
            if args.include_boundary_faces:
                # expand ls into padded zeros
                ls2 = np.zeros(np.shape(ls) + np.array([2, 2, 2]))
                ls2[1:-1, 1:-1, 1:-1] = ls
                coords, triangles = mcubes.marching_cubes(ls2, 0.5)
                coords -= np.array([1, 1, 1])
            else:
                coords, triangles = mcubes.marching_cubes(ls, 0.5)

            mm.points = coords * float(args.subsampling_factor)
            mm.triangles = triangles
            print('saving ', outfn_ply)
            mm.save(outfn_ply)

            # Save the level set data as a numpy or h5 file
            if args.saved_datatype == 'npy':
                # Save ls for this timepoint as npy file
                outfn_ls = outdir + args.outputfn_ls + timepoint + '.npy'
                print('saving ', outfn_ls)
                np.save(outfn_ls, ls)
            elif args.saved_datatype in ['h5', 'hdf5']:
                # Save ls for this timepoint as an hdf5 file
                outfn_ls = outdir + args.outputfn_ls + timepoint + '.h5'
                print('saving ', outfn_ls)
                msaux.save_ls_as_h5(outfn_ls, ls)
            else:
                print('skipping save of implicit ls...')

            # Make current level set into next iteration's guess
            init_ls = ls

            # Erode the init_ls several times to avoid spilling out of ROI on next round
            for _ in range(abs(args.init_ls_nu)):
                if args.init_ls_nu > 0:
                    init_ls = ndi.binary_dilation(init_ls)
                else:
                    init_ls = ndi.binary_erosion(init_ls)

    elif args.parameter_sweep:
        """
        Example usage 
        python ~/Dropbox/Soft_Matter/UCSB/gut_morphogenesis/gut_python/gut_python/run_morphsnakes.py -sweep -init_ls ./msls_apical_nu0p1_s1_pn1_ps1_l1_l2_000080.npy \
            -l1 1. -l2 2. -postnu 1 -postsmooth 1 \
            -o ./  -i ./Time_000090_c1_Probabilities.h5 \
            -ofn_ply mesh_apical_ms_000090 \
            -ofn_ls mesh_apical_ms_000090 -save
        """
        fn = args.input
        outputdir = os.path.join(args.outputdir, '')
        imdir = outputdir + 'morphsnakes_check/'
        ofn_ply_base = args.outputfn_ply
        ofn_ls_base = args.outputfn_ls
        nus = np.arange(0.0, 0.6, 0.1)[::-1]
        smooths = np.arange(3)

        if args.init_ls_fn == 'empty_string':
            init_ls = None
        else:
            init_ls = np.load(args.init_ls_fn)
            # Erode the init_ls several times to avoid spilling out of ROI on next round
            for _ in range(abs(args.init_ls_nu)):
                if args.init_ls_nu > 0:
                    init_ls = ndi.binary_dilation(init_ls)
                else:
                    init_ls = ndi.binary_erosion(init_ls)

        # check that it has loaded
        # print('init_ls = ', init_ls)

        for nu in nus:
            for smooth in smooths:
                name = '_nu{0:0.2f}'.format(nu).replace('.', 'p') + '_s{0:02d}'.format(smooth)
                name += '_pnu{0:02d}'.format(args.post_nu) + '_ps{0:02d}'.format(args.post_smoothing)
                name += '_l{0:0.2f}'.format(args.lambda1).replace('.', 'p')
                name += '_l{0:0.2f}'.format(args.lambda2).replace('.', 'p')

                ofn_ply = outputdir + ofn_ply_base + name + '.ply'
                ofn_ls = outputdir + ofn_ls_base + name

                ls = extract_levelset(fn, iterations=args.niters, channel=args.channel, init_ls=init_ls,
                                      smoothing=smooth, lambda1=args.lambda1, lambda2=args.lambda2,
                                      nu=nu, post_smoothing=args.post_smoothing, post_nu=args.post_nu,
                                      exit_thres=args.exit_thres,
                                      impath=imdir, plot_each=10, save_callback=args.save_callback,
                                      show_callback=args.show_callback,
                                      comparison_mesh=None, plot_diff=True)

                # Extract edges of level set
                coords, triangles = mcubes.marching_cubes(ls, 0.5)
                mm = mesh.Mesh()
                mm.points = coords
                mm.triangles = triangles
                print('saving ', ofn_ply)
                mm.save(ofn_ply)

                # Overwrite the extension if specified in the outputfn
                if ofn_ls[-4:] == '.npy':
                    args.saved_datatype = 'npy'
                    outfn_ls = ofn_ls + '.npy'
                elif ofn_ls[-3:] == '.h5' or ofn_ls[-5:] == '.hdf5':
                    args.saved_datatype = 'h5'
                    outfn_ls = ofn_ls + '.' + ofn_ls.split('.')[-1]

                # Now save it
                if args.saved_datatype == 'npy':
                    # Save ls for this timepoint as npy file
                    print('saving ', outfn_ls)
                    np.save(outfn_ls, ls)
                elif args.saved_datatype in ['h5', 'hdf5']:
                    # Save ls for this timepoint as an hdf5 file
                    print('saving ', outfn_ls)
                    msaux.save_ls_as_h5(outfn_ls, ls)
                else:
                    print('skipping save of implicit ls...')

                # Save the result as an image
                img = load_img(fn, args.channel, dset_name=args.dset_name)
                plt.close('all')
                msaux.plot_levelset_result(ls, img, name='ms' + name, imdir='./param_sweep/', fig=None, fig2=None,
                                           ax=None, title=None, comparison_mesh=None, save=True, show=False)
                plt.close('all')

    else:
        """Run morphological snakes on a single image to create implicit surface.
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

        # Clip the image if the parameter clip_floor was given as positive
        if args.clip_floor > 0:
            clip_floor = args.clip_floor
        else:
            clip_floor = None

        ls = extract_levelset(fn, iterations=args.niters, channel=channel, init_ls=init_ls,
                              smoothing=args.smoothing, lambda1=args.lambda1, lambda2=args.lambda2,
                              nu=args.nu, post_smoothing=args.post_smoothing, post_nu=args.post_nu,
                              exit_thres=args.exit_thres, dset_name=args.dset_name,
                              impath=imdir, plot_each=10, save_callback=args.save_callback,
                              show_callback=args.show_callback, axis_order=args.permute_axes,
                              comparison_mesh=None, radius_guess=radius_guess, center_guess=center_guess,
                              plot_mesh3d=args.plot_mesh3d, clip=clip, clip_floor=clip_floor,
                              labelcheckax=not args.hide_check_axis_ticks)
        print('Extracted level set')

        # Extract edges of level set
        mm = mesh.Mesh()
        # If desired, avoid chopping the boundaries by converting all boundary pixels to zero
        if args.include_boundary_faces:
            # expand ls into padded zeros
            ls2 = np.zeros(np.shape(ls) + np.array([2, 2, 2]))
            ls2[1:-1, 1:-1, 1:-1] = ls
            coords, triangles = mcubes.marching_cubes(ls2, 0.5)
            coords -= np.array([1, 1, 1])
        else:
            coords, triangles = mcubes.marching_cubes(ls, 0.5)

        mm.points = coords * float(args.subsampling_factor)
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
