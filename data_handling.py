import numpy as np
import matplotlib.path as mplpath
import copy
# from scipy.spatial import Voronoi
from scipy.spatial import cKDTree

try:
    import scipy.interpolate as interpolate
except:
    print('Could not import scipy.interpolate!')

'''
Description
===========
General functions for handling data, such as rotating arrays/tensors, converting classes into dictionaries,
finding unique rows of arrays, sorting, finding peaks in data, getting statistics of columns on 2d arrays,
smootihing, finding points inside polygons, etc
'''

# could put stuff in ilpm.networks here...


def first_true(arr):
    """Return the index of the first true element for each row in array arr.
    Warning: if all elements of a row of arr are zero, then this function returns a value of -1 for that index.
    Note that a workaround for this issue is to make the output a dictionary that would allow a value for None.

    Parameters
    ----------
    arr : 2d numpy array
        the array for which to identify nonzero elements

    Returns
    -------
    di : len(arr) x 1 int array
        the index where the first true element exists
        If all elements of the row of arr are zero, returns nan for that index

    Example Usage
    -------------
    arr = np.zeros((4, 5))
    arr[0, 2:] = 1.
    arr[2, 0:] = 1.
    arr[3, 3:] = 1.
    print first_true(arr)
    """
    di = np.zeros(np.shape(arr)[0], dtype=int)
    for i, ele in enumerate(np.argmax(arr, axis=1)):
        if ele == 0 and arr[i][0] == 0:
            di[i] = -1
        else:
            di[i] = ele

    return di


def argmaxn(arr, num_vals):
    """Return the indices of the largest num_vals values in arr

    Parameters
    ----------
    arr : n x 1 array
        the array for which to seek largest num_vals values
    num_vals: int
        how many big values to return

    Returns
    -------
    inds : num_vals x 1 int array
        the indices of arr where arr is largest, in decreasing order of magnitude
    """
    return arr.argsort()[-num_vals:][::-1]


def argminn(arr, num_vals):
    """Return the indices of the smallest num_vals values in arr -- ie arg_smallest_n() would be reasonable name

    Parameters
    ----------
    arr : n x 1 array
        the array for which to seek smallest num_vals values
    num_vals: int
        how many small values to return

    Returns
    -------
    inds : num_vals x 1 int array
        the indices of arr where arr is largest, in decreasing order of magnitude
    """
    return arr.argsort()[:num_vals]


def consecutive(data, stepsize=1):
    """Split data into chunks in which each chunck is increasing by stepsize.
    This allows to search for consecutive data chunks that are changing by a fixed amount.

    Parameters
    ----------
    data : 1d float or int array
        data to be split
    stepsize : float or int
        The increment that each chunck should be increasing by
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def class2dict(class_instance):
    """Put all attributes of a class into a dictionary, with their names as keys. Note that this would make a Java user
    cringe, but seems fine to do in python.

    Parameters
    ----------
    class_instance : instance of a class
        class instance for which to store all non-built-in attributes as key,val pairs in an output dictionary
    """
    dict = {}
    attrlist = [a for a in dir(class_instance) if not a.startswith('__') and not callable(getattr(class_instance, a))]
    for attr in attrlist:
        dict[attr] = class_instance.attr
    return dict


def bin_avg_minmaxstd(arr, bincol=0, tol=1e-7):
    """Get averages of data in an array, where one column denotes some number that is used to bin the rows of arr.
    Bin values of arr[:, bincol], use their digitization to group the rows of arr. Then take averages and look at
    statistics of those groups of data.

    Parameters
    ----------
    arr : numpy array
        The array of which to group elements and take statistics
    bincol : int
        the column which will be used to group entries in arr into bins
    tol : float
        The allowable difference between elements to put them into the same bin

    Returns
    -------
    binv : #unique bins x 1 float array
        The unique values of the binned column, sorted
    avgs : #unique bins x #ncols-1 float array
        The average values of all entries in each bin
    mins : #unique bins x #ncols-1 float array
        The min values of entries in each bin
    maxs : #unique bins x #ncols-1 float array
        The max values of entries in each bin
    stds : #unique bins x #ncols-1 float array
        The standard deviation of entries in each bin
    """
    binv, avgs, mins, maxs, stds, count = bin_avg_minmaxstdcount(arr, bincol=bincol, tol=tol)
    return binv, avgs, mins, maxs, stds


def bin_avg_minmaxstdcount(arr, bincol=0, tol=1e-7):
    """Get averages of data in an array, where one column denotes some number that is used to bin the rows of arr.
    Bin values of arr[:, bincol], use their digitization to group the rows of arr. Then take averages and look at
    statistics of those groups of data.

    Parameters
    ----------
    arr : numpy array
        The array of which to group elements and take statistics
    bincol : int
        the column which will be used to group entries in arr into bins
    tol : float
        The allowable difference between elements to put them into the same bin (this is the maximum possible binsize)

    Returns
    -------
    avgs : #unique bins x 1 float array
        The average values of all entries in each bin
    mins : #unique bins x #ncols-1 float array
        The min values of entries in each bin
    maxs : #unique bins x #ncols-1 float array
        The max values of entries in each bin
    stds : #unique bins x #ncols-1 float array
        The standard deviation of entries in each bin
    count : #unique bins x 1 float array
    """
    # get limits on the entries in bincolumn
    abc = arr[:, bincol]
    othercols = [x for x in range(len(arr[0, :])) if x != np.mod(bincol, len(arr[0, :]))]
    minbc = np.min(abc)
    maxbc = np.max(abc)
    # create a very small number to ensure that bin ranges enclose the values in abc
    eps = 1e-7 * np.min(np.abs(abc[np.nonzero(abc)[0]]))
    diffs = np.abs(diff_matrix(abc, abc).ravel())
    dx = np.min(diffs[np.where(diffs > tol)[0]])

    nbc = (maxbc - minbc) / dx + 2
    bins = np.linspace(minbc - eps, maxbc + eps, nbc)
    inds = np.digitize(abc, bins)

    uniq = np.unique(inds)

    # Create binv, the average value of the sorting id value in each bin
    binv = np.zeros(len(uniq))
    avgs = np.zeros((len(uniq), len(othercols)))
    mins = np.zeros((len(uniq), len(othercols)))
    maxs = np.zeros((len(uniq), len(othercols)))
    stds = np.zeros((len(uniq), len(othercols)))
    count = np.zeros(len(uniq))
    kk = 0
    for ii in uniq:
        # find which rows belong in the current bin labeled by ii
        inbin = np.where(inds == ii)[0]
        binarr = arr[inbin][:, othercols]
        avgs[kk] = np.mean(binarr, axis=0)
        mins[kk] = np.min(binarr, axis=0)
        maxs[kk] = np.max(binarr, axis=0)
        stds[kk] = np.std(binarr, axis=0)
        binv[kk] = np.mean(abc[inbin])
        count[kk] = len(inbin)
        kk += 1

    return binv, avgs, mins, maxs, stds, count


def binstats_extravariable(arr, bin0col=0, bin1col=1, tol=1e-7):
    """Get stats of data in an array, where one column denotes some number that is used to bin the rows of arr.
    Rows with different values of bin1col are not merged into the same bin, so there are effectively two binning
    columns. Bin values of arr[:, bin0col], use their digitization to group the rows of arr.
    Then take averages of each group that has bin0col==bin0val, where bin0val is each element in arr[:, bin0col],
    and look at statistics of those groups of data.

    Parameters
    ----------
    arr : numpy array
        The array of which to group elements and take statistics
    bin0col : int
        the first column which will be used to group entries of arr into bins
    bin1col : int
        the second column which is used to separate groups of entries of arr
    tol : float
        The allowable difference between elements to put them into the same bin

    Returns
    -------
    binvs : len(unique of arr[:, bin0col]) * len(unique of arr[:, bin1col]) x 1 float array
    avgs : #unique bins x 1 float array
        The average values of all entries in each bin
    mins : #unique bins x 1 float array
        The min values of entries in each bin
    maxs : #unique bins x 1 float array
        The max values of entries in each bin
    stds : #unique bins x 1 float array
        The standard deviation of entries in each bin
    cntvs :
        The number of data points in each bin
    bin0vs :
    """
    ii = 0
    for bin0val in np.sort(np.unique(arr[:, bin0col])):
        # print 'dh.binstats...: bin0val = ', bin0val
        arrslice = arr[arr[:, bin0col] == bin0val, :]
        # print 'dh.binstats...: arrslice = ', arrslice
        binv, avgs, mins, maxs, stds, count = bin_avg_minmaxstdcount(arrslice, bincol=bin1col, tol=tol)
        # print 'dh: bin1col--> binv = ', binv
        if ii == 0:
            binvs = binv
            avgvs = avgs
            minvs = mins
            maxvs = maxs
            stdvs = stds
            cntvs = count
            bin0v = bin0val * np.ones(len(mins))
        else:
            # print 'avgvs = ', avgvs
            # print 'avgs = ', avgs
            binvs = np.hstack((binvs, binv))
            avgvs = np.vstack((avgvs, avgs))
            minvs = np.vstack((minvs, mins))
            maxvs = np.vstack((maxvs, maxs))
            stdvs = np.vstack((stdvs, stds))
            print('cntvs = ', cntvs)
            print('count = ', count)
            cntvs = np.hstack((np.array(cntvs).ravel(), np.array(count).ravel()))
            bin0v = np.hstack((bin0v, bin0val * np.ones(len(mins))))
        ii += 1

    # print 'avgs = ', np.array(avgvs)

    return binvs, avgvs, minvs, maxvs, stdvs, cntvs, np.array(bin0v).ravel()


def approx_bounding_polygon(xy, ngridpts=100):
    """From a list of (possibly unstructured) 2d points, create a polygon which approximates the convex bounding polygon
    of the points to arbitrary precision.

    Example Usage
    -------------
    # Add approx bounding polygon to the axis
    polygons = approx_bounding_polygon(xy, ngridpts=100)
    polygon = polygons[0]
    poly = Polygon(polygon, closed=True, fill=True, lw=0.00, alpha=alpha, color=cmap(colorval), edgecolor=None)
    ax.add_artist(poly)


    Parameters
    ----------
    xy : N x 2 float array
        the xy values of the points
    ngridpts : int
        the number of points along the x dimension with which to estimate the bounding polygon

    Returns
    -------
    polygon : 2*(gridpts - 1) x 2 float array
        The convex bounding polygon for these points
    """
    # Get envelope for the bands by finding min, max pairs and making polygon (kxp for kx polygon)
    kx = xy[:, 0].ravel()
    yy = xy[:, 1].ravel()
    minkx = np.min(kx.ravel())
    maxkx = np.max(kx.ravel())
    kxp = np.linspace(minkx, maxkx, ngridpts)
    kxp_midpts = ((kxp + np.roll(kxp, -1)) * 0.5)[:-1]
    kxploop = np.hstack((kxp_midpts, kxp_midpts[::-1]))
    # print 'np.shape(kxploop) = ', np.shape(kxploop)

    # the y values as we walk right in kx and left in kx
    bandp_right = np.zeros(len(kxp) - 1, dtype=float)
    bandp_left = np.zeros(len(kxp) - 1, dtype=float)
    for kk in range(len(kxp) - 1):
        klow = kxp[kk]
        khi = kxp[kk + 1]
        inbin = np.logical_and(kx > klow, kx < khi)
        # print 'np.shape(yy) = ', np.shape(yy)
        # print 'np.shape(inbin) = ', np.shape(inbin)
        # print 'inbin = ', inbin
        try:
            bandp_right[kk] = np.max(yy[inbin])
            bandp_left[kk] = np.min(yy[inbin])
        except ValueError:
            if kk > 0:
                bandp_right[kk] = bandp_right[kk - 1]
                bandp_left[kk] = bandp_left[kk - 1]

    bandp = np.hstack((bandp_right, bandp_left[::-1]))

    # Check it
    # print 'yy = ', yy
    # print 'np.shape(kxploop) = ', np.shape(kxploop)
    # print 'np.shape(bandp) = ', np.shape(bandp)
    # plt.close('all')
    # plt.plot(bandpoly[-1][:, 0], bandpoly[-1][:, 1], 'b.-')
    # plt.show()

    return np.dstack((kxploop, bandp))[0]


def diff_matrix(AA, BB):
    """
    Compute the difference between all pairs of two sets of values, returning an array of differences.

    Parameters
    ----------
    pts: N x 1 array (float or int)
        points to measure distances from
    nbrs: M x 1 array (float or int)
        points to measure distances to

    Returns
    -------
    Mdiff : N x M float array
        i,jth element is difference between AA[i] and BB[j]
    """
    arr = np.ones((len(AA), len(BB)), dtype=float) * BB
    # gxy_x = np.array([gxy_Xarr[i] - xyip[i,0] for i in range(len(xyip)) ])
    Mdiff = arr - np.dstack(np.array([AA.tolist()] * np.shape(arr)[1]))[0]
    return Mdiff


def nanmedian(x):
    """Before numpy 1.9, there was only scipy.stats.nanmedian(), not numpy.nanmedian. This accomodates the difference"""
    try:
        return np.nanmedian(x)
    except:
        return np.median(x[np.isfinite(x)])


def setdiff2d(A, B):
    """Return row elements in A not in B.
    Used to be called remove_bonds_BL --> Remove bonds from bond list.

    Parameters
    ----------
    A : N1 x M array
        Array to take rows of not in B (could be BL, for ex)
    B : N2 x M
        Array whose rows to compare to those of A

    Returns
    ----------
    BLout : (usually N1-N2) x M array
        Rows in A that are not in B. If there are repeats in B, then length will differ from N1-N2.
    """
    a1_rows = A.view([('', A.dtype)] * A.shape[1])
    # print 'A.dtype = ', A.dtype
    # print 'B.dtype = ', B.dtype
    a2_rows = B.view([('', B.dtype)] * B.shape[1])
    # Now trim those bonds from BL
    C = np.setdiff1d(a1_rows, a2_rows).view(A.dtype).reshape(-1, A.shape[1])
    return C


def unique_nosort(arr):
    """Find unique values in array and return array in the same order as it was passed into this function (no sorting)

    Parameters
    ----------
    arr : numpy array
        the array to take the unique part of, without sorting

    Returns
    -------
    out
    """
    uniq, index = np.unique(arr, return_index=True)
    return uniq[index.argsort()]


def unique_count(a):
    """If using numpy version < 1.9 (accessed by numpy.version.version), then use this to count the occurrence of
    elements in an array of any datatype.
    If using numpy 1.9< use "unique, counts = np.unique(x, return_counts=True)"

    Returns
    -------
    numpy array of same type as a
        first column is unique elements, second column is count for that element
    """
    unique, inverse = np.unique(a, return_inverse=True)
    count = np.zeros(len(unique), np.int)
    np.add.at(count, inverse, 1)
    return np.vstack((unique, count)).T


def generate_gridpts_in_polygons(xlims, ylims, maskrois, dx=1., dy=1.):
    """Determine which grid points are in each polygon

    Parameters
    ----------
    xlims : tuple of two floats
        min and max of x positions for gridpoints
    ylims : tuple of two floats
        min and max of y positions for gridpoints
    maskrois : list of 2d arrays
        a list of polygons for which to find whether the points xgrid, ygrid are in the polygons
    dx : float
        The step in x between elements
    dy : float
        The step in y between elements

    Returns
    -------
    xygridpts : list of n x 2 float arrays
        The grid points in each maskroi
    """
    xx = np.arange(xlims[0], xlims[1] + dx, dx)
    yy = np.arange(ylims[0], ylims[1] + dy, dy)
    xgrid, ygrid = np.meshgrid(xx, yy)
    print('maskrois = ', maskrois)
    inroi = gridpts_in_polygons(xgrid, ygrid, maskrois)
    # print 'dh: maskrois = ', maskrois
    # print 'dh: inrois[0] = ', np.shape(inroi)
    # print 'dh: xgrid = ', np.shape(xgrid)
    # print 'dh: xgrid[inroi] = ', np.shape(xgrid[inroi])
    # print 'dh: ygrid[inroi] = ', np.shape(ygrid[inroi])
    xygridpts = np.dstack((xgrid[inroi].ravel(), ygrid[inroi].ravel()))[0]

    return xygridpts


def gridpts_in_polygons(xgrid, ygrid, maskrois):
    """Determine which grid points are in each polygon

    Parameters
    ----------
    xgrid : N x M float array
        x positions from gridpoints
    ygrid : N x M float array
        y positions from gridpoints
    maskrois : list of 2d arrays
        a list of polygons for which to find whether the points xgrid, ygrid are in the polygons

    Returns
    -------
    inrois : N x M bool array
    """
    first = True
    for roi in maskrois:
        # Determine which xgrid, ygrid are in this roi
        # print 'dh: np.shape(xgrid) = ', np.shape(xgrid)
        # print 'dh: np.shape(xgrid.ravel()) = ', np.shape(xgrid.ravel())
        # print 'dh: np.shape(xgrid) = ', np.shape(xgrid)
        # print 'dh: np.shape(ygrid) = ', np.shape(ygrid)
        # print 'dh: xgrid = ', xgrid
        xy = np.dstack((xgrid.ravel(), ygrid.ravel()))[0]
        bpath = mplpath.Path(roi)
        inroi = bpath.contains_points(xy)
        # print 'dh: inroi -> ', np.shape(inroi)
        inroi = inroi.reshape(np.shape(xgrid))
        # print 'dh: inroi reshaped -> ', np.shape(inroi)

        if first:
            inrois = inroi
            first = False
        else:
            inrois = np.logical_or(inroi, inrois)

    return inrois


def polygon_area(vertices):
    """Compute the area inside the closed polygon given by vertices. Compare also to poly_area() -- I believe this is
    slower.

    Parameters
    ----------
    vertices : n x 2 float array
        the corners of the poylgon

    Returns
    -------
    area : float
        the area enclosed by the polygon
    """
    nvtcs = len(vertices)
    area = 0.0
    for ii in range(nvtcs):
        jj = (ii + 1) % nvtcs
        area += vertices[ii][0] * vertices[jj][1]
        area -= vertices[jj][0] * vertices[ii][1]
    area = abs(area) / 2.0
    return area


def poly_area(xx, yy):
    """I believe this is a faster way

    Parameters
    ----------
    xx
    yy

    Returns
    -------

    """
    return 0.5 * np.abs(np.dot(xx, np.roll(yy, 1)) - np.dot(yy, np.roll(xx, 1)))


def unique_rows(a):
    """Clean up an array such that all its rows are unique.
    Reference:
    http://stackoverflow.com/questions/7989722/finding-unique-points-in-numpy-array

    Parameters
    ----------
    a : N x M array of variable dtype
        array from which to return only the unique rows
    """
    return np.array(list(set(tuple(p) for p in a)))


def unique_threshold(a, thres):
    """Clean up an array such that all its elements are at least 'thres' different in value.
    See also unique_rows_threshold() for a generalization to 2D.
    Reference:
    http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array

    Parameters
    ----------
    a : N x 1 array of variable dtype
        array from which to return only the unique rows
    thres : float
        threshold for deleting a row that has slightly different values from another row

    Returns
    ----------
    a : N x 1 array of variable dtype
        unique rows of input array
    """
    a = np.sort(a)
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = np.abs(diff) > thres

    return a[ui]


def unique_rows_threshold(a, thres):
    """Clean up an array such that all its rows are at least 'thres' different in value.
    Reference:
    http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array

    Parameters
    ----------
    a : N x M array of variable dtype
        array from which to return only the unique rows
    thres : float
        threshold for deleting a row that has slightly different values from another row

    Returns
    ----------
    a : N x M array of variable dtype
        unique rows of input array
    """
    if a.ndim > 1:
        # sort by ...
        order = np.lexsort(a.T)
        a = a[order]
        diff = np.diff(a, axis=0)
        ui = np.ones(len(a), 'bool')
        ui[1:] = (np.abs(diff) > thres).any(axis=1)
    else:
        a = np.sort(a)
        diff = np.diff(a, axis=0)
        ui = np.ones(len(a), 'bool')
        ui[1:] = np.abs(diff) > thres

    return a[ui]


def args_unique_rows_threshold(a, thres):
    """Clean up an array such that all its rows are at least 'thres' different in value.
    Reference:
    http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array

    Parameters
    ----------
    a : N x M array of variable dtype
        array from which to return only the unique rows
    thres : float
        threshold for deleting a row that has slightly different values from another row

    Returns
    ----------
    a : N x M array of variable dtype
        unique rows of input array
    order : N x 1 int array
        indices used to sort a in order
    ui : N x 1 boolean array
        True where row of a[order] is unique.

    """
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (np.abs(diff) > thres).any(axis=1)
    return a[ui], order, ui


def sortrows_2d(arr, priority=1, xdescending=False, ydescending=False):
    """

    Parameters
    ----------
    arr : N x 2 float or int array
        The array whose rows are to be sorted by the priority column, then by the other column
    priority : int (0 or 1) or 'x' or 'y'
        The column to use as a primary sorting column. Sorting will then proceed within each block where all elements of
        this column are identical, within machine precision
    xdescending : bool (default=False)
        Whether by sort column 0 in descending order rather than ascending
    ydescending : bool (default=False)
        Whether by sort column 1 in descending order rather than ascending

    Returns
    -------
    sortarr : N x 2 array of dtype(arr)
        The sorted 2d array; each row is found in input array arr
    """
    if priority in [1, 'y']:
        col0, col1 = 0, 1
    elif priority in [0, 'x']:
        col0, col1 = 1, 0
    else:
        raise RuntimeError("Argument 'priority' must be either 0 or 1 for 2d array")

    xascend = 1
    yascend = 1
    if xdescending:
        xascend = -1
    if ydescending:
        yascend = -1

    ind = np.lexsort((xascend * arr[:, col0], yascend * arr[:, col1]))

    # print 'dh.sortrows_2d: np.shape(arr) = ', np.shape(arr)
    # print 'dh.sortrows_2d: ind = ', ind
    return arr[ind]


def running_mean(x, N):
    """Compute running mean of an array x, averaged over a window of N elements.
    If the array x is 2d, then a running mean is performed on each row of the array.

    Parameters
    ----------
    x : N x (1 or 2) array
        The array to take a running average over
    N : int
        The window size of the running average

    Returns
    -------
    output : 1d or 2d array
        The averaged array (each row is averaged if 2d), preserving datatype of input array x
    """
    if len(np.shape(x)) == 1:
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / N
    elif len(np.shape(x)) == 2:
        # Apply same reasoning to the array row-by-row
        dmyi = 0
        for row in x:
            tmpsum = np.cumsum(np.insert(row, 0, 0))
            outrow = (tmpsum[N:] - tmpsum[:-N]) / N
            if dmyi == 0:
                outarray = np.zeros((np.shape(x)[0], len(outrow)), dtype=x.dtype)
            outarray[dmyi, :] = outrow
            dmyi += 1

        return outarray
    else:
        raise RuntimeError('Input array x in running_mean(x, N) must be 1d or 2d.')


def nearest_local_maximum(xarr, yarr, value, thres=None, npeaks=None):
    """Find the nearest local maximum in y array whose corresponding x array value is nearest to supplied value.
    If thres is specified, only consider peaks (y values) above thres, and if npeaks is supplied, only consider the
    largest npeaks.

    Parameters
    ----------
    xarr : n x 1 float array
        the x data for the intensity data (yarr)
    yarr : n x 1 float array
        Some intensity data in order of some parameter (xarr), to be searched for local maxima
    value : float
        look for a peak with an x value near this value
    thres : float (0-1) or None for no thresholding
        The threshold intensity as a fraction of the maximum
    npeaks : int or None
        If not None, only keep the npeaks biggest peaks

    Returns
    -------
    ind : int
        the index of xarr and yarr such that yarr is a peak with corresponding xarr closest to value.
    """
    inds = find_peaks(arr, thres=thres, npeaks=npeaks)
    ind = np.argmin(np.abs(xarr[inds] - value))
    return inds[ind]


def find_peaks(arr, xdata=None, thres=None, thres_curvature=None, normalize_for_curv=False, npeaks=None):
    """Identify local maxima in a 1d curve/array of data

    Parameters
    ----------
    arr : n x 1 float array
        Some intensity data in order of some parameter, to be searched for local maxima
    xdata :  n x 1 float array or None (optional, if thres_curvature is nonzero and unequally spaced data)
        The x data associated with the y data of arr. This is ONLY used if thres_curvature is not None -- ie, if there
        is thresholding in the magnitude of curvature to find a peak. This would exclude broad/short peaks
    thres : float (0-1) or None for no thresholding
        The threshold intensity as a fraction of the maximum
    thres_curvature : positive definite float or None for no curvature thresholding
        The absolute threshold curvature of the curve. The magnitude (np.abs()) of the curvature is considered, and
        corrections from the first derivative of the curve are neglected at the maximum. This means that we assume that
        the data is finely spaced enough that the identified peaks are very near the true local maxima.
    normalize_for_curv : bool
        whether to have the curvature be computed for the normalized signal, normalized such that the maximum intensity
        (positive or negative) is 1.0
    npeaks : int or None
        If not None, only keep the npeaks biggest peaks

    Returns
    -------
    inds : npeaks x 1 int array
        The indices of intensities which are peaks
    """
    inds = np.where((np.diff(arr)[:-1] > 0) & (np.diff(arr)[1:] < 0))[0] + 1
    # inds = np.r_[True, arr[1:] > arr[:-1]] & np.r_[arr[:-1] < arr[1:], True]

    # keep only the npeaks biggest peaks
    if npeaks is not None and npeaks > 0:
        inds = arr[inds].argsort()[-npeaks:][::-1]

    # perform thresholding
    if thres is not None:
        print('thresholding here: ', thres)
        print('inds = ', inds)
        tmp = np.where(arr[inds] > thres * np.max(arr))
        inds = inds[tmp]

    if thres_curvature is not None:
        # Note that curvature of a 1d curve is kappa = |f"(x) | / (1 + f'(x) **2 ) ** (3/2)
        # At the identified local maximum, the first derivative is approximately zero, so we neglect this correction
        if normalize_for_curv:
            # Note: avoid in-place redefinition here
            arr = arr / np.max(np.abs(arr))

        if xdata is not None:
            kappa = np.abs(np.gradient(xdata, np.gradient(xdata, arr)))
        else:
            kappa = np.gradient(np.gradient(arr))

        # Check it
        # import matplotlib.pyplot as plt
        # print 'kappa = ', kappa
        # plt.clf()
        # plt.plot(np.arange(len(kappa)), kappa, 'r-')
        # plt.show()

        inds = inds[np.where(kappa[inds] < -thres_curvature)[0]]

    return inds


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x: the input signal
    window_len: the dimension of the smoothing window; should be an odd integer
    window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        flat window will produce a moving average smoothing.

    Returns
    -------
        the smoothed signal

    Example usage
    -------------
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)]
    instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def interpol_meshgrid(x, y, z, n, method='nearest'):
    """Interpolate z on irregular or unordered grid data (x,y) by supplying # points along each dimension.
    Note that this does not guarantee a square mesh, if ranges of x and y differ.

    Parameters
    ----------
    x : unstructured 1D array
        data along first dimension
    y : unstructured 1D array
        data along second dimension
    z : 1D array
        values evaluated at x,y
    n : int
        number of spacings in meshgrid of unstructured xy data
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation. One of
        ``nearest``
          return the value at the data point closest to
          the point of interpolation.  See `NearestNDInterpolator` for
          more details.
        ``linear``
          tesselate the input point set to n-dimensional
          simplices, and interpolate linearly on each simplex.  See
          `LinearNDInterpolator` for more details.
        ``cubic`` (1-D)
          return the value determined from a cubic
          spline.
        ``cubic`` (2-D)
          return the value determined from a
          piecewise cubic, continuously differentiable (C1), and
          approximately curvature-minimizing polynomial surface. See
          `CloughTocher2DInterpolator` for more details.

    Returns
    -------
    X : n x 1 float array
        meshgrid of first dimension
    X : n x 1 float array
        meshgrid of second dimension
    Zm : n x 1 float array
        (interpolated) values of z on XY meshgrid, with nans masked
    """
    # define regular grid spatially covering input data
    xg = np.linspace(x.min(), x.max(), n)
    yg = np.linspace(y.min(), y.max(), n)
    X, Y = np.meshgrid(xg, yg)

    # interpolate Z values on defined grid
    Z = interpolate.griddata(np.vstack((x.flatten(), y.flatten())).T, np.vstack(z.flatten()),
                             (X, Y), method=method).reshape(X.shape)
    # mask nan values, so they will not appear on plot
    Zm = np.ma.masked_where(np.isnan(Z), Z)
    return X, Y, Zm


def interpolate_onto_mesh(xx, yy, zz, XX, YY, method='linear', mask=True):
    """Interpolate new data x,y,z onto grid data X,Y

    Parameters
    ----------
    xx : float array (any shape)
        x coordinates of points to be interpolated
    yy : float array (any shape)
        y coordinates of points to be interpolated
    zz : float array (any shape)
        the values evaluated on the xx, yy points to be interpolated as output

    Returns
    -------
    """
    # interpolate Z values on defined grid
    ZZ = griddata(np.vstack((xx.flatten(), yy.flatten())).T, np.vstack(zz.flatten()),
                  (XX, YY), method=method).reshape(XX.shape)
    # mask nan values, so they will not appear on a plot
    if mask:
        Zm = np.ma.masked_where(np.isnan(ZZ), ZZ)
        return Zm
    else:
        return ZZ


def round_thres(a, minclip):
    """Round a float value a to the nearest multiple of minclip

    Parameters
    ----------
    a : float
        float to round
    minclip : float or int
        resolution of the rounded result

    Returns
    -------
    float rounded to nearest multiple of minclip
    """
    return round(float(a) / minclip) * minclip


def round_thres_numpy(a, minclip):
    """Round elements of array to nearest multiple of minclip

    Parameters
    ----------
    a : numpy array
        array to round
    minclip : float or int
        resolution of the rounded result

    Returns
    -------
    array rounded to nearest multiple of minclip
    """
    return np.round(np.array(a, dtype=float) / minclip) * minclip


def round_sigfigs(x, sigfigs):
    """https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy
    Rounds the value(s) in x to the number of significant figures in sigfigs.

    Parameters
    ----------
    x : int, float, or array of real values
        the value(s) to round
    sigfigs : int
        the number of significant digits to include

    Restrictions
    ------------
    sigfigs must be an integer type and store a positive value.
    x must be a real value or an array like object containing only real values.
    """
    # The following constant was computed in maxima 5.35.1 using 64 bigfloat digits of precision
    __logBase10of2 = 3.010299956639811952137388947244930267681898814621085413104274611e-1

    if not (type(sigfigs) is int or np.issubdtype(sigfigs, np.integer)):
        raise TypeError("RoundToSigFigs: sigfigs must be an integer." )

    if not np.all(np.isreal(x)):
        raise TypeError("RoundToSigFigs: all x must be real.")

    if sigfigs <= 0:
        raise ValueError("RoundtoSigFigs: sigfigs must be positive." )

    xsgn = np.sign(x)
    absx = xsgn * x
    mantissas, binaryExponents = np.frexp( absx )

    decimalExponents = __logBase10of2 * binaryExponents
    intParts = np.floor(decimalExponents)

    mantissas *= 10.0**(decimalExponents - intParts)

    if type(mantissas) is float or np.issctype(np.dtype(mantissas)):
        if mantissas < 1.0:
            mantissas *= 10.0
            omags -= 1.0

    elif np.issubdtype(mantissas, np.ndarray):
        fixmsk = mantissas < 1.0
        mantissas[fixmsk] *= 10.0
        omags[fixmsk] -= 1.0

    return xsgn * np.around( mantissas, decimals=sigfigs - 1 ) * 10.0**intParts


def pts_in_polygon(xy, polygon):
    """Returns points in array xy that are located inside supplied polygon array.
    """
    bpath = mplpath.Path(polygon)
    inside = bpath.contains_points(xy)
    xy_out = xy[inside, :]
    return xy_out


def pts_outside_polygon(xy, polygon):
    """Returns points in array xy that are located inside supplied polygon array.
    """
    bpath = mplpath.Path(polygon)
    outside = np.logical_not(bpath.contains_points(xy))
    xy_out = xy[outside, :]
    return xy_out


def inds_in_polygon(xy, polygon):
    """Returns points in array xy that are located inside supplied polygon array.
    """
    bpath = mplpath.Path(polygon)
    inside = bpath.contains_points(xy)
    inds = np.where(inside)[0]
    return inds


def polygons_enclosing_pt(pt, polygons):
    """Returns points in array xy that are located inside supplied polygon array.

    Parameters
    ----------
    pt : 2 x 1 float array
        the point which we consider
    polygons : list of #vertices x 2 float arrays
        The vertices of the polygons to consider.

    Returns
    -------
    inds : list of ints
        The indices of polygons of the polygons which enclose pt
    """
    inds = []
    ind = 0
    for polygon in polygons:
        bpath = mplpath.Path(polygon)
        inside = bpath.contains_points(pt)
        encloses = np.where(inside)[0]
        if encloses:
            inds.append(ind)
        ind += 1
    return inds


def generate_random_xy_in_polygon(npts, polygon, sorted=False):
    """Generate random xy values inside a polygon (in order, for example, to get kx, ky inside Brillouin zone)
    Often searched for as generate_pts_in_polygon().

    Parameters
    ----------
    npts :
    polygon :
    sorted : bool
        sort the output by y value

    Returns
    -------

    """
    scale = max(np.max(polygon[:, 0]) - np.min(polygon[:, 0]), np.max(polygon[:, 1]) - np.min(polygon[:, 1]))
    cxy = np.mean(polygon, axis=0)
    xy0 = scale * (np.random.rand(npts, 2) - 0.5) + cxy
    xyout = pts_in_polygon(xy0, polygon)

    while np.shape(xyout)[0] < npts:
        np_new = npts - np.shape(xyout)[0] + 10
        xyadd = scale * (np.random.rand(np_new, 2) - 0.5) + cxy
        xyadd = pts_in_polygon(xyadd, polygon)
        xyout = np.vstack((xyout, xyadd))

    if sorted:
        xyout = xyout[xyout[:, 1].argsort()]

    return xyout[0:npts]


def pts_in_polyhedron(pts, polyhedron, boundary_is_inside=True, verbose=True):
    """

    Parameters
    ----------
    pts : N x D float array
        the points to test for being inside the polyhedron
    polyhedron : Polyhedron class instance
        The polyhedron for which to examine whether pts are inside
    boundary_is_inside : bool
        Consider points on the boundary of the polyhedron to be inside the polyhedron rather than outside
    verbose : bool
        Write to command line output to announce progress

    Returns
    -------

    """
    is_inside = np.zeros(np.shape(pts)[0])

    # Compute winding number of each point
    for (point, inpt, kk) in zip(pts, is_inside, range(len(pts))):
        if verbose:
            if kk % 100 == 0:
                print('examining pt #' + str(kk) + '/' + str(len(pts)))
        try:
            wn = polyhedron.winding_number(point)
            if wn == 1:
                inpt = True
            elif wn == 0:
                inpt = False
            else:
                raise RuntimeError("should never get here in pts_in_polygon()")
        except ValueError:
            if boundary_is_inside:
                inpt = True
            else:
                inpt = False

    return is_inside


def dist_pts(pts, nbrs, dim=-1, square_norm=False):
    """
    Compute the distance between all pairs of two sets of points, returning an array of distances, in an optimized way.

    Parameters
    ----------
    pts: N x D array (float or int)
        points to measure distances from
    nbrs: M x D array (float or int)
        points to measure distances to
    dim: int (default -1)
        dimension along which to measure distance. Default is -1, which measures the Euclidean distance in D dimensions
    square_norm: bool
        Abstain from taking square root, so that if dim==-1, returns the square norm (distance squared).

    Returns
    -------
    dist : N x M float array
        i,jth element is distance between pts[i] and nbrs[j], along dimension specified (default is normed distance)
    """
    if dim > 0:
        Xarr = np.ones((len(pts), len(nbrs)), dtype=float) * nbrs[:, dim]
        # Computing dist(x)
        dist_x = Xarr - np.dstack(np.array([pts[:, dim].tolist()] * np.shape(Xarr)[1]))[0]
    else:
        # Computing distance for each dim
        dists = []
        for axis in range(np.shape(pts)[1]):
            narr = np.ones((len(pts), len(nbrs)), dtype=float) * nbrs[:, axis]
            dists.append(narr - np.dstack(np.array([pts[:, 1].tolist()] * np.shape(narr)[1]))[0])

    if dim > 0:
        if square_norm:
            return dist_x ** 2
        else:
            return dist_x
    else:
        dist = 0.
        for dd in dists:
            dist += dd ** 2
        if not square_norm:
            dist = np.sqrt(dist)
        return dist


def nearest_pts(pts, nbrs, vectorization_cutoff=1e7, voronoi_cutoff=1e11, square_norm=False):
    """For each point in pts, match to the nearest point in neighbors, such that pts[ind, :] ~= nbrs -- ie return
    indices that maps pts to a pointset xy[ind] where each element xy[ind][ii] is closest to nbrs[ii].
    See also dist_nearest_pts() for the distances only.

    Parameters
    ----------
    pts : N x 2 float array
    nbrs : M x 2 float array
    vectorization_cutoff : int
        cutoff # total points above which we make a Voronoi tesselation
    voronoi_cutoff : int
        cutoff # total points above which we simply iterate over each. This is beneficial when n*Log(n) is too large.
    square_norm : bool
        whether to return the square norm or the true norm (allows for sparing the square root)

    Returns
    -------
    mins : N x 1 int array
        the distances of nbr such that each element in pts is measured to its nearest point in nbr
    """
    if len(pts) * len(nbrs) > vectorization_cutoff:
        if len(pts) * len(nbrs) > voronoi_cutoff:
            ##############################
            # Iterate over each
            ##############################
            print('data_handling.nearest_pts(): Iterating over each particle to compute nearest points')
            mins = np.zeros_like(pts[:, 0], dtype=float)
            inds = np.zeros_like(pts[:, 0], dtype=int)
            for (pt, kk) in zip(pts, np.arange(len(pts))):
                mins[kk], inds[kk] = closest_point(pt, nbrs)

            # Checking
            # print 'dist_pts = ', dist_pts(pts, nbrs)[0:10]
            # print 'dist_pts = ', np.min(dist_pts(pts, nbrs)[0:10], axis=0)
            # sys.exit()
        else:
            ##############################
            # Voronoi: O(n Log(n))
            ##############################
            print('data_handling.nearest_pts(): Computing Voronoi to compute nearest points')
            voronoi_kdtree = cKDTree(nbrs)
            # for (pt, kk) in zip(pts, range(len(pts))):
            print('Querying KDTree...')
            mins, inds = voronoi_kdtree.query(pts, k=1)
    else:
        # Checking
        # print 'dist_pts = ', dist_pts(pts, nbrs)[0:10]
        # print 'dist_pts = ', np.min(dist_pts(pts, nbrs)[0:10], axis=0)
        # sys.exit()
        dists = dist_pts(pts, nbrs, square_norm=True)
        mins = np.min(dists, axis=1)
        inds = np.argmin(dists, axis=1)

    if square_norm:
        return mins, inds
    else:
        return np.sqrt(mins), inds


def dist_nearest_pt(pts, nbrs, vectorization_cutoff=1e7, voronoi_cutoff=1e10, square_norm=False):
    """For each point in pts, match to the nearest point in neighbors, such that pts[ind, :] ~= nbrs -- ie return
    distance of indices that maps pts to a pointset xy[ind] where each element xy[ind][ii] is closest to nbrs[ii].
    See also match_points() for the indices instead of distances.

    Parameters
    ----------
    pts : N x 2 float array
    nbrs : M x 2 float array
    vectorization_cutoff : int
        cutoff # total points above which we make a Voronoi tesselation
    voronoi_cutoff : int
        cutoff # total points above which we simply iterate over each. This is beneficial when n*Log(n) is too large.
    square_norm : bool
        whether to return the square norm or the true norm (allows for sparing the square root)

    Returns
    -------
    mins : N x 1 int array
        the distances of nbr such that each element in pts is measured to its nearest point in nbr
    """
    mins, inds = nearest_pts(pts, nbrs, vectorization_cutoff=vectorization_cutoff,
                             voronoi_cutoff=voronoi_cutoff, square_norm=square_norm)
    return mins


def interparticle_separation(pts, nbr_cutoff=None, nnbrs=1, eps=1e-9):
    """Given a set of points, compute the average distance of a given point from nnbrs nearest points OR from all
    points within a distance of nbr_cutoff. Note that there is no network structure here, just points.

    Parameters
    ----------
    pts: N x 2 array (float or int)
        points to measure distances from
    nbr_cutoff : float or None
        If not None, computes average distance for each pt from all nearby points within a range of nbr_cutoff
    nnbrs : int (default=1)
        If nbr_cutoff is None, computes average distance for each pt from nnbrs nearest points
    eps : float
        short range cutoff to ignore if a particle's distance is less than eps (to eliminate the accidentally
        measured self-self distance)

    Returns
    -------
    separation : N x 1 float array
        average separation from nearby points
    """
    if nbr_cutoff is not None:
        dists = dist_pts(pts, pts)
        separation = np.array([dist[np.logical_and(dist < nbr_cutoff, dist > 0)] for dist in dists])
    else:
        dists = dist_pts(pts, pts)
        separation = np.array([dist[dist[dist > eps].argsort()[0:nnbrs]] for dist in dists])
    return separation


def dist_pts_periodic_2d(pts, nbrs, PV, dim=-1, square_norm=False):
    """
    Compute the distance between all pairs of two sets of points in a periodic rectangular system of dimension
    LL[0] x LL[1], returning an array of distances, in an optimized way. If particle a is closer to particle b across
    a periodic boundary, then the minimum distance is returned.
    Could generalize this to arbitrary periodic shape by using PV instead of LL, computing distances of each point to
    both interior and reflected points across each periodic boundary, taking the minimum.
    WARNING: There is no connectivity of points in this function. For getting bond lengths between particles when there
    are periodic bonds, use BMxy()

    Parameters
    ----------
    pts: N x 2 array (float or int)
        points to measure distances from
    nbrs: M x 2 array (float or int)
        points to measure distances to
    PV : 2 x 2 float array
        The vectors taking bottom left corner to bottom right and top left corners --> periodic vectors
    dim: int (default -1)
        dimension along which to measure distance. Default is -1, which measures the Euclidean distance in 2D
    square_norm: bool
        Abstain from taking square root, so that if dim==-1, returns the square norm (distance squared).

    Returns
    -------
    dist : N x M float array
        i,jth element is distance between pts[i] and nbrs[j], along dimension specified (default is normed distance)
    """
    if np.shape(PV) != (2, 2):
        raise RuntimeError('PV must be 2x2 float array')
    if dim < 1:
        Xarr = np.ones((len(pts), len(nbrs)), dtype=float) * nbrs[:, 0]
        # Computing dist(x)
        distsx = np.zeros((len(pts), len(nbrs), 5), dtype=float)
        dist_x0 = Xarr - np.dstack(np.array([pts[:, 0].tolist()] * np.shape(Xarr)[1]))[0]
        distsx[:, :, 0] = dist_x0
        distsx[:, :, 1] = dist_x0 - PV[0, 0]
        distsx[:, :, 2] = dist_x0 + PV[0, 0]
        distsx[:, :, 3] = dist_x0 - PV[1, 0]
        distsx[:, :, 4] = dist_x0 + PV[1, 0]
        # get x distance whose modulus is minimal
        minpick = np.argmin(np.abs(distsx), axis=2)
        dist_x = np.array(
            [[distsx[i, j, minpick[i, j]] for i in xrange(np.shape(distsx)[0])] for j in xrange(np.shape(distsx)[1])])
        # check it
        # print 'minpick = ', minpick
        # print 'dist_x = ', dist_x
        # plot_real_matrix(minpick, show=True)
        # plot_real_matrix(dist_x0, show=True)
        # plot_real_matrix(dist_x, show=True)
        # plot_real_matrix(dist_x0, show=True)
        # plot_real_matrix(dist_x, show=True)
        # plot_real_matrix(dist_x - dist_x0, show=True)
        # sys.exit()
    if np.abs(dim) > 0.5:
        Yarr = np.ones((len(pts), len(nbrs)), dtype=float) * nbrs[:, 1]
        # Computing dist(y)
        distsy = np.zeros((len(pts), len(nbrs), 5), dtype=float)
        dist_y0 = Yarr - np.dstack(np.array([pts[:, 1].tolist()] * np.shape(Yarr)[1]))[0]
        distsy[:, :, 0] = dist_y0
        distsy[:, :, 1] = dist_y0 - PV[0, 1]
        distsy[:, :, 2] = dist_y0 + PV[0, 1]
        distsy[:, :, 3] = dist_y0 - PV[1, 1]
        distsy[:, :, 4] = dist_y0 + PV[1, 1]
        # get x distance whose modulus is minimal
        minpick = np.argmin(np.abs(distsy), axis=2)
        dist_y = np.array(
            [[distsy[i, j, minpick[i, j]] for i in xrange(np.shape(distsy)[0])] for j in xrange(np.shape(distsy)[1])])

    if dim == -1:
        dist = dist_x ** 2 + dist_y ** 2
        if not square_norm:
            dist = np.sqrt(dist)
        return dist
    elif dim == 0:
        return dist_x
    elif dim == 1:
        return dist_y


def dist_pts_along_vec(pts, nbrs, vec):
    """Compute the distance between all pairs of two sets of points projected onto the vector vec, returning an array
    of projected distances.

    Parameters
    ----------
    pts: N x D array (float or int)
        points to measure distances from
    nbrs: M x D array (float or int)
        points to measure distances to
    vec : len(D) array (float or int)
        a vector along which to compute distance from each pt to nearest nbr in nbrs

    Returns
    -------
    dist_alongv : len(pts) x len(nbrs) float array
        the distances between pts and nbrs projected onto vec. dist_alongv[i,j] is the projected distance from pt[i]
        to nbrs[j]
    """
    # print 'dh.dist_pts_along_vec: pts = ', pts
    # print 'dh.dist_pts_along_vec: nbrs= ', nbrs

    # Allow for D dimensions here. Iterate over each dimension to get distance between all pairs of (pt, nbr)
    dist_xs = []
    for dd in range(np.shape(pts)[1]):
        dist_xs.append(dist_pts(pts, nbrs, dim=dd))

    if not dist_xs:
        raise RuntimeError('pts is zero dimensional. Exiting')

    # Append these together in unraveled arrays, one column for each dimension
    for (batch, dd) in zip(dist_xs, range(len(dist_xs))):
        if dd == 0:
            distv = batch.ravel()
            distv = distv.reshape((np.shape(distv)[0], 1))
        else:
            add_col = batch.ravel().reshape((np.shape(batch.ravel())[0], 1))
            # print 'np.shape(distv) = ', np.shape(distv)
            # print 'np.shape(add_col) = ', np.shape(add_col)
            distv = np.hstack((distv, add_col))
            # print 'distv = ', distv

    along = np.dot(distv, vec)
    # along = np.einsum('ij,ij->i', distv, vec)
    dist_alongv = along.reshape(np.shape(dist_xs[0]))
    return dist_alongv


def dist_pts_along_vecs(pts, nbrs, vecs):
    """Compute the distance between all pairs of two sets of points projected onto the vector vec, returning an array
    of projected distances.

    Parameters
    ----------
    pts: N x D array (float or int)
        points to measure distances from
    nbrs: M x D array (float or int)
        points to measure distances to
    vecs : N x D array (float or int)
        vectors for each pt in pts, along which to compute distance from each pt to nearest nbr in nbrs

    Returns
    -------
    dist_alongv : len(pts) x len(nbrs) float array
        The distances between pts and nbrs projected onto each vec in vecs.
        dist_alongv[i,j] is the projected distance from pt[i] to nbrs[j]
    """
    # print 'dh.dist_pts_along_vec: pts = ', pts
    # print 'dh.dist_pts_along_vec: nbrs= ', nbrs

    # Allow for D dimensions here. Iterate over each dimension to get distance between all pairs of (pt, nbr)
    dist_xs = []
    for dd in range(np.shape(pts)[1]):
        dist_xs.append(dist_pts(pts, nbrs, dim=dd))

    if not dist_xs:
        raise RuntimeError('pts is zero dimensional. Exiting')

    # Append these together in unraveled arrays, one column for each dimension
    for (batch, dd) in zip(dist_xs, range(len(dist_xs))):
        if dd == 0:
            distv = batch.ravel()
        else:
            distv = np.dstack((distv, batch.ravel()))[0]

    along = np.einsum('ij,ij->i', distv, vecs)
    dist_alongv = along.reshape(np.shape(dist_xs[0]))
    return dist_alongv


def closest_point(pt, xy):
    """Find the index of the point closest to the supplied single point pt

    Parameters
    ----------
    pt : 1 x D float array
        A single xy point
    xy : N x D float array or list
        the coordinates of the points to compare pt to
    """
    xy = np.asarray(xy)
    dist_2 = np.sum((xy - pt) ** 2, axis=1)
    return np.argmin(dist_2)


def closest_point(pt, xy):
    """Find the distance and index of the point closest to the supplied single point pt

    Parameters
    ----------
    pt : 1 x D float array
        A single xy point
    xy : N x D float array or list
        the coordinates of the points to compare pt to

    Returns
    -------
    dist : float
        the Euclidean distance of the point pt which is closest to xy
    ind : int
        index of the xy point which is closest to pt
    """
    xy = np.asarray(xy)
    dist_2 = np.sum((xy - pt) ** 2, axis=1)
    return np.sqrt(np.min(dist_2)), np.argmin(dist_2)


def dist_closest_point(pt, xy):
    """Find the distance of the point closest to the supplied single point pt

    Parameters
    ----------
    pt : 1 x D float array
        A single xy point
    xy : N x D float array or list
        the coordinates of the points to compare pt to
    """
    xy = np.asarray(xy)
    dist_2 = np.sum((xy - pt) ** 2, axis=1)
    return np.sqrt(np.min(dist_2))


def dist_closest_points_along_vecs(pts, nbrs, vecs, vectorization_cutoff=1e7):
    """Determine projected distance of a point in nbrs that is closest along a vector, for each point in pts.

    Parameters
    ----------
    pts : N x D float array
        the points for which to find the distance from pts[ii] to the nearest neighbor point along vec[ii]
    nbrs : M x D float array
        the points from which to find the distance from pts[ii] to the nearest neighbor point along vec[ii]
    vecs : N x D float array
        the vectors along which to project (visualize these emanating out from pts, like normal vecs of a surface,
        for example)
    vectorization_cutoff : int
        the largest size of len(pts) * len(nbrs) above which we iterate over each point in pts

    Returns
    -------
    dists : N x 1 float array
        the shortest projected distances from each pt in pts to the closest nbr point projected along each vec.
    """
    if len(pts) * len(nbrs) > vectorization_cutoff:
        dists = np.zeros(len(pts), dtype=float)
        # inds = np.zeros(len(pts), dtype=int)
        for (pt, vec, kk) in zip(pts, vecs, np.arange(len(pts))):
            if kk % 100 == 0:
                print('data_handling.dist_closest_points_along_vecs: pt #' + str(kk) + ' / ' + str(len(pts)))
            distpairs = dist_pts_along_vec(np.array([pt]), nbrs, vec).ravel()
            print('distpairs = ', distpairs)
            dists[kk] = np.min(distpairs[distpairs > 0])
    else:
        dists = dist_pts_along_vecs(pts, nbrs, vecs)
        dists = np.min(dists[dists > 0], axis=0)

    return dists


def match_points(pts, nbrs, vectorization_cutoff=1e7, voronoi_cutoff=1e10):
    """For each point in pts, match to the nearest point in neighbors, such that pts[ind, :] ~= nbrs -- ie return
    indices that maps pts to a pointset xy[ind] where each element xy[ind][ii] is closest to nbrs[ii].
    See also closest_point() for the single neighbor version.

    Parameters
    ----------
    pts : N x 2 float array
    nbrs : M x 2 float array

    Returns
    -------
    inds : N x 1 int array
        the indices of nbr such that each element in pts is mapped to its nearest point in nbr
    """
    dists, inds = nearest_pts(pts, nbrs, vectorization_cutoff=vectorization_cutoff, voronoi_cutoff=voronoi_cutoff,
                              square_norm=True)
    return inds
    # if len(pts) * len(nbrs) > vectorization_cutoff:
    #     inds = np.zeros_like(nbrs[:, 0], dtype=int)
    #     for (nbr, kk) in zip(nbrs, np.arange(len(nbrs))):
    #         ind = closest_point(nbr, pts)
    #         inds[kk] = ind
    # else:
    #     inds = np.argmin(dist_pts(pts, nbrs, square_norm=True), axis=0)
    # return inds


def match_values(vals, arr):
    """For each value in vals, match to the nearest value in arr

    Parameters
    ----------
    vals : N x 1 float or int array
    arr : M x 1 float or int array

    Returns
    -------
    inds : N x 1 int array
        the indices of arr such that each element in vals is mapped to its nearest value in arr -- ie vals[inds] ~= arr
    """
    arrv = np.ones((len(vals), len(arr)), dtype=float) * arr
    dist_x = arrv - np.dstack(np.array([vals.tolist()] * np.shape(arrv)[1]))[0]
    return np.argmin(np.abs(dist_x), axis=1)


def fill_nans_with_interpolation(arr):
    """Replace nans in an array with interpolated values"""
    mask = np.isnan(arr)
    f0 = np.flatnonzero(mask)
    f1 = np.flatnonzero(~mask)
    arr[mask] = np.interp(f0, f1, arr[~mask])

    arr2 = copy.deepcopy(arr.T)
    maskt = mask.T
    f0 = np.flatnonzero(maskt)
    f1 = np.flatnonzero(~maskt)
    arr2[mask] = np.interp(f0, f1, arr[~maskt])

    arrout = arr + arr2.T
    raise RuntimeError('Check that this works and is faster than interp2d')
    return arrout


def sort_arrays_by_first_array(arr2sort, arrayList2sort):
    """Sort many arrays in the same way, based on sorting of first array

    Parameters
    ----------
    arr2sort : N x 1 array
        Array to sort
    arrayList2sort : List of N x 1 arrays
        Other arrays to sort, by the indexing of arr2sort

    Returns
    ----------
    arr_s, arrList_s : sorted N x 1 arrays
    """
    IND = np.argsort(arr2sort)
    arr_s = arr2sort[IND]
    arrList_s = []
    for arr in arrayList2sort:
        arrList_s.append(arr[IND])
    return arr_s, arrList_s


def removekey(d, key):
    """Remove a key from a dictionary (Note: this is a shallow copy, not a deep copy)"""
    r = dict(d)
    del r[key]
    return r


def savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0):
    """ Apply a Savitzky-Golay filter to an array.
    This is a 1-d filter.  If `x`  has dimension greater than 1, `axis`
    determines the axis along which the filter is applied.
    Parameters
    ----------
    x : array_like
        The data to be filtered.  If `x` is not a single or double precision
        floating point array, it will be converted to type `numpy.float64`
        before filtering.
    window_length : int
        The length of the filter window (i.e. the number of coefficients).
        `window_length` must be a positive odd integer. If `mode` is 'interp',
        `window_length` must be less than or equal to the size of `x`.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute.  This must be a
        nonnegative integer.  The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.  Default is 1.0.
    axis : int, optional
        The axis of the array `x` along which the filter is to be applied.
        Default is -1.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.
    Returns
    -------
    y : ndarray, same shape as `x`
        The filtered data.
    See Also
    --------
    savgol_coeffs
    Notes
    -----
    Details on the `mode` options:
        'mirror':
            Repeats the values at the edges in reverse order.  The value
            closest to the edge is not included.
        'nearest':
            The extension contains the nearest input value.
        'constant':
            The extension contains the value given by the `cval` argument.
        'wrap':
            The extension contains the values from the other end of the array.
    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and
    `window_length` is 7, the following shows the extended data for
    the various `mode` options (assuming `cval` is 0)::
        mode       |   Ext   |         Input          |   Ext
        -----------+---------+------------------------+---------
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3
    .. versionadded:: 0.14.0
    Examples
    --------
    >>> from scipy.signal import savgol_filter
    >>> np.set_printoptions(precision=2)  # For compact display.
    >>> x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])
    Filter with a window length of 5 and a degree 2 polynomial.  Use
    the defaults for all other parameters.
    >>> savgol_filter(x, 5, 2)
    array([ 1.66,  3.17,  3.54,  2.86,  0.66,  0.17,  1.  ,  4.  ,  9.  ])
    Note that the last five values in x are samples of a parabola, so
    when mode='interp' (the default) is used with polyorder=2, the last
    three values are unchanged.  Compare that to, for example,
    `mode='nearest'`:
    >>> savgol_filter(x, 5, 2, mode='nearest')
    array([ 1.74,  3.03,  3.54,  2.86,  0.66,  0.17,  1.  ,  4.6 ,  7.97])
    """
    if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
        raise ValueError("mode must be 'mirror', 'constant', 'nearest' "
                         "'wrap' or 'interp'.")

    x = np.asarray(x)
    # Ensure that x is either single or double precision floating point.
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)

    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

    if mode == "interp":
        if window_length > x.size:
            raise ValueError("If mode is 'interp', window_length must be less "
                             "than or equal to the size of x.")

        # Do not pad.  Instead, for the elements within `window_length // 2`
        # of the ends of the sequence, use the polynomial that is fitted to
        # the last `window_length` elements.
        y = convolve1d(x, coeffs, axis=axis, mode="constant")
        _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)
    else:
        # Any mode other than 'interp' is passed on to ndimage.convolve1d.
        y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)

    return y


if __name__ == '__main__':
    from lepm.build.roipoly import RoiPoly
    import matplotlib.pyplot as plt

    # demonstrate gridpts_in_polygons()
    if False:
        nn = 30
        xgrid, ygrid = np.meshgrid(np.arange(nn), np.arange(nn), sparse=False)
        maskrois = []
        plt.imshow(np.random.rand(nn, nn))
        fig = plt.gcf()
        ax = plt.gca()
        while plt.fignum_exists(1):
            polygon = RoiPoly(fig=fig, ax=ax, roicolor='r')
            maskrois.append(np.dstack((polygon.allxpoints, polygon.allypoints))[0])

        print('maskrois = ', maskrois)
        print('maskrois[0] = ', maskrois[0])
        mask = gridpts_in_polygons(xgrid, ygrid, maskrois)
        plt.imshow(mask, interpolation='nearest')
        plt.savefig('/Users/npmitchell/Desktop/test.png')

    if True:
        arr = np.random.rand(10, 10)
        arr[0, 3] = np.nan
        arr[6, 4] = np.nan
        mask = np.isnan(arr)
        print('np.flatnonzero(mask) = ', np.flatnonzero(mask))
        arr = fill_nans_with_interpolation(arr)
        print('arr = ', arr)
        plt.imshow(arr)
        plt.show()

