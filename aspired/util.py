import logging
import numpy as np
from scipy import ndimage


def bfixpix(data, badmask, n=4, retdat=False):
    """
    Replace pixels flagged as nonzero in a bad-pixel mask with the
    average of their nearest four good neighboring pixels.

    Based on Ian Crossfield's code
    https://www.lpl.arizona.edu/~ianc/python/_modules/nsdata.html#bfixpix

    Parameters
    ----------
    data: numpy array (N, M)

    badmask: numpy array (N, M)
        Bad pixel mask.
    n: int
        number of nearby, good pixels to average over
    retdat: bool
        If True, return an array instead of replacing-in-place and do
        _not_ modify input array `data`.  This is always True if a 1D
        array is input!

    Returns
    -------
    another numpy array (if retdat is True)

    """

    nx, ny = data.shape

    badx, bady = np.nonzero(badmask)
    nbad = len(badx)

    if retdat:
        data = np.array(data, copy=True)

    for i in range(nbad):
        rad = 0
        numNearbyGoodPixels = 0

        while numNearbyGoodPixels < n:
            rad += 1
            xmin = max(0, badx[i] - rad)
            xmax = min(nx, badx[i] + rad)
            ymin = max(0, bady[i] - rad)
            ymax = min(ny, bady[i] + rad)
            x = np.arange(nx)[xmin:xmax + 1]
            y = np.arange(ny)[ymin:ymax + 1]
            yy, xx = np.meshgrid(y, x)

            rr = abs(xx + 1j * yy) * (1. -
                                      badmask[xmin:xmax + 1, ymin:ymax + 1])
            numNearbyGoodPixels = (rr > 0).sum()

        closestDistances = np.unique(np.sort(rr[rr > 0])[0:n])
        numDistances = len(closestDistances)
        localSum = 0.
        localDenominator = 0.
        for j in range(numDistances):
            localSum += data[xmin:xmax + 1,
                             ymin:ymax + 1][rr == closestDistances[j]].sum()
            localDenominator += (rr == closestDistances[j]).sum()

        data[badx[i], bady[i]] = 1.0 * localSum / localDenominator

    if retdat:
        ret = data
    else:
        ret = None

    return ret


def create_cutoff_mask(data,
                       cutoff=60000.,
                       grow=False,
                       iterations=1,
                       diagonal=False):
    """
    Create a simple mask from a numpy.ndarray, pixel values above
    (or below) the specified cutoff value(s) are masked as *BAD* pixels as
    True. If only one value is given, it will be treated as the upper limit.

    Parameters
    ----------
    data: numpy.ndarray
        Image data to be used for generating saturation mask
    cutoff: float
        This sets the (lower and) upper limit of electron count.
    grow: bool
        Set to True to grow the mask, see `grow_mask()`
    iterations: int
        The number of pixel growth along the Cartesian axes directions.
    diagonal: boolean
        Set to True to grow in the diagonal directions.

    Return
    ------
    cutoff_mask: numpy.ndarray
        Any pixel outside the cutoff values will be masked as True (bad).

    """

    if isinstance(cutoff, (list, np.ndarray)):

        if len(cutoff) == 2:

            lower_limit = cutoff[0]
            upper_limit = cutoff[1]

        else:

            err_msg = "Please supply a list or array for the cutoff. " +\
                "The given cutoff is {} and and a size of {}.".format(
                    cutoff, len(cutoff))
            logging.error(err_msg)
            raise RuntimeError(err_msg)

    elif isinstance(cutoff, (int, float)):

        lower_limit = -1e10
        upper_limit = cutoff

    else:

        err_msg = "Please supply a numeric value for the cutoff. " +\
            "The given cutoff is {} of type {}.".format(cutoff, type(cutoff))
        logging.error(err_msg)
        raise RuntimeError(err_msg)

    cutoff_mask = (data > upper_limit) | (data < lower_limit)

    if grow:

        cutoff_mask = grow_mask(cutoff_mask,
                                iterations=iterations,
                                diagonal=diagonal)

    return cutoff_mask


def create_bad_mask(data, grow=False, iterations=1, diagonal=False):
    """
    Create a simple mask from a 2D numpy.ndarray, pixel with non-numeric
    values will be masked as bad pixels (True).

    Parameters
    ----------
    data: numpy.ndarray
        Image data to be used for generating saturation mask
    grow: bool
        Set to True to grow the mask, see `grow_mask()`
    iterations: int
        The number of pixel growth along the Cartesian axes directions.
    diagonal: boolean
        Set to True to grow in the diagonal directions.

    Return
    ------
    bad_mask: numpy.ndarray
        Any pixel outside the cutoff values will be masked as True (bad).

    """

    bad_mask = ~np.isfinite(data) | np.isnan(data)

    if grow:

        bad_mask = grow_mask(mask=bad_mask,
                             iterations=iterations,
                             diagonal=diagonal)

    return bad_mask


def grow_mask(mask, iterations, diagonal):
    """
    This extends the mask by the given "iterations".

    The schematic of the combination of iterations and diagonal parameters to
    grow from 1 pixel to 5 by 5:

    .. code-block:: python

        0 0 0 0 0                                         0 0 0 0 0
        0 0 0 0 0     iterations = 1, diagonal = False    0 0 1 0 0
        0 0 1 0 0     ------------------------------>     0 1 1 1 0
        0 0 0 0 0                                         0 0 1 0 0
        0 0 0 0 0                                         0 0 0 0 0

        0 0 0 0 0                                         0 0 0 0 0
        0 0 0 0 0     iterations = 1, diagonal = True     0 1 1 1 0
        0 0 1 0 0     ------------------------------>     0 1 1 1 0
        0 0 0 0 0                                         0 1 1 1 0
        0 0 0 0 0                                         0 0 0 0 0

        0 0 0 0 0                                         0 0 1 0 0
        0 0 0 0 0     iterations = 2, diagonal = False    0 1 1 1 0
        0 0 1 0 0     ------------------------------>     1 1 1 1 1
        0 0 0 0 0                                         0 1 1 1 0
        0 0 0 0 0                                         0 0 1 0 0

        0 0 0 0 0                                         1 1 1 1 1
        0 0 0 0 0     iterations = 2, diagonal = True     1 1 1 1 1
        0 0 1 0 0     ------------------------------>     1 1 1 1 1
        0 0 0 0 0                                         1 1 1 1 1
        0 0 0 0 0                                         1 1 1 1 1

        These two will arrive at the same final mask.

        0 0 0 0 0                                         0 1 1 1 0
        0 0 1 0 0     iterations = 1, diagonal = True     1 1 1 1 1
        0 1 1 1 0     ------------------------------>     1 1 1 1 1
        0 0 1 0 0                                         1 1 1 1 1
        0 0 0 0 0                                         0 1 1 1 0

        0 0 0 0 0                                         0 1 1 1 0
        0 1 1 1 0     iterations = 1, diagonal = False    1 1 1 1 1
        0 1 1 1 0     ------------------------------>     1 1 1 1 1
        0 1 1 1 0                                         1 1 1 1 1
        0 0 0 0 0                                         0 1 1 1 0

    Parameters
    ----------
    mask: numpy.ndarray
        Input mask.
    iterations: int
        The number of pixel growth along the Cartesian axes directions.
    diagonal: boolean
        Set to True to grow in the diagonal directions.

    """

    if diagonal:

        struct = ndimage.generate_binary_structure(2, 2)

    else:

        struct = ndimage.generate_binary_structure(2, 1)

    mask_grown = ndimage.binary_dilation(input=mask,
                                         structure=struct,
                                         iterations=iterations)

    return mask_grown
