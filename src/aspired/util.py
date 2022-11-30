# -*- coding: utf-8 -*-
import logging
import numpy as np
from scipy import interpolate as itp
from scipy import ndimage
from statsmodels.nonparametric.smoothers_lowess import lowess


def bfixpix(data, badmask, n=4, retdat=False):
    """
    Replace pixels flagged as nonzero in a bad-pixel mask with the
    average of their nearest four good neighboring pixels.

    Taken and updated from Ian Crossfield's code
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
            x = np.arange(nx)[xmin : xmax + 1]
            y = np.arange(ny)[ymin : ymax + 1]
            yy, xx = np.meshgrid(y, x)

            rr = abs(xx + 1j * yy) * (
                1.0 - badmask[xmin : xmax + 1, ymin : ymax + 1]
            )
            numNearbyGoodPixels = (rr > 0).sum()

        closestDistances = np.unique(np.sort(rr[rr > 0])[0:n])
        numDistances = len(closestDistances)
        localSum = 0.0
        localDenominator = 0.0

        for j in range(numDistances):
            localSum += data[xmin : xmax + 1, ymin : ymax + 1][
                rr == closestDistances[j]
            ].sum()
            localDenominator += (rr == closestDistances[j]).sum()

        data[badx[i], bady[i]] = 1.0 * localSum / localDenominator

    if retdat:

        ret = data

    else:

        ret = None

    return ret


def create_cutoff_mask(
    data, cutoff=65535.0, grow=False, iterations=1, diagonal=False
):
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

            err_msg = (
                "Please supply a list or array for the cutoff. "
                + "The given cutoff is {} and and a size of {}.".format(
                    cutoff, len(cutoff)
                )
            )
            logging.error(err_msg)
            raise RuntimeError(err_msg)

    elif isinstance(cutoff, (int, float)):

        lower_limit = -1e10
        upper_limit = cutoff

    else:

        err_msg = (
            "Please supply a numeric value for the cutoff. "
            + "The given cutoff is {} of type {}.".format(cutoff, type(cutoff))
        )
        logging.error(err_msg)
        raise RuntimeError(err_msg)

    cutoff_mask = (data > upper_limit) | (data < lower_limit)

    if grow:

        cutoff_mask = grow_mask(
            cutoff_mask, iterations=iterations, diagonal=diagonal
        )

    if (data > upper_limit).any():

        logging.warning("Saturated pixels detected.")
        return cutoff_mask, True

    else:

        return cutoff_mask, False


def create_bad_pixel_mask(data, grow=False, iterations=1, diagonal=False):
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
    bad_pixel_mask: numpy.ndarray
        Any pixel outside the cutoff values will be masked as True (bad).

    """

    bad_pixel_mask = ~np.isfinite(data) | np.isnan(data)

    if grow:

        bad_pixel_mask = grow_mask(
            mask=bad_pixel_mask, iterations=iterations, diagonal=diagonal
        )

    if bad_pixel_mask.any():

        logging.warning("Bad pixels detected.")
        return bad_pixel_mask, True

    else:

        return bad_pixel_mask, False


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

    mask_grown = ndimage.binary_dilation(
        input=mask, structure=struct, iterations=iterations
    )

    return mask_grown


def get_continuum(x, y, **kwargs):
    """
    This is a wrapper function of the lowess function from statsmodels that
    uses a different frac default value that is more appropriate in getting
    a first guess continuum which reject "outliers" much more aggressively.
    This function also takes in values in a Pythonic way that of providing
    arguments: "first x then y".

    Parameters
    ----------
    x: list or numpy.ndarray
        Absicissa (conventionally the first number of a coordinate pair)
    y: list or numpy.ndarray
        Ordinate (conventionally the second number of a coordinate pair)
    **kwargs: dict
        The keyword arguments for the lowess function.

    """

    if "frac" not in kwargs:

        kwargs["frac"] = 0.15

    if "return_sorted" not in kwargs:

        kwargs["return_sorted"] = False

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(y) & ~np.isnan(y) & (y > 0.0)

    x_smoothed = x[mask]
    y_smoothed = lowess(y[mask], x_smoothed, **kwargs)

    return itp.interp1d(
        x_smoothed, y_smoothed, kind="cubic", fill_value="extrapolate"
    )(x)
