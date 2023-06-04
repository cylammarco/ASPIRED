#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""To construct the line spread function"""

from typing import Union

import numpy as np
from astropy.modeling import fitting, models


def build_line_spread_profile(
    spectrum2D: np.ndarray,
    trace: Union[list, np.ndarray],
    trace_width: int = 15,
):
    """
    build an empirical LSP from data

    Parameters
    ----------
    spectrum2D: 2D numpy array (M, N) or (N, M)
        The 2D spectral image.
    trace: 1D list or array (M or N)
        The trace of the spectrum in pixel coordinate.
    trace_width: float or int
        The distance from the trace to be used for building a LSF.

    """

    trace = np.asarray(trace)
    _a, _b = np.shape(spectrum2D)
    if _a == len(trace):
        spatial_size = _b
        # rotate here so the for loop will go across the image spatially
        _spectrum2D = spectrum2D
    elif _b == len(trace):
        spatial_size = _a
        _spectrum2D = np.rot90(spectrum2D)
    else:
        raise ValueError(
            f"length of trace ({len(trace)}) is different from the lengths "
            "in both dimensions of the spectral image "
            f"({np.shape(spectrum2D)})."
        )

    # Get the centre of the upsampled spectrum
    first_pix = trace - trace_width
    last_pix = trace + trace_width + 1

    first_pix = np.around(first_pix).astype("int")
    last_pix = np.around(last_pix).astype("int")

    spectrum = np.zeros((len(trace), int(2 * trace_width + 1)))

    # compute ONE sigma for each trace
    for i, spec in enumerate(_spectrum2D):
        if first_pix[i] < 0:
            start = 0
            start_pad = start - first_pix[i]
        else:
            start = first_pix[i]
            start_pad = 0
        if last_pix[i] > spatial_size:
            end = spatial_size
            end_pad = last_pix[i] - spatial_size
        else:
            end = last_pix[i]
            end_pad = 0
        spectrum[i] = np.concatenate(
            (np.zeros(start_pad), spec[start:end], np.zeros(end_pad))
        )

    line_spread_profile = np.nanmedian(spectrum, axis=0)
    line_spread_profile[np.isnan(line_spread_profile)] = np.nanmin(
        line_spread_profile
    )
    line_spread_profile -= np.nanmin(line_spread_profile)

    return line_spread_profile


def get_line_spread_function(
    trace: Union[list, np.ndarray],
    line_spread_profile: Union[list, np.ndarray],
    bounds: dict = None,
):
    """
    function refers to the fitted model

    Parameters
    ----------
    trace: 1D list or array (M)
        The trace of the spectrum in pixel coordinate.
    line_spread_profile: 1D list or array (N)
        The line spread profile to be fitted with a gaussian and a linear
        background.
    bounds: dict
        Limits of the gaussian function: amplitude, mean and stddev.

    """

    if bounds is None:
        bounds = {}

    # impose some weak constraint on the amplitude
    if "amplitude" not in bounds:
        bounds["amplitude"] = [np.nanmedian(line_spread_profile), None]

    elif bounds["amplitude"] is None:
        bounds["amplitude"] = [np.nanmedian(line_spread_profile), None]

    else:
        pass

    # impose some weak constraint on the mean
    if "mean" not in bounds:
        bounds["mean"] = [0, None]

    elif bounds["mean"] is None:
        bounds["mean"] = [0, None]

    else:
        pass

    # impose some weak constraint on the standard deviation
    if "stddev" not in bounds:
        bounds["stddev"] = [0.5, 10.0]

    elif bounds["stddev"] is None:
        bounds["stddev"] = [0.5, 10.0]

    else:
        pass

    # construct the guassian and background profile
    gauss_prof = models.Gaussian1D(
        amplitude=np.nanmax(line_spread_profile),
        mean=np.nanmean(trace),
        stddev=2.5,
        bounds=bounds,
    )
    bkg_prof = models.Linear1D(
        slope=0.0,
        intercept=np.nanpercentile(line_spread_profile, 5.0),
    )

    # combined profile
    total_prof = gauss_prof + bkg_prof

    pix = (
        np.arange(len(line_spread_profile))
        - len(line_spread_profile) // 2
        + np.nanmean(trace)
    )

    # Fit the profile
    fitter = fitting.LevMarLSQFitter()
    fitted_profile_func = fitter(
        total_prof,
        pix,
        line_spread_profile,
    )

    return fitted_profile_func
