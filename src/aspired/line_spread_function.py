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

    # If trace provided as sparse (x, y) points, interpolate to full length
    if trace.ndim == 2 and trace.shape[1] == 2:
        xs = trace[:, 0]
        ys = trace[:, 1]
        # Choose axis length that matches x coordinate range best
        target_len = _b if np.nanmax(xs) <= _b - 1 else _a
        full_x = np.arange(target_len)
        trace = np.interp(full_x, xs, ys)

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

    # Get window around trace
    first_pix = trace - trace_width
    last_pix = trace + trace_width + 1

    first_pix = np.floor(first_pix).astype(int)
    last_pix = np.ceil(last_pix).astype(int)

    win_len = int(2 * trace_width + 1)
    spectrum = np.zeros((len(trace), win_len))

    for i, spec in enumerate(_spectrum2D):
        start = max(first_pix[i], 0)
        end = min(last_pix[i], spatial_size)
        seg = spec[start:end]
        pad_left = max(0, -first_pix[i])
        pad_right = max(0, last_pix[i] - spatial_size)
        row = np.concatenate((np.zeros(pad_left), seg, np.zeros(pad_right)))
        # enforce fixed window length
        if len(row) > win_len:
            row = row[:win_len]
        elif len(row) < win_len:
            row = np.concatenate((row, np.zeros(win_len - len(row))))
        spectrum[i] = row

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
