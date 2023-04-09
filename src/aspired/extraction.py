#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Functions for spectral extraction"""

import numpy as np
from scipy import ndimage, special
from statsmodels.nonparametric.smoothers_lowess import lowess

from .util import bfixpix, gaus


def tophat_extraction(
    source_slice: np.ndarray,
    sky_source_slice: np.ndarray,
    var_sky: float,
    pix_frac: float,
    gain: float,
    sky_width_dn: int,
    sky_width_up: int,
    width_dn: int,
    width_up: int,
    source_bad_mask: np.ndarray = None,
    sky_source_bad_mask: np.ndarray = None,
):
    """
    Make sure the counts are the number of photoelectrons or an equivalent
    detector unit, and not counts per second.

    Parameters
    ----------
    source_slice: 1-d numpy array (N)
        The counts along the profile for aperture extraction.
    sky_source_slice: 1-d numpy array (M)
        Count of the fitted sky along the pix, has to be the same
        length
    var_sky: float
        The variance of the sky_source_slice (measured, not fitted)
    pix_frac: float
        The decimal places of the centroid.
    gain: float
        Detector gain, in electrons per ADU
    sky_width_dn: int
        Number of pixels used for sky modelling on the lower side of the
        spectrum.
    sky_width_up: int
        Number of pixels used for sky modelling on the upper side of the
        spectrum.
    width_dn: int
        Number of pixels used for aperture extraction on the lower side
        of the spectrum.
    width_up: int
        Number of pixels used for aperture extraction on the upper side
        of the spectrum.
    source_bad_mask: 1-d numpy array (N, default: None)
        Masking the unusable pixels for extraction.
    sky_source_bad_mask: 1-d numpy array (M, default: None)
        Masking the unusable pixels for sky subtraction.

    """

    if source_bad_mask is not None:
        _source_slice = source_slice[~source_bad_mask]
    else:
        _source_slice = source_slice

    if source_bad_mask is not None:
        _sky_source_slice = sky_source_slice[~sky_source_bad_mask]
    else:
        _sky_source_slice = sky_source_slice

    # Get the total count
    source_plus_sky = (
        np.nansum(_source_slice)
        - pix_frac * _source_slice[0]
        - (1 - pix_frac) * _source_slice[-1]
    )

    # finally, compute the error in this pixel
    sky = (
        np.nansum(_sky_source_slice)
        - pix_frac * _sky_source_slice[0]
        - (1 - pix_frac) * _sky_source_slice[-1]
    )

    # number of bkgd pixels
    nB = sky_width_dn + sky_width_up - np.sum(np.isnan(_sky_source_slice))
    # number of aperture pixels
    nA = width_dn + width_up - np.sum(np.isnan(_source_slice))

    # Based on aperture phot err description by F. Masci,
    # Caltech:
    # http://wise2.ipac.caltech.edu/staff/fmasci/
    #   ApPhotUncert.pdf
    # All the counts are in per second already, so need to
    # multiply by the exposure time when computing the
    # uncertainty
    _signal = source_plus_sky - sky
    _noise = np.sqrt(_signal / gain + (nA + nA**2.0 / nB) * var_sky)

    return _signal, _noise, False


def optimal_extraction_horne86(
    source_slice: np.ndarray,
    sky: np.ndarray,
    profile: np.ndarray = None,
    tol: float = 1e-6,
    max_iter: int = 99,
    gain: float = 1.0,
    readnoise: float = 0.0,
    cosmicray_sigma: float = 5.0,
    forced: bool = False,
    variances: np.ndarray = None,
    bad_mask: np.ndarray = None,
):
    """
    Make sure the counts are the number of photoelectrons or an equivalent
    detector unit, and not counts per second or ADU.

    Iterate to get the optimal signal. Following the algorithm on
    Horne, 1986, PASP, 98, 609 (1986PASP...98..609H). The 'steps' in the
    inline comments are in reference to this article.

    The LOWESS setting can be found at:
    https://www.statsmodels.org/dev/generated/
        statsmodels.nonparametric.smoothers_lowess.lowess.html

    Parameters
    ----------
    source_slice: 1D numpy.ndarray (N)
        The counts along the profile for extraction, including the sky
        regions to be fitted and subtracted from. (NOT count per second)
    sky: 1D numpy.ndarray (N)
        Count of the fitted sky along the pix, has to be the same
        length as pix
        The width of the Gaussian
    profile: 1D numpy.ndarray (N)
        The gaussian profile (only used if model='gauss')
    tol: float
        The tolerance limit for the covergence
    max_iter: int
        The maximum number of iteration in the optimal extraction
    gain: float (Default: 1.0)
        Detector gain, in electrons per ADU
    readnoise: float
        Detector readnoise, in electrons.
    cosmicray_sigma: int (Default: 5)
        Sigma-clipping threshold for cleaning & cosmic ray rejection.
    forced: bool
        Forced extraction with the given weights.
    variances: 1D numpy.ndarray (N)
        The 1/weights of used for optimal extraction, has to be the
        same length as the pix. Only used if forced is True.
    bad_mask: list or None (Default: None)
        Mask of the bad or usable pixels.

    Returns
    -------
    signal: float
        The optimal signal.
    noise: float
        The noise associated with the optimal signal.
    is_optimal: bool
        List indicating whether the extraction at that pixel was
        optimal or not. True = optimal, False = suboptimal.
    P: numpy array
        The line spread function of the extraction
    var_f: float
        The variance in the extraction.

    """

    # step 2 - initial variance estimates
    var1 = readnoise**2.0 + np.abs(source_slice) / gain

    # step 4a - extract standard spectrum
    f = source_slice - sky
    f1 = np.nansum(f)

    # step 4b - variance of standard spectrum
    v1 = 1.0 / np.nansum(1.0 / var1)

    # step 5 - construct the spatial profile
    P = profile

    f_diff = 1
    v_diff = 1
    i = 0
    is_optimal = True

    while (f_diff > tol) | (v_diff > tol):
        mask_cr = np.ones(len(P), dtype=bool)

        if bad_mask is not None:
            mask_cr = mask_cr & ~bad_mask.astype(bool)

        if forced:
            var_f = variances

        f0 = f1
        v0 = v1

        # step 6 - revise variance estimates
        # var_f is the V in Horne87
        if not forced:
            var_f = readnoise**2.0 + np.abs(P * f0 + sky) / gain

        # step 7 - cosmic ray mask, only start considering after the
        # 2nd iteration. 1 pixel is masked at a time until convergence,
        # once the pixel is masked, it will stay masked.
        if i > 1:
            ratio = (cosmicray_sigma**2.0 * var_f) / (f - P * f0) ** 2.0

            if (ratio > 1).any():
                mask_cr[np.argmax(ratio)] = False

        denom = np.nansum((P**2.0 / var_f)[mask_cr])

        # step 8a - extract optimal signal
        f1 = np.nansum((P * f / var_f)[mask_cr]) / denom

        # step 8b - variance of optimal signal
        v1 = np.nansum(P[mask_cr]) / denom

        f_diff = abs((f1 - f0) / f0)
        v_diff = abs((v1 - v0) / v0)

        i += 1

        if i == int(max_iter):
            is_optimal = False
            break

    signal = f1
    noise = np.sqrt(v1)

    return signal, noise, is_optimal, P, var_f


def optimal_extraction_marsh89(
    frame,
    residual_frame: np.ndarray,
    variance: np.ndarray,
    trace: np.ndarray,
    spectrum: np.ndarray = None,
    readnoise: float = 0.0,
    apwidth: int = 7,
    goodpixelmask: np.ndarray = None,
    npoly: int = 21,
    polyspacing: int = 1,
    pord: int = 2,
    cosmicray_sigma: float = 5.0,
    qmode: str = "slow-nearest",
    nreject: int = 100,
):
    """
    Optimally extract curved spectra taken and updated from
    Ian Crossfield's code

    https://people.ucsc.edu/~ianc/python/_modules/spec.html#superExtract,
    following Marsh 1989.

    Parameters
    ----------
    frame: 2-d Numpy array (M, N)
        The calibrated frame from which to extract spectrum. In units
        of electrons count.
    residual_frame: 2-d Numpy array (M, N)
        The sky background only frame.
    variance: 2-d Numpy array (M, N)
        Variances of pixel values in 'frame'.
    trace: 1-d numpy array (N)
        :ocation of spectral trace.
    spectrum: 1-d numpy array (M) (Default: None)
        The extracted spectrum for initial guess.
    gain: float (Default: 1.0)
        Detector gain, in electrons per ADU
    readnoise: float (Default: 0.0)
        Detector readnoise, in electrons.
    apwidth: int or list of int (default: 7)
        The size of the aperture for extraction.
    goodpixelmask : 2-d numpy array (M, N) (Default: None)
        Equals 0 for bad pixels, 1 for good pixels
    npoly: int (Default: 21)
        Number of profile to be use for polynomial fitting to evaluate
        (Marsh's "K"). For symmetry, this should be odd.
    polyspacing: float (Default: 1)
        Spacing between profile polynomials, in pixels. (Marsh's "S").
        A few cursory tests suggests that the extraction precision
        (in the high S/N case) scales as S^-2 -- but the code slows down
        as S^2.
    pord: int (Default: 2)
        Order of profile polynomials; 1 = linear, etc.
    cosmicray_sigma: int (Default: 5)
        Sigma-clipping threshold for cleaning & cosmic-ray rejection.
    qmode: str (Default: 'slow-nearest')
        How to compute Marsh's Q-matrix. Valid inputs are 'fast-linear',
        'slow-linear', 'fast-nearest', and 'slow-nearest'. These select
        between various methods of integrating the nearest-neighbor or
        linear interpolation schemes as described by Marsh; the 'linear'
        methods are preferred for accuracy. Use 'slow' if you are
        running out of memory when using the 'fast' array-based methods.
    nreject: int (Default: 100)
        Number of outlier-pixels to reject at each iteration.

    Returns
    -------
    spectrum_marsh:
        The optimal signal.
    spectrum_err_marsh:
        The noise associated with the optimal signal.
    is_optimal:
        List indicating whether the extraction at that pixel was
        optimal or not (this list is always all optimal).
    profile:
        The line spread functions of the extraction
    variance0:
        The variance in the extraction.

    """

    frame = frame.transpose()
    residual_frame = residual_frame.transpose()
    variance = variance.transpose()

    if isinstance(apwidth, (float, int)):
        # first do the aperture count
        width_dn = apwidth
        width_up = apwidth

    elif isinstance(apwidth, (list, np.ndarray)) & (len(apwidth) == 2):
        width_dn = apwidth[0]
        width_up = apwidth[1]

    else:
        width_dn = 7
        width_up = 7

    if goodpixelmask is not None:
        goodpixelmask = goodpixelmask.transpose()
        goodpixelmask = np.array(goodpixelmask, copy=True).astype(bool)

    else:
        goodpixelmask = np.ones_like(frame, dtype=bool)

    goodpixelmask *= np.isfinite(frame) * np.isfinite(variance)

    variance[~goodpixelmask] = frame[goodpixelmask].max() * 1e9
    spectral_size, spatial_size = frame.shape

    # (my 3a: mask any bad values)
    bad_residual_frame_mask = ~np.isfinite(residual_frame)
    residual_frame[bad_residual_frame_mask] = 0.0

    sky_subframe = frame - residual_frame
    # Interpolate and fix bad pixels for extraction of standard
    # spectrum -- otherwise there can be 'holes' in the spectrum from
    # ill-placed bad pixels.
    sky_subframe = bfixpix(
        sky_subframe, ~goodpixelmask, n_nearest=8, retdat=True
    )

    # Define new indices (in Marsh's appendix):
    N = pord + 1
    mm = np.tile(np.arange(N).reshape(N, 1), (npoly)).ravel()
    nn = mm.copy()
    ll = np.tile(np.arange(npoly), N)
    kk = ll.copy()
    pp = N * ll + mm
    qq = N * kk + nn

    ii = np.arange(spatial_size)  # column (i.e., spatial direction)
    jjnorm = np.linspace(-1, 1, spectral_size)  # normalized X-coordinate
    _pow = np.arange(2 * N - 1).reshape(2 * N - 1, 1, 1)
    jjnorm_pow = jjnorm.reshape(1, 1, spectral_size) ** _pow

    # Marsh eq. 9, defining centers of each polynomial:
    constant = 0.0  # What is it for???
    poly_centers = (
        np.array(trace).reshape(spectral_size, 1)
        + polyspacing * np.arange(-npoly / 2 + 1, npoly / 2 + 1)
        + constant
    )

    # Marsh eq. 11, defining Q_kij    (via nearest-neighbor interpolation)
    #    Q_kij =  max(0, min(S, (S+1)/2 - abs(x_kj - i)))
    if qmode == "fast-nearest":  # Array-based nearest-neighbor mode.
        Q = np.array(
            [
                np.zeros((npoly, spatial_size, spectral_size)),
                np.array(
                    [
                        polyspacing
                        * np.ones((npoly, spatial_size, spectral_size)),
                        0.5 * (polyspacing + 1)
                        - np.abs(
                            (
                                poly_centers - ii.reshape(spatial_size, 1, 1)
                            ).transpose(2, 0, 1)
                        ),
                    ]
                ).min(0),
            ]
        ).max(0)

    elif qmode == "slow-linear":  # Code is a mess, but it works.
        invs = 1.0 / polyspacing
        poly_centers_over_s = poly_centers / polyspacing
        xps_mat = poly_centers + polyspacing
        xms_mat = poly_centers - polyspacing
        Q = np.zeros((npoly, spatial_size, spectral_size))
        for i in range(spatial_size):
            ip05 = i + 0.5
            im05 = i - 0.5
            for j in range(spectral_size):
                for k in range(npoly):
                    xkj = poly_centers[j, k]
                    xkjs = poly_centers_over_s[j, k]
                    # xkj + polyspacing
                    xps = xps_mat[j, k]
                    # xkj - polyspacing
                    xms = xms_mat[j, k]

                    if (ip05 <= xms) or (im05 >= xps):
                        qval = 0.0
                    elif (im05) > xkj:
                        lim1 = im05
                        lim2 = min(ip05, xps)
                        qval = (lim2 - lim1) * (
                            1.0 + xkjs - 0.5 * invs * (lim1 + lim2)
                        )
                    elif (ip05) < xkj:
                        lim1 = max(im05, xms)
                        lim2 = ip05
                        qval = (lim2 - lim1) * (
                            1.0 - xkjs + 0.5 * invs * (lim1 + lim2)
                        )
                    else:
                        lim1 = max(im05, xms)
                        lim2 = min(ip05, xps)
                        qval = (
                            lim2
                            - lim1
                            + invs
                            * (
                                xkj * (-xkj + lim1 + lim2)
                                - 0.5 * (lim1 * lim1 + lim2 * lim2)
                            )
                        )
                    Q[k, i, j] = max(0, qval)

    # Code is a mess, but it's faster than 'slow' mode
    elif qmode == "fast-linear":
        invs = 1.0 / polyspacing
        xps_mat = poly_centers + polyspacing
        Q = np.zeros((npoly, spatial_size, spectral_size))
        for j in range(spectral_size):
            xkj_vec = np.tile(
                poly_centers[j, :].reshape(npoly, 1), (1, spatial_size)
            )
            xps_vec = np.tile(
                xps_mat[j, :].reshape(npoly, 1), (1, spatial_size)
            )
            xms_vec = xps_vec - 2 * polyspacing

            ip05_vec = np.tile(np.arange(spatial_size) + 0.5, (npoly, 1))
            im05_vec = ip05_vec - 1
            ind00 = (ip05_vec <= xms_vec) + (im05_vec >= xps_vec)
            ind11 = (im05_vec > xkj_vec) * ~ind00
            ind22 = (ip05_vec < xkj_vec) * ~ind00
            ind33 = ~(ind00 + ind11 + ind22)
            ind11 = ind11.nonzero()
            ind22 = ind22.nonzero()
            ind33 = ind33.nonzero()

            n_ind11 = len(ind11[0])
            n_ind22 = len(ind22[0])
            n_ind33 = len(ind33[0])

            if n_ind11 > 0:
                ind11_3d = ind11 + (np.ones(n_ind11, dtype=int) * j,)
                lim2_ind11 = np.array((ip05_vec[ind11], xps_vec[ind11])).min(0)
                Q[ind11_3d] = (
                    (lim2_ind11 - im05_vec[ind11])
                    * invs
                    * (
                        polyspacing
                        + xkj_vec[ind11]
                        - 0.5 * (im05_vec[ind11] + lim2_ind11)
                    )
                )

            if n_ind22 > 0:
                ind22_3d = ind22 + (np.ones(n_ind22, dtype=int) * j,)
                lim1_ind22 = np.array((im05_vec[ind22], xms_vec[ind22])).max(0)
                Q[ind22_3d] = (
                    (ip05_vec[ind22] - lim1_ind22)
                    * invs
                    * (
                        polyspacing
                        - xkj_vec[ind22]
                        + 0.5 * (ip05_vec[ind22] + lim1_ind22)
                    )
                )

            if n_ind33 > 0:
                ind33_3d = ind33 + (np.ones(n_ind33, dtype=int) * j,)
                lim1_ind33 = np.array((im05_vec[ind33], xms_vec[ind33])).max(0)
                lim2_ind33 = np.array((ip05_vec[ind33], xps_vec[ind33])).min(0)
                Q[ind33_3d] = (lim2_ind33 - lim1_ind33) + invs * (
                    xkj_vec[ind33]
                    * (-xkj_vec[ind33] + lim1_ind33 + lim2_ind33)
                    - 0.5 * (lim1_ind33 * lim1_ind33 + lim2_ind33 * lim2_ind33)
                )

    # 'slow' Loop-based nearest-neighbor mode: requires less memory
    else:
        Q = np.zeros((npoly, spatial_size, spectral_size))
        for k in range(npoly):
            for i in range(spatial_size):
                for j in range(spectral_size):
                    Q[k, i, j] = max(
                        0,
                        min(
                            polyspacing,
                            0.5 * (polyspacing + 1)
                            - np.abs(poly_centers[j, k] - i),
                        ),
                    )

    # Some quick math to find out which dat columns are important, and
    # which contain no useful spectral information:
    q_mask = Q.sum(0).transpose() > 0
    q_ind = q_mask.transpose().nonzero()
    q_cols = [q_ind[0].min(), q_ind[0].max()]
    q_sm = Q[:, q_cols[0] : q_cols[1] + 1, :]

    # Prepar to iteratively clip outliers
    new_bad_pixels = True
    i = -1

    spectrum_marsh = np.zeros(spectral_size)
    spectrum_err_marsh = np.zeros(spectral_size)
    is_optimal = np.zeros(spectral_size)
    profile = np.zeros((spatial_size, spectral_size))
    model_spectrum = (
        np.array(spectrum).reshape(spectral_size, 1) * profile.transpose()
    )
    model_data = model_spectrum + residual_frame
    variance0 = np.abs(model_data) + readnoise**2

    while new_bad_pixels:
        i += 1

        # Compute pixel fractions (Marsh Eq. 5):
        #     (Note that values outside the desired polynomial region
        #     have Q=0, and so do not contribute to the fit)
        inv_e_variance = (
            np.array(spectrum).reshape(spectral_size, 1) ** 2 / variance
        ).transpose()
        weighted_e = (
            sky_subframe
            * np.array(spectrum).reshape(spectral_size, 1)
            / variance
        ).transpose()  # E / var_E
        inv_e_variance_subset = inv_e_variance[q_cols[0] : q_cols[1] + 1, :]

        # Define X vector (Marsh Eq. A3):
        X = np.zeros(N * npoly)
        for q in qq:
            X[q] = (
                weighted_e[q_cols[0] : q_cols[1] + 1, :]
                * q_sm[kk[q], :, :]
                * jjnorm_pow[nn[q]]
            ).sum()

        # Define C matrix (Marsh Eq. A3)
        C = np.zeros((N * npoly, N * npoly))

        # C-matrix computation buffer (to be sure we don't miss any pixels)
        buffer = 1.1

        # Compute *every* element of C (though most equal zero!)
        for p in pp:
            qp = q_sm[ll[p], :, :]
            for q in qq:
                #  Check that we need to compute C:
                if np.abs(kk[q] - ll[p]) <= (1.0 / polyspacing + buffer):
                    if q >= p:
                        # Only compute over non-zero columns:
                        C[q, p] = (
                            q_sm[kk[q], :, :]
                            * qp
                            * jjnorm_pow[nn[q] + mm[p]]
                            * inv_e_variance_subset
                        ).sum()
                    if q > p:
                        C[p, q] = C[q, p]

        # Solve for the profile-polynomial coefficients (Marsh Eq. A4):
        if np.abs(np.linalg.det(C)) < 1e-10:
            b_soln = np.dot(np.linalg.pinv(C), X)
        else:
            b_soln = np.linalg.solve(C, X)

        a_soln = b_soln.reshape(N, npoly).transpose()

        # Define G_kj, the profile-defining polynomial profiles
        # (Marsh Eq. 8)
        g_soln = np.zeros((npoly, spectral_size))
        for n in range(npoly):
            g_soln[n] = np.polyval(
                a_soln[n, ::-1], jjnorm
            )  # reorder polynomial coef.

        # Compute the profile (Marsh eq. 6) and normalize it:
        for i in range(spatial_size):
            profile[i, :] = (Q[:, i, :] * g_soln).sum(0)

        if profile.min() < 0:
            profile[profile < 0] = 0.0
        profile /= np.nansum(profile, axis=0)
        profile[~np.isfinite(profile)] = 0.0

        # Step6: Revise variance estimates
        model_spectrum = (
            np.array(spectrum).reshape(spectral_size, 1) * profile.transpose()
        )
        model_data = model_spectrum + residual_frame
        variance0 = np.abs(model_data) + readnoise**2
        variance = variance0 / (
            goodpixelmask + 1e-9
        )  # De-weight bad pixels, avoiding infinite variance

        outlier_variances = (frame - model_data) ** 2 / variance

        if outlier_variances.max() > cosmicray_sigma**2:
            new_bad_pixels = True
            # nreject-counting on pixels within the spectral trace
            max_rejected_value = max(
                cosmicray_sigma**2,
                np.sort(outlier_variances[q_mask])[-nreject],
            )
            worst_outliers = (
                outlier_variances >= max_rejected_value
            ).nonzero()
            goodpixelmask[worst_outliers] = False
            # number_rejected = len(worst_outliers[0])
        else:
            new_bad_pixels = False
            # number_rejected = 0

        # Optimal Spectral Extraction: (Horne, Step 8)
        for i in range(spectral_size):
            aperture = np.arange(
                int(trace[i]) - width_dn, int(trace[i]) + width_up + 1
            ).astype(int)

            # Horne86 notation
            P = profile[aperture, i]
            V = variance0[i, aperture]
            D = sky_subframe[i, aperture]

            denom = np.nansum(P**2.0 / V)

            if denom == 0:
                spectrum_marsh[i] = 0.0
                spectrum_err_marsh[i] = 9e9
            else:
                spectrum_marsh[i] = np.nansum(P / V * D) / denom
                spectrum_err_marsh[i] = np.sqrt(np.nansum(P) / denom)
                is_optimal[i] = True

    return (
        spectrum_marsh,
        spectrum_err_marsh,
        is_optimal,
        profile.T,
        variance0,
    )
