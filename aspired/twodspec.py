from functools import partial

import numpy as np
from scipy import signal
from scipy import interpolate as itp
from scipy.optimize import curve_fit
from astropy.io import fits
from matplotlib import pyplot as plt
try:
    from astroscrappy import detect_cosmics
except ImportError:
    warn(AstropyWarning(
        'astroscrappy is not present, so ap_trace will clean ' +
        'cosmic rays with a 2D-median filter of size 5.'
        ))
    detect_cosmics = partial(signal.medfilt2d, kernel_size=5)

try:
    from matplotlib import pyplot as plt
except ImportError:
    warn(AstropyWarning(
        'matplotlib is not present, diagnostic plots cannot be generated.'
        ))


def _gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function, for internal use only
    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """

    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def _find_peaks(img, spec_size, spatial_size, ydata, ztot, f_height,
                display=False):
    """
    Identify peaks assuming the spatial and spectral directions are
    aligned with the X and Y direction within a few degrees.
    ----------
    img : 2-d numpy array
        The data to evaluate the Gaussian over
    Saxis : int
        Set which axis the spatial dimension is along. 1 = Y axis, 0 = X.
        (Default is 1)
    Waxis : int
        The perpendicular axis of Saxis
    f_height : float
        The minimum intensity as a fraction of maximum height
    
    Returns
    -------
    peaks_y :
        Array or float of the pixel values of the detected peaks
    heights_y :
        Array or float of the integrated counts at the peaks 
    """
    
    # get the height thershold
    height = max(ztot) * f_height

    # identify peaks 
    peaks_y, heights_y = signal.find_peaks(ztot, height=height)
    heights_y = heights_y['peak_heights']

    # sort by strength
    mask = np.argsort(heights_y)
    peaks_y = peaks_y[mask][::-1]
    heights_y = heights_y[mask][::-1]

    # display disgnostic plot
    if display:
        # set a side-by-side subplot
        fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(10,10))
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.08, wspace=0)

        # show the image on the left
        ax0.cla()
        ax0.imshow(
            np.log10(img),
            origin='lower',
            interpolation="nearest",
            aspect='auto'
            )
        ax0.set_xlim(0, spatial_size)
        ax0.set_ylim(0, spec_size)
        ax0.set_xlabel('Spectral Direction / pixel')
        ax0.set_ylabel('Spatial Direction / pixel')

        # plot the integrated count and the detected peaks on the right
        ax1.cla()
        ax1.plot(ztot, ydata, color='black')
        ax1.scatter(heights_y, ydata[peaks_y])
        ax1.set_xlim(min(ztot[ztot>0]),max(ztot))
        ax1.set_ylim(0, len(ztot))
        ax1.set_xlabel('Integrated Count')
        ax1.set_xscale('log')
        plt.show()
    
    return peaks_y, heights_y


def ap_trace(img, nsteps=20, spatial_mask=(1, ), spec_mask=(1,0),
             cosmic=True, n_spec=1, recenter=False, prevtrace=(0, ),
             bigbox=8, Saxis=1, nomessage=False, display=False):
    """
    Trace the spectrum aperture in an image
    Assumes wavelength axis is along the X, spatial axis along the Y.
    Chops image up in bins along the wavelength direction, fits a Gaussian
    within each bin to determine the spatial center of the trace. Finally,
    draws a cubic spline through the bins to up-sample the trace.
    Parameters
    ----------
    img : 2d numpy array
        This is the image, stored as a normal numpy array. Can be read in
        using astropy.io.fits like so:
        >>> hdu = fits.open('file.fits')  # doctest: +SKIP
        >>> img = hdu[0].data  # doctest: +SKIP
    nsteps : int, optional
        Keyword, number of bins in X direction to chop image into. Use
        fewer bins if ap_trace is having difficulty, such as with faint
        targets (default is 20, minimum is 4)
    fmask : array-like, optional
        A list of illuminated rows in the spatial direction (Y), as
        returned by flatcombine.
    recenter : bool, optional
        Set to True to use previous trace, but allow small shift in
        position. Currently only allows linear shift (Default is False)
    bigbox : float, optional
        The number of sigma away from the main aperture to allow to trace
    Saxis : int, optional
        Set which axis the spatial dimension is along. 1 = Y axis, 0 = X.
        (Default is 1)

    Returns
    -------
    my : array
        The spatial (Y) positions of the trace, interpolated over the
        entire wavelength (X) axis
    y_sigma : array
        The sigma measured at the nsteps 
    """

    # define the wavelength axis
    Waxis = 0
    # add a switch in case the spatial/wavelength axis is swapped
    if Saxis is 0:
        Waxis = 1
    if not nomessage:
        print('Tracing Aperture using nsteps=' + str(nsteps))

    # the valid y-range of the chip
    if (len(spatial_mask) > 1):
        img = img[spatial_mask]

    if (len(spec_mask) > 1):
        img = img[:,spec_mask]

    # get the length in the spectral and spatial directions
    spec_size = np.shape(img)[Waxis]
    spatial_size = np.shape(img)[Saxis]

    # the valid y-range of the chip (an array of int)
    ydata = np.arange(spec_size)
    ztot = np.sum(img, axis=Saxis)

    # need at least 4 samples along the trace. sometimes can get away with very few
    if (nsteps < 4):
        nsteps = 4

    # clean cosmic rays
    if cosmic:
        img = detect_cosmics(img)
        # astroscrappy returns [mask, np.array]
        # medfilt2d returns np.array
        if type(img)==tuple:
            img = img[1]


    # detect peaks by summing in the spatial direction
    peak, peak_height = _find_peaks(
        img, spec_size, spatial_size, ydata, ztot, 0.05, display=False
        )

    if display:
        fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(10,10))
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.08, wspace=0)

        # show the image on the left
        ax0.imshow(
            np.log10(img),
            origin='lower',
            interpolation="nearest",
            aspect='auto'
            )
        ax0.set_xlim(0, spatial_size)
        ax0.set_ylim(0, spec_size)
        ax0.set_xlabel('Spectral Direction / pixel')
        ax0.set_ylabel('Spatial Direction / pixel')

        # plot the integrated count and the detected peaks on the right
        ax1.plot(ztot, ydata, color='black')
        ax1.set_xlim(min(ztot[ztot>0]),max(ztot))
        ax1.set_ylim(0, len(ztot))
        ax1.set_xlabel('Integrated Count')
        ax1.set_xscale('log')
        plt.gca().set_prop_cycle(None)

    my = np.zeros((n_spec, spatial_size))
    y_sigma = np.zeros((n_spec, nsteps))

    # trace each individual spetrum one by one
    for i in range(n_spec):

        peak_guess = [peak_height[i], np.nanmedian(ztot), peak[i], 2.]

        # use middle of previous trace as starting guess
        if (recenter is True) and (len(prevtrace) > 10):
            peak_guess[2] = np.nanmedian(prevtrace)

        # fit a Gaussian to peak
        try:
            pgaus, pcov = curve_fit(
                _gaus,
                ydata[np.isfinite(ztot)],
                ztot[np.isfinite(ztot)],
                p0=peak_guess,
                bounds=((0., 0., peak_guess[2]-5, 1.),
                        (np.inf, np.inf, peak_guess[2]+5, 3.))
                )
        except:
            if not nomessage:
                print('Spectrum ' + str(i) + ' of ' + str(n_spec) +
                      ' is likely to be (1) too faint, (2) in a crowed'
                      ' field, or (3) an extended source. Automated' +
                      ' tracing is sub-optimal. Please (1) reduce n_spec,' +
                      ' or (2) reduce n_steps, or (3) provide prevtrace.')

        if display:
            ax1.plot(
                _gaus(ydata, pgaus[0], pgaus[1], pgaus[2], pgaus[3]),
                ydata,
                label='Spectrum ' + str(i+1)
                )

        # only allow data within a box around this peak
        ydata2 = ydata[np.where((ydata >= pgaus[2] - pgaus[3] * bigbox) &
                                (ydata <= pgaus[2] + pgaus[3] * bigbox))]
        yi = np.arange(spec_size)[ydata2]

        # define the X-bin edges
        xbins = np.linspace(0, spatial_size, nsteps)
        ybins = np.zeros_like(xbins)
        ybins_sigma = np.zeros_like(xbins)

        # loop through each bin
        for j in range(0, len(xbins) - 1):
            # fit gaussian w/j each window
            if Saxis is 1:
                zi = np.sum(
                    img[ydata2,
                        int(np.floor(xbins[j])):int(np.ceil(xbins[j + 1]))],
                    axis=Saxis)
            else:
                zi = np.sum(
                    img[int(np.floor(xbins[j])):int(np.ceil(xbins[j +
                                                                  1])), ydata2],
                    axis=Saxis)

            # fit gaussian w/j each window
            if sum(zi) == 0:
                break
            else:
                pguess = [np.nanmax(zi), np.nanmedian(zi), yi[np.nanargmax(zi)], 2.]
            try:
                popt, pcov = curve_fit(_gaus, yi, zi, p0=pguess)
            except:
                if not nomessage:
                  print('Step ' + str(j+1) + ' of ' + str(nsteps) +
                        ' of spectrum ' + str(i+1) + ' of ' + str(n_spec) +
                        ' cannot be fitted.')
                break

            # if the peak is lower than background, sigma is too broad or
            # gaussian fits off chip, then use chip-integrated answer
            if ((popt[2] <= min(ydata) + 25) or
                (popt[2] >= max(ydata) - 25) or
                (popt[0] < 0) or
                (popt[3] < 0) or
                (popt[3] > 10)):
                ybins[j] = pgaus[2]
                popt = pgaus
                if not nomessage:
                    print('Step ' + str(j+1) + ' of ' + str(nsteps) +
                          ' of spectrum ' + str(i+1) + ' of ' + str(n_spec) +
                          ' has a poor fit. Initial guess is used instead.')
            else:
                ybins[j] = popt[2]
                ybins_sigma[j] = popt[3]


        # recenter the bin positions, trim the unused bin off in Y
        mxbins = (xbins[:-1] + xbins[1:]) / 2.
        mybins = ybins[:-1]

        # run a cubic spline thru the bins
        ap_spl = itp.UnivariateSpline(mxbins, mybins, ext=0, k=3)

        # interpolate the spline to 1 position per column
        mx = np.arange(0, spatial_size)
        my[i] = ap_spl(mx)
        y_sigma[i] = ybins_sigma

        if display:
            ax0.plot(mx, my[i])

        if not nomessage:
            if np.sum(ybins_sigma) == 0:
                print('Spectrum ' + str(i+1) + ' of ' + str(n_spec) +
                      ' is likely to be (1) too faint, (2) in a crowed'
                      ' field, or (3) an extended source. Automated' +
                      ' tracing is sub-optimal. Please disable multi-source' +
                      ' mode and (1) reduce n_spec, or (2) reduce n_steps,' +
                      '  or (3) provide prevtrace, or (4) all of above.')

            print('Spectrum ' + str(i+1) + ' : Trace gaussian width = ' +
                  str(ybins_sigma) + ' pixels')

    ax1.legend()
    plt.show()

    # add the minimum pixel value from fmask before returning
    if len(spatial_mask)>1:
        my += min(spatial_mask)

    return my, y_sigma
