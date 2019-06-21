from functools import partial

import numpy as np
from scipy import signal
from scipy import stats
from scipy import interpolate as itp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from astropy.io import fits
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
    from matplotlib.patches import Rectangle
except ImportError:
    warn(AstropyWarning(
        'matplotlib is not present, diagnostic plots cannot be generated.'
        ))


def _gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function.

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

    Parameters
    ----------
    img : 2-d numpy array
        The data to evaluate the Gaussian over
    Saxis : int, optional
        Set the axis of the spatial dimension. 1 = Y axis, 0 = X axis.
        (Default is 1, i.e. spectrum in the left-right direction.)
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
        ax1.grid()
        plt.show()
    
    return peaks_y, heights_y


def _optimal_signal(pix, xslice, sky, mu, sigma, rn, gain, display, cr_sigma):
    """
    Iterate to get optimal signal, for internal use only

    Parameters
    ----------
    pix : 1-d numpy array
        pixel number along the spatial direction
    xslice : 1-d numpy array
        ADU along the pix
    sky : 1-d numpy array
        ADU of the fitted sky along the pix
    my : float
        The center of the Gaussian
    sigma : float
        The width of the Gaussian
    rn : float
        The read noise of the detector
    gain : float
        The gain value of the detector.
    display : tuple
        Set to show diagnostic plot.
    cr_sigma : float
        The cosmic ray rejection sigma-clipping threshold.

    Returns
    -------
    signal : float
        The optimal signal. 
    noise : float
        The noise associated with the optimal signal.

    """

    # construct the Gaussian model
    gauss = _gaus(pix, 1., 0., mu, sigma)
    
    # weight function and initial values
    P = gauss / np.sum(gauss)
    signal0 = np.sum(xslice - sky)
    var0 = rn + np.abs(xslice) / gain
    variance0 = 1. / np.sum(P**2. / var0)

    for i in range(100):

        # cosmic ray mask
        mask_cr = ((xslice - sky - P*signal0)**2. < cr_sigma**2. * var0)

        # compute signal and noise
        signal1 = np.sum((P * (xslice - sky) / var0)[mask_cr]) / \
            np.sum((P**2. / var0)[mask_cr])
        var1 = rn + np.abs(P*signal1 + sky) / gain
        variance1 = 1. / np.sum((P**2. / var1)[mask_cr])

        # iterate
        if (((signal1 - signal0) / signal1 > 0.001) or 
            ((variance1 - variance0) / variance1 > 0.001)):
            signal0 = signal1
            var0 = var1
            variance0 = variance1
        else:
            break

    if i==99:
        print('Unable to obtain optimal signal, please try a longer ' + 
              'iteration or revert to unit-weighted extraction. Values ' +
              'returned (if at all) are sub-optimal at best.')

    signal = signal1
    noise = np.sqrt(variance1)

    if display:
        fit = _gaus(pix, max(xslice-sky), 0., mu, sigma) + sky
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        ax.plot(pix, xslice)
        ax.plot(pix, fit)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Count')
        #print(signal, variance)
        #print(np.sum(xslice-sky_const))
    
    return signal, noise 


def ap_trace(img, nsteps=20, Saxis=1, spatial_mask=(1, ), spec_mask=(1, ),
             cosmic=True, n_spec=1, recenter=False, prevtrace=(0, ),
             fittype='spline', order=3, bigbox=8, silence=False,
             display=False):
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
    Saxis : int, optional
        Set the axis of the spatial dimension. 1 = Y axis, 0 = X axis.
        (Default is 1, i.e. spectrum in the left-right direction.)
    spatial_mask : 1-d numpy array (M), optional
        An array of 0/1 or True/False in the spatial direction (Y).
    spec_mask : 1-d numpy array (N), optional
        An array of 0/1 or True/False in the spectral direction (X).
    cosmic : tuple, optional
        Set to apply cosmic ray removal beefore tracing using astroscrappy if
        available, otherwise with a 2D median filter of size 5. It does not
        alter the base image. (default is True)
    n_spec : int, optional
        Number of spectra to be extracted, guaranteed to return the requested
        number of trace, but not guaranteed to contain signals.
    recenter : bool, optional
        Set to True to use previous trace, allow small shift in position
        along the spatial direction. Not doing anything if prevtrace is not
        supplied. (Default is False)
    prevtrace : 1-d numpy array, optional
        Provide first guess or refitting the center with different parameters.
    fittype : string, optional
        Set to 'spline' or 'polynomial', using
        scipy.interpolate.UnivariateSpline and numpy.polyfit
    order : string, optional
        Degree of the spline or polynomial. Spline must be <= 5.
        (default is k=3)
    bigbox : float, optional
        The number of sigma away from the main aperture to allow to trace
    silence : tuple, optional
        Set to disable warning/error messages. (Default is False)
    display : tuple, optional
        Set to show diagnostic plots. (Default is True)

    Returns
    -------
    my : array (N, nspec)
        The spatial (Y) positions of the trace, interpolated over the
        entire wavelength (X) axis
    y_sigma : array (N, nspec)
        The sigma measured at the nsteps.

    """

    # define the wavelength axis
    Waxis = 0
    # add a switch in case the spatial/wavelength axis is swapped
    if Saxis is 0:
        Waxis = 1
    if not silence:
        print('Tracing Aperture using nsteps=' + str(nsteps))

    # the valid y-range of the chip
    if (len(spatial_mask) > 1):
        if Saxis is 1:
            img = img[spatial_mask]
        else:
            img = img[:,spatial_mask]

    if (len(spec_mask) > 1):
        if Saxis is 1:
            img = img[:,spec_mask]
        else:
            img = img[spec_mask]

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
    y_sigma = np.zeros((n_spec, spatial_size))

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
                bounds=((0., 0., peak_guess[2]-10, 0.),
                        (np.inf, np.inf, peak_guess[2]+10, np.inf))
                )
        except:
            if not silence:
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
                if not silence:
                  print('Step ' + str(j+1) + ' of ' + str(nsteps) +
                        ' of spectrum ' + str(i+1) + ' of ' + str(n_spec) +
                        ' cannot be fitted.')
                break

            # if the peak is lower than background, sigma is too broad or
            # gaussian fits off chip, then use chip-integrated answer
            if ((popt[0] < 0) or
                (popt[3] < 0) or
                (popt[3] > 10)):
                ybins[j] = pgaus[2]
                popt = pgaus
                if not silence:
                    print('Step ' + str(j+1) + ' of ' + str(nsteps) +
                          ' of spectrum ' + str(i+1) + ' of ' + str(n_spec) +
                          ' has a poor fit. Initial guess is used instead.')
            else:
                ybins[j] = popt[2]
                ybins_sigma[j] = popt[3]

        # recenter the bin positions, trim the unused bin off in Y
        mxbins = (xbins[:-1] + xbins[1:]) / 2.
        mybins = ybins[:-1]
        mx = np.arange(0, spatial_size)

        if (fittype=='spline'):
            # run a cubic spline thru the bins
            interpolated = itp.UnivariateSpline(mxbins, mybins, ext=0, k=order)
            # interpolate 1 position per column
            my[i] = interpolated(mx)

        elif (fittype=='polynomial'):
            # linear fit
            npfit = np.polyfit(mxbins, mybins, deg=order)
            # interpolate 1 position per column
            my[i] = np.polyval(npfit, mx)

        else:
            if not silence:
                print('Unknown fitting type, please choose from ' + 
                      '(1) \'spline\'; or (2) \'polynomial\'.')


        # get the uncertainties in the spatial direction along the spectrum
        slope, intercept, r_value, p_value, std_err =\
                stats.linregress(mxbins, ybins_sigma[:-1])
        y_sigma[i] = slope * mx + intercept

        if display:
            ax0.plot(mx, my[i])

        if not silence:
            if np.sum(ybins_sigma) == 0:
                print('Spectrum ' + str(i+1) + ' of ' + str(n_spec) +
                      ' is likely to be (1) too faint, (2) in a crowed'
                      ' field, or (3) an extended source. Automated' +
                      ' tracing is sub-optimal. Please disable multi-source' +
                      ' mode and (1) reduce n_spec, or (2) reduce n_steps,' +
                      '  or (3) provide prevtrace, or (4) all of above.')

            print('Spectrum ' + str(i+1) + ' : Trace gaussian width = ' +
                  str(ybins_sigma) + ' pixels')

    if display:
        ax1.legend()
        ax1.grid()
        plt.show()

    # add the minimum pixel value from fmask before returning
    #if len(spatial_mask)>1:
    #    my += min(spatial_mask)

    return my, y_sigma


def ap_extract(img, trace, apwidth=7, trace_sigma=(1, ), Saxis=1,
               spatial_mask=(1, ), spec_mask=(1, ), skysep=3, skywidth=7,
               skydeg=0, optimal=True, cr_sigma=5., gain=1.0, rn=5.0,
               silence=False, display=False):
    """
    1. Extract the spectrum using the trace. Simply add up all the flux
    around the aperture within a specified +/- width.
    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An major simplification at present. To be changed!
    2. Fits a polynomial to the sky at each column
    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An important simplification.
    3. Computes the uncertainty in each pixel
    ** In the description, we have the spectral direction on the X-axis and
       the spatial direction on the Y-axis.

    Parameters
    ----------
    img : 2d numpy array (M, N)
        This is the image, stored as a normal numpy array. Can be read in
        using astropy.io.fits like so:
        >>> hdu = fits.open('file.fits') # doctest: +SKIP
        >>> img = hdu[0].data # doctest: +SKIP
    trace : 1-d numpy array (N)
        The spatial positions (Y axis) corresponding to the center of the
        trace for every wavelength (X axis), as returned from ap_trace
    trace_sigma : float, or 1-d array (1 or N)
        Tophat extraction : Float is accepted but will be rounded to an int,
                            which gives the constant aperture size on either
                            side of the trace to extract.
        Optimal extraction: Float or 1-d array of the same size as the trace.
                            If a float is supplied, a fixed standard deviation
                            will be used to construct the gaussian weight
                            function along the entire spectrum.
    Saxis : int, optional
        Set the axis of the spatial dimension. 1 = Y axis, 0 = X axis.
        (Default is 1, i.e. spectrum in the left-right direction.)
    spatial_mask : 1-d numpy array (M), optional
        An array of 0/1 or True/False in the spatial direction (Y).
    spec_mask : 1-d numpy array (N), optional
        An array of 0/1 or True/False in the spectral direction (X).
    skysep : int, optional
        The separation in pixels from the aperture to the sky window.
        (Default is 3)
    skywidth : int, optional
        The width in pixels of the sky windows on either side of the
        aperture. (Default is 7)
    skydeg : int, optional
        The polynomial order to fit between the sky windows.
        (Default is 0, i.e. constant flat sky level)
    optimal : tuple, optional
        Set optimal extraction. (Default is True)
    cr_sigma : float, optional
        Cosmic ray sigma clipping limit, only used if extraction is optimal.
        (Default is 5)
    gain : float, optional
        Gain of the detector. (Deafult is 1.0)
    rn : float, optional
        Readnoise of the detector. (Deafult is 5.0)
    silence : tuple, optional
        Set to disable warning/error messages. (Default is False)
    display : tuple, optional
        Set to show diagnostic plots. (Default is True)

    Returns
    -------
    onedspec : 1-d array
        The summed flux at each column about the trace. Note: is not
        sky subtracted!
    skyflux : 1-d array
        The integrated sky values along each column, suitable for
        subtracting from the output of ap_extract
    fluxerr : 1-d array
        the uncertainties of the flux values
    """

    skyflux = np.zeros_like(trace)
    fluxerr = np.zeros_like(trace)
    flux = np.zeros_like(trace)
    median_trace = int(np.median(trace))
    len_trace = len(trace)

    # cosmic ray rejection
    img = detect_cosmics(img)

    # Depending on the cosmic ray rejection, need to take [1] if processed
    # by astroscrappy
    if type(img)==tuple:
        img = img[1]

    # the valid y-range of the chip
    if (len(spatial_mask) > 1):
        if Saxis is 1:
            img = img[spatial_mask]
        else:
            img = img[:,spatial_mask]

    if (len(spec_mask) > 1):
        if Saxis is 1:
            img = img[:,spec_mask]
        else:
            img = img[spec_mask]

    if Saxis is 0:
        img = np.transpose(img)
    for i, pos in enumerate(trace):

        itrace = int(pos)

        # first do the aperture flux
        widthup = apwidth
        widthdn = apwidth
        # fix width if trace is too close to the edge
        if (itrace+widthup > img.shape[0]):
            widthup = img.shape[0]-trace[i] - 1
        if (itrace-widthdn < 0):
            widthdn = itrace - 1 # i.e. starting at pixel row 1

        # simply add up the total flux around the trace +/- width
        xslice = img[itrace-widthdn:itrace+widthup+1, i]
        flux_ap = np.sum(xslice)

        # get the indexes of the sky regions
        y0 = max(itrace - widthdn - skysep - skywidth, 0)
        y1 = max(itrace - widthdn - skysep, 0)
        y2 = min(itrace + widthup + skysep + 1, img.shape[0])
        y3 = min(itrace + widthup + skysep + skywidth + 1, img.shape[0])
        y = np.append(np.arange(y0, y1),
                      np.arange(y2, y3))

        z = img[y,i]
        if (skydeg > 0):
            # fit a polynomial to the sky in this column
            pfit = np.polyfit(y, z, skydeg)
            # define the aperture in this column
            ap = np.arange(itrace-apwidth, itrace+apwidth+1)
            # evaluate the polynomial across the aperture, and sum
            skyflux[i] = np.sum(np.polyval(pfit, ap))
        elif (skydeg==0):
            skyflux[i] = np.sum(np.ones(apwidth*2 + 1) * np.nanmean(z))

        flux[i] = flux_ap - skyflux[i]

        #-- finally, compute the error in this pixel
        sigB = np.std(z) # stddev in the background data
        N_B = len(y) # number of bkgd pixels
        N_A = apwidth*2. + 1 # number of aperture pixels

        # based on aperture phot err description by F. Masci, Caltech:
        # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
        fluxerr[i] = np.sqrt(np.sum((flux_ap-skyflux[i])) / gain +
                             (N_A + N_A**2. / N_B) * (sigB**2.))

        # if optimal extraction
        if optimal:
            pix = range(itrace-widthdn,itrace+widthup+1)
            if (skydeg > 0):
                sky = np.polyval(pfit, pix)
            else:
                sky = np.ones(len(pix)) * np.nanmean(z)
            flux[i], fluxerr[i] = _optimal_signal(
                pix, xslice, sky, trace[i], trace_sigma[i], rn, gain,
                display=False, cr_sigma=cr_sigma
                )

    if display:
        fig2, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(10,10))
        fig2.tight_layout()
        plt.subplots_adjust(bottom=0.08, hspace=0)

        # show the image on the left
        ax0.imshow(
            np.log10(
                img[max(0, median_trace-widthdn-skysep-skywidth-1):
                    min(median_trace+widthup+skysep+skywidth, len(img[0])), :]),
            origin='lower',
            interpolation="nearest",
            aspect='auto',
            extent=[0,
                    len_trace,
                    max(0, median_trace-widthdn-skysep-skywidth-1),
                    min(median_trace+widthup+skysep+skywidth, len(img[0]))]
            )
        ax0.add_patch(
            Rectangle(
                (0, median_trace-widthdn-1),
                width=len_trace,
                height=(apwidth*2 + 1),
                linewidth=2,
                edgecolor='k',
                facecolor='none',
                zorder=1
                )
            )
        if (itrace-widthdn >= 0):
            ax0.add_patch(
                Rectangle(
                    (0, max(0, median_trace-widthdn-skysep-(y1-y0)-1)),
                    width=len_trace,
                    height=min(skywidth, (y1-y0)),
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none',
                    zorder=1
                    )
                )
        if (itrace+widthup <= img.shape[0]):
            ax0.add_patch(
                Rectangle(
                    (0, min(median_trace+widthup+skysep, len(img[0]))),
                    width=len_trace,
                    height=min(skywidth, (y3-y2)),
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none',
                    zorder=1
                    )
                )
        ax0.set_xlim(0-1, len_trace+1)
        ax0.set_ylim(max(0, median_trace-widthdn-skysep-skywidth-1)-1,
                     min(median_trace+widthup+skysep+skywidth, len(img[0]))+1)
        ax0.set_ylabel('Spatial Direction / pixel')

        # plot the spectrum of the target, sky and uncertainty
        ax1.plot(range(len_trace), flux, label='Target flux')
        ax1.plot(range(len_trace), skyflux, label='Sky flux')
        ax1.plot(range(len_trace), fluxerr, label='Uncertainty')
        ax1.set_xlabel('Spectral Direction / pixel')
        ax1.set_ylabel('Flux / count')
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=10)
        ax1.legend()
        ax1.grid()

        # plot the SNR
        ax2 = ax1.twinx()
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)
        ax2.plot(range(len_trace), flux/fluxerr, color='lightgrey',
            label='Signal-to-Noise Ratio')
        ax2.set_ylabel('Signal-to-Noise Ratio')
        ax2.set_ylim(bottom=0)
        ax2.legend(loc='upper left')

        plt.gca().set_prop_cycle(None)

    return flux, skyflux, fluxerr

