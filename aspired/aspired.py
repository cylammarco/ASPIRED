import difflib
import os
import sys
from functools import partial
import warnings

from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from ccdproc import Combiner
import numpy as np
from scipy import signal
from scipy import stats
from scipy import interpolate as itp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
try:
    from astroscrappy import detect_cosmics
except ImportError:
    warn(
        AstropyWarning('astroscrappy is not present, so ap_trace will clean ' +
                       'cosmic rays with a 2D-median filter of size 5.'))
    detect_cosmics = partial(signal.medfilt2d, kernel_size=5)
try:
    from spectres import spectres
    spectres_imported = True
except ImportError:
    warnings.warn(
        'spectres is not present, spectral resampling cannot be performed. '
        'Flux calibration is suboptimal. Flux is not conserved.')
    spectres_imported = False
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    plotly_imported = True
except ImportError:
    warnings.warn(
        'plotly is not present, diagnostic plots cannot be generated.')

from standard_list import *


class ImageReduction:
    def __init__(self, filelistpath, ftype='csv', combinetype='median'):
        '''
        bias, dark, flat, light
        filepath include HDU number if not [0]  
        '''
        self.filelistpath = filelistpath
        self.ftype = ftype
        if ftype == 'csv':
            self.delimiter = ','
        elif ftype == 'tsv':
            self.delimiter = '\t'
        elif ftype == 'ascii':
            self.delimiter = ' '
        if combinetype == 'average':
            self.combinetype = combinetype
        elif combinetype == 'median':
            self.combinetype = combinetype
        else:
            warnings.warn('Unknown image combine type, '
                          'median_combine() is used.')
            self.combinetype = 'median'

        # FITS keyword standard recommends XPOSURE, but most observatories
        # use EXPTIME for supporting iraf. Also included a few other keywords
        # which are the proxy-exposure times at best. ASPIRED will use the
        # first keyword found on the list, if all failed, an exposure time of
        # 1 second will be applied.
        self.exptime_keyword = [
            'XPOSURE', 'EXPTIME', 'EXPOSED', 'TELAPSED', 'ELAPSED'
        ]

        self.bias_list = None
        self.dark_list = None
        self.flat_list = None
        self.light = None

        self.bias_master = None
        self.dark_master = None
        self.flat_master = None
        self.light_master = None

        self.light_filename = []
        self.bias_filename = []
        self.dark_filename = []
        self.flat_filename = []

        # import file with first column as image type and second column as
        # file path
        filelist = np.genfromtxt(self.filelistpath,
                                 delimiter=self.delimiter,
                                 dtype='str',
                                 autostrip=True)
        imtype = filelist[:, 0]
        impath = filelist[:, 1]

        self.bias_list = impath[imtype == 'bias']
        self.dark_list = impath[imtype == 'dark']
        self.flat_list = impath[imtype == 'flat']
        self.light_list = impath[imtype == 'light']

        # If there is no science frames, nothing to process.
        assert (self.light_list.size > 0), 'There is no light frame.'

        # Only load the science data, other types of image data are loaded by
        # separate methods.
        light_CCDData = []
        light_time = []

        for i in range(self.light_list.size):
            # Open all the light frames
            light = fits.open(self.light_list[i])[0]
            light_CCDData.append(CCDData(light.data, unit=u.adu))

            self.light_filename.append(self.light_list[i].split('/')[-1])

            # Get the exposure time for the light frames
            for exptime in self.exptime_keyword:
                try:
                    light_time.append(light.header[exptime])
                    break
                except:
                    continue

        # Put data into a Combiner
        light_combiner = Combiner(light_CCDData)
        # Free memory
        del light_CCDData

        # Image combine by median or average
        if self.combinetype == 'median':
            self.light_master = light_combiner.median_combine()
            self.light_time = np.median(light_time)
        elif self.combinetype == 'average':
            self.light_master = light_combiner.average_combine()
            self.light_time = np.mean(light_time)
        else:
            raise ValueError('ASPIRED: Unknown combinetype.')

        # Free memory
        del light_combiner

        # If exposure time cannot be found from the header, use 1 second
        if len(light_time) == 0:
            self.light_time = 1.
            warnings.warn('Light frame exposure time cannot be found. '
                          '1 second is used as the exposure time.')

        # Frame in unit of ADU per second
        self.light_master = self.light_master

    def _bias_subtract(self):

        bias_CCDData = []

        for i in range(self.bias_list.size):
            # Open all the bias frames
            bias = fits.open(self.bias_list[i])[0]
            bias_CCDData.append(CCDData(biad.data, unit=u.adu))

            self.bias_filename.append(self.bias_list[i].split('/')[-1])

        # Put data into a Combiner
        bias_combiner = Combiner(bias_CCDData)

        # Image combine by median or average
        if self.combinetype == 'median':
            self.bias_master = bias_combiner.median_combine()
        elif self.combinetype == 'average':
            self.bias_master = bias_combiner.average_combine()
        else:
            raise ValueError('ASPIRED: Unknown combinetype.')

        # Bias subtract
        self.light_master = self.light_master - self.bias_master

        # Free memory
        del bias_CCDData
        del bias_combiner

    def _dark_subtract(self):

        dark_CCDData = []
        dark_time = []

        for i in range(self.dark_list.size):
            # Open all the dark frames
            dark = fits.open(self.dark_list[i])[0]
            dark_CCDData.append(CCDData(dark.data, unit=u.adu))

            self.dark_filename.append(self.dark_list[i].split('/')[-1])

            # Get the exposure time for the dark frames
            for exptime in self.exptime_keyword:
                try:
                    dark_time.append(dark.header[exptime])
                    break
                except:
                    continue

        # Put data into a Combiner
        dark_combiner = Combiner(dark_CCDData)

        # Image combine by median or average
        if self.combinetype == 'median':
            self.dark_master = dark_combiner.median_combine()
            self.dark_time = np.median(dark_time)
        elif self.combinetype == 'average':
            self.dark_master = dark_combiner.average_combine()
            self.dark_time = np.mean(dark_time)
        else:
            raise ValueError('ASPIRED: Unknown combinetype.')

        # If exposure time cannot be found from the header, use 1 second
        if len(dark_time) == 0:
            warnings.warn('Dark frame exposure time cannot be found. '
                          '1 second is used as the exposure time.')
            self.dark_time = 1.

        # Frame in unit of ADU per second
        self.light_master =\
            self.light_master -\
            self.dark_master / self.dark_time * self.light_time

        # Free memory
        del dark_CCDData
        del dark_combiner

    def _flatfield(self):

        flat_CCDData = []

        for i in range(self.flat_list.size):
            # Open all the flatfield frames
            flat = fits.open(self.flat_list[i])[0]
            flat_CCDData.append(CCDData(flat.data, unit=u.adu))

            self.flat_filename.append(self.flat_list[i].split('/')[-1])

        # Put data into a Combiner
        flat_combiner = Combiner(flat_CCDData)

        # Image combine by median or average
        if self.combinetype == 'median':
            self.flat_master = flat_combiner.median_combine()
        elif self.combinetype == 'average':
            self.flat_master = flat_combiner.average_combine()
        else:
            raise ValueError('ASPIRED: Unknown combinetype.')

        # Field-flattening
        self.light_master = self.light_master / self.flat_master

        # Free memory
        del flat_CCDData
        del flat_combiner

    def reduce(self, display=False, log=True):

        if self.bias_list.size > 0:
            self._bias_subtract()
        else:
            warnings.warn('No bias frames. Bias subtraction is not performed.')

        if self.dark_list.size > 0:
            self._dark_subtract()
        else:
            warnings.warn('No dark frames. Dark subtraction is not performed.')

        if self.flat_list.size > 0:
            self._flatfield()
        else:
            warnings.warn('No flat frames. Field-flattening is not performed.')

        self.light_master = np.array((self.light_master))
        self.fits_data = fits.PrimaryHDU(self.light_master)

        if len(self.light_filename) > 0:
            for i in range(len(self.light_filename)):
                self.fits_data.header.set('light' + str(i + 1),
                                          self.light_filename[i],
                                          'Light frames')
        if len(self.bias_filename) > 0:
            for i in range(len(self.bias_filename)):
                self.fits_data.header.set('bias' + str(i + 1),
                                          self.bias_filename[i], 'Bias frames')
        if len(self.dark_filename) > 0:
            for i in range(len(self.dark_filename)):
                self.fits_data.header.set('dark' + str(i + 1),
                                          self.dark_filename[i], 'Dark frames')
        if len(self.flat_filename) > 0:
            for i in range(len(self.flat_filename)):
                self.fits_data.header.set('flat' + str(i + 1),
                                          self.flat_filename[i], 'Flat frames')

        if display:
            self.inspect(log=log)

    def savefits(self, filepath='reduced_image.fits', overwrite=False):
        self.fits_data.writeto(filepath, overwrite=overwrite)

    def inspect(self, log=True, renderer='default', verbose=False):

        # Generate plot with plotly can be imported
        if plotly_imported:
            if log:
                fig = go.Figure(data=go.Heatmap(z=np.log10(self.light_master),
                                                colorscale="Viridis"))
            else:
                fig = go.Figure(
                    data=go.Heatmap(z=self.light_master, colorscale="Viridis"))
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

            if verbose:
                return fig.to_json()
        else:
            warnings.warn('plotly is not present, diagnostic plots cannot be '
                          'generated.')

    def list_files(self):
        pass


class TwoDSpec:
    def __init__(self,
                 img,
                 Saxis=1,
                 spatial_mask=(1, ),
                 spec_mask=(1, ),
                 n_spec=1,
                 cr=True,
                 cr_sigma=5.,
                 rn=5.,
                 gain=1.,
                 seeing=1.2,
                 exptime=1.,
                 silence=False,
                 display=True,
                 renderer='default',
                 verbose=False):
        '''
        cr_sigma : float, optional
            Cosmic ray sigma clipping limit, only used if extraction is optimal.
            (Default is 5)
        gain : float, optional
            Gain of the detector. (Deafult is 1.0)
        rn : float, optional
            Readnoise of the detector. (Deafult is 5.0)
        '''

        self.Saxis = Saxis
        if Saxis is 0:
            self.Waxis = 1
        else:
            self.Waxis = 0
        self.spatial_mask = spatial_mask
        self.spec_mask = spec_mask
        self.n_spec = n_spec
        self.cr_sigma = cr_sigma
        self.rn = rn
        self.gain = gain
        self.seeing = seeing
        self.exptime = exptime
        self.silence = silence
        self.display = display
        if not plotly_imported:
            self.display = False
        self.renderer = renderer
        self.verbose = verbose

        # cosmic ray rejection
        if cr:
            img = detect_cosmics(img,
                                 sigclip=cr_sigma,
                                 readnoise=rn,
                                 gain=gain,
                                 fsmode='convolve',
                                 psffwhm=seeing)[1]

        # the valid y-range of the chip (i.e. spatial direction)
        if (len(spatial_mask) > 1):
            if Saxis is 1:
                img = img[spatial_mask]
            else:
                img = img[:, spatial_mask]

        # the valid x-range of the chip (i.e. spectral direction)
        if (len(spec_mask) > 1):
            if Saxis is 1:
                img = img[:, spec_mask]
            else:
                img = img[spec_mask]

        # get the length in the spectral and spatial directions
        self.spec_size = np.shape(img)[self.Waxis]
        self.spatial_size = np.shape(img)[Saxis]
        self.img = img

    def _gaus(self, x, a, b, x0, sigma):
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

    def _identify_spectrum(self, f_height, Saxis, display, renderer):
        """
        Identify peaks assuming the spatial and spectral directions are
        aligned with the X and Y direction within a few degrees.

        Parameters
        ----------
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
        ydata = np.arange(self.spec_size)
        ztot = np.sum(self.img, axis=Saxis)

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
            fig = go.Figure()

            # show the image on the left
            if Saxis == 1:
                fig.add_trace(
                    go.Heatmap(z=np.log10(self.img),
                               colorscale="Viridis",
                               xaxis='x',
                               yaxis='y'))
            else:
                fig.add_trace(
                    go.Heatmap(z=np.log10(np.transpose(self.img)),
                               colorscale="Viridis",
                               xaxis='x',
                               yaxis='y'))

            # plot the integrated count and the detected peaks on the right
            fig.add_trace(
                go.Scatter(x=ztot,
                           y=ydata,
                           line=dict(color='black'),
                           xaxis='x2'))
            fig.add_trace(
                go.Scatter(x=heights_y,
                           y=ydata[peaks_y],
                           marker=dict(color='firebrick'),
                           xaxis='x2'))
            fig.update_layout(autosize=True,
                              yaxis_title='Spatial Direction / pixel',
                              xaxis=dict(zeroline=False,
                                         domain=[0, 0.5],
                                         showgrid=False,
                                         title='Spectral Direction / pixel'),
                              xaxis2=dict(zeroline=False,
                                          domain=[0.5, 1],
                                          showgrid=True,
                                          title='Integrated Count'),
                              bargap=0,
                              hovermode='closest',
                              showlegend=False)

            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

            if verbose:
                return fig.to_json()

        self.peak = peaks_y
        self.peak_height = heights_y

    def _optimal_signal(self, pix, xslice, sky, mu, sigma, display, renderer):
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
        display : tuple
            Set to show diagnostic plot.

        Returns
        -------
        signal : float
            The optimal signal. 
        noise : float
            The noise associated with the optimal signal.

        """

        # construct the Gaussian model
        gauss = self._gaus(pix, 1., 0., mu, sigma)

        # weight function and initial values
        P = gauss / np.sum(gauss)
        signal0 = np.sum(xslice - sky)
        var0 = self.rn + np.abs(xslice) / self.gain
        variance0 = 1. / np.sum(P**2. / var0)

        for i in range(100):

            # cosmic ray mask
            mask_cr = ((xslice - sky - P * signal0)**2. <
                       self.cr_sigma**2. * var0)

            # compute signal and noise
            signal1 = np.sum((P * (xslice - sky) / var0)[mask_cr]) / \
                np.sum((P**2. / var0)[mask_cr])
            var1 = self.rn + np.abs(P * signal1 + sky) / self.gain
            variance1 = 1. / np.sum((P**2. / var1)[mask_cr])

            # iterate
            if (((signal1 - signal0) / signal1 > 0.001)
                    or ((variance1 - variance0) / variance1 > 0.001)):
                signal0 = signal1
                var0 = var1
                variance0 = variance1
            else:
                break

        if i == 99:
            print('Unable to obtain optimal signal, please try a longer ' +
                  'iteration or revert to unit-weighted extraction. Values ' +
                  'returned (if at all) are sub-optimal at best.')

        signal = signal1
        noise = np.sqrt(variance1)

        if display:
            fit = self._gaus(pix, max(xslice - sky), 0., mu, sigma) + sky
            fig, ax = plt.subplots(ncols=1, figsize=(10, 10))
            ax.plot(pix, xslice)
            ax.plot(pix, fit)
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Count')
            #print(signal, variance)
            #print(np.sum(xslice-sky_const))

        return signal, noise

    def ap_trace(self,
                 nsteps=20,
                 recenter=False,
                 prevtrace=(0, ),
                 fittype='spline',
                 order=3,
                 bigbox=8):
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
        spatial_mask : 1-d numpy array (<=M), optional
            An array of pixel number or True/False in the spatial direction (Y).
        spec_mask : 1-d numpy array (<=N), optional
            An array of pixel number or True/False in the spectral direction (X).
        cr_sigma : float (default is None)
            Set to apply cosmic ray removal beefore tracing using astroscrappy if
            available, otherwise with a 2D median filter of size 5. It does not
            alter the base image. (default is True)
        rn : float (deafult is 5.)
            Readnoise of the detector (ADU)
        gain : float (default is 1.)
            Gain of the detector (e- per ADU)
        seeing : float (default is 1.)
            Seeing condition during exposure (or the combined image).
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

        if not self.silence:
            print('Tracing Aperture using nsteps=' + str(nsteps))

        # the valid y-range of the chip (an array of int)
        ydata = np.arange(self.spec_size)
        ztot = np.sum(self.img, axis=self.Saxis)

        # need at least 4 samples along the trace. sometimes can get away with very few
        if (nsteps < 4):
            nsteps = 4

        # detect peaks by summing in the spatial direction
        self._identify_spectrum(0.01,
                                self.Saxis,
                                display=False,
                                renderer=self.renderer)

        if self.display:
            # set a side-by-side subplot
            fig = go.Figure()

            # show the image on the left
            if self.Saxis == 1:
                fig.add_trace(
                    go.Heatmap(z=np.log10(self.img),
                               colorscale="Viridis",
                               xaxis='x',
                               yaxis='y',
                               colorbar=dict(title='log(ADU)')))
            else:
                fig.add_trace(
                    go.Heatmap(z=np.log10(np.transpose(self.img)),
                               colorscale="Viridis",
                               xaxis='x',
                               yaxis='y',
                               colorbar=dict(title='log(ADU)')))

            # plot the integrated count and the detected peaks on the right
            fig.add_trace(
                go.Scatter(x=np.log10(ztot),
                           y=ydata,
                           line=dict(color='black'),
                           xaxis='x2'))
            fig.add_trace(
                go.Scatter(x=np.log10(self.peak_height),
                           y=self.peak,
                           mode='markers',
                           marker=dict(color='firebrick'),
                           xaxis='x2'))

        my = np.zeros((self.n_spec, self.spatial_size))
        y_sigma = np.zeros((self.n_spec))

        # trace each individual spetrum one by one
        for i in range(self.n_spec):

            peak_guess = [
                self.peak_height[i],
                np.nanmedian(ztot), self.peak[i], 2.
            ]

            if (recenter is False) and (len(prevtrace) > 10):
                self.trace = prevtrace
                self.trace_sigma = np.ones(len(prevtrace)) * self.seeing
                if self.display:
                    fig.add_trace(
                        go.Scatter(x=[min(ztot[ztot > 0]),
                                      max(ztot)],
                                   y=[min(self.trace),
                                      max(self.trace)],
                                   mode='lines',
                                   xaxis='x2'))
                    fig.add_trace(
                        go.Scatter(x=np.arange(len(self.trace)),
                                   y=self.trace,
                                   mode='lines',
                                   xaxis='x2'))
                break

            # use middle of previous trace as starting guess
            elif (recenter is True) and (len(prevtrace) > 10):
                peak_guess[2] = np.nanmedian(prevtrace)

            else:
                # fit a Gaussian to peak
                try:
                    pgaus, pcov = curve_fit(
                        self._gaus,
                        ydata[np.isfinite(ztot)],
                        ztot[np.isfinite(ztot)],
                        p0=peak_guess,
                        bounds=((0., 0., peak_guess[2] - 10, 0.),
                                (np.inf, np.inf, peak_guess[2] + 10, np.inf)))
                    #print(pgaus, pcov)
                except:
                    if not self.silence:
                        print(
                            'Spectrum ' + str(i) + ' of ' + str(self.n_spec) +
                            ' is likely to be (1) too faint, (2) in a crowed'
                            ' field, or (3) an extended source. Automated' +
                            ' tracing is sub-optimal. Please (1) reduce n_spec,'
                            +
                            ' or (2) reduce n_steps, or (3) provide prevtrace.'
                        )

                if self.display:
                    fig.add_trace(
                        go.Scatter(x=np.log10(
                            self._gaus(ydata, pgaus[0], pgaus[1], pgaus[2],
                                       pgaus[3])),
                                   y=ydata,
                                   mode='lines',
                                   xaxis='x2'))

                # only allow data within a box around this peak
                ydata2 = ydata[
                    np.where((ydata >= pgaus[2] - pgaus[3] * bigbox)
                             & (ydata <= pgaus[2] + pgaus[3] * bigbox))]
                yi = np.arange(self.spec_size)[ydata2]

                # define the X-bin edges
                xbins = np.linspace(0, self.spatial_size, nsteps)
                ybins = np.zeros_like(xbins)
                ybins_sigma = np.zeros_like(xbins)

                # loop through each bin
                for j in range(0, len(xbins) - 1):
                    # fit gaussian w/j each window
                    if self.Saxis is 1:
                        zi = np.sum(self.img[ydata2,
                                             int(np.floor(xbins[j])
                                                 ):int(np.ceil(xbins[j + 1]))],
                                    axis=self.Saxis)
                    else:
                        zi = np.sum(
                            self.img[int(np.floor(xbins[j])
                                         ):int(np.ceil(xbins[j + 1])), ydata2],
                            axis=self.Saxis)

                    # fit gaussian w/j each window
                    if sum(zi) == 0:
                        break
                    else:
                        pguess = [
                            np.nanmax(zi),
                            np.nanmedian(zi), yi[np.nanargmax(zi)], 2.
                        ]
                    try:
                        popt, pcov = curve_fit(self._gaus, yi, zi, p0=pguess)
                    except:
                        if not self.silence:
                            print('Step ' + str(j + 1) + ' of ' + str(nsteps) +
                                  ' of spectrum ' + str(i + 1) + ' of ' +
                                  str(self.n_spec) + ' cannot be fitted.')
                        break

                    # if the peak is lower than background, sigma is too broad or
                    # gaussian fits off chip, then use chip-integrated answer
                    if ((popt[0] < 0) or (popt[3] < 0) or (popt[3] > 10)):
                        ybins[j] = pgaus[2]
                        popt = pgaus
                        if not self.silence:
                            print(
                                'Step ' + str(j + 1) + ' of ' + str(nsteps) +
                                ' of spectrum ' + str(i + 1) + ' of ' +
                                str(self.n_spec) +
                                ' has a poor fit. Initial guess is used instead.'
                            )
                    else:
                        ybins[j] = popt[2]
                        ybins_sigma[j] = popt[3]

                # recenter the bin positions, trim the unused bin off in Y
                mxbins = (xbins[:-1] + xbins[1:]) / 2.
                mybins = ybins[:-1]
                mx = np.arange(0, self.spatial_size)

                if (fittype == 'spline'):
                    # run a cubic spline thru the bins
                    interpolated = itp.UnivariateSpline(mxbins,
                                                        mybins,
                                                        ext=0,
                                                        k=order)
                    # interpolate 1 position per column
                    my[i] = interpolated(mx)

                elif (fittype == 'polynomial'):
                    # linear fit
                    npfit = np.polyfit(mxbins, mybins, deg=order)
                    # interpolate 1 position per column
                    my[i] = np.polyval(npfit, mx)

                else:
                    if not self.silence:
                        print('Unknown fitting type, please choose from ' +
                              '(1) \'spline\'; or (2) \'polynomial\'.')

                # get the uncertainties in the spatial direction along the spectrum
                slope, intercept, r_value, p_value, std_err =\
                        stats.linregress(mxbins, ybins_sigma[:-1])
                y_sigma[i] = np.nanmedian(slope * mx + intercept)

                if self.display:
                    fig.add_trace(
                        go.Scatter(x=mx, y=my[i], mode='lines', xaxis='x'))

                if not self.silence:
                    if np.sum(ybins_sigma) == 0:
                        print(
                            'Spectrum ' + str(i + 1) + ' of ' +
                            str(self.n_spec) +
                            ' is likely to be (1) too faint, (2) in a crowed'
                            ' field, or (3) an extended source. Automated' +
                            ' tracing is sub-optimal. Please disable multi-source'
                            +
                            ' mode and (1) reduce n_spec, or (2) reduce n_steps,'
                            +
                            '  or (3) provide prevtrace, or (4) all of above.')

                    print('Spectrum ' + str(i + 1) +
                          ' : Trace gaussian width = ' + str(ybins_sigma) +
                          ' pixels')

            if self.display:
                fig.update_layout(autosize=True,
                                  yaxis_title='Spatial Direction / pixel',
                                  xaxis=dict(
                                      zeroline=False,
                                      domain=[0, 0.5],
                                      showgrid=False,
                                      title='Spectral Direction / pixel'),
                                  xaxis2=dict(zeroline=False,
                                              domain=[0.5, 1],
                                              showgrid=True,
                                              title='Integrated Count'),
                                  bargap=0,
                                  hovermode='closest',
                                  showlegend=False,
                                  height=800)
            if self.renderer == 'default':
                fig.show()
            else:
                fig.show(self.renderer)

            if self.verbose:
                return fig.to_json()

            # add the minimum pixel value from fmask before returning
            #if len(spatial_mask)>1:
            #    my += min(spatial_mask)

            self.trace = my[0]
            self.trace_sigma = y_sigma[0]

    def ap_extract(self,
                   apwidth=7,
                   skysep=3,
                   skywidth=7,
                   skydeg=0,
                   optimal=True):

        #
        # CURRENTLY ONLY WORK FOR A SINGLE TRACE. EVEN THOUGH ap_trace
        # RETURNS MULTIPLE TRACES IN ONE GO
        #
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
        silence : tuple, optional
            Set to disable warning/error messages. (Default is False)
        display : tuple, optional
            Set to show diagnostic plots. (Default is True)

        Returns
        -------
        onedspec : 1-d array
            The summed adu at each column about the trace. Note: is not
            sky subtracted!
        skyadu : 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        aduerr : 1-d array
            the uncertainties of the adu values
        """

        skyadu = np.zeros_like(self.trace)
        aduerr = np.zeros_like(self.trace)
        adu = np.zeros_like(self.trace)
        median_trace = int(np.median(self.trace))
        len_trace = len(self.trace)

        # convert to numpy array of length spec_mask
        trace_sigma = np.ones(len(self.trace)) * self.trace_sigma

        for i, pos in enumerate(self.trace):

            itrace = int(round(pos))

            # first do the aperture adu
            widthup = apwidth
            widthdn = apwidth
            # fix width if trace is too close to the edge
            if (itrace + widthup > self.spatial_size):
                widthup = spatial_size - itrace - 1
            if (itrace - widthdn < 0):
                widthdn = itrace - 1  # i.e. starting at pixel row 1

            # simply add up the total adu around the trace +/- width
            xslice = self.img[itrace - widthdn:itrace + widthup + 1, i]
            adu_ap = np.sum(xslice)

            if skywidth > 0:
                # get the indexes of the sky regions
                y0 = max(itrace - widthdn - skysep - skywidth, 0)
                y1 = max(itrace - widthdn - skysep, 0)
                y2 = min(itrace + widthup + skysep + 1, self.spatial_size)
                y3 = min(itrace + widthup + skysep + skywidth + 1,
                         self.spatial_size)
                y = np.append(np.arange(y0, y1), np.arange(y2, y3))

                z = self.img[y, i]
                if (skydeg > 0):
                    # fit a polynomial to the sky in this column
                    pfit = np.polyfit(y, z, skydeg)
                    # define the aperture in this column
                    ap = np.arange(itrace - apwidth, itrace + apwidth + 1)
                    # evaluate the polynomial across the aperture, and sum
                    skyadu[i] = np.sum(np.polyval(pfit, ap))
                elif (skydeg == 0):
                    skyadu[i] = np.sum(
                        np.ones(apwidth * 2 + 1) * np.nanmean(z))

                #-- finally, compute the error in this pixel
                sigB = np.std(z)  # stddev in the background data
                N_B = len(y)  # number of bkgd pixels
                N_A = apwidth * 2. + 1  # number of aperture pixels

                # based on aperture phot err description by F. Masci, Caltech:
                # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
                aduerr[i] = np.sqrt(
                    np.sum((adu_ap - skyadu[i])) / self.gain +
                    (N_A + N_A**2. / N_B) * (sigB**2.))

            adu[i] = adu_ap - skyadu[i]

            # if optimal extraction
            if optimal:
                pix = range(itrace - widthdn, itrace + widthup + 1)
                if (skydeg > 0):
                    sky = np.polyval(pfit, pix)
                else:
                    sky = np.ones(len(pix)) * np.nanmean(z)
                adu[i], aduerr[i] = self._optimal_signal(
                    pix,
                    xslice,
                    sky,
                    self.trace[i],
                    trace_sigma[i],
                    display=False,
                    renderer=self.renderer)

        if self.display:
            fig = go.Figure()

            # show the image on the top
            fig.add_trace(
                go.Heatmap(
                    x=np.arange(len_trace),
                    y=np.arange(
                        max(0, median_trace - widthdn - skysep - skywidth - 1),
                        min(median_trace + widthup + skysep + skywidth,
                            len(self.img[0]))),
                    z=np.log10(self.img[max(
                        0, median_trace - widthdn - skysep - skywidth -
                        1):min(median_trace + widthup + skysep +
                               skywidth, len(self.img[0])), :]),
                    colorscale="Viridis",
                    xaxis='x',
                    yaxis='y',
                    colorbar=dict(title='log(ADU)')))

            # Middle black box on the image
            fig.add_trace(
                go.Scatter(x=[0, len_trace, len_trace, 0, 0],
                           y=[
                               median_trace - widthdn - 1,
                               median_trace - widthdn - 1,
                               median_trace - widthdn - 1 + (apwidth * 2 + 1),
                               median_trace - widthdn - 1 + (apwidth * 2 + 1),
                               median_trace - widthdn - 1
                           ],
                           xaxis='x',
                           yaxis='y',
                           mode='lines',
                           line_color='black',
                           showlegend=False))

            # Lower red box on the image
            if (itrace - widthdn >= 0):
                fig.add_trace(
                    go.Scatter(
                        x=[0, len_trace, len_trace, 0, 0],
                        y=[
                            max(
                                0, median_trace - widthdn - skysep -
                                (y1 - y0) - 1),
                            max(
                                0, median_trace - widthdn - skysep -
                                (y1 - y0) - 1),
                            max(
                                0, median_trace - widthdn - skysep -
                                (y1 - y0) - 1) + min(skywidth, (y1 - y0)),
                            max(
                                0, median_trace - widthdn - skysep -
                                (y1 - y0) - 1) + min(skywidth, (y1 - y0)),
                            max(
                                0, median_trace - widthdn - skysep -
                                (y1 - y0) - 1)
                        ],
                        line_color='red',
                        xaxis='x',
                        yaxis='y',
                        mode='lines',
                        showlegend=False))

            # Upper red box on the image
            if (itrace + widthup <= self.spatial_size):
                fig.add_trace(
                    go.Scatter(
                        x=[0, len_trace, len_trace, 0, 0],
                        y=[
                            min(median_trace + widthup + skysep,
                                len(self.img[0])),
                            min(median_trace + widthup + skysep,
                                len(self.img[0])),
                            min(median_trace + widthup + skysep,
                                len(self.img[0])) + min(skywidth, (y3 - y2)),
                            min(median_trace + widthup + skysep,
                                len(self.img[0])) + min(skywidth, (y3 - y2)),
                            min(median_trace + widthup + skysep,
                                len(self.img[0]))
                        ],
                        xaxis='x',
                        yaxis='y',
                        mode='lines',
                        line_color='red',
                        showlegend=False))
            # extrated source, sky and uncertainty
            fig.add_trace(
                go.Scatter(x=np.arange(len_trace),
                           y=adu,
                           xaxis='x2',
                           yaxis='y2',
                           line=dict(color='royalblue'),
                           name='Target ADU'))
            fig.add_trace(
                go.Scatter(x=np.arange(len_trace),
                           y=skyadu,
                           xaxis='x2',
                           yaxis='y2',
                           line=dict(color='firebrick'),
                           name='Sky ADU'))
            fig.add_trace(
                go.Scatter(x=np.arange(len_trace),
                           y=aduerr,
                           xaxis='x2',
                           yaxis='y2',
                           line=dict(color='orange'),
                           name='Uncertainty'))
            # plot the SNR
            fig.add_trace(
                go.Scatter(x=np.arange(len_trace),
                           y=adu / aduerr,
                           xaxis='x2',
                           yaxis='y3',
                           line=dict(color='slategrey'),
                           name='Signal-to-Noise Ratio'))

            # Decorative stuff
            fig.update_layout(autosize=True,
                              xaxis=dict(showticklabels=False),
                              yaxis=dict(zeroline=False,
                                         domain=[0.5, 1],
                                         showgrid=False,
                                         title='Spatial Direction / pixel'),
                              yaxis2=dict(
                                  range=[
                                      max(np.nanmin(np.log10(adu)), 1),
                                      np.nanmax(np.log10(skyadu))
                                  ],
                                  zeroline=False,
                                  domain=[0, 0.5],
                                  showgrid=True,
                                  type='log',
                                  title='log(ADU / Count)',
                              ),
                              yaxis3=dict(title='S/N ratio',
                                          anchor="x2",
                                          overlaying="y2",
                                          side="right"),
                              xaxis2=dict(title='Spectral Direction / pixel',
                                          anchor="y2",
                                          matches="x"),
                              legend=go.layout.Legend(x=0,
                                                      y=0.45,
                                                      traceorder="normal",
                                                      font=dict(
                                                          family="sans-serif",
                                                          size=12,
                                                          color="black"),
                                                      bgcolor='rgba(0,0,0,0)'),
                              bargap=0,
                              hovermode='closest',
                              showlegend=True,
                              height=800)
            if self.renderer == 'default':
                fig.show()
            else:
                fig.show(self.renderer)

            if self.verbose:
                return fig.to_json()

        self.adu = adu
        self.skyadu = skyadu
        self.aduerr = aduerr


class WavelengthPolyFit:
    def __init__(self,
                 spec,
                 distance=3.,
                 percentile=20.,
                 pix_scale=10.,
                 min_wave=3000.,
                 max_wave=9000.,
                 sample_size=5,
                 max_tries=100,
                 top_n=100,
                 n_slopes=10000):
        '''
        spec : TwoDSpec object of the arc image
        '''
        self.spec = spec
        self.distance = distance
        self.percentile = percentile
        self.pix_scale = pix_scale
        self.min_wave = min_wave
        self.max_wave = max_wave
        self.sample_size = sample_size
        self.max_tries = max_tries
        self.top_n = top_n
        self.n_slopes = n_slopes

    def find_arc_lines(self, display=False):
        '''
        # pixelscale in unit of A/pix

        '''

        p = np.percentile(self.spec, self.percentile)
        self.peaks, _ = signal.find_peaks(self.spec,
                                          distance=self.distance,
                                          prominence=p)

        if diaplay & plotly_imported:
            fig = go.Figure()

            # show the image on the top
            fig.add_trace(
                go.Scatter(x=np.arange(self.spec),
                           y=self.spec,
                           line=dict(color='royalblue', width=4)))

            for i in peaks:
                fig.add_trace(
                    go.Scatter(x=[i, i],
                               y=[
                                   self.spec[self.peaks.astype('int')],
                                   self.spec.max() * 1.1
                               ],
                               line=dict(color='firebrick', width=4)))

            fig.update_layout(autosize=True,
                              xaxis_title='Pixel',
                              yaxis_title='Count',
                              hovermode='closest',
                              showlegend=False,
                              height=800)

        if self.renderer == 'default':
            fig.show()
        else:
            fig.show(self.renderer)

        if self.verbose:
            return fig.to_json()

    def calibrate(self,
                  elements=["Hg", "Ar", "Xe", "CuNeAr", "Kr"],
                  display=False,
                  **kwargs):
        '''
        thresh (A) :: the individual line fitting tolerance to accept as a valid fitting point
        fit_tolerance (A) :: the RMS
        '''

        c = Calibrator(self.peaks,
                       elements=self.elements,
                       min_wavelength=self.min_wave,
                       max_wavelength=self.max_wave)

        c.set_fit_constraints(**kwargs)

        p = c.fit(sample_size=self.sample_size,
                  max_tries=self.max_tries,
                  top_n=self.top_n,
                  n_slope=self.n_slope)

        pfit = c.match_peaks_to_atlas(p)[0]

        if display & plotly_imported:
            c.plot_fit(spec, pfit, tolerance=pix_scale)

        self.pfit = pfit


class StandardFlux:
    def __init__(self,
                 target,
                 group,
                 cutoff=0.4,
                 ftype='flux',
                 silence=False,
                 renderer='default',
                 verbose=False):
        self.target = target
        self.group = group
        self.cutoff = cutoff
        self.ftype = ftype
        self.silence = silence
        self.renderer = renderer
        self.verbose = verbose
        self._lookup_standard()

    def _lookup_standard(self):
        '''
        Check if the requested standard and library exist.

        '''

        try:
            target_list = eval(self.group)
        except:
            raise ValueError('Requested standard star library does not exist.')

        if self.target not in target_list:
            best_match = difflib.get_close_matches(self.target,
                                                   target_list,
                                                   cutoff=cutoff)
            raise ValueError(
                'Requested standard star is not in the library.', '',
                'The requrested spectrophotometric library contains: ',
                target_list, '', 'Are you looking for these: ', best_match)

    def load_standard(self, display=False):
        '''
        Read the standard flux/magnitude file. And return the wavelength and
        flux/mag in units of

        wavelength: \AA
        flux:       ergs / cm / cm / s / \AA
        mag:        mag (AB) 

        '''

        flux_multiplier = 1.
        if self.group[:4] == 'iraf':
            target_name = self.target + '.dat'
        else:
            if self.ftype == 'flux':
                target_name = 'f' + self.target + '.dat'
                if self.group != 'xshooter':
                    flux_multiplier = 1e-16
            elif self.ftype == 'mag':
                target_name = 'm' + self.target + '.dat'
            else:
                raise ValueError('The type has to be \'flux\' of \'mag\'.')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(dir_path, '..', 'standards',
                                str(self.group) + 'stan', target_name)

        if self.group[:4] == 'iraf':
            f = np.loadtxt(filepath, skiprows=1)
        else:
            f = np.loadtxt(filepath)

        wave = f[:, 0]
        if (self.group[:4] == 'iraf') & (self.ftype == 'flux'):
            fluxmag = 10.**(-(f[:, 1] / 2.5)) * 3630.780548 / 3.34e4 / wave**2
        else:
            fluxmag = f[:, 1] * flux_multiplier

        self.wave_std = wave
        self.fluxmag_std = fluxmag

        if display & plotly_imported:
            self.inspect_standard()

    def inspect_standard(self):
        fig = go.Figure()

        # show the image on the top
        fig.add_trace(
            go.Scatter(x=self.wave_std,
                       y=self.fluxmag_std,
                       line=dict(color='royalblue', width=4)))

        fig.update_layout(
            autosize=True,
            title=self.group + ' : ' + self.target + ' ' + self.ftype,
            xaxis_title=r'$\text{Wavelength / A}$',
            yaxis_title=
            r'$\text{Flux / ergs cm}^{-2} \text{s}^{-1} \text{A}^{-1}$',
            hovermode='closest',
            showlegend=False,
            height=800)

        if self.renderer == 'default':
            fig.show()
        else:
            fig.show(self.renderer)

        if self.verbose:
            return fig.to_json()


class OneDSpec:
    def __init__(self,
                 science,
                 wave_cal,
                 standard=None,
                 wave_cal_std=None,
                 flux_cal=None,
                 renderer='default',
                 verbose=False):
        '''
        twodspec : TwoDSpec object
        wavelength_calibrate : WavelengthPolyFit object
        flux_calibrate : StandardFlux object
        '''

        self.renderer = renderer
        self.verbose = verbose
        try:
            self.adu = science.adu
            self.aduerr = science.aduerr
            self.skyadu = science.skyadu
            self.exptime = science.exptime
        except:
            raise TypeError('Please provide a valid TwoDSpec.')

        try:
            self.set_wave_cal(wave_cal, 'science')
        except:
            raise TypeError('Please provide a WavelengthPolyFit.')

        if standard is not None:
            self.set_standard(standard)
            self.standard_imported = True
        else:
            self.standard_imported = False
            warnings.warn('The TwoDSpec of the standard observation is not '
                          'available. Flux calibration will not be performed.')

        if wave_cal_std is not None:
            self.set_wave_cal(wave_cal_std, 'standard')
            self.wav_cal_std_imported = True

        if (wave_cal_std is None) & (standard is not None):
            self.set_wave_cal(wave_cal, 'standard')
            self.wav_cal_std_imported = True
            warnings.warn(
                'The WavelengthPolyFit of the standard observation '
                'is not available. The wavelength calibration for the science '
                'frame is applied to the standard.')

        if flux_cal is not None:
            self.set_flux_cal(flux_cal)
            self.flux_imported = True
        else:
            self.flux_imported = False
            warnings.warn('The StandardFlux of the standard star is not '
                          'available. Flux calibration will not be performed.')

    def set_standard(self, standard):
        try:
            self.adu_std = standard.adu
            self.aduerr_std = standard.aduerr
            self.skyadu_std = standard.skyadu
            self.exptime_std = standard.exptime
        except:
            raise TypeError('Please provide a valid TwoDSpec.')

    def set_wave_cal(self, wave_cal, stype):
        if stype == 'science':
            try:
                self.pfit_type = wave_cal.pfit_type
                self.pfit = wave_cal.pfit
                if self.pfit_type == 'poly':
                    self.polyval = np.polyval
                elif self.pfit_type == 'legendre':
                    self.polyval = np.polynomial.legendre.legval
                elif self.pfit_type == 'chebyshev':
                    self.polyval = np.polynomial.chebyshev.chebval
                else:
                    raise ValueError(
                        'fittype must be: (1) poly; (2) legendre; or '
                        '(3) chebyshev')
            except:
                raise TypeError('Please provide a valid WavelengthPolyFit.')
        elif stype == 'standard':
            try:
                self.pfit_type_std = wave_cal.pfit_type
                self.pfit_std = wave_cal.pfit
                if self.pfit_type_std == 'poly':
                    self.polyval_std = np.polyval
                elif self.pfit_type_std == 'legendre':
                    self.polyval_std = np.polynomial.legendre.legval
                elif self.pfit_type_std == 'chebyshev':
                    self.polyval_std = np.polynomial.chebyshev.chebval
                else:
                    raise ValueError(
                        'fittype must be: (1) poly; (2) legendre; or '
                        '(3) chebyshev')
            except:
                raise TypeError('Please provide a valid WavelengthPolyFit.')
        elif stype == 'all':
            try:
                self.pfit_type = wave_cal.pfit_type
                self.pfit_type_std = wave_cal.pfit_type
                self.pfit = wave_cal.pfit
                self.pfit_std = wave_cal.pfit
                if self.pfit_type == 'poly':
                    self.polyval = np.polyval
                    self.polyval_std = np.polyval
                elif self.pfit_type == 'legendre':
                    self.polyval = np.polynomial.legendre.legval
                    self.polyval_std = np.polynomial.legendre.legval
                elif self.pfit_type == 'chebyshev':
                    self.polyval = np.polynomial.chebyshev.chebval
                    self.polyval_std = np.polynomial.chebyshev.chebval
                else:
                    raise ValueError(
                        'fittype must be: (1) poly; (2) legendre; or '
                        '(3) chebyshev')
            except:
                raise TypeError('Please provide a valid WavelengthPolyFit.')
        else:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')

    def set_flux_cal(self, flux_cal):
        try:
            self.group = flux_cal.group
            self.target = flux_cal.target
            self.wave_std_true = flux_cal.wave_std
            self.fluxmag_std_true = flux_cal.fluxmag_std
        except:
            raise TypeError('Please provide a valid StandardFlux.')

    def apply_wavelength_calibration(self, stype):
        if stype == 'science':
            pix = np.arange(len(self.adu))
            self.wave = self.polyval(self.pfit, pix)
        elif stype == 'standard':
            if self.standard_imported:
                pix_std = np.arange(len(self.adu_std))
                self.wave_std = self.polyval(self.pfit_std, pix_std)
            else:
                raise AttributeError(
                    'The TwoDSpec of the standard '
                    'observation is not available. Flux calibration will not '
                    'be performed.')
        elif stype == 'all':
            pix = np.arange(len(self.adu))
            pix_std = np.arange(len(self.adu_std))
            self.wave = self.polyval(self.pfit, pix)
            self.wave_std = self.polyval(self.pfit_std, pix_std)
        else:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')

    def compute_sencurve(self,
                         kind=3,
                         smooth=False,
                         slength=5,
                         sorder=2,
                         display=False):
        '''
        Get the standard flux or magnitude of the given target and group
        based on the given array of wavelengths. Some standard libraries
        contain the same target with slightly different values.

        Parameters
        ----------
        kind : string
            interpolation kind
            >>> [‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
                 ‘previous’, ‘next’]
            (default is 'cubic')
        smooth : tuple
            set to smooth the input ADU/flux/mag with scipy.signal.savgol_filter
            (default is True)
        slength : int
            SG-filter window size
        sorder : int
            SG-filter polynomial order

        Returns
        -------
        A scipy interp1d object.

        '''

        # Get the standard flux/magnitude
        self.slength = slength
        self.sorder = sorder
        self.smooth = smooth

        # Compute bin sizes such that the bin is roughly 10 A wide
        #wave_range = self.wave_std[-1] - self.wave_std[0]
        #bin_size = self.wave_std[1] - self.wave_std[0]

        # Find the centres of the first and last bins such that
        # the old and new spectra covers identical wavelength range
        #wave_lhs = self.wave_std[0] - bin_size / 2.
        #wave_rhs = self.wave_std[-1] + bin_size / 2.
        #wave_obs = np.arange(wave_lhs, wave_rhs, bin_size)
        #wave_std = self.wave_std

        if spectres_imported:
            # resampling both the observed and the database standard spectra
            # in unit of flux per second
            flux_std = spectres(self.wave_std_true, self.wave_std,
                                self.adu_std / self.exptime_std)
            flux_std_true = self.fluxmag_std_true
        else:
            flux_std = flux_std / self.exptime_std
            flux_std_true = itp.interp1d(self.wave_std_true,
                                         self.fluxmag_std_true)(self.wave_obs)
        # Get the sensitivity curve
        sensitivity = flux_std_true / flux_std
        mask = (np.isfinite(sensitivity) & (sensitivity > 0.) &
                ((self.wave_std_true < 6850.) | (self.wave_std_true > 7000.)) &
                ((self.wave_std_true < 7150.) | (self.wave_std_true > 7400.)) &
                ((self.wave_std_true < 7575.) | (self.wave_std_true > 7775.)))

        sensitivity = sensitivity[mask]
        wave_std = self.wave_std_true[mask]
        flux_std = flux_std[mask]

        # apply a Savitzky-Golay filter to remove noise and Telluric lines
        if smooth:
            sensitivity = signal.savgol_filter(sensitivity, slength, sorder)

        sencurve = itp.interp1d(wave_std,
                                np.log10(sensitivity),
                                kind=kind,
                                fill_value='extrapolate')

        self.sensitivity = sensitivity
        self.sencurve = sencurve
        self.wave_sen = wave_std
        self.flux_sen = flux_std

        # Diagnostic plot
        if display & plotly_imported:
            self.inspect_sencurve()

    def inspect_sencurve(self):
        fig = go.Figure()
        # show the image on the top
        fig.add_trace(
            go.Scatter(x=self.wave_sen,
                       y=self.flux_sen,
                       line=dict(color='royalblue', width=4),
                       name='ADU (Observed)'))

        fig.add_trace(
            go.Scatter(x=self.wave_sen,
                       y=self.sensitivity,
                       yaxis='y2',
                       line=dict(color='firebrick', width=4),
                       name='Sensitivity Curve'))

        fig.add_trace(
            go.Scatter(x=self.wave_sen,
                       y=10.**self.sencurve(self.wave_sen),
                       yaxis='y2',
                       line=dict(color='black', width=2),
                       name='Best-fit Sensitivity Curve'))

        if self.smooth:
            fig.update_layout(title='SG(' + str(self.slength) + ', ' +
                              str(self.sorder) + ')-Smoothed ' + self.group +
                              ' : ' + self.target,
                              yaxis_title='Smoothed ADU')
        else:
            fig.update_layout(title=self.group + ' : ' + self.target,
                              yaxis_title='ADU')

        fig.update_layout(autosize=True,
                          hovermode='closest',
                          showlegend=True,
                          xaxis_title=r'$\text{Wavelength / A}$',
                          yaxis=dict(title='ADU'),
                          yaxis2=dict(title='Sensitivity Curve',
                                      type='log',
                                      anchor="x",
                                      overlaying="y",
                                      side="right"),
                          legend=go.layout.Legend(x=0,
                                                  y=1,
                                                  traceorder="normal",
                                                  font=dict(
                                                      family="sans-serif",
                                                      size=12,
                                                      color="black"),
                                                  bgcolor='rgba(0,0,0,0)'),
                          height=800)

        if self.renderer == 'default':
            fig.show()
        else:
            fig.show(self.renderer)

        if self.verbose:
            print(fig.to_json())

    def apply_flux_calibration(self, stype='all'):
        if stype == 'science':
            self.flux = 10.**self.sencurve(self.wave) * self.adu
            self.fluxerr = 10.**self.sencurve(self.wave) * self.aduerr
            self.skyflux = 10.**self.sencurve(self.wave) * self.skyadu
        elif stype == 'standard':
            self.flux_std = 10.**self.sencurve(self.wave_std) * self.adu_std
            self.fluxerr_std = 10.**self.sencurve(
                self.wave_std) * self.aduerr_std
            self.skyflux_std = 10.**self.sencurve(
                self.wave_std) * self.skyadu_std
        elif stype == 'all':
            self.flux = 10.**self.sencurve(self.wave) * self.adu
            self.fluxerr = 10.**self.sencurve(self.wave) * self.aduerr
            self.skyflux = 10.**self.sencurve(self.wave) * self.skyadu
            self.flux_std = 10.**self.sencurve(self.wave_std) * self.adu_std
            self.fluxerr_std = 10.**self.sencurve(
                self.wave_std) * self.aduerr_std
            self.skyflux_std = 10.**self.sencurve(
                self.wave_std) * self.skyadu_std
        else:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')

    def inspect_reduced_spectrum(self, stype='all'):
        if stype == 'science':
            fig = go.Figure()
            # show the image on the top
            fig.add_trace(
                go.Scatter(x=self.wave,
                           y=self.flux,
                           line=dict(color='royalblue'),
                           name='Flux'))
            fig.add_trace(
                go.Scatter(x=self.wave,
                           y=self.fluxerr,
                           line=dict(color='firebrick'),
                           name='Flux Uncertainty'))
            fig.add_trace(
                go.Scatter(x=self.wave,
                           y=self.skyflux,
                           line=dict(color='orange'),
                           name='Sky Flux'))
            fig.update_layout(autosize=True,
                              hovermode='closest',
                              showlegend=True,
                              xaxis_title='Wavelength / A',
                              yaxis=dict(title='Flux', type='log'),
                              legend=go.layout.Legend(x=0,
                                                      y=1,
                                                      traceorder="normal",
                                                      font=dict(
                                                          family="sans-serif",
                                                          size=12,
                                                          color="black"),
                                                      bgcolor='rgba(0,0,0,0)'),
                              height=800)

            if self.renderer == 'default':
                fig.show()
            else:
                fig.show(self.renderer)

            if self.verbose:
                return fig.to_json()

        elif stype == 'standard':
            fig = go.Figure()
            # show the image on the top
            fig.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.flux_std,
                           line=dict(color='royalblue'),
                           name='Flux'))
            fig.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.fluxerr_std,
                           line=dict(color='orange'),
                           name='Flux Uncertainty'))
            fig.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.skyflux_std,
                           line=dict(color='firebrick'),
                           name='Sky Flux'))
            fig.add_trace(
                go.Scatter(x=self.wave_std_true,
                           y=self.fluxmag_std_true,
                           line=dict(color='black'),
                           name='Standard'))
            fig.update_layout(autosize=True,
                              hovermode='closest',
                              showlegend=True,
                              xaxis_title='Wavelength / A',
                              yaxis=dict(title='Flux', type='log'),
                              legend=go.layout.Legend(x=0,
                                                      y=1,
                                                      traceorder="normal",
                                                      font=dict(
                                                          family="sans-serif",
                                                          size=12,
                                                          color="black"),
                                                      bgcolor='rgba(0,0,0,0)'),
                              height=800)

            fig.show(self.renderer)
        elif stype == 'all':
            fig = go.Figure()
            # show the image on the top
            fig.add_trace(
                go.Scatter(x=self.wave,
                           y=self.flux,
                           line=dict(color='royalblue'),
                           name='Flux'))
            fig.add_trace(
                go.Scatter(x=self.wave,
                           y=self.fluxerr,
                           line=dict(color='orange'),
                           name='Flux Uncertainty'))
            fig.add_trace(
                go.Scatter(x=self.wave,
                           y=self.skyflux,
                           line=dict(color='firebrick'),
                           name='Sky Flux'))
            fig.update_layout(autosize=True,
                              hovermode='closest',
                              showlegend=True,
                              xaxis_title='Wavelength / A',
                              yaxis=dict(title='Flux', type='log'),
                              legend=go.layout.Legend(x=0,
                                                      y=1,
                                                      traceorder="normal",
                                                      font=dict(
                                                          family="sans-serif",
                                                          size=12,
                                                          color="black"),
                                                      bgcolor='rgba(0,0,0,0)'),
                              height=800)

            if self.renderer == 'default':
                fig.show()
            else:
                fig.show(self.renderer)

            fig2 = go.Figure()
            # show the image on the top
            fig2.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.flux_std,
                           line=dict(color='royalblue'),
                           name='Flux'))
            fig2.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.fluxerr_std,
                           line=dict(color='orange'),
                           name='Flux Uncertainty'))
            fig2.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.skyflux_std,
                           line=dict(color='firebrick'),
                           name='Sky Flux'))
            fig2.add_trace(
                go.Scatter(x=self.wave_std_true,
                           y=self.fluxmag_std_true,
                           line=dict(color='black'),
                           name='Standard'))
            fig2.update_layout(autosize=True,
                               hovermode='closest',
                               showlegend=True,
                               xaxis_title='Wavelength / A',
                               yaxis=dict(title='Flux', type='log'),
                               legend=go.layout.Legend(
                                   x=0,
                                   y=1,
                                   traceorder="normal",
                                   font=dict(family="sans-serif",
                                             size=12,
                                             color="black"),
                                   bgcolor='rgba(0,0,0,0)'),
                               height=800)

            if self.renderer == 'default':
                fig2.show()
            else:
                fig2.show(self.renderer)

            if self.verbose:
                return fig.to_json(), fig2.to_json()
        else:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')
