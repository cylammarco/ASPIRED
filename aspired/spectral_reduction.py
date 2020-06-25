import difflib
import json
import os
import sys
import warnings
from itertools import chain

import numpy as np
from astroscrappy import detect_cosmics
from astropy.io import fits
from astropy.stats import sigma_clip
from rascal.calibrator import Calibrator
from rascal.util import load_calibration_lines
from rascal.util import refine_peaks
from plotly import graph_objects as go
from plotly import io as pio
from scipy import signal
from scipy import interpolate as itp
from scipy.optimize import curve_fit
from spectres import spectres

from .image_reduction import ImageReduction

base_dir = os.path.dirname(__file__)


class TwoDSpec:
    def __init__(self,
                 data,
                 header=None,
                 saxis=1,
                 spatial_mask=(1, ),
                 spec_mask=(1, ),
                 flip=False,
                 cosmicray=False,
                 cosmicray_sigma=5.,
                 readnoise=None,
                 gain=None,
                 seeing=None,
                 exptime=None,
                 silence=False):
        '''
        This is a class for processing a 2D spectral image, the read noise,
        detector gain, seeing and exposure time will be automatically extracted
        from the FITS header if it conforms with the IAUFWG FITS standard.

        Currently, there is no automated way to decide if a flip is needed.

        The supplied file should contain 2 or 3 columns with the following
        structure:

        | column 1: one of bias, dark, flat or light
        | column 2: file location
        | column 3: HDU number (default to 0 if not given)

        If the 2D spectrum is

        +--------+--------+-------+-------+
        |  blue  |   red  | saxis |  flip |
        +========+========+=======+=======+
        |  left  |  right |   1   | False |
        +--------+--------+-------+-------+
        |  right |  left  |   1   |  True |
        +--------+--------+-------+-------+
        |   top  | bottom |   0   | False |
        +--------+--------+-------+-------+
        | bottom |   top  |   0   |  True |
        +--------+--------+-------+-------+

        Spectra are sorted by their brightness. If there are multiple spectra
        on the image, and the target is not the brightest source, use at least
        the number of spectra visible to eye and pick the one required later.
        The default automated outputs is the brightest one, which is the
        most common case for images from a long-slit spectrograph.

        Parameters
        ----------
        data: 2D numpy array (M x N) OR astropy.io.fits object
            2D spectral image in either format
        header: FITS header
            THIS WILL OVERRIDE the header from the astropy.io.fits object
        saxis: int
            Spectral direction, 0 for vertical, 1 for horizontal.
            (Default is 1)
        spatial_mask: 1D numpy array (N)
            Mask in the spatial direction, can be the indices of the pixels
            to be included (size <N) or a 1D numpy array of True/False (size N)
            (Default is (1,) i.e. keep everything)
        spec_mask: 1D numpy array (M)
            Mask in the spectral direction, can be the indices of the pixels
            to be included (size <M) or a 1D numpy array of True/False (size M)
            (Default is (1,) i.e. keep everything)
        flip: boolean
            If the frame has to be left-right flipped, set to True.
            (Deafult is False)
        cosmicray: boolean
            Set to True to apply cosmic ray rejection by sigma clipping with
            astroscrappy if available, otherwise a 2D median filter of size 5
            would be used. (default is True)
        cosmicray_sigma: float
            Cosmic ray sigma clipping limit (Deafult is 5.0)
        readnoise: float
            Readnoise of the detector, not important if noise estimation is
            not needed.
            (Deafult is None, which will be replaced with 1.0)
        gain: float
            Gain of the detector, not important if noise estimation is
            not needed.
            (Deafult is None, which will be replaced with 1.0)
        seeing: float
            Seeing in unit of arcsec, use as the first guess of the line
            spread function of the spectra.
            (Deafult is None, which will be replaced with 1.0)
        exptime: float
            Esposure time for the observation, not important if absolute flux
            calibration is not needed.
            (Deafult is None, which will be replaced with 1.0)
        silence: boolean
            Set to True to suppress all verbose warnings.
        '''

        # If data provided is an numpy array
        if isinstance(data, np.ndarray):
            img = data
            self.header = header
        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(data, fits.hdu.image.PrimaryHDU) or isinstance(
                data, fits.hdu.image.ImageHDU):
            img = data.data
            self.header = data.header
        # If it is an ImageReduction object
        elif isinstance(data, ImageReduction):
            # If the data is not reduced, reduce it here. Error handling is
            # done by the ImageReduction class
            if data.image_fits is None:
                data._create_image_fits()
            img = data.image_fits.data
            self.header = data.image_fits.header
        # If a filepath is provided
        elif isinstance(data, str):
            # If HDU number is provided
            if data[-1] == ']':
                filepath, hdunum = data.split('[')
                hdunum = hdunum[:-1]
            # If not, assume the HDU idnex is 0
            else:
                filepath = data
                hdunum = 0

            # Load the file and dereference it afterwards
            fitsfile_tmp = fits.open(filepath)[hdunum]
            img = fitsfile_tmp.data
            self.header = fitsfile_tmp.header
            fitsfile_tmp = None
        else:
            raise TypeError(
                'Please provide a numpy array, an ' +
                'astropy.io.fits.hdu.image.PrimaryHDU object or an ' +
                'ImageReduction object.')

        self.saxis = saxis
        if self.saxis == 1:
            self.waxis = 0
        else:
            self.waxis = 1
        self.spatial_mask = spatial_mask
        self.spec_mask = spec_mask
        self.flip = flip
        self.cosmicray_sigma = cosmicray_sigma

        # Default values if not supplied or cannot be automatically identified
        # from the header
        self.readnoise = 0.
        self.gain = 1.
        self.seeing = 1.
        self.exptime = 1.

        # Default keywords to be searched in the order in the list
        self.readnoise_keyword = ['RDNOISE', 'RNOISE', 'RN']
        self.gain_keyword = ['GAIN']
        self.seeing_keyword = ['SEEING', 'L1SEEING', 'ESTSEE']
        self.exptime_keyword = [
            'XPOSURE', 'EXPTIME', 'EXPOSED', 'TELAPSED', 'ELAPSED'
        ]

        # Get the Read Noise
        if readnoise is not None:
            if isinstance(readnoise, str):
                # use the supplied keyword
                self.readnoise = float(self.header[readnoise])
            elif np.isfinite(readnoise):
                # use the given readnoise value
                self.readnoise = float(readnoise)
            else:
                warnings.warn(
                    'readnoise has to be None, a numeric value or the ' +
                    'FITS header keyword, ' + str(readnoise) + ' is ' +
                    'given. It is set to 0.')
        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:
                readnoise_keyword_matched = np.in1d(self.readnoise_keyword,
                                                    self.header)
                if readnoise_keyword_matched.any():
                    self.readnoise = data.header[self.readnoise_keyword[
                        np.where(readnoise_keyword_matched)[0][0]]]
                else:
                    warnings.warn('Read Noise value cannot be identified. ' +
                                  'It is set to 0.')
            else:
                warnings.warn('Header is not provided. ' +
                              'Read Noise value is not provided. ' +
                              'It is set to 0.')

        # Get the Gain
        if gain is not None:
            if isinstance(gain, str):
                # use the supplied keyword
                self.gain = float(self.header[gain])
            elif np.isfinite(gain):
                # use the given gain value
                self.gain = float(gain)
            else:
                warnings.warn('Gain has to be None, a numeric value or the ' +
                              'FITS header keyword, ' + str(gain) + ' is ' +
                              'given. It is set to 1.')
        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:
                gain_keyword_matched = np.in1d(self.gain_keyword, self.header)
                if gain_keyword_matched.any():
                    self.gain = self.header[self.gain_keyword[np.where(
                        gain_keyword_matched)[0][0]]]
                else:
                    warnings.warn('Gain value cannot be identified. ' +
                                  'It is set to 1.')
            else:
                warnings.warn('Header is not provide. ' +
                              'Gain value is not provided. ' +
                              'It is set to 1.')

        # Get the Seeing
        if seeing is not None:
            if isinstance(seeing, str):
                # use the supplied keyword
                self.seeing = float(data.header[seeing])
            elif np.isfinite(gain):
                # use the given gain value
                self.seeing = float(seeing)
            else:
                warnings.warn(
                    'Seeing has to be None, a numeric value or the ' +
                    'FITS header keyword, ' + str(seeing) + ' is ' +
                    'given. It is set to 1.')
        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:
                seeing_keyword_matched = np.in1d(self.seeing_keyword,
                                                 self.header)
                if seeing_keyword_matched.any():
                    self.seeing = self.header[self.seeing_keyword[np.where(
                        seeing_keyword_matched)[0][0]]]
                else:
                    warnings.warn('Seeing value cannot be identified. ' +
                                  'It is set to 1.')
            else:
                warnings.warn('Header is not provide. ' +
                              'Seeing value is not provided. ' +
                              'It is set to 1.')

        # Get the Exposure Time
        if exptime is not None:
            if isinstance(exptime, str):
                # use the supplied keyword
                self.exptime = float(self.header[exptime])
            elif np.isfinite(gain):
                # use the given gain value
                self.exptime = float(exptime)
            else:
                warnings.warn(
                    'Exposure Time has to be None, a numeric value or the ' +
                    'FITS header keyword, ' + str(exptime) + ' is ' +
                    'given. It is set to 1.')
        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:
                exptime_keyword_matched = np.in1d(self.exptime_keyword,
                                                  self.header)
                if exptime_keyword_matched.any():
                    self.exptime = self.header[self.exptime_keyword[np.where(
                        exptime_keyword_matched)[0][0]]]
                else:
                    warnings.warn(
                        'Exposure Time value cannot be identified. ' +
                        'It is set to 1.')
            else:

                warnings.warn('Header is not provide. ' +
                              'Exposure Time value is not provided. ' +
                              'It is set to 1.')

        img = img / self.exptime
        self.silence = silence

        # cosmic ray rejection
        if cosmicray:
            img = detect_cosmics(img,
                                 sigclip=self.cosmicray_sigma,
                                 readnoise=self.readnoise,
                                 gain=self.gain,
                                 fsmode='convolve',
                                 psfmodel='gaussy',
                                 psfsize=31,
                                 psffwhm=self.seeing)[1]

        # the valid y-range of the chip (i.e. spatial direction)
        if (len(self.spatial_mask) > 1):
            if self.saxis == 1:
                img = img[self.spatial_mask]
            else:
                img = img[:, self.spatial_mask]

        # the valid x-range of the chip (i.e. spectral direction)
        if (len(self.spec_mask) > 1):
            if self.saxis == 1:
                img = img[:, self.spec_mask]
            else:
                img = img[self.spec_mask]

        # get the length in the spectral and spatial directions
        self.spec_size = np.shape(img)[self.waxis]
        self.spatial_size = np.shape(img)[self.saxis]
        if self.saxis == 0:
            self.img = np.transpose(img)
            img = None
        else:
            self.img = img
            img = None

        if self.flip:
            self.img = np.flip(self.img)

        # set the 2D histogram z-limits
        img_log = np.log10(self.img)
        img_log_finite = img_log[np.isfinite(img_log)]
        self.zmin = np.nanpercentile(img_log_finite, 5)
        self.zmax = np.nanpercentile(img_log_finite, 95)

    def set_readnoise_keyword(self, keyword_list, append=False):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        '''

        if append:
            self.readnoise_keyword += list(keyword_list)
        else:
            self.readnoise_keyword = list(keyword_list)

    def set_gain_keyword(self, keyword_list, append=False):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        '''

        if append:
            self.gain_keyword += list(keyword_list)
        else:
            self.gain_keyword = list(keyword_list)

    def set_seeing_keyword(self, keyword_list, append=False):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        '''

        if append:
            self.seeing_keyword += list(keyword_list)
        else:
            self.seeing_keyword = list(keyword_list)

    def set_exptime_keyword(self, keyword_list, append=False):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        '''

        if append:
            self.exptime_keyword += list(keyword_list)
        else:
            self.exptime_keyword = list(keyword_list)

    def set_header(self, header):
        '''
        Set/replace the header.

        Parameters
        ----------
        header: astropy.io.fits.header.Header
            FITS header from a single HDU.
        '''

        # If it is a fits.hdu.header.Header object
        if isinstance(header, fits.header.Header):
            self.header = data.header
        else:
            raise TypeError(
                'Please provide an astropy.io.fits.header.Header object.')

    def _gaus(self, x, a, b, x0, sigma):
        """
        Simple Gaussian function.

        Parameters
        ----------
        x: float or 1-d numpy array
            The data to evaluate the Gaussian over
        a: float
            the amplitude
        b: float
            the constant offset
        x0: float
            the center of the Gaussian
        sigma: float
            the width of the Gaussian

        Returns
        -------
        Array or float of same type as input (x).
        """

        return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b

    def _identify_spectra(self, f_height, display, renderer, jsonstring,
                          iframe, open_iframe):
        """
        Identify peaks assuming the spatial and spectral directions are
        aligned with the X and Y direction within a few degrees.

        Parameters
        ----------
        f_height: float
            The minimum intensity as a fraction of maximum height.
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        iframe: boolean
            Save as an iframe, can work concurrently with other renderer
            apart from exporting jsonstring.
        open_iframe: boolean
            Open the iframe in the default browser if set to True.

        Returns
        -------
        peaks_y :
            Array or float of the pixel values of the detected peaks
        heights_y :
            Array or float of the integrated counts at the peaks
        """

        ydata = np.arange(self.spec_size)
        ztot = np.nanmedian(self.img, axis=1)

        # get the height thershold
        height = np.nanmax(ztot) * f_height

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
            fig.add_trace(
                go.Heatmap(z=np.log10(self.img),
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

            if jsonstring:
                return fig.to_json()
            if iframe:
                if open_iframe:
                    pio.write_html(fig, 'identify_spectra.html')
                else:
                    pio.write_html(fig,
                                   'identify_spectra.html',
                                   auto_open=False)
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

        self.peak = peaks_y
        self.peak_height = heights_y

    def _optimal_signal(self, pix, xslice, sky, mu, sigma, tol=1e-6):
        """
        Iterate to get the optimal signal. Following the algorithm on
        Horne, 1986, PASP, 98, 609 (1986PASP...98..609H).

        Parameters
        ----------
        pix: 1-d numpy array
            pixel number along the spatial direction
        xslice: 1-d numpy array
            ADU along the pix
        sky: 1-d numpy array
            ADU of the fitted sky along the pix
        mu: float
            The center of the Gaussian
        sigma: float
            The width of the Gaussian

        Returns
        -------
        signal: float
            The optimal signal.
        noise: float
            The noise associated with the optimal signal.
        """

        # construct the Gaussian model
        P = self._gaus(pix, 1., 0., mu, sigma)
        P /= np.nansum(P)

        #
        signal = xslice - sky
        signal[signal < 0] = 0.
        # weight function and initial values
        signal1 = np.nansum(signal)
        var1 = self.readnoise + np.abs(xslice) / self.gain
        variance1 = 1. / np.nansum(P**2. / var1)

        sky_median = np.median(sky)

        signal_diff = 1
        variance_diff = 1
        i = 0
        suboptimal = False

        mask = np.ones(len(P), dtype=bool)

        while (signal_diff > tol) | (variance_diff > tol):

            signal0 = signal1
            var0 = var1
            variance0 = variance1

            mask_cr = mask.copy()

            # cosmic ray mask, only start considering after the 1st iteration
            # masking at most 2 pixels
            if i > 0:
                ratio = (self.cosmicray_sigma**2. * var0) / (signal -
                                                             P * signal0)**2.
                comparison = np.sum(ratio > 1)
                if comparison == 1:
                    mask_cr[np.argmax(ratio)] = False
                if comparison >= 2:
                    mask_cr[np.argsort(ratio)[-2:]] = False

            # compute signal and noise
            signal1 = np.nansum((P * signal / var0)[mask_cr]) / \
                np.nansum((P**2. / var0)[mask_cr])
            var1 = self.readnoise + np.abs(P * signal1 + sky) / self.gain
            variance1 = 1. / np.nansum((P[mask_cr]**2. / var1[mask_cr]))

            signal_diff = abs((signal1 - signal0) / signal0)
            variance_diff = abs((variance1 - variance0) / variance0)

            i += 1

            if i == 99:
                suboptimal = True
                break

        signal = signal1
        noise = np.sqrt(variance1)

        return signal, noise, suboptimal

    def ap_trace(self,
                 nspec=1,
                 nwindow=25,
                 spec_sep=5,
                 resample_factor=10,
                 rescale=False,
                 scaling_min=0.995,
                 scaling_max=1.005,
                 scaling_step=0.001,
                 percentile=5,
                 tol=3,
                 polydeg=3,
                 ap_faint=10,
                 display=False,
                 renderer='default',
                 jsonstring=False,
                 iframe=False,
                 open_iframe=False):
        '''
        Aperture tracing by first using cross-correlation then the peaks are
        fitting with a polynomial with an order of floor(nwindow, 10) with a
        minimum order of 1. Nothing is returned unless jsonstring of the
        plotly graph is set to be returned.

        Each spectral slice is convolved with the adjacent one in the spectral
        direction. Basic tests show that the geometrical distortion from one
        end to the other in the spectral direction is small. With LT/SPRAT, the
        linear distortion is less than 0.5%, thus, even provided as an option,
        the rescale option is set to False by default. Given how unlikely a
        geometrical distortion correction is needed, higher order correction
        options are not provided.

        A rough estimation on the background level is done by taking the
        n-th percentile percentile of the slice, a rough guess can improve the
        cross-correlation process significantly due to low dynamic range in a
        typical spectral image. The removing of the "background" can massively
        improve the contrast between the peaks and the relative background,
        hence the correlation method is more likely to yield a true positive.

        The trace(s), i.e. the spatial positions of the spectra (Y-axis),
        found will be stored as the properties of the TwoDSpec object as a
        1D numpy array, of length N, which is the size of the spectrum after
        applying the spec_mask. The line spread function is stored in
        trace_sigma, by fitting a gaussian on the shift-corrected stack of the
        spectral slices. Given the scaling was found to be small, reporting
        a single value of the averaged gaussian sigma is sufficient as the
        first guess to be used by the aperture extraction function.

        Parameters
        ----------
        nspec: int
            Number of spectra to be extracted.
        nwindow: int
            Number of spectral slices (subspectra) to be produced for
            cross-correlation.
        spec_sep: int
            Minimum separation between sky lines.
        resample_factor: int
            Number of times the collapsed 1D slices in the spatial directions
            are to be upsampled.
        rescale: boolean
            Fit for the linear scaling factor between adjacent slices.
        scaling_min: float
            Minimum scaling factor to be fitted.
        scaling_max: float
            Maximum scaling factor to be fitted.
        scaling_step: float
            Steps of the scaling factor.
        percentile: float
            The percentile of the flux to be used as the estimate of the
            background sky level to the first order. [ADU]
        tol: float
            Maximum allowed shift between neighbouring slices, this value is
            referring to native pixel size without the application of the
            resampling or rescaling. [pix]
        polydeg: int
            Degree of the polynomial fit of the trace.
        ap_faint: float
            The percentile threshold of ADU aperture to be used for fitting
            the trace. Note that this percentile is of the ADU, not of the
            number of subspectra.
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        iframe: boolean
            Save as an iframe, can work concurrently with other renderer
            apart from exporting jsonstring.
        open_iframe: boolean
            Open the iframe in the default browser if set to True.

        Returns
        -------
        json string if jsonstring is True, otherwise only an image is displayed
        '''

        # Get the shape of the 2D spectrum and define upsampling ratio
        nwave = len(self.img[0])
        nspatial = len(self.img)

        nresample = nspatial * resample_factor

        # window size
        w_size = nwave // nwindow
        img_split = np.array_split(self.img, nwindow, axis=1)
        start_window_idx = nwindow // 2

        lines_ref_init = np.nanmedian(img_split[start_window_idx], axis=1)
        lines_ref_init[np.isnan(lines_ref_init)] = 0.
        lines_ref_init_resampled = signal.resample(lines_ref_init, nresample)

        # linear scaling limits
        if rescale:
            scaling_range = np.arange(scaling_min, scaling_max, scaling_step)
        else:
            scaling_range = np.ones(1)

        # estimate the n-th percentile as the sky background level
        lines_ref = lines_ref_init_resampled - np.percentile(
            lines_ref_init_resampled, percentile)

        shift_solution = np.zeros(nwindow)
        scale_solution = np.ones(nwindow)

        # maximum shift (SEMI-AMPLITUDE) from the neighbour (pixel)
        tol_len = int(tol * resample_factor)

        spec_spatial = np.zeros(nresample)

        pix_init = np.arange(nresample)
        pix_resampled = pix_init

        # Scipy correlate method
        for i in chain(range(start_window_idx, nwindow),
                       range(start_window_idx - 1, -1, -1)):

            # smooth by taking the median
            lines = np.nanmedian(img_split[i], axis=1)
            lines[np.isnan(lines)] = 0.
            lines = signal.resample(lines, nresample)
            lines = lines - np.nanpercentile(lines, percentile)

            # cross-correlation values and indices
            corr_val = np.zeros(len(scaling_range))
            corr_idx = np.zeros(len(scaling_range))

            # upsample by the same amount as the reference
            for j, scale in enumerate(scaling_range):

                # Upsampling the reference lines
                lines_ref_j = signal.resample(lines_ref,
                                              int(nresample * scale))

                # find the linear shift
                corr = signal.correlate(lines_ref_j, lines)

                # only consider the defined range of shift tolerance
                corr = corr[nresample - 1 - tol_len:nresample + tol_len]

                # Maximum corr position is the shift
                corr_val[j] = np.nanmax(corr)
                corr_idx[j] = np.nanargmax(corr) - tol_len

            # Maximum corr_val position is the scaling
            shift_solution[i] = corr_idx[np.nanargmax(corr_val)]
            scale_solution[i] = scaling_range[np.nanargmax(corr_val)]

            # Align the spatial profile before stacking
            if i == (start_window_idx - 1):
                pix_resampled = pix_init
            pix_resampled = pix_resampled * scale_solution[i] + shift_solution[
                i]

            spec_spatial += spectres(np.arange(nresample),
                                     pix_resampled,
                                     lines,
                                     verbose=False)

            # Update (increment) the reference line
            if (i == nwindow - 1):
                lines_ref = lines_ref_init_resampled
            else:
                lines_ref = lines

        nscaled = (nresample * scale_solution).astype('int')

        # Find the spectral position in the middle of the gram in the upsampled
        # pixel location location
        peaks = signal.find_peaks(spec_spatial,
                                  distance=spec_sep,
                                  prominence=0)

        # update the number of spectra if the number of peaks detected is less
        # than the number requested
        self.nspec = min(len(peaks[0]), nspec)

        # Sort the positions by the prominences, and return to the original
        # scale (i.e. with subpixel position)
        spec_init = np.sort(peaks[0][np.argsort(-peaks[1]['prominences'])]
                            [:self.nspec]) / resample_factor

        # Create array to populate the spectral locations
        spec_idx = np.zeros((len(spec_init), len(img_split)))

        # Populate the initial values
        spec_idx[:, start_window_idx] = spec_init

        # Pixel positions of the mid point of each data_split (spectral)
        spec_pix = np.arange(len(img_split)) * w_size + w_size / 2.

        # Looping through pixels larger than middle pixel
        for i in range(start_window_idx + 1, nwindow):
            spec_idx[:,
                     i] = (spec_idx[:, i - 1] * resample_factor * nscaled[i] /
                           nresample - shift_solution[i]) / resample_factor

        # Looping through pixels smaller than middle pixel
        for i in range(start_window_idx - 1, -1, -1):
            spec_idx[:, i] = (spec_idx[:, i + 1] * resample_factor -
                              shift_solution[i + 1]) / (
                                  int(nresample * scale_solution[i + 1]) /
                                  nresample) / resample_factor

        ap = np.zeros((len(spec_idx), nwave))
        ap_sigma = np.zeros(len(spec_idx))

        for i in range(len(spec_idx)):

            # Get the median of the subspectrum and then get the ADU at the
            # centre of the aperture
            ap_val = np.zeros(nwindow)
            for j in range(nwindow):
                # rounding
                idx = int(spec_idx[i][j] + 0.5)
                ap_val[j] = np.nanmedian(img_split[j], axis=1)[idx]

            # Mask out the faintest ap_faint percentile
            mask = (ap_val > np.percentile(ap_val, ap_faint))

            # fit the trace
            ap_p = np.polyfit(spec_pix[mask], spec_idx[i][mask], int(polydeg))
            ap[i] = np.polyval(ap_p, np.arange(nwave))

            # Get the centre of the upsampled spectrum
            ap_centre_idx = ap[i][start_window_idx] * resample_factor

            # Get the indices for the 10 pixels on the left and right of the
            # spectrum, and apply the resampling factor.
            start_idx = int(ap_centre_idx - 10 * resample_factor + 0.5)
            end_idx = start_idx + 20 * resample_factor + 1

            # compute ONE sigma for each trace
            pguess = [
                np.nanmax(spec_spatial[start_idx:end_idx]),
                np.nanpercentile(spec_spatial, 10), ap_centre_idx, 3.
            ]

            popt, pcov = curve_fit(self._gaus,
                                   range(start_idx, end_idx),
                                   spec_spatial[start_idx:end_idx],
                                   p0=pguess)
            ap_sigma[i] = popt[3] / resample_factor

        self.trace = ap
        self.trace_sigma = ap_sigma

        # Plot
        if display:

            fig = go.Figure()

            fig.add_trace(
                go.Heatmap(z=np.log10(self.img),
                           zmin=self.zmin,
                           zmax=self.zmax,
                           colorscale="Viridis",
                           colorbar=dict(title='log(ADU / s)')))
            for i in range(len(spec_idx)):
                fig.add_trace(
                    go.Scatter(x=np.arange(nwave),
                               y=ap[i],
                               line=dict(color='black')))
                fig.add_trace(
                    go.Scatter(x=spec_pix,
                               y=spec_idx[i],
                               mode='markers',
                               marker=dict(color='grey')))
            fig.add_trace(
                go.Scatter(x=np.ones(len(spec_idx)) *
                           spec_pix[start_window_idx],
                           y=spec_idx[:, start_window_idx],
                           mode='markers',
                           marker=dict(color='firebrick')))
            fig.update_layout(autosize=True,
                              yaxis_title='Spatial Direction / pixel',
                              xaxis=dict(zeroline=False,
                                         showgrid=False,
                                         title='Spectral Direction / pixel'),
                              bargap=0,
                              hovermode='closest',
                              showlegend=False,
                              height=800)
            if jsonstring:
                return fig.to_json()
            if iframe:
                if open_iframe:
                    pio.write_html(fig, 'ap_trace.html')
                else:
                    pio.write_html(fig, 'ap_trace.html', auto_open=False)
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

    def add_trace(self, trace, trace_sigma):
        '''
        Add user-supplied trace. The trace has to have the size as the 2D
        spectral image in the spectral direction.

        Parameters
        ----------
        trace: 1D numpy array of list of 1D numpy array
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: float or list of float or numpy array of float
            Standard deviation of the Gaussian profile of a trace

        '''

        # If only one trace is provided
        if np.shape(np.shape(trace))[0] == 1:
            self.nspec = 1
            self.trace = [trace]

            if isinstance(trace_sigma, float) or isinstance(
                    trace_sigma, np.ndarray):
                self.trace_sigma = np.array(trace_sigma).reshape(-1)
            else:
                raise TypeError('The trace_sigma has to be a float. A ' +\
                          str(type(trace_sigma)) + ' is given.')

        # If there are multiple traces
        else:
            self.nspec = np.shape(trace)[0]
            if np.shape(trace)[1] != 1:
                self.trace = np.array(trace)
            else:
                self.trace = np.zeros((self.nspec, self.spec_size))
                for i in range(self.nspec):
                    self.trace[i] = [np.ones(self.spec_size) * trace[i]]

            # If all traces have the same line spread function
            if isinstance(trace_sigma, float):
                self.trace_sigma = np.ones(self.nspec) * trace_sigma
            elif (len(trace_sigma) == self.nspec):
                self.trace_sigma = np.array(trace_sigma)
            else:
                raise ValueError(
                    'The trace_sigma should be a single float or an '
                    'array of a size of the number the of traces.')

    def ap_extract(self,
                   apwidth=7,
                   skysep=3,
                   skywidth=5,
                   skydeg=1,
                   optimal=True,
                   display=False,
                   renderer='default',
                   jsonstring=False,
                   iframe=False,
                   open_iframe=False):
        """
        Extract the spectra using the traces, support tophat or optimal
        extraction. The sky background is fitted in one dimention only. The
        uncertainty at each pixel is also computed, but the values are only
        meaningful if correct gain and read noise are provided.

        Tophat extraction: Float is accepted but will be rounded to an int,
                           which gives the constant aperture size on either
                           side of the trace to extract.
        Optimal extraction: Float or 1-d array of the same size as the trace.
                            If a float is supplied, a fixed standard deviation
                            will be used to construct the gaussian weight
                            function along the entire spectrum.

        Nothing is returned unless jsonstring of the plotly graph is set to be
        returned. The adu, adusky and aduerr are stored as properties of the
        TwoDSpec object.

        adu: 1-d array
            The summed adu at each column about the trace. Note: is not
            sky subtracted!
        adusky: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        aduerr: 1-d array
            the uncertainties of the adu values

        Parameters
        ----------
        apwidth: int (or list of int)
            Half the size of the aperature (fixed value for tophat extraction).
            If a list of two ints are provided, the first element is the
            lower half of the aperture  and the second one is the upper half
            (up and down refers to large and small pixel values)
        skysep: int
            The separation in pixels from the aperture to the sky window.
            (Default is 3)
        skywidth: int
            The width in pixels of the sky windows on either side of the
            aperture. Zero (0) means ignore sky subtraction. (Default is 7)
        skydeg: int
            The polynomial order to fit between the sky windows.
            (Default is 0, i.e. constant flat sky level)
        optimal: boolean
            Set optimal extraction. (Default is True)
        silence: boolean
            Set to disable warning/error messages. (Default is False)
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        iframe: boolean
            Save as an iframe, can work concurrently with other renderer
            apart from exporting jsonstring.
        open_iframe: boolean
            Open the iframe in the default browser if set to True.

        """

        len_trace = len(self.trace[0])
        adusky = np.zeros((self.nspec, len_trace))
        aduerr = np.zeros((self.nspec, len_trace))
        adu = np.zeros((self.nspec, len_trace))
        suboptimal = np.zeros((self.nspec, len_trace), dtype=bool)
        for j in range(self.nspec):

            for i, pos in enumerate(self.trace[j]):
                itrace = int(pos)
                pix_frac = pos - itrace

                if isinstance(apwidth, int):
                    # first do the aperture adu
                    widthdn = apwidth
                    widthup = apwidth
                elif len(apwidth) == 2:
                    widthdn = apwidth[0]
                    widthup = apwidth[1]
                else:
                    raise TypeError(
                        'apwidth can only be an int or a list of two ints')

                # fix width if trace is too close to the edge
                if (itrace + widthup > self.spatial_size):
                    widthup = spatial_size - itrace - 1
                if (itrace - widthdn < 0):
                    widthdn = itrace - 1  # i.e. starting at pixel row 1

                # simply add up the total adu around the trace +/- width
                xslice = self.img[itrace - widthdn:itrace + widthup + 1, i]
                adu_ap = np.sum(xslice) - pix_frac * xslice[0] - (
                    1 - pix_frac) * xslice[-1]

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
                        polyfit = np.polyfit(y, z, skydeg)
                        # define the aperture in this column
                        ap = np.arange(itrace - widthdn, itrace + widthup + 1)
                        # evaluate the polynomial across the aperture, and sum
                        adusky_slice = np.polyval(polyfit, ap)
                        adusky[j][i] = np.sum(
                            adusky_slice) - pix_frac * adusky_slice[0] - (
                                1 - pix_frac) * adusky_slice[-1]
                    elif (skydeg == 0):
                        adusky[j][i] = (widthdn + widthup) * np.nanmean(z)

                else:
                    polyfit = [0., 0.]

                # if optimal extraction
                if optimal:
                    pix = np.arange(itrace - widthdn, itrace + widthup + 1)
                    # Fit the sky background
                    if (skydeg > 0):
                        sky = np.polyval(polyfit, pix)
                    else:
                        sky = np.ones(len(pix)) * np.nanmean(z)
                    # Get the optimal signals
                    adu[j][i], aduerr[j][i], suboptimal[j][
                        i] = self._optimal_signal(pix, xslice, sky,
                                                  self.trace[j][i],
                                                  self.trace_sigma[j])
                else:
                    #-- finally, compute the error in this pixel
                    sigB = np.nanstd(z)  # standarddev in the background data
                    nB = len(y)  # number of bkgd pixels
                    nA = apwidth * 2. + 1  # number of aperture pixels

                    # based on aperture phot err description by F. Masci, Caltech:
                    # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
                    aduerr[j][i] = np.sqrt((adu_ap - adusky[j][i]) /
                                           self.gain + (nA + nA**2. / nB) *
                                           (sigB**2.))
                    adu[j][i] = adu_ap - adusky[j][i]

            # If more than a third of the spectrum is extracted suboptimally
            if np.sum(suboptimal[j]) / i > 0.333:
                if not self.silence:
                    print(
                        'Signal extracted is likely to be suboptimal, please try '
                        'a longer iteration, larger tolerance or revert to '
                        'top-hat extraction.')

            if display:
                min_trace = int(min(self.trace[j]) + 0.5)
                max_trace = int(max(self.trace[j]) + 0.5)

                fig = go.Figure()
                # the 3 is to show a little bit outside the extraction regions
                img_display = np.log10(self.img[
                    max(0, min_trace - widthdn - skysep - skywidth -
                        3):min(max_trace + widthup + skysep +
                               skywidth, len(self.img[0])) + 3, :])

                # show the image on the top
                # the 3 is the show a little bit outside the extraction regions
                fig.add_trace(
                    go.Heatmap(
                        x=np.arange(len_trace),
                        y=np.arange(
                            max(0,
                                min_trace - widthdn - skysep - skywidth - 3),
                            min(max_trace + widthup + skysep + skywidth + 3,
                                len(self.img[0]))),
                        z=img_display,
                        colorscale="Viridis",
                        zmin=self.zmin,
                        zmax=self.zmax,
                        xaxis='x',
                        yaxis='y',
                        colorbar=dict(title='log(ADU / s)')))

                # Middle black box on the image
                fig.add_trace(
                    go.Scatter(x=list(
                        np.concatenate(
                            (np.arange(len_trace), np.arange(len_trace)[::-1],
                             np.zeros(1)))),
                               y=list(
                                   np.concatenate(
                                       (self.trace[j] - widthdn - 1,
                                        self.trace[j][::-1] + widthup + 1,
                                        np.ones(1) *
                                        (self.trace[j][0] - widthdn - 1)))),
                               xaxis='x',
                               yaxis='y',
                               mode='lines',
                               line_color='black',
                               showlegend=False))

                # Lower red box on the image
                lower_redbox_upper_bound = self.trace[j] - widthdn - skysep - 1
                lower_redbox_lower_bound = self.trace[
                    j][::-1] - widthdn - skysep - max(skywidth, (y1 - y0) - 1)

                if (itrace - widthdn >= 0) & (skywidth > 0):
                    fig.add_trace(
                        go.Scatter(x=list(
                            np.concatenate(
                                (np.arange(len_trace),
                                 np.arange(len_trace)[::-1], np.zeros(1)))),
                                   y=list(
                                       np.concatenate(
                                           (lower_redbox_upper_bound,
                                            lower_redbox_lower_bound,
                                            np.ones(1) *
                                            lower_redbox_upper_bound[0]))),
                                   line_color='red',
                                   xaxis='x',
                                   yaxis='y',
                                   mode='lines',
                                   showlegend=False))

                # Upper red box on the image
                upper_redbox_upper_bound = self.trace[
                    j] + widthup + skysep + min(skywidth, (y3 - y2) + 1)
                upper_redbox_lower_bound = self.trace[
                    j][::-1] + widthup + skysep + 1

                if (itrace + widthup <= self.spatial_size) & (skywidth > 0):
                    fig.add_trace(
                        go.Scatter(x=list(
                            np.concatenate(
                                (np.arange(len_trace),
                                 np.arange(len_trace)[::-1], np.zeros(1)))),
                                   y=list(
                                       np.concatenate(
                                           (upper_redbox_upper_bound,
                                            upper_redbox_lower_bound,
                                            np.ones(1) *
                                            upper_redbox_upper_bound[0]))),
                                   xaxis='x',
                                   yaxis='y',
                                   mode='lines',
                                   line_color='red',
                                   showlegend=False))
                # plot the SNR
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=adu[j] / aduerr[j],
                               xaxis='x2',
                               yaxis='y3',
                               line=dict(color='slategrey'),
                               name='Signal-to-Noise Ratio'))

                # extrated source, sky and uncertainty
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=adusky[j],
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='firebrick'),
                               name='Sky ADU / s'))
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=aduerr[j],
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='orange'),
                               name='Uncertainty'))
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=adu[j],
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='royalblue'),
                               name='Target ADU / s'))

                # Decorative stuff
                fig.update_layout(
                    autosize=True,
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(zeroline=False,
                               domain=[0.5, 1],
                               showgrid=False,
                               title='Spatial Direction / pixel'),
                    yaxis2=dict(
                        range=[
                            min(
                                np.nanmin(
                                    sigma_clip(np.log10(adu[j]),
                                               sigma=5.,
                                               masked=False)),
                                np.nanmin(
                                    sigma_clip(np.log10(aduerr[j]),
                                               sigma=5.,
                                               masked=False)),
                                np.nanmin(
                                    sigma_clip(np.log10(adusky[j]),
                                               sigma=5.,
                                               masked=False)), 1),
                            max(np.nanmax(np.log10(adu[j])),
                                np.nanmax(np.log10(adusky[j])))
                        ],
                        zeroline=False,
                        domain=[0, 0.5],
                        showgrid=True,
                        type='log',
                        title='log(ADU / Count / s)',
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
                                            font=dict(family="sans-serif",
                                                      size=12,
                                                      color="black"),
                                            bgcolor='rgba(0,0,0,0)'),
                    bargap=0,
                    hovermode='closest',
                    showlegend=True,
                    height=800)
                if jsonstring:
                    return fig.to_json()
                if iframe:
                    if open_iframe:
                        pio.write_html(fig, 'ap_extract_' + str(j) + 'html')
                    else:
                        pio.write_html(fig,
                                       'ap_extract_' + str(j) + 'html',
                                       auto_open=False)
                if renderer == 'default':
                    fig.show()
                else:
                    fig.show(renderer)

        self.adu = adu
        self.aduerr = aduerr
        self.adusky = adusky

    def _create_trace_fits(self):
        # Put the reduced data in FITS format with an image header
        self.trace_hdulist = np.array([None] * self.nspec, dtype='object')
        for i in range(self.nspec):
            self.trace_hdulist[i] = fits.HDUList([
                fits.ImageHDU(self.trace[i]),
                fits.ImageHDU(np.array(self.trace_sigma[i], ndmin=1))
            ])

    def _create_adu_fits(self):
        # Put the reduced data in FITS format with an image header
        self.adu_hdulist = np.array([None] * self.nspec, dtype='object')
        for i in range(self.nspec):
            self.adu_hdulist[i] = fits.HDUList([
                fits.ImageHDU(self.adu[i]),
                fits.ImageHDU(self.aduerr[i]),
                fits.ImageHDU(self.adusky[i])
            ])

    def save_fits(self,
                  output='trace+adu',
                  filename='TwoDSpecExtracted',
                  overwrite=False):
        '''
        Save the reduced image to disk.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            adu: 3 HDUs
                Flux, uncertainty and sky (bin width = per wavelength)
            trace: 2 HDU
                Pixel position of the trace in the spatial direction
                and the best fit gaussian line spread function sigma
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        overwrite: boolean
            Default is False.

        '''

        filename = os.path.splitext(filename)[0]

        output_split = output.split('+')

        if 'adu' in output_split:
            self._create_adu_fits()

        if 'trace' in output_split:
            self._create_trace_fits()

        # Save each trace as a separate FITS file
        for i in range(self.nspec):

            # Empty list for appending HDU lists
            hdu_output = fits.HDUList()
            if 'adu' in output_split:
                hdu_output.extend(self.adu_hdulist[i])

            if 'trace' in output_split:
                hdu_output.extend(self.trace_hdulist[i])

            # Convert the first HDU to PrimaryHDU
            hdu_output[0] = fits.PrimaryHDU(hdu_output[0].data,
                                            hdu_output[0].header)
            hdu_output.update_extend()

            # Save file to disk
            hdu_output.writeto(filename + '_' + str(i) + '.fits',
                               overwrite=overwrite)


class WavelengthCalibration():
    def __init__(self, silence=False):
        '''
        This is a wrapper for using RASCAL to perform wavelength calibration,
        which can handle arc lamps containing Xe, Cu, Ar, Hg, He, Th, Fe. This
        guarantees to provide something sensible or nothing at all. It will
        require some fine-tuning when using the first time. The more GOOD
        initial guesses provided, the faster the solution converges and with
        better fit. Knowing the dispersion, wavelength ranges and one or two
        known lines will significantly improve the fit. Conversely, wrong
        values supplied by the user will siginificantly distort the solution
        as any user supplied information will be treated as the ground truth.

        Deatils of how RASCAL works should be referred to

            https://rascal.readthedocs.io/en/latest/

        Parameters
        ----------
        silence: boolean
            Set to True to suppress all verbose warnings.
        '''

        self.silence = silence
        self.arc = None
        self.trace = None
        self.nspec = None
        self.polyfit = None
        self.arcspec = None
        self.pixel_list = None
        self.polyfit_coeff = None
        self.polyfit_type = None
        self.polyval = None
        self.peaks_raw = None
        self.peaks = None
        self.wavecal_hdulist = None
        self.polyfit_hdulist = None

    def add_arc_lines(self, peaks):
        '''
        Provide the pixel locations of the arc lines.

        Parameters
        ----------
        peaks: list of list or list of arrays
            The pixel locations of the arc lines. Multiple traces of the arc
            can be provided as list of list or list of arrays.
        '''

        if not isinstance(peaks, list):
            self.peaks = [peaks]
        else:
            self.peaks = peaks

        self.nspec = len(peaks)

    def add_arcspec(self, spectrum):
        '''
        Provide the collapsed 1D spectrum/a of the arc image.

        Parameters
        ----------
        spectrum: list of list or list of arrays
            The ADU/flux of the 1D arc spectrum/a. Multiple spectrum/a
            can be provided as list of list or list of arrays.
        '''

        if not isinstance(arcspec, list):
            self.arcspec = [arcspec]
        else:
            self.arcspec = arcspec

        self.nspec = len(arcspec)

    def add_arc(self, arc):
        '''
        To add or replace an arc image. Make sure left (small index) is blue,
        right (large index) is red.

        Parameters
        ----------
        arc: 2D numpy array, PrimaryHDU object or ImageReduction object
            The image of the arc image.
        '''

        # If data provided is an numpy array
        if isinstance(arc, np.ndarray):
            self.arc = arc
        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(arc, fits.hdu.image.PrimaryHDU) or isinstance(
                arc, fits.hdu.image.ImageHDU):
            self.arc = arc.data
        # If it is an ImageReduction object
        elif isinstance(arc, ImageReduction):
            self.arc = arc.arc_master
        else:
            raise TypeError(
                'Please provide a numpy array, an ' +
                'astropy.io.fits.hdu.image.PrimaryHDU object or an ' +
                'ImageReduction object.')

    def add_spec(self, adu, aduerr=None, adusky=None):

        if not np.shape(np.shape(adu))[0] == 2:
            self.adu = [adu]
            self.nspec = np.shape(np.shape(adu))[0]
        else:
            self.adu = adu
            self.nspec = np.shape(adu)[0]

        if aduerr is not None:
            if not np.shape(np.shape(aduerr))[0] == 2:
                aduerr = [aduerr]
            assert len(aduerr) == len(self.adu), ('There should be the same '
                                                  'number of aduerr and adu.')
            assert len(aduerr[0]) == len(
                self.adu[0]), ('The length of each '
                               'aduerr and adu should be the same.')

        self.aduerr = aduerr

        if adusky is not None:
            if not np.shape(np.shape(adusky))[0] == 2:
                adusky = [adusky]
            assert len(adusky) == len(self.adu), ('There should be the same '
                                                  'number of adusky and adu.')
            assert len(adusky[0]) == len(
                self.adu[0]), ('The length of each '
                               'aduerr and adu should be the same.')

        self.adusky = adusky

    def add_trace(self, trace, trace_sigma, pixel_list=None):
        '''
        Add user-supplied trace. The trace has to be the size as the 2D
        spectral image in the spectral direction. Make sure the trace pixels
        are corresponding to the arc image, left (small index) is blue, right
        (large index) is red.

        Parameters
        ----------
        trace: 1D numpy array of list of 1D numpy array
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        pixel_list: list or numpy array
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(len(spec.trace[0])), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        '''

        # If only one trace is provided
        if np.shape(np.shape(trace))[0] == 1:
            self.nspec = 1
            self.trace = [trace]
            self.pixel_mapping_itp = np.array([None] * self.nspec,
                                              dtype='object')

            if pixel_list is None:
                # assume continuous pixel, which is the case for a single CCD
                self.pixel_list = [np.arange(len(trace))]
            elif len(pixel_list) == len(self.trace):
                # check the pixel_list has the same length as the trace
                self.pixel_list = [pixel_list]
            else:
                raise ValueError(
                    'pixel_list has to be the same length as the trace.')
            self.pixel_mapping_itp[0] = itp.interp1d(np.arange(
                len(self.trace[0])),
                                                     self.pixel_list[0],
                                                     kind='cubic',
                                                     fill_value='extrapolate')

            if isinstance(trace_sigma, float):
                self.trace_sigma = np.array(trace_sigma).reshape(-1)
            elif isinstance(trace_sigma, np.ndarray) and len(trace_sigma) == 1:
                self.trace_sigma = np.array(trace_sigma).reshape(-1)
            else:
                raise TypeError('The trace_sigma has to be a float. ' +\
                          str(type(trace_sigma)) + ' is given.')

        # If there are multiple traces
        else:
            self.nspec = np.shape(trace)[0]
            self.trace = np.array(trace)
            self.pixel_mapping_itp = np.array([None] * self.nspec,
                                              dtype='object')

            if pixel_list is None:
                # assume continuous pixel, which is the case for a single CCD
                self.pixel_list = np.zeros_like(trace)
                for i in range(self.nspec):
                    self.pixel_list[i] = np.arange(len(self.trace[i]))
                    self.pixel_mapping_itp[i] = itp.interp1d(
                        np.arange(len(self.pixel_list[i])),
                        self.pixel_list[i],
                        kind='cubic',
                        fill_value='extrapolate')
            elif len(pixel_list[0]) == len(self.trace[0]):
                self.pixel_list = pixel_list
                for i in range(self.nspec):
                    self.pixel_list[i] = np.arange(len(self.trace[i]))
                    self.pixel_mapping_itp[i] = itp.interp1d(
                        np.arange(len(self.pixel_list[i])),
                        self.pixel_list[i],
                        kind='cubic',
                        fill_value='extrapolate')
            else:
                raise ValueError(
                    'pixel_list should be of the same shape as trace.')

            # If all traces have the same line spread function
            if isinstance(trace_sigma, float):
                self.trace_sigma = np.ones(self.nspec) * trace_sigma
            # Each trace has its own line spread function
            elif (len(trace_sigma) == self.nspec):
                self.trace_sigma = np.array(trace_sigma)
            else:
                raise ValueError(
                    'The trace_sigma should be a single float or an '
                    'array of a size of the number the of traces.')

    def add_polyfit(self, polyfit_coeff, polyfit_type=['poly']):
        '''
        To provide the polynomial coefficients and polynomial type for science,
        standard or both. Science stype can provide multiple traces. Standard
        stype can only accept one trace.

        Parameters
        ----------
        polyfit_coeff: list or list of list
            Polynomial fit coefficients.
        polyfit_type: str or list of str
            Strings starting with 'poly', 'leg' or 'cheb' for polynomial,
            legendre and chebyshev fits. Case insensitive.

        '''

        if not isinstance(polyfit_coeff, list):
            self.polyfit_coeff = [polyfit_coeff]
        else:
            self.polyfit_coeff = polyfit_coeff

        if not isinstance(polyfit_type, list):
            self.polyfit_type = [polyfit_type]
        else:
            self.polyfit_type = polyfit_type

        self.polyval = np.array([None] * len(polyfit_type), dtype='object')

        if len(self.polyfit_coeff) == self.nspec:
            for i in range(self.nspec):
                if self.polyfit_type[i].lower().startswith('poly'):
                    self.polyval[i] = np.polynomial.polynomial.polyval
                elif self.polyfit_type[i].lower().startswith('leg'):
                    self.polyval[i] = np.polynomial.legendre.legval
                elif self.polyfit_type[i].lower().startswith('cheb'):
                    self.polyval[i] = np.polynomial.chebyshev.chebval
                else:
                    raise ValueError(
                        'polyfit_type must be: (1) poly(nomial); (2) leg(endre); '
                        'or (3) cheb(yshev)')
        else:
            raise ValueError(
                'Number of polynomial fits should be the same as the number '
                'of traces.')

    def add_twodspec(self, twodspec, pixel_list=None):
        '''
        To add a TwoDSpec object or numpy array to provide the traces, line
        spread function of the traces, optionally the pixel values
        correcponding to the traces. If the arc is provided, the saxis and flip
        properties of the TwoDSpec will be applied to the arc, and then
        the spec_mask and the spatial_mask from the TwoDSpec object will be
        applied.

        Parameters
        ----------
        twodspec: TwoDSpec object
            TwoDSpec of the science/standard image containin the trace(s) and
            trace_sigma(s).
        pixel_list:


        '''

        if pixel_list is not None:
            if not isinstance(pixel_list, list):
                pixel_list = [pixel_list]
            else:
                pixel_list = pixel_list

        self.saxis = twodspec.saxis
        self.flip = twodspec.flip
        self.spec_mask = twodspec.spec_mask
        self.spatial_mask = twodspec.spatial_mask

        # get the length in the spectral and spatial directions
        self.nspec = int(np.shape(twodspec.trace)[0])

        self.trace = []
        self.trace_sigma = []
        self.pixel_list = []
        self.pixel_mapping_itp = np.array([None] * self.nspec, dtype='object')

        self.adu = []
        self.aduerr = self.adu.copy()
        self.adusky = self.adu.copy()

        for i in range(self.nspec):
            self.trace.append(twodspec.trace[i])
            self.trace_sigma.append(
                np.ones(len(self.trace[i])) * twodspec.trace_sigma[i])
            if pixel_list is not None:
                self.pixel_list.append(pixel_list[i])
            else:
                self.pixel_list.append(np.arange(len(self.trace[i])))
            self.pixel_mapping_itp[i] = itp.interp1d(np.arange(
                len(self.pixel_list[i])),
                                                     self.pixel_list[i],
                                                     kind='cubic',
                                                     fill_value='extrapolate')
            self.adu.append(twodspec.adu[i])
            self.aduerr.append(twodspec.aduerr[i])
            self.adusky.append(twodspec.adusky[i])

    def apply_twodspec_mask(self):

        if self.saxis == 0:
            self.arc = np.transpose(self.arc)

        if self.flip:
            self.arc = np.flip(self.arc)

        self.apply_spec_mask(self.spec_mask)
        self.apply_spatial_mask(self.spatial_mask)

    def apply_spec_mask(self, spec_mask):
        # the valid x-range of the chip (i.e. spectral direction)
        if (len(spec_mask) > 1):
            if self.saxis == 1:
                self.arc = self.arc[:, spec_mask]
            else:
                self.arc = self.arc[spec_mask]

    def apply_spatial_mask(self, spatial_mask):
        # the valid y-range of the chip (i.e. spatial direction)
        if (len(spatial_mask) > 1):
            if self.saxis == 1:
                self.arc = self.arc[spatial_mask]
            else:
                self.arc = self.arc[:, spatial_mask]

    def extract_arcspec(self,
                        use_pixel_list=True,
                        display=False,
                        jsonstring=False,
                        renderer='default',
                        iframe=False,
                        open_iframe=False):
        '''
        This function applies the trace(s) to the arc image then take median
        average of the stripe before identifying the arc lines (peaks) with
        scipy.signal.find_peaks(), where only the distance and the prominence
        keywords are used. Distance is the minimum separation between peaks,
        the default value is roughly twice the nyquist sampling rate (i.e.
        pixel size is 2.3 times smaller than the object that is being resolved,
        hence, the sepration between two clearly resolved peaks are ~5 pixels
        apart). A crude estimate of the background can exclude random noise
        which look like small peaks.

        Parameters
        ----------
        use_pixel_list: boolean
            Use the user supplied pixel if available.
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        iframe: boolean
            Save as an iframe, can work concurrently with other renderer
            apart from exporting jsonstring.
        open_iframe: boolean
            Open the iframe in the default browser if set to True.
        '''

        if self.arc is None:
            raise ValueError(
                'arc is not provided. Please provide arc by using add_arc() '
                'or with add_twodspec() before executing find_arc_lines().')

        if self.trace is None:
            raise ValueError(
                'arc is not provided. Please provide trace(s) by using '
                'add_trace() or with add_twodspec() before executing '
                'extract_arc_lines().')

        trace_shape = np.shape(self.trace)
        self.nspec = trace_shape[0]

        self.arcspec = np.zeros(trace_shape)

        fig = np.array([None] * self.nspec, dtype='object')

        for i in range(self.nspec):
            trace = int(np.mean(self.trace[i]))
            width = int(np.mean(self.trace_sigma[i]) * 3)

            arc_trace = self.arc[max(0, trace - width -
                                     1):min(trace +
                                            width, len(self.trace[i])), :]

            self.arcspec[i] = np.median(arc_trace, axis=0)

            # note that the display is adjusted for the chip gaps
            if display:
                fig[i] = go.Figure()

                if use_pixel_list:
                    fig[i].add_trace(
                        go.Scatter(x=self.pixel_list[i],
                                   y=self.arcspec[i],
                                   mode='lines',
                                   line=dict(color='royalblue', width=1)))
                else:
                    fig[i].add_trace(
                        go.Scatter(x=[np.arange(len(self.arcspec[i]))],
                                   y=self.arcspec[i],
                                   mode='lines',
                                   line=dict(color='royalblue', width=1)))

                fig[i].update_layout(
                    autosize=True,
                    xaxis=dict(zeroline=False,
                               title='Spectral Direction / pixel'),
                    yaxis=dict(zeroline=False,
                               range=[0, max(self.arcspec[i])],
                               title='ADU per second'),
                    hovermode='closest',
                    showlegend=False,
                    height=600)

                if jsonstring:
                    return fig[i].to_json()
                if iframe:
                    if open_iframe:
                        pio.write_html(fig[i], 'arcspec_' + str(i) + 'html')
                    else:
                        pio.write_html(fig[i],
                                       'arcspec_' + str(i) + 'html',
                                       auto_open=False)
                if renderer == 'default':
                    fig[i].show()
                else:
                    fig[i].show(renderer)

    def find_arc_lines(self,
                       percentile=25.,
                       prominence=10.,
                       distance=5.,
                       display=False,
                       jsonstring=False,
                       renderer='default',
                       iframe=False,
                       open_iframe=False):
        '''
        This function applies the trace(s) to the arc image then take median
        average of the stripe before identifying the arc lines (peaks) with
        scipy.signal.find_peaks(), where only the distance and the prominence
        keywords are used. Distance is the minimum separation between peaks,
        the default value is roughly twice the nyquist sampling rate (i.e.
        pixel size is 2.3 times smaller than the object that is being resolved,
        hence, the sepration between two clearly resolved peaks are ~5 pixels
        apart). A crude estimate of the background can exclude random noise
        which look like small peaks.

        Parameters
        ----------
        percentile: float
            The percentile of the flux to be used as the estimate of the
            background sky level to the first order. [ADU]
        distance: float
            Minimum separation between peaks
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        iframe: boolean
            Save as an iframe, can work concurrently with other renderer
            apart from exporting jsonstring.
        open_iframe: boolean
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if jsonstring is set to True
        '''

        if self.arcspec is None:
            raise ValueError(
                '1D arc spectrum is not available. Please provide the '
                'spectrum(a) by using add_arcspec() or generate from '
                'extract_arcspec() before executing find_arc_lines().')

        p = np.nanpercentile(self.arc, percentile)

        self.peaks_raw = []
        self.peaks = []

        fig = np.array([None] * self.nspec, dtype='object')
        for j in range(self.nspec):
            peaks_raw, _ = signal.find_peaks(self.arcspec[j],
                                             distance=distance,
                                             height=p,
                                             prominence=prominence)

            # Fine tuning
            self.peaks_raw.append(
                refine_peaks(self.arcspec[j], peaks_raw, window_width=5))

            # Adjust for the pixel mapping (e.g. chip gaps increment)
            if np.diff(self.pixel_list[j]).max() != 1:
                self.peaks.append(self.pixel_mapping_itp[j](self.peaks_raw[j]))
            else:
                self.peaks.append(self.peaks_raw[j])

            if display:
                fig[j] = go.Figure()

                # show the image on the top
                fig[j].add_trace(
                    go.Heatmap(x=np.arange(self.arc.shape[0]),
                               y=np.arange(self.arc.shape[1]),
                               z=np.log10(self.arc),
                               colorscale="Viridis",
                               colorbar=dict(title='log(ADU / s)')))

                # note that the image is not adjusted for the chip gaps
                # peaks_raw are plotted instead of the peaks
                trace = int(np.mean(self.trace[j]))
                width = int(np.mean(self.trace_sigma[j]) * 3)
                for i in self.peaks_raw[j]:
                    fig[j].add_trace(
                        go.Scatter(x=[i, i],
                                   y=[trace - width - 1, trace + width],
                                   mode='lines',
                                   line=dict(color='firebrick', width=1)))

                fig[j].update_layout(
                    autosize=True,
                    xaxis=dict(zeroline=False,
                               range=[0, self.arc.shape[1]],
                               title='Spectral Direction / pixel'),
                    yaxis=dict(zeroline=False,
                               range=[0, self.arc.shape[0]],
                               title='Spatial Direction / pixel'),
                    hovermode='closest',
                    showlegend=False,
                    height=600)

                if jsonstring:
                    return fig[j].to_json()
                if iframe:
                    if open_iframe:
                        pio.write_html(fig[j], 'arc_lines_' + str(j) + 'html')
                    else:
                        pio.write_html(fig[j],
                                       'arc_lines_' + str(j) + 'html',
                                       auto_open=False)
                if renderer == 'default':
                    fig[j].show()
                else:
                    fig[j].show(renderer)

    def initialise_calibrator(self,
                              peaks=None,
                              num_pix=None,
                              pixel_list=None,
                              min_wavelength=3000,
                              max_wavelength=9000,
                              plotting_library='plotly',
                              log_level='info'):

        self.calibrator = []
        for i in range(self.nspec):

            if peaks is None:
                peaks = self.peaks[i]

            if num_pix is None:
                num_pix = len(self.arcspec[i])

            self.calibrator.append(
                Calibrator(peaks=peaks,
                           num_pix=num_pix,
                           pixel_list=pixel_list,
                           min_wavelength=min_wavelength,
                           max_wavelength=max_wavelength,
                           plotting_library=plotting_library,
                           log_level=log_level))

    def set_known_pairs(self, pix=None, wave=None, idx=None):

        if isinstance(idx, int):
            idx = [idx]

        if idx is None:
            idx = range(self.nspec)

        assert (pix is not None) & (
            wave is not None), "pix and wave cannot be None."

        assert (np.shape(pix)
                == np.shape(wave)) & (np.shape(np.shape(pix)) == np.shape(
                    np.shape(wave))), "pix and wave must have the same shape."

        for i in idx:
            if np.shape(np.shape(pix))[0] == 1:
                self.calibrator[i].set_known_pairs(pix=pix, wave=wave)
            elif np.shape(np.shape(pix))[0] == 2:
                self.calibrator[i].set_known_pairs(pix=pix[i], wave=wave[i])

    def set_fit_constraints(self,
                            num_slopes=5000,
                            range_tolerance=500,
                            fit_tolerance=10.,
                            polydeg=4,
                            candidate_thresh=15.,
                            linearity_thresh=1.5,
                            ransac_thresh=3,
                            num_candidates=25,
                            xbins=100,
                            ybins=100,
                            brute_force=False,
                            fittype='poly',
                            idx=None):
        '''
        num_slopes : int (default: 1000)
            Number of slopes to consider during Hough transform
        range_tolerance : float (default: 500)
            Estimation of the error on the provided spectral range
            e.g. 3000-5000 with tolerance 500 will search for
            solutions that may satisfy 2500-5500
        fit_tolerance : float (default: 10)
            Sets a tolerance on whether a fit found by RANSAC is considered
            acceptable
        polydeg : int (default: 4)
            Degree of the polynomial fit.
        candidate_thresh : float (default: 15)
            Threshold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        linearity_thresh : float (default: 1.5)
            A threshold (Angstroms) that expresses how non-linear the solution
            can be. This mostly affects which atlas points are included and
            should be reasonably large, e.g. 500A.
        ransac_thresh : float (default: 1)
            The distance criteria  (Angstroms) to be considered an inlier to a
            fit. This should be close to the size of the expected residuals on
            the final fit (e.g. 1A is typical)
        num_candidates: int (default: 25)
            Number of best trial Hough pairs.
        xbins : int (default: 50)
            Number of bins for Hough accumulation
        ybins : int (default: 50)
            Number of bins for Hough accumulation
        brute_force : boolean (default: False)
            Set to True to try all possible combination in the given parameter
            space
        fittype : string (default: 'poly')
            One of 'poly', 'legendre' or 'chebyshev'
        '''

        if isinstance(idx, int):
            idx = [idx]

        if idx is None:
            idx = range(self.nspec)

        self.polyfit_type = []
        self.polyval = np.array([None] * len(idx), dtype='object')

        for i in idx:
            self.polyfit_type.append(fittype)

            if self.polyfit_type[i].lower().startswith('poly'):
                self.polyval[i] = np.polynomial.polynomial.polyval
            elif self.polyfit_type[i].lower().startswith('leg'):
                self.polyval[i] = np.polynomial.legendre.legval
            elif self.polyfit_type[i].lower().startswith('cheb'):
                self.polyval[i] = np.polynomial.chebyshev.chebval
            else:
                raise ValueError(
                    'polyfit_type must be: (1) poly(nomial); (2) leg(endre); '
                    'or (3) cheb(yshev)')

            self.calibrator[i].set_fit_constraints(
                num_slopes, range_tolerance, fit_tolerance, polydeg,
                candidate_thresh, linearity_thresh, ransac_thresh,
                num_candidates, xbins, ybins, brute_force, fittype)

    def add_atlas(self,
                  elements,
                  min_atlas_wavelength=0,
                  max_atlas_wavelength=15000,
                  min_intensity=0,
                  min_distance=0,
                  vacuum=False,
                  pressure=101325.,
                  temperature=273.15,
                  relative_humidity=0.,
                  constrain_poly=False,
                  idx=None):

        '''
            elements: string or list of string
                String or list of strings of Chemical symbol. Case insensitive.
            min_atlas_wave: float
                Minimum wavelength of the bluest arc line, NOT OF THE SPECTRUM.
            max_atlas_wave: float
                Maximum wavelength of the reddest arc line, NOT OF THE SPECTRUM.

        '''
        if isinstance(idx, int):
            idx = [idx]

        if idx is None:
            idx = range(self.nspec)

        for i in idx:
            self.calibrator[i].add_atlas(elements, min_atlas_wavelength,
                                         max_atlas_wavelength, min_intensity,
                                         min_distance, vacuum, pressure,
                                         temperature, relative_humidity,
                                         constrain_poly)

    def fit(self,
            sample_size=5,
            top_n=10,
            max_tries=5000,
            progress=True,
            coeff=None,
            linear=True,
            weighted=True,
            filter_close=False,
            display=False,
            savefig=False,
            filename=None,
            idx=None):
        '''
        A wrapper function to perform wavelength calibration with RASCAL.

        As of 14 January 2020, it supports He, Ne, Ar, Cu, Kr, Cd, Xe,
        Hg and Th from NIST:

            https://physics.nist.gov/PhysRefData/ASD/lines_form.html

        If there is already a set of good coefficienes, use calibrate_polyfit()
        instead.

        Parameters
        ----------
        sample_size: int
            Number of lines to be fitted in each loop.
        top_n: int
            Top ranked lines to be fitted.
        max_tries: int
            Number of trials of polynomial fitting.
        progress: boolean
            Set to show the progress using tdqm (if imported).
        coeff: list
            List of the polynomial fit coefficients for the first guess.
        lines: boolean (default: True)
            Set to True to sample candidates with straight lines in hough space,
            else sample candidates with coeff
        weighted: boolean (default: True)
            Set to True to use convolution in Hough space in the cost function
        filter_close: boolean (default: False)

        idx: None or int or List (default: None)
            The indices of the calibrator. If set to None, all of the
            calibrators will be fitted with the same given parameters. If set
            to int, it will turn into a list with a single integer. If a list of
            integers are given, those calibrators will be used to fit with
            the input parameters
        display: boolean
            Set to show diagnostic plot.
        savefig: string
            Set to save figure.
        filename: string
            Filename of the figure. Only work if display and savefig are set
            to True.

        '''

        self.polyfit_coeff = []
        self.rms = []
        self.residual = []
        self.peak_utilisation = []

        if isinstance(idx, int):
            idx = [idx]

        if idx is None:
            idx = range(self.nspec)

        for i in idx:
            polyfit_coeff, rms, residual, peak_utilisation = self.calibrator[
                i].fit(sample_size, top_n, max_tries, progress, coeff, linear,
                       weighted, filter_close)

            self.polyfit_coeff.append(polyfit_coeff)
            self.rms.append(rms)
            self.residual.append(residual)
            self.peak_utilisation.append(peak_utilisation)

            if display:
                if savefig:
                    self.calibrator[i].plot_fit(self.arcspec[i],
                               self.polyfit_coeff[i],
                               plot_atlas=True,
                               log_spectrum=False,
                               tolerance=1.0,
                               savefig=True,
                               filename=filename)
                else:
                    self.calibrator[i].plot_fit(self.arcspec[i],
                               self.polyfit_coeff[i],
                               plot_atlas=True,
                               log_spectrum=False,
                               tolerance=1.0)

    def refine_fit(self,
                   polyfit_coeff=None,
                   n_delta=None,
                   refine=True,
                   tolerance=10.,
                   method='Nelder-Mead',
                   convergence=1e-6,
                   robust_refit=True,
                   polydeg=None,
                   display=False,
                   savefig=False,
                   filename=None,
                   idx=None):
        '''
        A wrapper function to fine tune wavelength calibration with RASCAL
        when there is already a set of good coefficienes.

        As of 14 January 2020, it supports He, Ne, Ar, Cu, Kr, Cd, Xe,
        Hg and Th from NIST:

            https://physics.nist.gov/PhysRefData/ASD/lines_form.html

        Parameters
        ----------
        elements: string or list of string
            String or list of strings of Chemical symbol. Case insensitive.
        min_wavelength: float
            Minimum wavelength of the spectrum, NOT of the arc.
        max_wavelength: float
            Maximum wavelength of the spectrum, NOT of the arc.
        tolerance : float
            Absolute difference between fit and model.
        polydeg: int
            Degree of the polynomial
        display: boolean
            Set to show diagnostic plot.
        savefig: string
            Set to save figure.
        filename: string
            Filename of the figure. Only work if display and savefig are set
            to True.

        '''

        polyfit_new = []
        rms_new = []
        residual_new = []
        peak_utilisation_new = []

        if idx is None:
            idx = range(self.nspec)

        if isinstance(idx, int):
            idx = [idx]

        for i in idx:

            if polyfit_coeff is None:
                polyfit_coeff = self.polyfit_coeff[i]

            if polydeg is None:
                polydeg = len(polyfit_coeff) - 1

            if n_delta is None:
                n_delta = len(polyfit_coeff) - 1

            polyfit_coeff, _, _, residual, peak_utilisation = self.calibrator[
                i].match_peaks(polyfit_coeff,
                               n_delta=n_delta,
                               refine=refine,
                               tolerance=tolerance,
                               method=method,
                               convergence=convergence,
                               robust_refit=robust_refit,
                               polydeg=polydeg)

            polyfit_new.append(polyfit_coeff)
            rms_new.append(np.sqrt(np.mean(residual**2)))
            residual_new.append(residual)
            peak_utilisation_new.append(peak_utilisation)

            if display:
                if savefig:
                    self.calibrator[i].plot_fit(self.arcspec[i],
                               polyfit_new[i],
                               plot_atlas=True,
                               log_spectrum=False,
                               tolerance=1.0,
                               savefig=True,
                               filename=filename)
                else:
                    self.calibrator[i].plot_fit(self.arcspec[i],
                               polyfit_new[i],
                               plot_atlas=True,
                               log_spectrum=False,
                               tolerance=1.0)

        self.polyfit_coeff = polyfit_new
        self.residual = residual_new
        self.rms = rms_new
        self.peak_utilisation = peak_utilisation_new

    def apply_wavelength_calibration(self,
                                     wave_start=None,
                                     wave_end=None,
                                     wave_bin=None,
                                     idx=None):
        '''
        Apply the wavelength calibration. Because the polynomial fit can run
        away at the two ends, the default minimum and maximum are limited to
        1,000 and 12,000 A, respectively. This can be overridden by providing
        user's choice of wave_start and wave_end.

        Parameters
        ----------
        wave_start: float
            Provide the minimum wavelength for resampling.
        wave_end: float
            Provide the maximum wavelength for resampling
        wave_bin: float
            Provide the resampling bin size
        '''

        self.wave = []
        self.wave_resampled = []

        self.wave_bin = []
        self.wave_start = []
        self.wave_end = []

        self.adu_wcal = []
        self.aduerr_wcal = []
        self.adusky_wcal = []

        if idx is None:
            idx = range(self.nspec)

        if isinstance(idx, int):
            idx = [idx]

        for i in idx:

            # Adjust for pixel shift dur to chip gaps, map to itself
            # if it was not defined in WavelengthCalibration
            self.wave.append(self.polyval[i](self.pixel_list[i],
                                             self.polyfit_coeff[i]))

            # compute the new equally-spaced wavelength array
            if wave_bin is not None:
                self.wave_bin.append(wave_bin)
            else:
                self.wave_bin.append(np.median(np.ediff1d(self.wave[i])))

            if wave_start is not None:
                self.wave_start.append(wave_start)
            else:
                self.wave_start.append(self.wave[i][0])

            if wave_end is not None:
                self.wave_end.append(wave_end)
            else:
                self.wave_end.append(self.wave[i][-1])

            new_wave = np.arange(self.wave_start[i], self.wave_end[i],
                                 self.wave_bin[i])

            # apply the flux calibration and resample
            self.adu_wcal.append(
                spectres(new_wave, self.wave[i], self.adu[i], verbose=False))
            if self.aduerr is not None:
                self.aduerr_wcal.append(
                    spectres(new_wave,
                             self.wave[i],
                             self.aduerr[i],
                             verbose=False))
            if self.adusky is not None:
                self.adusky_wcal.append(
                    spectres(new_wave,
                             self.wave[i],
                             self.adusky[i],
                             verbose=False))

            self.wave_resampled.append(new_wave)

        self.wave = np.array(self.wave)
        self.wave_resampled = np.array(self.wave_resampled)
        self.wave_bin = np.array(self.wave_bin)
        self.wave_start = np.array(self.wave_start)
        self.wave_end = np.array(self.wave_end)
        self.adu_wcal = np.array(self.adu_wcal)
        self.aduerr_wcal = np.array(self.aduerr_wcal)
        self.adusky_wcal = np.array(self.adusky_wcal)


    def _create_adu_fits(self, stype):

        stype_split = stype.split('+')

        # Put the reduced data in FITS format with an image header
        self.adu_hdulist = np.array([None] * self.nspec, dtype='object')
        for i in range(self.nspec):
            self.adu_hdulist[i] = fits.HDUList([
                fits.ImageHDU(self.adu[i]),
                fits.ImageHDU(self.aduerr[i]),
                fits.ImageHDU(self.adusky[i])
            ])

    def _create_arcspec_fits(self):
        # Put the 1D arc spectrum in FITS format with an image header
        self.arcspec_hdulist = np.array([None] * self.nspec,
                                        dtype='object')
        for i in range(self.nspec):
            self.arcspec_hdulist[i] = fits.HDUList(
                [fits.ImageHDU(self.arcspec[i])])

    def _create_wavecal_fits(self):
        # Put the polynomial(s) in FITS format with an image header
        self.wavecal_hdulist = np.array([None] * self.nspec,
                                        dtype='object')
        for i in range(self.nspec):
            self.wavecal_hdulist[i] = fits.HDUList(
                [fits.ImageHDU(self.polyfit_coeff[i])])

    def save_fits(self,
                  output='arcspec+wavecal',
                  filename="wavecal",
                  overwrite=True):
        '''
        Save the wavelength calibration polynomial coefficients.

        Parameters
        ----------
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        extension: String
            File extension without the dot.
        overwrite: boolean
            Default is False.

        '''

        filename = os.path.splitext(filename)[0]

        output_split = output.split('+')

        if (self.arcspec is None) and (self.polyfit_coeff is None):
            raise ValueError('Neither arcspec is extracted nor wavelength is '
                             'calibrated.')

        if 'arcspec' in output_split:
            if self.arcspec is None:
                warnings.warn("Spectrum of the arc is not extracted yet.")
            else:
                self._create_arcspec_fits()

        if 'wavecal' in output_split:

            if self.polyfit_coeff is None:
                warnings.warn("Wavelength calibration is not performed yet.")
            else:
                self._create_wavecal_fits()

        # Save each trace as a separate FITS file
        for i in range(self.nspec):
            hdu_output = fits.HDUList()

            if 'arcspec' in output_split:
                if self.arcspec is not None:
                    hdu_output.extend(self.arcspec_hdulist[i])

            if 'wavecal' in output_split:
                if self.polyfit_coeff is not None:
                    hdu_output.extend(self.wavecal_hdulist[i])

            hdu_output[0] = fits.PrimaryHDU(hdu_output[0].data,
                                            hdu_output[0].header)
            hdu_output.update_extend()

            hdu_output[i].writeto(filename + '_' + str(i) + '.fits',
                                  overwrite=overwrite)


class StandardLibrary:
    def __init__(self, silence=False):
        '''
        This class handles flux calibration by comparing the extracted and
        wavelength-calibrated standard observation to the "ground truth"
        from

        https://github.com/iraf-community/iraf/tree/master/noao/lib/onedstandards
        https://www.eso.org/sci/observing/tools/standards/spectra.html

        See explanation notes at those links for details.

        The list of targets and libraries can be listed with

        list_all()

        Parameters
        ----------
        silence: boolean
            Set to True to suppress all verbose warnings.
        '''

        self.silence = silence

        self._load_standard_dictionary()

    def _load_standard_dictionary(self):
        '''
        Load the dictionaries
        '''

        self.lib_to_uname = json.load(
            open(os.path.join(base_dir, 'standards/lib_to_uname.json')))
        self.uname_to_lib = json.load(
            open(os.path.join(base_dir, 'standards/uname_to_lib.json')))

    def _get_eso_standard(self):

        folder = self.library

        # first letter of the file name
        if self.ftype == 'flux':
            filename = 'f'
        else:
            filename = 'm'

        # the rest of the file name
        filename += self.target

        # the extension
        filename += '.dat'

        filepath = os.path.join(base_dir, 'standards', folder, filename)

        f = np.loadtxt(filepath)

        self.wave_standard_true = f[:, 0]
        self.fluxmag_standard_true = f[:, 1]

        if self.ftype == 'flux':
            if self.library != 'esoxshooter':
                self.fluxmag_standard_true *= 1e-16

    def _get_ing_standard(self):

        folder = self.library.split("_")[0]

        # the first part of the file name
        filename = self.target
        extension = self.library.split('_')[-1]

        # last letter (or nothing) of the file name
        if self.ftype == 'flux':
            # .mas only contain magnitude files
            if extension == 'mas':
                filename += 'a'
            if ((filename == 'g24') or
                (filename == 'g157')) and (extension == 'fg'):
                filename += 'a'
            if (filename == 'h102') and (extension == 'sto'):
                filename += 'a'
        else:
            filename += 'a'

        # the extension
        filename += '.' + extension

        filepath = os.path.join(base_dir, 'standards', folder, filename)

        f = open(filepath)
        wave = []
        fluxmag = []
        for line in f.readlines():
            if line[0] in ['*', 'S']:
                if line.startswith('SET .Z.UNITS = '):
                    # remove all special characters and white spaces
                    unit = ''.join(e for e in line.split('"')[1].lower()
                                   if e.isalnum())
            else:
                l = line.strip().strip(':').split()
                wave.append(l[0])
                fluxmag.append(l[1])

        f.close()
        self.wave_standard_true = np.array(wave).astype('float')
        self.fluxmag_standard_true = np.array(fluxmag).astype('float')

        if self.ftype == 'flux':
            # Trap the ones without flux files
            if ((extension == 'mas') | (filename == 'g24a.fg') |
                (filename == 'g157a.fg') | (filename == 'h102a.sto')):
                self.fluxmag_standard_true = 10.**(
                    -(self.fluxmag_standard_true / 2.5)
                ) * 3630.780548 / 3.33564095e4 / self.wave_standard_true**2

            # convert milli-Jy into F_lambda
            if unit == 'mjy':
                self.fluxmag_standard_true * 1e-3 * 3.33564095e4 * self.wave_standard_true**2

            # convert micro-Jy into F_lambda
            if unit == 'microjanskys':
                self.fluxmag_standard_true * 1e-6 * 3.33564095e4 * self.wave_standard_true**2

    def _get_iraf_standard(self):
        # iraf is always in AB magnitude

        folder = self.library

        # file name and extension
        filename = self.target + '.dat'

        filepath = os.path.join(base_dir, 'standards', folder, filename)

        f = np.loadtxt(filepath, skiprows=1)

        self.wave_standard_true = f[:, 0]
        self.fluxmag_standard_true = f[:, 1]
        if self.ftype == 'flux':
            # Convert from AB mag to flux
            self.fluxmag_standard_true = 10.**(
                -(self.fluxmag_standard_true / 2.5)
            ) * 3630.780548 / 3.33564095e4 / self.wave_standard_true**2

    def lookup_standard_libraries(self, target, cutoff=0.4):
        '''
        Check if the requested standard and library exist. Return the three
        most similar words if the requested one does not exist. See

            https://docs.python.org/3.7/library/difflib.html
        '''

        # Load the list of targets in the requested library
        try:
            libraries = self.uname_to_lib[target]
            return libraries, True

        except:
            # If the requested target is not in any library, suggest the
            # closest match, Top 5 are returned.
            # difflib uses Gestalt pattern matching.
            target_list = difflib.get_close_matches(
                target, list(self.uname_to_lib.keys()), n=5, cutoff=cutoff)
            if len(target_list) > 0:

                if not self.silence:
                    warnings.warn(
                        'Requested standard star cannot be found, a list of ' +
                        'the closest matching names are returned:' +
                        str(target_list))

                return target_list, False

            else:

                raise ValueError(
                    'Please check the name of your standard star, nothing '
                    'share a similarity above ' + str(cutoff) + '.')

    def load_standard(self,
                      target,
                      library=None,
                      ftype='flux',
                      cutoff=0.4,
                      display=False,
                      renderer='default',
                      jsonstring=False,
                      iframe=False,
                      open_iframe=False):
        '''
        Read the standard flux/magnitude file. And return the wavelength and
        flux/mag. The units of the data are always in

        | wavelength: A
        | flux:       ergs / cm / cm / s / A
        | mag:        mag (AB)


        Returns
        -------
        target: string
            Name of the standard star
        library: string
            Name of the library of standard star
        ftype: string
            'flux' or 'mag'
        cutoff: float
            The threshold for the word similarity in the range of [0, 1].
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        iframe: boolean
            Save as an iframe, can work concurrently with other renderer
            apart from exporting jsonstring.
        open_iframe: boolean
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if jsonstring is set to True
        '''

        self.target = target
        self.ftype = ftype
        self.cutoff = cutoff

        libraries, success = self.lookup_standard_libraries(self.target)

        if success:
            if library in libraries:
                self.library = library
            else:
                self.library = libraries[0]
                warnings.warn(
                    'The request standard star cannot be found in the given '
                    'library, using ' + self.library + ' instead.')
        else:
            # If not, search again with the first one returned from lookup.
            self.target = libraries[0]
            libraries, _ = self.lookup_standard_libraries(self.target)
            self.library = libraries[0]
            print('The requested library does not exist, ' + self.library +
                  ' is used because it has the closest matching name.')

        if not self.silence:
            if library is None:
                # Use the default library order
                if not self.silence:
                    print('Standard library is not given, ' + self.library +
                          ' is used.')

        if self.library.startswith('iraf'):
            self._get_iraf_standard()

        if self.library.startswith('ing'):
            self._get_ing_standard()

        if self.library.startswith('eso'):
            self._get_eso_standard()

        # Note that if the renderer does not generate any image (e.g. JSON)
        # nothing will be displayed
        if display:
            self.inspect_standard(renderer, jsonstring, iframe, open_iframe)

    def inspect_standard(self,
                         renderer='default',
                         jsonstring=False,
                         iframe=False,
                         open_iframe=False):
        '''
        Display the standard star plot.

        Parameters
        ----------
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        iframe: boolean
            Save as an iframe, can work concurrently with other renderer
            apart from exporting jsonstring.
        open_iframe: boolean
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if jsonstring is set to True
        '''
        fig = go.Figure(layout=dict(updatemenus=list([
            dict(
                active=0,
                buttons=list([
                    dict(label='Log Scale',
                         method='update',
                         args=[{
                             'visible': [True, True]
                         }, {
                             'title': 'Log scale',
                             'yaxis': {
                                 'type': 'log'
                             }
                         }]),
                    dict(label='Linear Scale',
                         method='update',
                         args=[{
                             'visible': [True, False]
                         }, {
                             'title': 'Linear scale',
                             'yaxis': {
                                 'type': 'linear'
                             }
                         }])
                ]),
            )
        ]),
                                    title='Log scale'))

        # show the image on the top
        fig.add_trace(
            go.Scatter(x=self.wave_standard_true,
                       y=self.fluxmag_standard_true,
                       line=dict(color='royalblue', width=4)))

        fig.update_layout(
            autosize=True,
            title=self.library + ': ' + self.target + ' ' + self.ftype,
            xaxis_title=r'$\text{Wavelength / A}$',
            yaxis_title=
            r'$\text{Flux / ergs cm}^{-2} \text{s}^{-1} \text{A}^{-1}$',
            hovermode='closest',
            showlegend=False,
            height=800)

        if jsonstring:
            return fig.to_json()
        if iframe:
            if open_iframe:
                pio.write_html(fig, 'standard.html')
            else:
                pio.write_html(fig, 'standard.html', auto_open=False)
        if renderer == 'default':
            fig.show()
        else:
            fig.show(renderer)

    def query(self):
        pass


class FluxCalibration(StandardLibrary):
    def __init__(self, silence=False):
        # Load the dictionary
        super().__init__()
        self.science_imported = False
        self.standard_imported = False
        self.flux_science_calibrated = False
        self.flux_standard_calibrated = False
        self.adu = None
        self.aduerr = None
        self.adusky = None
        self.flux = None
        self.fluxerr = None
        self.fluxsky = None
        self.flux_raw = None
        self.fluxerr_raw = None
        self.fluxsky_raw = None
        self.adu_standard = None
        self.aduerr_standard = None
        self.adusky_standard = None
        self.flux_standard = None
        self.fluxerr_standard = None
        self.fluxsky_standard = None
        self.fluxmag_standard_true = None
        self.flux_standard_raw = None
        self.fluxerr_standard_raw = None
        self.fluxsky_standard_raw = None

    def add_wavelength(self, wave, stype):

        if stype == 'standard':
            self.wave_standard = wave
            self.wave_standard_resampled = wave
            self.wave_standard_bin = wave[1] - wave[0]
            self.wave_standard_start = wave[0]
            self.wave_standard_end = wave[-1]

        elif stype == 'science':
            if np.shape(wave) == np.shape(self.adu):
                self.wave = wave
                self.wave_resampled = wave
                self.wave_bin = wave[0][0] - wave[0][1]
                self.wave_start = wave[0][0]
                self.wave_end = wave[0][-1]
            elif (np.shape(np.shape(wave))[0] == 1) and (np.shape(self.adu[0])
                                                         == (np.shape(wave))):
                self.wave = [wave]
                self.wave_resampled = [wave]
                self.wave_bin = wave[1] - wave[0]
                self.wave_start = wave[0]
                self.wave_end = wave[-1]
            else:
                raise ValueError('Size of the wavelength array has to be the '
                                 'same size of the adu array.')

    def add_spec(self, adu, aduerr=None, adusky=None, stype='standard'):

        if stype == 'standard':

            self.adu_standard = adu

            if aduerr is not None:
                self.aduerr_standard = aduerr
                assert len(aduerr_standard) == len(self.adu_standard), (
                    'The length of each aduerr_standard and '
                    'adu_standard should be the same.')

            if adusky is not None:
                self.adusky_standard = adusky
                assert len(adudky_standard[0]) == len(self.adu_standard[0]), (
                    'The length of each adudky_standard and'
                    'adu_standard should be the same.')

        elif stype == 'science':

            if not np.shape(np.shape(adu))[0] == 2:
                self.adu = [adu]
                self.nspec = np.shape(np.shape(adu))[0]
            else:
                self.adu = adu
                self.nspec = np.shape(adu)[0]

            if aduerr is not None:
                if not np.shape(np.shape(aduerr))[0] == 2:
                    aduerr = [aduerr]
                assert len(aduerr) == len(
                    self.adu), ('There should be the same '
                                'number of aduerr and adu.')
                assert len(aduerr[0]) == len(
                    self.adu[0]), ('The length of each '
                                   'aduerr and adu should be the same.')
            self.aduerr = aduerr

            if adusky is not None:
                if not np.shape(np.shape(adusky))[0] == 2:
                    adusky = [adusky]
                assert len(adusky) == len(
                    self.adu), ('There should be the same '
                                'number of adusky and adu.')
                assert len(adusky[0]) == len(
                    self.adu[0]), ('The length of each '
                                   'adudky and adu should be the same.')
            self.adusky = adusky

        else:
            if stype not in ['science', 'standard']:
                raise ValueError('Unknown stype, please choose from '
                                 '(1) science; and/or (2) standard.')

    def add_twodspec(self, twodspec, pixel_list=None, stype='standard'):
        '''
        To add a TwoDSpec object or numpy array to provide the traces, line
        spread function of the traces, optionally the pixel values
        correcponding to the traces.

        Science type twodspec can contain more than 1 spectrum.

        Parameters
        ----------
        twodspec: TwoDSpec object
            TwoDSpec of the science/standard image containin the trace(s)
        arc: 2D numpy array, PrimaryHDU object or ImageReduction object
            The image of the arc image.
        pixel_list:

        stype:


        '''

        if stype == 'standard':
            self.trace_standard = twodspec.trace[0]
            if pixel_list is not None:
                self.pixel_list_standard = pixel_list
            else:
                self.pixel_list_standard = np.arange(len(self.trace_standard))

            self.pixel_mapping_itp = itp.interp1d(np.arange(
                len(self.pixel_list_standard)),
                                                  self.pixel_list_standard,
                                                  kind='cubic',
                                                  fill_value='extrapolate')
            self.adu_standard = twodspec.adu[0]
            self.aduerr_standard = twodspec.aduerr[0]
            self.adusky_standard = twodspec.adusky[0]

            self.standard_imported = True

        if stype == 'science':
            if pixel_list is not None:
                if not isinstance(pixel_list, list):
                    pixel_list = [pixel_list]
                else:
                    pixel_list = pixel_list

            # get the length in the spectral and spatial directions
            self.nspec = np.shape(twodspec.trace)[0]

            self.trace = np.zeros((self.nspec, len(twodspec.trace[0])))
            self.pixel_list = self.trace.copy()
            self.pixel_mapping_itp = np.array([None] * self.nspec,
                                              dtype='object')

            self.adu = np.zeros((self.nspec, len(twodspec.adu[0])))
            self.aduerr = self.adu.copy()
            self.adusky = self.adu.copy()

            for i in range(self.nspec):
                self.trace[i] = twodspec.trace[i]
                if pixel_list is not None:
                    self.pixel_list[i] = pixel_list[i]
                else:
                    self.pixel_list[i] = np.arange(len(self.trace[i]))
                self.pixel_mapping_itp[i] = itp.interp1d(
                    np.arange(len(self.pixel_list[i])),
                    self.pixel_list[i],
                    kind='cubic',
                    fill_value='extrapolate')
                self.adu[i] = twodspec.adu[i]
                self.aduerr[i] = twodspec.aduerr[i]
                self.adusky[i] = twodspec.adusky[i]

            self.science_imported = True

    def add_wavecal(self, wavecal, stype='standard'):

        stype_split = stype.split('+')

        if 'standard' in stype_split:
            self.wave_standard = wavecal.wave[0]
            self.wave_standard_bin = wavecal.wave_bin[0]
            self.wave_standard_start = wavecal.wave_start[0]
            self.wave_standard_end = wavecal.wave_end[0]
            self.wave_standard_resampled = wavecal.wave_resampled[0]

        if 'science' in stype_split:
            self.wave = wavecal.wave
            self.wave_bin = wavecal.wave_bin
            self.wave_start = wavecal.wave_start
            self.wave_end = wavecal.wave_end
            self.wave_resampled = wavecal.wave_resampled

    def compute_sensitivity(self,
                            kind=3,
                            smooth=False,
                            slength=5,
                            sorder=3,
                            mask_range=[[6850, 6960], [7150,
                                                       7400], [7575, 7700],
                                        [8925, 9050], [9265, 9750]],
                            display=False,
                            renderer='default',
                            jsonstring=False,
                            iframe=False,
                            open_iframe=False):
        '''
        The sensitivity curve is computed by dividing the true values by the
        wavelength calibrated standard spectrum, which is resampled with the
        spectres.spectres(). The curve is then interpolated with a cubic spline
        by default and is stored as a scipy interp1d object.

        A Savitzky-Golay filter is available for smoothing before the
        interpolation but it is not used by default.

        6850 - 7000 A,  7150 - 7400 A and 7575 - 7775 A are masked by default.

        Parameters
        ----------
        kind: string or integer [1,2,3,4,5 only]
            interpolation kind is one of [‘linear’, ‘nearest’, ‘zero’,
             ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’]
        smooth: boolean
            set to smooth the input spectrum with scipy.signal.savgol_filter
        slength: int
            SG-filter window size
        sorder: int
            SG-filter polynomial order
        mask_range: None or list of list
            Masking out regions not suitable for fitting the sensitivity curve.
            None for no mask. List of list has the pattern
            [[min1, max1], [min2, max2],...]
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        iframe: boolean
            Save as an iframe, can work concurrently with other renderer
            apart from exporting jsonstring.
        open_iframe: boolean
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if jsonstring is set to True.
        '''

        # Get the standard flux/magnitude
        self.slength = slength
        self.sorder = sorder
        self.smooth = smooth

        # resampling both the observed and the database standard spectra
        # in unit of flux per second. The higher resolution spectrum is
        # resampled to match the lower resolution one.
        if np.median(np.ediff1d(self.wave_standard)) < np.median(
                np.ediff1d(self.wave_standard_true)):
            flux_standard = spectres(self.wave_standard_true,
                                     self.wave_standard,
                                     self.adu_standard,
                                     verbose=False)
            flux_standard_true = self.fluxmag_standard_true
            wave_standard_true = self.wave_standard_true
        else:
            flux_standard = self.adu_standard
            flux_standard_true = spectres(self.wave_standard,
                                          self.wave_standard_true,
                                          self.fluxmag_standard_true,
                                          verbose=False)
            wave_standard_true = self.wave_standard

        # Get the sensitivity curve
        sens = flux_standard_true / flux_standard

        if mask_range is None:
            mask = np.isfinite(sens)
        else:
            mask = np.isfinite(sens)
            for m in mask_range:
                mask = mask & ((wave_standard_true < m[0]) |
                               (wave_standard_true > m[1]))

        sens = sens[mask]
        wave_standard = wave_standard_true[mask]
        flux_standard = flux_standard[mask]

        # apply a Savitzky-Golay filter to remove noise and Telluric lines
        if smooth:
            sens = signal.savgol_filter(sens, slength, sorder)

        sensitivity_itp = itp.interp1d(wave_standard,
                                       np.log10(sens),
                                       kind=kind,
                                       fill_value='extrapolate')

        self.sens = sens
        self.sensitivity_itp = sensitivity_itp
        self.wave_fcal = wave_standard
        self.flux_fcal = flux_standard

        # Diagnostic plot
        if display:
            self.inspect_sensitivity(renderer, jsonstring, iframe, open_iframe)

    def add_sensitivity_itp(self, sensitivity_itp):
        self.sensitivity_itp = sensitivity_itp

    def inspect_sensitivity(self,
                            renderer='default',
                            jsonstring=False,
                            iframe=False,
                            open_iframe=False):
        '''
        Display the computed sensitivity curve.

        Parameters
        ----------
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        iframe: boolean
            Save as an iframe, can work concurrently with other renderer
            apart from exporting jsonstring.
        open_iframe: boolean
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if jsonstring is set to True.
        '''

        fig = go.Figure(layout=dict(updatemenus=list([
            dict(
                active=0,
                buttons=list([
                    dict(label='Log Scale',
                         method='update',
                         args=[{
                             'visible': [True, True]
                         }, {
                             'title': 'Log scale',
                             'yaxis': {
                                 'type': 'log'
                             }
                         }]),
                    dict(label='Linear Scale',
                         method='update',
                         args=[{
                             'visible': [True, False]
                         }, {
                             'title': 'Linear scale',
                             'yaxis': {
                                 'type': 'linear'
                             }
                         }])
                ]),
            )
        ]),
                                    title='Log scale'))
        # show the image on the top
        fig.add_trace(
            go.Scatter(x=self.wave_fcal,
                       y=self.flux_fcal,
                       line=dict(color='royalblue', width=4),
                       name='ADU / s (Observed)'))

        fig.add_trace(
            go.Scatter(x=self.wave_fcal,
                       y=self.sens,
                       yaxis='y2',
                       line=dict(color='firebrick', width=4),
                       name='Sensitivity Curve'))

        fig.add_trace(
            go.Scatter(x=self.wave_fcal,
                       y=10.**self.sensitivity_itp(self.wave_fcal),
                       yaxis='y2',
                       line=dict(color='black', width=2),
                       name='Best-fit Sensitivity Curve'))

        if self.smooth:
            fig.update_layout(title='SG(' + str(self.slength) + ', ' +
                              str(self.sorder) + ')-Smoothed ' + self.library +
                              ': ' + self.target,
                              yaxis_title='Smoothed ADU / s')
        else:
            fig.update_layout(title=self.library + ': ' + self.target,
                              yaxis_title='ADU / s')

        fig.update_layout(autosize=True,
                          hovermode='closest',
                          showlegend=True,
                          xaxis_title=r'$\text{Wavelength / A}$',
                          yaxis=dict(title='ADU / s'),
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
        if jsonstring:
            return fig.to_json()
        if iframe:
            if open_iframe:
                pio.write_html(fig, 'senscurve.html')
            else:
                pio.write_html(fig, 'senscurve.html', auto_open=False)
        if renderer == 'default':
            fig.show()
        else:
            fig.show(renderer)

    def apply_flux_calibration(self, stype='science+standard'):
        '''
        Apply the computed sensitivity curve. And resample the spectra to
        match the highest resolution (the smallest wavelength bin) part of the
        spectrum.

        Parameters
        ----------
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        '''

        stype_split = stype.split('+')

        # Can be multiple spectra in the science frame
        if 'science' in stype_split:

            self.flux = []
            self.fluxerr = []
            self.fluxsky = []

            self.flux_raw = []
            self.fluxerr_raw = []
            self.fluxsky_raw = []

            self.sensitivity_raw = []
            self.sensitivity = []

            for i in range(self.nspec):
                # apply the flux calibration
                self.sensitivity_raw.append(10.**self.sensitivity_itp(
                    self.wave[i]))

                self.flux_raw.append(self.sensitivity_raw[i] * self.adu[i])
                if self.aduerr is not None:
                    self.fluxerr_raw.append(self.sensitivity_raw[i] *
                                            self.aduerr[i])
                if self.adusky is not None:
                    self.fluxsky_raw.append(self.sensitivity_raw[i] *
                                            self.adusky[i])

                self.flux.append(
                    spectres(self.wave_resampled[i],
                             self.wave[i],
                             self.flux_raw[i],
                             verbose=False))
                if self.aduerr is not None:
                    self.fluxerr.append(
                        spectres(self.wave_resampled[i],
                                 self.wave[i],
                                 self.fluxerr_raw[i],
                                 verbose=False))
                if self.adusky is not None:
                    self.fluxsky.append(
                        spectres(self.wave_resampled[i],
                                 self.wave[i],
                                 self.fluxsky_raw[i],
                                 verbose=False))

                # Only computed for diagnostic
                self.sensitivity.append(
                    spectres(self.wave_resampled[i],
                             self.wave[i],
                             self.sensitivity_raw[i],
                             verbose=False))

            self.flux_science_calibrated = True

            self.flux = np.array(self.flux)
            self.fluxerr = np.array(self.fluxerr)
            self.fluxsky = np.array(self.fluxsky)

            self.flux_raw = np.array(self.flux_raw)
            self.fluxerr_raw = np.array(self.fluxerr_raw)
            self.fluxsky_raw = np.array(self.fluxsky_raw)

            self.sensitivity_raw = np.array(self.sensitivity_raw)
            self.sensitivity = np.array(self.sensitivity)

        if 'standard' in stype_split:

            # apply the flux calibration and resample
            self.sensitivity_standard_raw = 10.**self.sensitivity_itp(
                self.wave_standard)

            self.flux_standard_raw = self.sensitivity_standard_raw * self.adu_standard
            if self.aduerr_standard is not None:
                self.fluxerr_standard_raw = self.sensitivity_standard_raw * self.aduerr_standard
            if self.adusky_standard is not None:
                self.fluxsky_standard_raw = self.sensitivity_standard_raw * self.adusky_standard

            self.flux_standard = spectres(self.wave_standard_resampled,
                                          self.wave_standard,
                                          self.flux_standard_raw,
                                          verbose=False)
            if self.aduerr_standard is not None:
                self.fluxerr_standard = spectres(self.wave_standard_resampled,
                                                 self.wave_standard,
                                                 self.fluxerr_standard_raw,
                                                 verbose=False)
            if self.adusky_standard is not None:
                self.fluxsky_standard = spectres(self.wave_standard_resampled,
                                                 self.wave_standard,
                                                 self.fluxsky_standard_raw,
                                                 verbose=False)

            # Only computed for diagnostic
            self.sensitivity_standard = spectres(self.wave_standard_resampled,
                                                 self.wave_standard,
                                                 self.sensitivity_standard_raw,
                                                 verbose=False)

            self.flux_standard_calibrated = True

        if ('science' not in stype_split) and ('standard' not in stype_split):
            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

    def inspect_reduced_spectrum(self,
                                 stype='science+standard',
                                 wave_min=4000.,
                                 wave_max=8000.,
                                 renderer='default',
                                 jsonstring=False,
                                 iframe=False,
                                 open_iframe=False):
        '''
        Display the reduced spectra.

        Parameters
        ----------
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        wave_min: float
            Minimum wavelength to display
        wave_max: float
            Maximum wavelength to display
        renderer: string
            Plotly renderer options.
        jsonstring: boolean
            Set to True to return json string that can be rendered by Plotly
            in any support language.
        iframe: boolean
            Save as an iframe, can work concurrently with other renderer
            apart from exporting jsonstring.
        open_iframe: boolean
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if jsonstring is set to True.
        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:
            fig_sci = np.array([None] * self.nspec, dtype='object')
            for i in range(self.nspec):

                if self.flux_science_calibrated:
                    wave_mask = ((self.wave_resampled[i] > wave_min) &
                                 (self.wave_resampled[i] < wave_max))
                    flux_mask = (
                        (self.flux[i] >
                         np.nanpercentile(self.flux[i][wave_mask], 5) / 1.5) &
                        (self.flux[i] <
                         np.nanpercentile(self.flux[i][wave_mask], 95) * 1.5))
                    flux_min = np.log10(np.nanmin(self.flux[i][flux_mask]))
                    flux_max = np.log10(np.nanmax(self.flux[i][flux_mask]))
                else:
                    warnings.warn('Flux calibration is not available.')
                    wave_mask = ((self.wave[i] > wave_min) &
                                 (self.wave[i] < wave_max))
                    flux_mask = (
                        (self.adu[i] >
                         np.nanpercentile(self.adu[i][wave_mask], 5) / 1.5) &
                        (self.adu[i] <
                         np.nanpercentile(self.adu[i][wave_mask], 95) * 1.5))
                    flux_min = np.log10(np.nanmin(self.adu[i][flux_mask]))
                    flux_max = np.log10(np.nanmax(self.adu[i][flux_mask]))

                fig_sci[i] = go.Figure(layout=dict(updatemenus=list([
                    dict(
                        active=0,
                        buttons=list([
                            dict(label='Log Scale',
                                 method='update',
                                 args=[{
                                     'visible': [True, True]
                                 }, {
                                     'title': 'Log scale',
                                     'yaxis': {
                                         'type': 'log'
                                     }
                                 }]),
                            dict(label='Linear Scale',
                                 method='update',
                                 args=[{
                                     'visible': [True, False]
                                 }, {
                                     'title': 'Linear scale',
                                     'yaxis': {
                                         'type': 'linear'
                                     }
                                 }])
                        ]),
                    )
                ]),
                                                   title='Log scale'))
                # show the image on the top
                if self.flux_science_calibrated:
                    fig_sci[i].add_trace(
                        go.Scatter(x=self.wave_resampled[i],
                                   y=self.flux[i],
                                   line=dict(color='royalblue'),
                                   name='Flux'))
                    if self.fluxerr != []:
                        fig_sci[i].add_trace(
                            go.Scatter(x=self.wave_resampled[i],
                                       y=self.fluxerr[i],
                                       line=dict(color='firebrick'),
                                       name='Flux Uncertainty'))
                    if self.fluxsky != []:
                        fig_sci[i].add_trace(
                            go.Scatter(x=self.wave_resampled[i],
                                       y=self.fluxsky[i],
                                       line=dict(color='orange'),
                                       name='Sky Flux'))
                else:
                    fig_sci[i].add_trace(
                        go.Scatter(x=self.wave[i],
                                   y=self.adu[i],
                                   line=dict(color='royalblue'),
                                   name='ADU / s'))
                    if self.aduerr is not None:
                        fig_sci[i].add_trace(
                            go.Scatter(x=self.wave[i],
                                       y=self.aduerr[i],
                                       line=dict(color='firebrick'),
                                       name='ADU Uncertainty / s'))
                    if self.adusky is not None:
                        fig_sci[i].add_trace(
                            go.Scatter(x=self.wave[i],
                                       y=self.adusky[i],
                                       line=dict(color='orange'),
                                       name='Sky ADU / s'))
                fig_sci[i].update_layout(
                    autosize=True,
                    hovermode='closest',
                    showlegend=True,
                    xaxis=dict(title='Wavelength / A',
                               range=[wave_min, wave_max]),
                    yaxis=dict(title='Flux',
                               range=[flux_min, flux_max],
                               type='log'),
                    legend=go.layout.Legend(x=0,
                                            y=1,
                                            traceorder="normal",
                                            font=dict(family="sans-serif",
                                                      size=12,
                                                      color="black"),
                                            bgcolor='rgba(0,0,0,0)'),
                    height=800)

                if jsonstring:
                    return fig_sci[j].to_json()
                if iframe:
                    if open_iframe:
                        pio.write_html(fig_sci[i],
                                       'spectrum_' + str(i) + '.html')
                    else:
                        pio.write_html(fig_sci[i],
                                       'spectrum_' + str(i) + '.html',
                                       auto_open=False)
                if renderer == 'default':
                    fig_sci[i].show()
                else:
                    fig_sci[i].show(renderer)

        if 'standard' in stype_split:

            if self.flux_standard_calibrated:
                wave_standard_mask = (
                    (self.wave_standard_resampled > wave_min) &
                    (self.wave_standard_resampled < wave_max))
                flux_standard_mask = (
                    (self.flux_standard > np.nanpercentile(
                        self.flux_standard[wave_standard_mask], 5) / 1.5) &
                    (self.flux_standard < np.nanpercentile(
                        self.flux_standard[wave_standard_mask], 95) * 1.5))
                flux_standard_min = np.log10(
                    np.nanmin(self.flux_standard[flux_standard_mask]))
                flux_standard_max = np.log10(
                    np.nanmax(self.flux_standard[flux_standard_mask]))
            else:
                warnings.warn('Flux calibration is not available.')
                wave_standard_mask = ((self.wave_standard > wave_min) &
                                      (self.wave_standard < wave_max))
                flux_standard_mask = (
                    (self.adu_standard > np.nanpercentile(
                        self.adu_standard[wave_standard_mask], 5) / 1.5) &
                    (self.adu_standard < np.nanpercentile(
                        self.adu_standard[wave_standard_mask], 95) * 1.5))
                flux_standard_min = np.log10(
                    np.nanmin(self.adu_standard[flux_standard_mask]))
                flux_standard_max = np.log10(
                    np.nanmax(self.adu_standard[flux_standard_mask]))

            fig_standard = go.Figure(layout=dict(updatemenus=list([
                dict(
                    active=0,
                    buttons=list([
                        dict(label='Log Scale',
                             method='update',
                             args=[{
                                 'visible': [True, True]
                             }, {
                                 'title': 'Log scale',
                                 'yaxis': {
                                     'type': 'log'
                                 }
                             }]),
                        dict(label='Linear Scale',
                             method='update',
                             args=[{
                                 'visible': [True, False]
                             }, {
                                 'title': 'Linear scale',
                                 'yaxis': {
                                     'type': 'linear'
                                 }
                             }])
                    ]),
                )
            ]),
                                                 title='Log scale'))
            # show the image on the top
            if self.flux_standard_calibrated:
                fig_standard.add_trace(
                    go.Scatter(x=self.wave_standard_resampled,
                               y=self.flux_standard,
                               line=dict(color='royalblue'),
                               name='Flux'))
                if self.fluxerr_standard is not None:
                    fig_standard.add_trace(
                        go.Scatter(x=self.wave_standard_resampled,
                                   y=self.fluxerr_standard,
                                   line=dict(color='firebrick'),
                                   name='Flux Uncertainty'))
                if self.fluxsky_standard is not None:
                    fig_standard.add_trace(
                        go.Scatter(x=self.wave_standard_resampled,
                                   y=self.fluxsky_standard,
                                   line=dict(color='orange'),
                                   name='Sky Flux'))
                if self.fluxmag_standard_true is not None:
                    fig_standard.add_trace(
                        go.Scatter(x=self.wave_standard_true,
                                   y=self.fluxmag_standard_true,
                                   line=dict(color='black'),
                                   name='Standard'))
            else:
                fig_standard.add_trace(
                    go.Scatter(x=self.wave_standard_resampled,
                               y=self.adu_standard,
                               line=dict(color='royalblue'),
                               name='ADU / s'))
                if self.aduerr_standard is not None:
                    fig_standard.add_trace(
                        go.Scatter(x=self.wave_standard_resampled,
                                   y=self.aduerr_standard,
                                   line=dict(color='firebrick'),
                                   name='ADU Uncertainty / s'))
                if self.adusky_standard is not None:
                    fig_standard.add_trace(
                        go.Scatter(x=self.wave_standard_resampled,
                                   y=self.adusky_standard,
                                   line=dict(color='orange'),
                                   name='Sky ADU / s'))

            fig_standard.update_layout(
                autosize=True,
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='Wavelength / A', range=[wave_min, wave_max]),
                yaxis=dict(title='Flux',
                           range=[flux_standard_min, flux_standard_max],
                           type='log'),
                legend=go.layout.Legend(x=0,
                                        y=1,
                                        traceorder="normal",
                                        font=dict(family="sans-serif",
                                                  size=12,
                                                  color="black"),
                                        bgcolor='rgba(0,0,0,0)'),
                height=800)

            if jsonstring:
                return fig_standard.to_json()
            if iframe:
                if open_iframe:
                    pio.write_html(fig_standard, 'spectrum_standard.html')
                else:
                    pio.write_html(fig_standard,
                                   'spectrum_standard.html',
                                   auto_open=False)
            if renderer == 'default':
                fig_standard.show()
            else:
                fig_standard.show(renderer)

        if ('science' not in stype_split) and ('standard' not in stype_split):
            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

    def _create_adu_fits(self, stype):

        stype_split = stype.split('+')

        # Put the reduced data in FITS format with an image header
        if 'science' in stype_split:

            self.adu_hdulist = np.array([None] * self.nspec,
                                        dtype='object')
            for i in range(self.nspec):
                self.adu_hdulist[i] = fits.HDUList([
                    fits.ImageHDU(self.adu[i]),
                    fits.ImageHDU(self.aduerr[i]),
                    fits.ImageHDU(self.adusky[i])
                ])

        if 'standard' in stype_split:

            self.adu_standard_hdulist = fits.HDUList([
                fits.ImageHDU(self.adu_standard),
                fits.ImageHDU(self.aduerr_standard),
                fits.ImageHDU(self.adusky_standard)
            ])

    def _create_flux_fits(self, stype):
        '''
        Create HDU list of the reuqested list(s) of spectra, uncertainty and
        sky as a function of wavelength at the native pixel sampling.

        Parameters
        ----------
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        # wavelengths are in the natively extracted bins
        # Put the reduced data in FITS format with an image header
        if 'science' in stype_split:

            self.science_hdulist = np.array([None] * self.nspec,
                                            dtype='object')
            for i in range(self.nspec):
                # Note that wave_start is the centre of the starting bin
                flux_wavecal_fits = fits.ImageHDU(self.flux_raw[i])
                fluxerr_wavecal_fits = fits.ImageHDU(self.fluxerr_raw[i])
                fluxsky_wavecal_fits = fits.ImageHDU(self.fluxsky_raw[i])

                sensitivity_fits = fits.ImageHDU(self.sensitivity_raw[i])

                self.science_hdulist[i] = fits.HDUList([
                    flux_wavecal_fits, fluxerr_wavecal_fits,
                    fluxsky_wavecal_fits, sensitivity_fits
                ])

        if 'standard' in stype_split:

            # Note that wave_start is the centre of the starting bin
            flux_wavecal_fits = fits.ImageHDU(self.flux_standard_raw)
            fluxerr_wavecal_fits = fits.ImageHDU(self.fluxerr_standard_raw)
            fluxsky_wavecal_fits = fits.ImageHDU(self.fluxsky_standard_raw)

            sensitivity_fits = fits.ImageHDU(self.sensitivity_standard_raw)

            self.flux_standard_hdulist = fits.HDUList([
                flux_wavecal_fits, fluxerr_wavecal_fits, fluxsky_wavecal_fits,
                sensitivity_fits
            ])

        if ('science' not in stype_split) and ('standard' not in stype_split):
            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

    def _create_flux_resampled_fits(self, stype):
        '''
        Create HDU list of the reuqested list(s) of spectra, uncertainty and
        sky as a function of wavelength at fixed interval. This can be
        directely plotted with iraf/splot.

        Parameters
        ----------
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        # iraf splot readable format where wavelength are in equal-sized bins
        # Put the reduced data in FITS format with an image header
        if 'science' in stype_split:

            self.science_resampled_hdulist = np.array([None] * self.nspec,
                                            dtype='object')

            for i in range(self.nspec):
                # Note that wave_start is the centre of the starting bin
                flux_wavecal_fits = fits.ImageHDU(self.flux[i])
                flux_wavecal_fits.header['LABEL'] = 'Flux'
                flux_wavecal_fits.header['CRPIX1'] = 1.00E+00
                flux_wavecal_fits.header['CDELT1'] = self.wave_bin[i]
                flux_wavecal_fits.header['CRVAL1'] = self.wave_start[i]
                flux_wavecal_fits.header['CTYPE1'] = 'Wavelength'
                flux_wavecal_fits.header['CUNIT1'] = 'Angstroms'
                flux_wavecal_fits.header[
                    'BUNIT'] = 'erg/(s*cm**2*Angstrom)'
                self.science_resampled_hdulist[i] = fits.HDUList(flux_wavecal_fits)

                if self.fluxerr is not None:
                    fluxerr_wavecal_fits = fits.ImageHDU(self.fluxerr[i])
                    fluxerr_wavecal_fits.header['LABEL'] = 'Flux'
                    fluxerr_wavecal_fits.header['CRPIX1'] = 1.00E+00
                    fluxerr_wavecal_fits.header['CDELT1'] = self.wave_bin[
                        i]
                    fluxerr_wavecal_fits.header[
                        'CRVAL1'] = self.wave_start[i]
                    fluxerr_wavecal_fits.header['CTYPE1'] = 'Wavelength'
                    fluxerr_wavecal_fits.header['CUNIT1'] = 'Angstroms'
                    fluxerr_wavecal_fits.header[
                        'BUNIT'] = 'erg/(s*cm**2*Angstrom)'
                    self.science_resampled_hdulist[i].append(
                        fluxerr_wavecal_fits)

                if self.fluxsky is not None:
                    fluxsky_wavecal_fits = fits.ImageHDU(self.fluxsky[i])
                    fluxsky_wavecal_fits.header['LABEL'] = 'Flux'
                    fluxsky_wavecal_fits.header['CRPIX1'] = 1.00E+00
                    fluxsky_wavecal_fits.header['CDELT1'] = self.wave_bin[
                        i]
                    fluxsky_wavecal_fits.header[
                        'CRVAL1'] = self.wave_start[i]
                    fluxsky_wavecal_fits.header['CTYPE1'] = 'Wavelength'
                    fluxsky_wavecal_fits.header['CUNIT1'] = 'Angstroms'
                    fluxsky_wavecal_fits.header[
                        'BUNIT'] = 'erg/(s*cm**2*Angstrom)'
                    self.science_resampled_hdulist[i].append(
                        fluxsky_wavecal_fits)

                if self.sensitivity is not None:
                    sensitivity_fits = fits.ImageHDU(self.sensitivity[i])
                    self.science_resampled_hdulist[i].append(
                        sensitivity_fits)

        if 'standard' in stype_split:

            self.standard_resampled_hdulist = fits.HDUList()

            # Note that wave_start is the centre of the starting bin
            flux_wavecal_fits = fits.ImageHDU(self.flux_standard)
            flux_wavecal_fits.header['LABEL'] = 'Flux'
            flux_wavecal_fits.header['CRPIX1'] = 1.00E+00
            flux_wavecal_fits.header['CDELT1'] = self.wave_standard_bin
            flux_wavecal_fits.header['CRVAL1'] = self.wave_standard_start
            flux_wavecal_fits.header['CTYPE1'] = 'Wavelength'
            flux_wavecal_fits.header['CUNIT1'] = 'Angstroms'
            flux_wavecal_fits.header['BUNIT'] = 'erg/(s*cm**2*Angstrom)'
            self.standard_resampled_hdulist.append(flux_wavecal_fits)

            if self.fluxerr_standard is not None:
                fluxerr_wavecal_fits = fits.ImageHDU(self.fluxerr_standard)
                fluxerr_wavecal_fits.header['LABEL'] = 'Flux'
                fluxerr_wavecal_fits.header['CRPIX1'] = 1.00E+00
                fluxerr_wavecal_fits.header['CDELT1'] = self.wave_standard_bin
                fluxerr_wavecal_fits.header[
                    'CRVAL1'] = self.wave_standard_start
                fluxerr_wavecal_fits.header['CTYPE1'] = 'Wavelength'
                fluxerr_wavecal_fits.header['CUNIT1'] = 'Angstroms'
                fluxerr_wavecal_fits.header['BUNIT'] = 'erg/(s*cm**2*Angstrom)'
                self.standard_resampled_hdulist.append(fluxerr_wavecal_fits)

            if self.fluxsky_standard is not None:
                fluxsky_wavecal_fits = fits.ImageHDU(self.fluxsky_standard)
                fluxsky_wavecal_fits.header['LABEL'] = 'Flux'
                fluxsky_wavecal_fits.header['CRPIX1'] = 1.00E+00
                fluxsky_wavecal_fits.header['CDELT1'] = self.wave_standard_bin
                fluxsky_wavecal_fits.header[
                    'CRVAL1'] = self.wave_standard_start
                fluxsky_wavecal_fits.header['CTYPE1'] = 'Wavelength'
                fluxsky_wavecal_fits.header['CUNIT1'] = 'Angstroms'
                fluxsky_wavecal_fits.header['BUNIT'] = 'erg/(s*cm**2*Angstrom)'
                self.standard_resampled_hdulist.append(fluxsky_wavecal_fits)

            if self.sensitivity_standard is not None:
                sensitivity_fits = fits.ImageHDU(self.sensitivity_standard)
                self.standard_resampled_hdulist.append(sensitivity_fits)

        if ('science' not in stype_split) and ('standard' not in stype_split):
            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')


class OneDSpec():
    def __init__(self, silence=False):
        '''
        This class applies the wavelength calibrations and compute & apply the
        flux calibration to the extracted 1D spectra. The standard TwoDSpec
        object is not required for data reduction, but the flux calibrated
        standard observation will not be available for diagnostic.

        Parameters
        ----------
        science: TwoDSpec object
            The TwoDSpec object with the extracted science target
        wave_cal: WavelengthPolyFit object
            The WavelengthPolyFit object for the science target, flux will
            not be calibrated if this is not provided.
        standard: TwoDSpec object
            The TwoDSpec object with the extracted standard target
        wave_cal_standard: WavelengthPolyFit object
            The WavelengthPolyFit object for the standard target, flux will
            not be calibrated if this is not provided.
        flux_cal: StandardFlux object
            The true mag/flux values.
        silence: boolean
            Set to True to suppress all verbose warnings.
        '''
        self.science_imported = False
        self.standard_imported = False
        self.wavecal_science_imported = False
        self.wavecal_standard_imported = False
        self.arc_science_imported = False
        self.arc_standard_imported = False
        self.arc_lines_science_imported = False
        self.arc_lines_standard_imported = False
        self.flux_imported = False

        self.wavecal_science = WavelengthCalibration(silence)
        self.wavecal_standard = WavelengthCalibration(silence)
        self.fluxcal = FluxCalibration(silence)

    def add_fluxcalibration(self, fluxcal):
        pass
        '''
        Provide the pre-calibrated FluxCalibration object.

        Parameters
        ----------
        fluxcal: FluxCalibration object
            The true mag/flux values.

        try:
            self.fluxcal = fluxcal
            self.flux_imported = True
        except:
            raise TypeError('Please provide a valid StandardFlux.')
        '''

    def add_wavelengthcalibration(self, wavecal, stype):
        pass
        '''
        Provide the pre-calibrated WavelengthCalibration object.

        Parameters
        ----------
        wavecal: WavelengthCalibration object
            The true mag/flux values.
        self.fluxcal.add_wavecal(wavecal, stype)
        '''

    def add_wavelength(self, wave, stype):
        self.fluxcal.add_wavelength(wave, stype)

    def add_spec(self, adu, aduerr=None, adusky=None, stype='science'):

        self.fluxcal.add_spec(adu, aduerr, adusky, stype)

        if stype == 'science':
            self.wavecal_science.add_spec(adu, aduerr, adusky)
            self.nspec = self.wavecal_science.nspec

        if stype == 'standard':
            self.wavecal_standard.add_spec(adu, aduerr, adusky)

    def add_twodspec(self, twodspec, pixel_list=None, stype='science'):
        '''
        Extract the required information from the TwoDSpec object of the
        standard.

        Parameters
        ----------
        standard: TwoDSpec object
            The TwoDSpec object with the extracted standard target
        '''

        self.fluxcal.add_twodspec(twodspec, pixel_list, stype)

        if stype == 'science':
            self.wavecal_science.add_twodspec(twodspec, pixel_list)
            self.nspec = self.wavecal_science.nspec

        if stype == 'standard':
            self.wavecal_standard.add_twodspec(twodspec, pixel_list)

    def apply_twodspec_mask(self, stype='science'):

        if stype == 'science':
            self.wavecal_science.apply_twodspec_mask()
        if stype == 'standard':
            self.wavecal_standard.apply_twodspec_mask()

    def apply_spec_mask(self, spec_mask, stype='science'):
        self.apply_spec_mask(spec_mask)

    def apply_spatial_mask(self, spatial_mask, stype='science'):
        self.apply_spatial_mask(spec_mask)

    def add_arc_lines(self, peaks, stype='science'):

        if stype == 'science':
            self.wavecal_science.add_arc_lines(peaks)
            self.arc_lines_science_imported = True
        if stype == 'standard':
            self.wavecal_standard.add_arc_lines(peaks)
            self.arc_lines_standard_imported = True

    def add_arcspec(self, spectrum, stype='science'):

        if stype == 'science':
            self.wavecal_science.add_arcspec(spectrum)
            self.arc_science_imported = True
        if stype == 'standard':
            self.wavecal_standard.add_arcspec(spectrum)
            self.arc_standard_imported = True

    def add_arc(self, arc, stype='science'):

        if stype == 'science':
            self.wavecal_science.add_arc(arc)
            self.arc_science_imported = True
        if stype == 'standard':
            self.wavecal_standard.add_arc(arc)
            self.arc_standard_imported = True

    def add_trace(self, trace, trace_sigma, pixel_list=None, stype='science'):

        if stype == 'science':
            self.wavecal_science.add_trace(trace, trace_sigma, pixel_list)
            self.nspec = self.wavecal_science.nspec
        if stype == 'standard':
            self.wavecal_standard.add_trace(trace, trace_sigma, pixel_list)

    def add_polyfit(self,
                    polyfit_coeff,
                    polyfit_type=['poly'],
                    stype='science'):

        if stype == 'science':
            self.wavecal_science.add_polyfit(polyfit_coeff, polyfit_type)
        if stype == 'standard':
            self.wavecal_standard.add_polyfit(polyfit_coeff, polyfit_type)

    def extract_arcspec(self,
                        use_pixel_list=True,
                        display=False,
                        jsonstring=False,
                        renderer='default',
                        iframe=False,
                        open_iframe=False,
                        stype='science'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.extract_arcspec(use_pixel_list=use_pixel_list,
                                                 display=display,
                                                 jsonstring=jsonstring,
                                                 renderer=renderer,
                                                 iframe=iframe,
                                                 open_iframe=open_iframe)
        if 'standard' in stype_split:
            self.wavecal_standard.extract_arcspec(use_pixel_list=use_pixel_list,
                                                  display=display,
                                                  jsonstring=jsonstring,
                                                  renderer=renderer,
                                                  iframe=iframe,
                                                  open_iframe=open_iframe)

    def find_arc_lines(self,
                       percentile=25.,
                       prominence=10.,
                       distance=5.,
                       display=False,
                       jsonstring=False,
                       renderer='default',
                       iframe=False,
                       open_iframe=False,
                       stype='science'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.find_arc_lines(percentile=percentile,
                                                prominence=prominence,
                                                distance=distance,
                                                display=display,
                                                jsonstring=jsonstring,
                                                renderer=renderer,
                                                iframe=iframe,
                                                open_iframe=open_iframe)
        if 'standard' in stype_split:
            self.wavecal_standard.find_arc_lines(percentile=percentile,
                                                 prominence=prominence,
                                                 distance=distance,
                                                 display=display,
                                                 jsonstring=jsonstring,
                                                 renderer=renderer,
                                                 iframe=iframe,
                                                 open_iframe=open_iframe)

    def initialise_calibrator(self,
                              peaks=None,
                              num_pix=None,
                              pixel_list=None,
                              min_wavelength=3000,
                              max_wavelength=9000,
                              plotting_library='plotly',
                              log_level='info',
                              stype='science'):
        '''
        If the peaks were found with find_arc_lines(), peaks and num_pix can
        be None.
        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.initialise_calibrator(
                peaks=peaks,
                num_pix=num_pix,
                pixel_list=pixel_list,
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                plotting_library=plotting_library,
                log_level=log_level)
        if 'standard' in stype_split:
            self.wavecal_standard.initialise_calibrator(
                peaks=peaks,
                num_pix=num_pix,
                pixel_list=pixel_list,
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                plotting_library=plotting_library,
                log_level=log_level)

    def set_known_pairs(self, pix=None, wave=None, idx=None, stype='standard'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.set_known_pairs(pix, wave, idx)
        if 'standard' in stype_split:
            self.wavecal_science.set_known_pairs(pix, wave, idx=0)

    def set_fit_constraints(self,
                            num_slopes=5000,
                            range_tolerance=500,
                            fit_tolerance=10.,
                            polydeg=4,
                            candidate_thresh=15.,
                            linearity_thresh=1.5,
                            ransac_thresh=3,
                            num_candidates=25,
                            xbins=100,
                            ybins=100,
                            brute_force=False,
                            fittype='poly',
                            idx=None,
                            stype='science'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.set_fit_constraints(
                num_slopes=num_slopes,
                range_tolerance=range_tolerance,
                fit_tolerance=fit_tolerance,
                polydeg=polydeg,
                candidate_thresh=candidate_thresh,
                linearity_thresh=linearity_thresh,
                ransac_thresh=ransac_thresh,
                num_candidates=num_candidates,
                xbins=xbins,
                ybins=ybins,
                brute_force=brute_force,
                fittype=fittype,
                idx=idx)
        if 'standard' in stype_split:
            self.wavecal_standard.set_fit_constraints(
                num_slopes=num_slopes,
                range_tolerance=range_tolerance,
                fit_tolerance=fit_tolerance,
                polydeg=polydeg,
                candidate_thresh=candidate_thresh,
                linearity_thresh=linearity_thresh,
                ransac_thresh=ransac_thresh,
                num_candidates=num_candidates,
                xbins=xbins,
                ybins=ybins,
                brute_force=brute_force,
                fittype=fittype,
                idx=0)

    def add_atlas(self,
                  elements,
                  min_atlas_wavelength=0,
                  max_atlas_wavelength=15000,
                  min_intensity=0,
                  min_distance=0,
                  vacuum=False,
                  pressure=101325.,
                  temperature=273.15,
                  relative_humidity=0.,
                  constrain_poly=False,
                  idx=None,
                  stype='science'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.add_atlas(
                elements=elements,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=min_intensity,
                min_distance=min_distance,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
                constrain_poly=constrain_poly,
                idx=idx)

        if 'standard' in stype_split:
            self.wavecal_standard.add_atlas(
                elements=elements,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=min_intensity,
                min_distance=min_distance,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
                constrain_poly=constrain_poly,
                idx=0)

    def fit(self,
            sample_size=5,
            top_n=10,
            max_tries=5000,
            progress=True,
            coeff=None,
            linear=True,
            weighted=True,
            filter_close=False,
            display=False,
            savefig=False,
            filename=None,
            idx=None,
            stype='standard'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.fit(sample_size=sample_size,
                                     top_n=top_n,
                                     max_tries=max_tries,
                                     progress=progress,
                                     coeff=coeff,
                                     linear=linear,
                                     weighted=weighted,
                                     filter_close=filter_close,
                                     display=display,
                                     savefig=savefig,
                                     filename=filename,
                                     idx=idx)

        if 'standard' in stype_split:
            self.wavecal_standard.fit(sample_size=sample_size,
                                      top_n=top_n,
                                      max_tries=max_tries,
                                      progress=progress,
                                      coeff=coeff,
                                      linear=linear,
                                      weighted=weighted,
                                      filter_close=filter_close,
                                      display=display,
                                      savefig=savefig,
                                      filename=filename,
                                      idx=0)

    def refine_fit(self,
                   polyfit_coeff=None,
                   n_delta=None,
                   refine=True,
                   tolerance=10.,
                   method='Nelder-Mead',
                   convergence=1e-6,
                   robust_refit=True,
                   polydeg=None,
                   display=False,
                   savefig=False,
                   filename=None,
                   idx=None,
                   stype='standard'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.refine_fit(polyfit_coeff=polyfit_coeff,
                                            n_delta=n_delta,
                                            refine=refine,
                                            tolerance=tolerance,
                                            method=method,
                                            convergence=convergence,
                                            robust_refit=robust_refit,
                                            polydeg=polydeg,
                                            display=display,
                                            savefig=savefig,
                                            filename=filename,
                                            idx=idx)
        if 'standard' in stype_split:
            self.wavecal_standard.refine_fit(polyfit_coeff=polyfit_coeff,
                                             n_delta=n_delta,
                                             refine=refine,
                                             tolerance=tolerance,
                                             method=method,
                                             convergence=convergence,
                                             robust_refit=robust_refit,
                                             polydeg=polydeg,
                                             display=display,
                                             savefig=savefig,
                                             filename=filename,
                                             idx=0)

    def apply_wavelength_calibration(self,
                                     wave_start=None,
                                     wave_end=None,
                                     wave_bin=None,
                                     stype='science+standard'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.apply_wavelength_calibration(
                wave_start, wave_end, wave_bin)
            self.fluxcal.add_wavecal(self.wavecal_science, stype='science')
        if 'standard' in stype_split:
            self.wavecal_standard.apply_wavelength_calibration(
                wave_start, wave_end, wave_bin)
            self.fluxcal.add_wavecal(self.wavecal_standard, stype='standard')

    def lookup_standard_libraries(self, target, cutoff=0.4):
        self.fluxcal.lookup_standard_libraries(target, cutoff)

    def load_standard(self,
                      target,
                      library=None,
                      ftype='flux',
                      cutoff=0.4,
                      display=False,
                      renderer='default',
                      jsonstring=False,
                      iframe=False,
                      open_iframe=False):
        self.fluxcal.load_standard(target, library, ftype, cutoff, display,
                                   renderer, jsonstring, iframe, open_iframe)

    def inspect_standard(self,
                         renderer='default',
                         jsonstring=False,
                         iframe=False,
                         open_iframe=False):

        self.fluxcal.inspect_standard(renderer, jsonstring, iframe,
                                      open_iframe)

    def compute_sensitivity(self,
                            kind=3,
                            smooth=False,
                            slength=5,
                            sorder=3,
                            mask_range=[[6850, 6960], [7150,
                                                       7400], [7575, 7700],
                                        [8925, 9050], [9265, 9750]],
                            display=False,
                            renderer='default',
                            jsonstring=False,
                            iframe=False,
                            open_iframe=False):

        self.fluxcal.compute_sensitivity(kind, smooth, slength, sorder,
                                         mask_range, display, renderer,
                                         jsonstring, iframe, open_iframe)

    def add_sensitivity_itp(self, sensitivity_itp):

        self.fluxcal.sensitivity_itp = sensitivity_itp

    def inspect_sensitivity(self,
                            renderer='default',
                            jsonstring=False,
                            iframe=False,
                            open_iframe=False):

        self.fluxcal.inspect_sensitivity(renderer, jsonstring, iframe,
                                         open_iframe)

    def apply_flux_calibration(self, stype='science+standard'):

        stype_split = stype.split('+')

        self.fluxcal.apply_flux_calibration(stype)

    def inspect_reduced_spectrum(self,
                                 stype='science+standard',
                                 wave_min=4000.,
                                 wave_max=8000.,
                                 renderer='default',
                                 jsonstring=False,
                                 iframe=False,
                                 open_iframe=False):

        self.fluxcal.inspect_reduced_spectrum(stype, wave_min, wave_max,
                                              renderer, jsonstring, iframe,
                                              open_iframe)

    def _create_wavelength_fits(self, stype, resample):
        '''
        Create HDU list of the reuqested list(s) of wavelength of the spectra.

        Parameters
        ----------
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        resample: boolean
            set to True to return the array of wavelengths in fixed intervals
            of wavelength; False to return the array of wavelengths at the
            native pixel sampling.

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavelength_hdulist = np.array([None] * self.nspec,
                                               dtype='object')
            for i in range(self.nspec):
                if resampled:
                    self.wavelength_hdulist = fits.ImageHDU(
                        self.wave_resampled[i])
                else:
                    self.wavelength_hdulist = fits.ImageHDU(self.wave[i])

        if 'standard' in stype_split:
            if resampled:
                self.wavelength_standard_hdulist = [
                    fits.ImageHDU(self.wave_standard_resampled)
                ]
            else:
                self.wavelength_standard_hdulist = [
                    fits.ImageHDU(self.wave_standard)
                ]

        if ('science' not in stype_split) and ('standard' not in stype_split):
            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

    def save_fits(self,
                  output='flux+wavecal+fluxraw+adu',
                  filename='reduced',
                  stype='science',
                  overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        stype: String
            Spectral type: science or standard
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            flux: 3 HDUs
                Flux, uncertainty and sky (bin width = per wavelength)
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            fluxraw: 3 HDUs
                Flux, uncertainty and sky (bin width = per pixel)
            adu: 3 HDUs
                ADU, uncertainty and sky (bin width = per pixel)
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        extension: String
            File extension without the dot.
        overwrite: boolean
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        output_split = output.split('+')
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if 'flux' in output_split:
                if self.fluxcal.flux is None:
                    warnings.warn("Spectrum is not flux calibrated.")
                else:
                    self.fluxcal._create_flux_resampled_fits(
                        'science')

            if 'wavecal' in output_split:
                if self.wavecal_science.polyfit_coeff is None:
                    warnings.warn("Spectrum is not wavelength calibrated.")
                else:
                    self.wavecal_science._create_wavecal_fits()

            if 'fluxraw' in output_split:
                if self.fluxcal.flux_raw != []:
                    warnings.warn("Spectrum is not flux calibrated.")
                else:
                    self.fluxcal._create_flux_fits('science')

            if 'adu' in output_split:
                if self.fluxcal.adu is not None:
                    self.fluxcal._create_adu_fits('science')
                elif self.wavecal_science.adu is not None:
                    self.wavecal_science._create_adu_fits()
                else:
                    warnings.warn("ADU does not exist. Have you included "
                                  "a spectrum of anysort?")

            for i in range(self.nspec):

                # Prepare multiple extension HDU
                hdu_output = fits.HDUList()
                if 'flux' in output_split:
                    if self.fluxcal.flux is not None:
                        hdu_output.extend(
                            self.fluxcal.science_resampled_hdulist[i])

                if 'wavecal' in output_split:
                    if self.wavecal_science.polyfit_coeff is not None:
                        hdu_output.extend(
                            self.wavecal_science.wavecal_hdulist[i])

                if 'fluxraw' in output_split:
                    if self.fluxcal.flux_raw != []:
                        hdu_output.extend(
                            self.fluxcal.flux_science_hdulist[i])

                if 'adu' in output_split:
                    if (self.fluxcal.adu
                            is not None) or (self.wavecal_science.adu
                                             is not None):
                        hdu_output.extend(
                            self.fluxcal.adu_hdulist[i])

                # Convert the first HDU to PrimaryHDU
                hdu_output[0] = fits.PrimaryHDU(hdu_output[0].data,
                                                hdu_output[0].header)
                hdu_output.update_extend()

                # Save file to disk
                hdu_output.writeto(filename + '_science_' + str(i) + '.fits',
                                   overwrite=overwrite)

        if 'standard' in stype_split:
            # Prepare multiple extension HDU
            hdu_output = fits.HDUList()
            if 'flux' in output_split:
                if self.fluxcal.flux_standard is None:
                    warnings.warn("Spectrum is not flux calibrated.")
                else:
                    self.fluxcal._create_flux_resampled_fits(
                        'standard')
                    hdu_output.extend(self.fluxcal.standard_resampled_hdulist)

            if 'wavecal' in output_split:
                if self.wavecal_standard.polyfit_coeff is None:
                    warnings.warn("Spectrum is not wavelength calibrated.")
                else:
                    self.wavecal_standard._create_wavecal_fits()
                    hdu_output.extend(self.wavecal_standard.wavecal_hdulist)

            if 'fluxraw' in output_split:
                if self.fluxcal.flux_standard_raw is None:
                    warnings.warn("Spectrum is not flux calibrated.")
                else:
                    self.fluxcal._create_flux_fits('standard')
                    hdu_output.extend(self.fluxcal.flux_standard_hdulist)

            if 'adu' in output_split:
                if self.fluxcal.adu_standard is not None:
                    self.fluxcal._create_adu_fits('standard')
                    hdu_output.extend(self.fluxcal.adu_standard_hdulist)
                elif self.wavecal_standard.adu_standard is not None:
                    self.wavecal_standard._create_adu_fits()
                    hdu_output.extend(
                        self.wavecal_standard.adu_standard_hdulist)
                else:
                    warnings.warn("ADU does not exist. Have you included "
                                  "a spectrum of anysort?")

            # Convert the first HDU to PrimaryHDU
            hdu_output[0] = fits.PrimaryHDU(hdu_output[0].data,
                                            hdu_output[0].header)
            hdu_output.update_extend()

            # Save file to disk
            hdu_output.writeto(filename + '_standard.fits',
                               overwrite=overwrite)

        if ('science' not in stype_split) and ('standard' not in stype_split):
            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')
