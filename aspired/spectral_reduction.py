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
from scipy import stats
from scipy import interpolate as itp
from scipy.optimize import curve_fit
from spectres import spectres

from .image_reduction import ImageReduction

base_dir = os.path.dirname(__file__)


class spectrum1D():
    '''
    Base class of a 1D spectral object

    [To Do]
    support masking
    '''
    def __init__(self, spec_id):

        # spectrum ID
        self.spec_id = spec_id

        # Detector properties
        self.gain = None
        self.readnoise = None
        self.exptime = None

        # Observing Condition
        self.relative_humidity = None
        self.pressure = None
        self.temperature = None

        # Trace properties
        self.trace = None
        self.trace_sigma = None
        self.len_trace = None
        self.pixel_list = None
        self.pixel_mapping_itp = None
        self.background = None
        self.adu = None
        self.adu_err = None
        self.adu_sky = None
        self.extraction_type = None
        self.optimal_pixel = None

        # Wavelength calibration properties
        self.arc_spec = None
        self.peaks_raw = None
        self.peaks_pixel = None
        self.polyfit_type = None

        # fit constrains
        self.calibrator = None
        self.min_atlas_wavelength = None
        self.max_atlas_wavelength = None
        self.num_slopes = None
        self.range_tolerance = None
        self.fit_tolerance = None
        self.polydeg = None
        self.candidate_thresh = None
        self.linearity_thresh = None
        self.ransac_thresh = None
        self.num_candidates = None
        self.xbins = None
        self.ybins = None
        self.brute_force = None

        # fit config
        self.sample_size = None
        self.top_n = None
        self.max_tries = None
        self.intput_coeff = None
        self.linear = None
        self.weighted = None
        self.filter_close = None

        # fit output
        self.polyfit_coeff = None
        self.rms = None
        self.residual = None
        self.peak_utilisation = None

        # fitted solution
        self.wave = None
        self.wave_bin = None
        self.wave_start = None
        self.wave_end = None
        self.wave_resampled = None
        self.adu_resampled = None
        self.adu_err_resampled = None
        self.adu_sky_resampled = None

        # Fluxes
        self.flux = None
        self.flux_err = None
        self.flux_sky = None
        self.flux_resampled = None
        self.flux_err_resampled = None
        self.flux_sky_resampled = None

        # Sensitivity curve properties
        self.smooth = None
        self.slength = None
        self.sorder = None

        self.sensitivity = None
        self.sensitivity_resampled = None
        self.sensitivity_itp = None
        self.wave_literature = None
        self.flux_literature = None

    def add_trace(self, trace, trace_sigma, pixel_list=None):
        assert type(trace) == list, 'trace has to be a list'
        assert type(trace_sigma) == list, 'trace_sigma has to be a list'
        assert len(trace_sigma) == len(trace), 'trace and trace_sigma have to '
        ' be the same size.'

        if pixel_list is None:
            pixel_list = list(np.arange(len(trace)).astype('int'))
        else:
            assert type(pixel_list) == list, 'pixel_list has to be a list'
            assert len(pixel_list) == len(trace), 'trace and pixel_list have '
            'to be the same size.'

        pixel_mapping_itp = itp.interp1d(np.arange(len(trace)),
                                         pixel_list,
                                         kind='cubic',
                                         fill_value='extrapolate')

        # Only add if all assertions are passed.
        self.trace = trace
        self.trace_sigma = trace_sigma
        self.len_trace = len(trace)
        self.add_pixel_list(pixel_list)
        self.add_pixel_mapping_itp(pixel_mapping_itp)

    def remove_trace(self):
        self.trace = None
        self.trace_sigma = None
        self.len_trace = None
        self.remove_pixel_list()
        self.remove_pixel_mapping_itp()

    def add_adu(self, adu, adu_err=None, adu_sky=None):
        assert type(adu) == list, 'adu has to be a list'

        if adu_err is not None:
            assert type(adu_err) == list, 'adu_err has to be a list'
            assert len(adu_err) == len(
                adu), 'adu_err has to be the same size as adu'

        if adu_sky is not None:
            assert type(adu_sky) == list, 'adu_sky has to be a list'
            assert len(adu_sky) == len(
                adu), 'adu_sky has to be the same size as adu'

        # Only add if all assertions are passed.
        self.adu = adu
        if adu_err is not None:
            self.adu_err = adu_err
        else:
            self.adu_err = np.zeros_like(self.adu)
        if adu_sky is not None:
            self.adu_sky = adu_sky
        else:
            self.adu_sky = np.zeros_like(self.adu)

    def remove_adu(self):
        self.adu = None
        self.adu_err = None
        self.adu_sky = None

    def add_arc_spec(self, arc_spec):
        assert type(arc_spec) == list, 'arc_spec has to be a list'
        self.arc_spec = arc_spec

    def remove_arc_spec(self):
        self.arc_spec = None

    def add_pixel_list(self, pixel_list):
        assert type(pixel_list) == list, 'pixel_list has to be a list'
        self.pixel_list = pixel_list

    def remove_pixel_list(self):
        self.pixel_list = None

    def add_pixel_mapping_itp(self, pixel_mapping_itp):
        assert type(
            pixel_mapping_itp
        ) == itp.interpolate.interp1d, 'pixel_mapping_itp has to be a scipy.interpolate.interpolate.interp1d '
        'object.'
        self.pixel_mapping_itp = pixel_mapping_itp

    def remove_pixel_mapping_itp(self):
        self.pixel_mapping_itp = None

    def add_peaks_raw(self, peaks_raw):
        assert type(peaks_raw) == list, 'peaks_raw has to be a list'
        self.peaks_raw = peaks_raw

    def remove_peaks_raw(self):
        self.peaks_raw = None

    def add_peaks_pixel(self, peaks_pixel):
        assert type(peaks_pixel) == list, 'peaks_pixel has to be a list'
        self.peaks_pixel = peaks_pixel

    def remove_peaks_pixel(self):
        self.peaks_pixel = None

    def add_peaks_wave(self, peaks_wave):
        assert type(peaks_wave) == list, 'peaks_wave has to be a list'
        self.peaks_wave = peaks_wave

    def remove_peaks_pixel(self):
        self.peaks_wave = None

    def add_background(self, background):
        # background ADU level
        assert np.isfinite(background), 'background has to be finite.'
        self.background = background

    def remove_background(self):
        self.background = None

    def add_calibrator(self, calibrator):
        assert type(
            calibrator
        ) == rascal.Calibrator, 'calibrator has to be a rascal.Calibrator object.'
        self.calibrator = calibrator

    def remove_calibrator(self):
        self.calibrator = None

    def add_atlas_wavelength_range(self, min_atlas_wavelength,
                                   max_atlas_wavelength):
        assert np.isfinite(
            min_atlas_wavelength), 'min_atlas_wavelength has to be finite.'
        assert np.isfinite(
            max_atlas_wavelength), 'max_atlas_wavelength has to be finite.'
        self.min_atlas_wavelength = min_atlas_wavelength
        self.max_atlas_wavelength = max_atlas_wavelength

    def remove_atlas_wavelength_range(self):
        self.min_atlas_wavelength = None
        self.max_atlas_wavelength = None

    def add_min_atlas_intensity(self, min_atlas_intensity):
        assert np.isfinite(
            min_atlas_intensity), 'min_atlas_intensity has to be finite.'
        self.min_atlas_intensity = min_atlas_intensity

    def remove_min_atlas_intensity(self):
        self.min_atlas_intensity = None

    def add_min_atlas_distance(self, min_atlas_distance):
        assert np.isfinite(
            min_atlas_distance), 'min_atlas_distance has to be finite.'
        self.min_atlas_distance = min_atlas_distance

    def remove_min_atlas_distance(self):
        self.min_atlas_distance = None

    def add_weather_condition(self, pressure, temperature, relative_humidity):
        assert np.isfinite(pressure), 'pressure has to be finite.'
        assert np.isfinite(temperature), 'temperature has to be finite.'
        assert np.isfinite(
            relative_humidity), 'relative_humidity has to be finite.'
        self.pressure = pressure
        self.temperature = temperature
        self.relative_humidity = relative_humidity

    def remove_weather_condition(self, pressure, temperature,
                                 relative_humidity):
        self.pressure = None
        self.temperature = None
        self.relative_humidity = None

    def add_polyfit_type(self, polyfit_type):
        assert type(polyfit_type) == str, 'polyfit_type has to be a string'
        assert polyfit_type in ['poly', 'leg', 'cheb'], 'polyfit_type must be '
        '(1) poly(nomial); (2) leg(endre); or (3) cheb(yshev).'
        self.polyfit_type = polyfit_type

    def remove_polyfit_type(self):
        self.polyfit_type = None

    def add_polyfit_coeff(self, polyfit_coeff):
        assert type(polyfit_coeff) == list, 'polyfit_coeff has to be a list.'
        self.polyfit_coeff = polyfit_coeff

    def remove_polyfit_coeff(self):
        self.polyfit_coeff = None

    def add_fit_constraints(self, num_slopes, range_tolerance, fit_tolerance,
                            polydeg, candidate_thresh, linearity_thresh,
                            ransac_thresh, num_candidates, xbins, ybins,
                            brute_force):
        assert type(num_slopes) == int, 'num_slopes has to be int.'
        assert np.isfinite(
            range_tolerance), 'range_tolerance has to be finite.'
        assert np.isfinite(fit_tolerance), 'fit_tolerance has to be finite.'
        assert type(polydeg) == int, 'polydeg has to be int.'
        assert np.isfinite(
            candidate_thresh), 'candidate_thresh has to be finite.'
        assert np.isfinite(
            linearity_thresh), 'linearity_thresh has to be finite.'
        assert np.isfinite(ransac_thresh), 'ransac_thresh has to be finite.'
        assert type(num_candidates) == int, 'num_candidates has to be int.'
        assert type(xbins) == int, 'xbins has to be int.'
        assert type(ybins) == int, 'ybins has to be int.'
        assert type(brute_force) == bool, 'brute_force has to be boolean.'

        # Only populate if all assertions pass
        self.num_slopes = num_slopes
        self.range_tolerance = range_tolerance
        self.fit_tolerance = fit_tolerance
        self.polydeg = polydeg
        self.candidate_thresh = candidate_thresh
        self.linearity_thresh = linearity_thresh
        self.ransac_thresh = ransac_thresh
        self.num_candidates = num_candidates
        self.xbins = xbins
        self.ybins = ybins
        self.brute_force = brute_force

    def remove_fit_constraints(self):
        self.num_slopes = None
        self.range_tolerance = None
        self.fit_tolerance = None
        self.polydeg = None
        self.candidate_thresh = None
        self.linearity_thresh = None
        self.ransac_thresh = None
        self.num_candidates = None
        self.xbins = None
        self.ybins = None
        self.brute_force = None
        self.polyfit_type = None

    def add_fit_config(self, sample_size, top_n, max_tries, input_coeff,
                       linear, weighted, filter_close):
        # add assertion here
        self.sample_size = sample_size
        self.top_n = top_n
        self.max_tries = max_tries
        self.intput_coeff = input_coeff
        self.linear = linear
        self.weighted = weighted
        self.filter_close = filter_close

    def remove_fit_config(self):
        self.sample_size = None
        self.top_n = None
        self.max_tries = None
        self.intput_coeff = None
        self.linear = None
        self.weighted = None
        self.filter_close = None

    def add_fit_output_final(self, polyfit_coeff, rms, residual,
                             peak_utilisation):
        # add assertion here
        self.polyfit_coeff = polyfit_coeff
        self.rms = rms
        self.residual = residual
        self.peak_utilisation = peak_utilisation

    def remove_fit_output_final(self):
        self.polyfit_coeff = None
        self.rms = None
        self.residual = None
        self.peak_utilisation = None

    def add_fit_output_rascal(self, polyfit_coeff, rms, residual,
                              peak_utilisation):
        # add assertion here
        self.polyfit_coeff_rascal = polyfit_coeff
        self.rms_rascal = rms
        self.residual_rascal = residual
        self.peak_utilisation_rascal = peak_utilisation
        self.add_fit_output_final(polyfit_coeff, rms, residual,
                                  peak_utilisation)

    def remove_fit_output_rascal(self):
        self.polyfit_coeff_rascal = None
        self.rms_rascal = None
        self.residual_rascal = None
        self.peak_utilisation_rascal = None

    def add_fit_output_refine(self, polyfit_coeff, rms, residual,
                              peak_utilisation):
        # add assertion here
        self.polyfit_coeff_refine = polyfit_coeff
        self.rms_refine = rms
        self.residual_refine = residual
        self.peak_utilisation_refine = peak_utilisation
        self.add_fit_output_final(polyfit_coeff, rms, residual,
                                  peak_utilisation)

    def remove_fit_output_refine(self):
        self.polyfit_coeff_refine = None
        self.rms_refine = None
        self.residual_refine = None
        self.peak_utilisation_refine = None

    def add_wavelength(self, wave):
        # add assertion here
        self.wave = wave

    def remove_wavelength(self):
        self.wave = None

    def add_wavelength_resampled(self, wave_bin, wave_start, wave_end,
                                 wave_resampled):
        # add assertion here
        self.wave_bin = wave_bin
        self.wave_start = wave_start
        self.wave_end = wave_end
        self.wave_resampled = wave_resampled

    def remove_wavelength_resampled(self):
        self.wave_bin = None
        self.wave_start = None
        self.wave_end = None
        self.wave_resampled = None

    def add_adu_resampled(self, adu_resampled, adu_err_resampled,
                          adu_sky_resampled):
        # add assertion here
        self.adu_resampled = adu_resampled
        self.adu_err_resampled = adu_err_resampled
        self.adu_sky_resampled = adu_sky_resampled

    def remove_adu_resampled(self):
        self.adu_resampled = None
        self.adu_err_resampled = None
        self.adu_sky_resampled = None

    def add_smoothing(self, smooth, slength, sorder):
        # add assertion here
        self.smooth = smooth
        self.slength = slength
        self.sorder = sorder

    def remove_smoothing(self):
        self.smooth = None
        self.slength = None
        self.sorder = None

    def add_sensitivity_itp(self, sensitivity_itp):
        # add assertion here
        self.sensitivity_itp = sensitivity_itp

    def remove_sensitivity_itp(self):
        self.sensitivity_itp = None

    def add_sensitivity(self, sensitivity):
        # add assertion here
        self.sensitivity = sensitivity

    def remove_sensitivity(self):
        self.sensitivity = None

    def add_literature_standard(self, wave_literature, flux_literature):
        # add assertion here
        self.wave_literature = wave_literature
        self.flux_literature = flux_literature

    def remove_literature_standard(self):
        self.wave_literature = None
        self.flux_literature = None

    def add_flux(self, flux, flux_err, flux_sky):
        # add assertion here
        self.flux = flux
        self.flux_err = flux_err
        self.flux_sky = flux_sky

    def remove_flux(self):
        self.flux = None
        self.flux_err = None
        self.flux_sky = None

    def add_flux_resampled(self, flux_resampled, flux_err_resampled,
                           flux_sky_resampled):
        # add assertion here
        self.flux_resampled = flux_resampled
        self.flux_err_resampled = flux_err_resampled
        self.flux_sky_resampled = flux_sky_resampled

    def remove_flux_resampled(self):
        self.flux_resampled = None
        self.flux_err_resampled = None
        self.flux_sky_resampled = None

    def add_sensitivity_resampled(self, sensitivity_resampled):
        # add assertion here
        self.sensitivity_resampled = sensitivity_resampled

    def remove_sensitivity_resampled(self):
        self.sensitivity_resampled = None


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
        self.spec_size = np.shape(img)[self.saxis]
        self.spatial_size = np.shape(img)[self.waxis]
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

        self.spectrum_list = {}

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

    def _identify_spectra(self, f_height, display, renderer, savejsonstring,
                          saveiframe, open_iframe):
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
        savejsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        saveiframe: boolean
            Save as an saveiframe, can work concurrently with other renderer
            apart from exporting savejsonstring.
        open_iframe: boolean
            Open the saveiframe in the default browser if set to True.

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

        self.peak = peaks_y
        self.peak_height = heights_y

        if display or saveiframe or savejsonstring:

            # set a side-by-side subplot
            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width))

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
            fig.update_layout(yaxis_title='Spatial Direction / pixel',
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

            if saveiframe:
                pio.write_html(fig,
                               'identify_spectra.html',
                               auto_open=open_iframe)

            # display disgnostic plot
            if display:
                if renderer == 'default':
                    fig.show()
                else:
                    fig.show(renderer)

            if savejsonstring:
                return fig.to_json()

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

        sky_median = np.nanmedian(sky)

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
                 width=1280,
                 height=720,
                 savejsonstring=False,
                 saveiframe=False,
                 open_iframe=False):
        '''
        Aperture tracing by first using cross-correlation then the peaks are
        fitting with a polynomial with an order of floor(nwindow, 10) with a
        minimum order of 1. Nothing is returned unless savejsonstring of the
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
        savejsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        saveiframe: boolean
            Save as an saveiframe, can work concurrently with other renderer
            apart from exporting savejsonstring.
        open_iframe: boolean
            Open the saveiframe in the default browser if set to True.

        Returns
        -------
        json string if savejsonstring is True, otherwise only an image is displayed
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
        self.nspec_traced = min(len(peaks[0]), nspec)

        # Sort the positions by the prominences, and return to the original
        # scale (i.e. with subpixel position)
        spec_init = np.sort(peaks[0][np.argsort(-peaks[1]['prominences'])]
                            [:self.nspec_traced]) / resample_factor

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

        for i in range(len(spec_idx)):

            # Get the median of the subspectrum and then get the ADU at the
            # centre of the aperture
            ap_val = np.zeros(nwindow)
            for j in range(nwindow):
                # rounding
                idx = int(spec_idx[i][j] + 0.5)
                ap_val[j] = np.nanmedian(img_split[j], axis=1)[idx]

            # Mask out the faintest ap_faint percentile
            mask = (ap_val > np.nanpercentile(ap_val, ap_faint))

            # fit the trace
            ap_p = np.polyfit(spec_pix[mask], spec_idx[i][mask], int(polydeg))
            ap = np.polyval(ap_p, np.arange(nwave))

            # Get the centre of the upsampled spectrum
            ap_centre_idx = ap[start_window_idx] * resample_factor

            # Get the indices for the 10 pixels on the left and right of the
            # spectrum, and apply the resampling factor.
            start_idx = int(ap_centre_idx - 10 * resample_factor + 0.5)
            end_idx = start_idx + 20 * resample_factor + 1

            start_idx = max(0, start_idx)
            end_idx = min(nspatial * resample_factor, end_idx)

            if start_idx == end_idx:
                ap_sigma = np.nan
                continue

            # compute ONE sigma for each trace
            pguess = [
                np.nanmax(spec_spatial[start_idx:end_idx]),
                np.nanpercentile(spec_spatial, 10), ap_centre_idx, 3.
            ]

            non_nan_mask = np.isnan(spec_spatial[start_idx:end_idx]) == False
            popt, pcov = curve_fit(
                self._gaus,
                np.arange(start_idx, end_idx)[non_nan_mask],
                spec_spatial[start_idx:end_idx][non_nan_mask],
                p0=pguess)
            ap_sigma = popt[3] / resample_factor

            self.spectrum_list[i] = spectrum1D(i)
            self.spectrum_list[i].add_trace(list(ap), [ap_sigma] * len(ap))

        # Plot
        if saveiframe or display or savejsonstring:

            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width))

            fig.add_trace(
                go.Heatmap(z=np.log10(self.img),
                           zmin=self.zmin,
                           zmax=self.zmax,
                           colorscale="Viridis",
                           colorbar=dict(title='log(ADU / s)')))
            for i in range(len(spec_idx)):
                fig.add_trace(
                    go.Scatter(x=np.arange(nwave),
                               y=self.spectrum_list[i].trace,
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
            fig.update_layout(yaxis_title='Spatial Direction / pixel',
                              xaxis=dict(zeroline=False,
                                         showgrid=False,
                                         title='Spectral Direction / pixel'),
                              bargap=0,
                              hovermode='closest',
                              showlegend=False)
            if saveiframe:
                pio.write_html(fig, 'ap_trace.html', auto_open=open_iframe)

            if display:
                if renderer == 'default':
                    fig.show()
                else:
                    fig.show(renderer)

            if savejsonstring:
                return fig.to_json()

    def add_trace(self, spec_id, trace, trace_sigma):
        '''
        Add user-supplied trace. The trace has to have the size as the 2D
        spectral image in the spectral direction.

        Parameters
        ----------
        trace: list (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: list (N)
            Standard deviation of the Gaussian profile of a trace

        '''

        assert type(spec_id) == int, 'spec_id has to be an integer.'
        assert type(trace) == list, 'trace has to be a list.'
        assert type(trace_sigma) == list, 'trace_sigma has to be a list.'
        assert len(trace) == len(trace_sigma), 'trace and trace_sigma have to '
        'be the same size.'

        if spec_id in spectrum_list:
            self.spectrum_list[spec_id].add_trace(trace, trace_sigma)
        else:
            raise ValueError("{spec_id: %s} is not in the list of spectra." %
                             spec_id)

    def remove_trace(self, spec_id):
        if spec_id in spectrum_list:
            self.spectrum_list[spec_id].remove_trace()
        else:
            raise ValueError("{spec_id: %s} is not in the list of spectra." %
                             spec_id)

    def ap_extract(self,
                   apwidth=7,
                   skysep=3,
                   skywidth=5,
                   skydeg=1,
                   spec_id=None,
                   optimal=True,
                   display=False,
                   renderer='default',
                   width=1280,
                   height=720,
                   savejsonstring=False,
                   saveiframe=False,
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

        Nothing is returned unless savejsonstring of the plotly graph is set to be
        returned. The adu, adu_sky and adu_err are stored as properties of the
        TwoDSpec object.

        adu: 1-d array
            The summed adu at each column about the trace. Note: is not
            sky subtracted!
        adu_sky: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        adu_err: 1-d array
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
        savejsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        saveiframe: boolean
            Save as an saveiframe, can work concurrently with other renderer
            apart from exporting savejsonstring.
        open_iframe: boolean
            Open the saveiframe in the default browser if set to True.

        """

        if spec_id is not None:
            assert np.in1d(spec_id,
                           list(self.spectrum_list.keys())).all(), 'Some '
            'spec_id provided are not in the spectrum_list.'
        else:
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        for j in spec_id:
            len_trace = len(self.spectrum_list[j].trace)
            adu_sky = np.zeros(len_trace)
            adu_err = np.zeros(len_trace)
            adu = np.zeros(len_trace)
            suboptimal = np.zeros(len_trace, dtype=bool)

            for i, pos in enumerate(self.spectrum_list[j].trace):
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

                if isinstance(skysep, int):
                    # first do the aperture adu
                    sepdn = skysep
                    sepup = skysep
                elif len(skysep) == 2:
                    sepdn = skysep[0]
                    sepup = skysep[1]
                else:
                    raise TypeError(
                        'skysep can only be an int or a list of two ints')

                if isinstance(skywidth, int):
                    # first do the aperture adu
                    skywidthdn = skywidth
                    skywidthup = skywidth
                elif len(skywidth) == 2:
                    skywidthdn = skywidth[0]
                    skywidthup = skywidth[1]
                else:
                    raise TypeError(
                        'skywidth can only be an int or a list of two ints')

                # fix width if trace is too close to the edge
                if (itrace + widthup > self.spatial_size):
                    widthup = spatial_size - itrace - 1
                if (itrace - widthdn < 0):
                    widthdn = itrace - 1  # i.e. starting at pixel row 1

                # simply add up the total adu around the trace +/- width
                xslice = self.img[itrace - widthdn:itrace + widthup + 1, i]
                adu_ap = np.sum(xslice) - pix_frac * xslice[0] - (
                    1 - pix_frac) * xslice[-1]

                if (skywidthup > 0) or (skywidthdn > 0):
                    # get the indexes of the sky regions
                    y0 = max(itrace - widthdn - sepdn - skywidthdn, 0)
                    y1 = max(itrace - widthdn - sepdn, 0)
                    y2 = min(itrace + widthup + sepup + 1, self.spatial_size)
                    y3 = min(itrace + widthup + sepup + skywidthup + 1,
                             self.spatial_size)
                    y = np.append(np.arange(y0, y1), np.arange(y2, y3))
                    z = self.img[y, i]

                    if (skydeg > 0):
                        # fit a polynomial to the sky in this column
                        polyfit = np.polyfit(y, z, skydeg)
                        # define the aperture in this column
                        ap = np.arange(itrace - widthdn, itrace + widthup + 1)
                        # evaluate the polynomial across the aperture, and sum
                        adu_sky_slice = np.polyval(polyfit, ap)
                        adu_sky[i] = np.sum(
                            adu_sky_slice) - pix_frac * adu_sky_slice[0] - (
                                1 - pix_frac) * adu_sky_slice[-1]
                    elif (skydeg == 0):
                        adu_sky[i] = (widthdn + widthup) * np.nanmean(z)

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
                    adu[i], adu_err[i], suboptimal[i] = self._optimal_signal(
                        pix, xslice, sky, self.spectrum_list[j].trace[i],
                        self.spectrum_list[j].trace_sigma[i])
                else:
                    #-- finally, compute the error in this pixel
                    sigB = np.nanstd(z)  # standarddev in the background data
                    nB = len(y)  # number of bkgd pixels
                    nA = apwidth * 2. + 1  # number of aperture pixels

                    # based on aperture phot err description by F. Masci, Caltech:
                    # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
                    adu_err[i] = np.sqrt((adu_ap - adu_sky[i]) / self.gain +
                                         (nA + nA**2. / nB) * (sigB**2.))
                    adu[i] = adu_ap - adu_sky[i]

            self.spectrum_list[j].add_adu(list(adu), list(adu_err),
                                          list(adu_sky))
            self.spectrum_list[j].gain = self.gain
            self.spectrum_list[j].optimal_pixel = suboptimal

            if optimal:
                self.spectrum_list[j].extraction_type = "Optimal"
            else:
                self.spectrum_list[j].extraction_type = "Aperture"

            # If more than a third of the spectrum is extracted suboptimally
            if np.sum(suboptimal) / i > 0.333:
                if not self.silence:
                    print(
                        'Signal extracted is likely to be suboptimal, please try '
                        'a longer iteration, larger tolerance or revert to '
                        'top-hat extraction.')

            if saveiframe or display or savejsonstring:
                min_trace = int(min(self.spectrum_list[j].trace) + 0.5)
                max_trace = int(max(self.spectrum_list[j].trace) + 0.5)

                fig = go.Figure(
                    layout=dict(autosize=False, height=height, width=width))
                # the 3 is to show a little bit outside the extraction regions
                img_display = np.log10(self.img[
                    max(0, min_trace - widthdn - sepdn - skywidthdn -
                        3):min(max_trace + widthup + sepup +
                               skywidthup, len(self.img[0])) + 3, :])

                # show the image on the top
                # the 3 is the show a little bit outside the extraction regions
                fig.add_trace(
                    go.Heatmap(
                        x=np.arange(len_trace),
                        y=np.arange(
                            max(0,
                                min_trace - widthdn - sepdn - skywidthdn - 3),
                            min(max_trace + widthup + sepup + skywidthup + 3,
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
                    go.Scatter(
                        x=list(
                            np.concatenate(
                                (np.arange(len_trace),
                                 np.arange(len_trace)[::-1], np.zeros(1)))),
                        y=list(
                            np.concatenate(
                                (np.array(self.spectrum_list[j].trace) -
                                 widthdn - 1,
                                 np.array(self.spectrum_list[j].trace[::-1]) +
                                 widthup + 1,
                                 np.ones(1) * (self.spectrum_list[j].trace[0] -
                                               widthdn - 1)))),
                        xaxis='x',
                        yaxis='y',
                        mode='lines',
                        line_color='black',
                        showlegend=False))

                # Lower red box on the image
                lower_redbox_upper_bound = np.array(
                    self.spectrum_list[j].trace) - widthdn - sepdn - 1
                lower_redbox_lower_bound = np.array(
                    self.spectrum_list[j].trace)[::-1] - widthdn - sepdn - max(
                        skywidthdn, (y1 - y0) - 1)

                if (itrace - widthdn >= 0) & (skywidthdn > 0):
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
                upper_redbox_upper_bound = np.array(
                    self.spectrum_list[j].trace) + widthup + sepup + min(
                        skywidthup, (y3 - y2) + 1)
                upper_redbox_lower_bound = np.array(
                    self.spectrum_list[j].trace)[::-1] + widthup + sepup + 1

                if (itrace + widthup <= self.spatial_size) & (skywidthup > 0):
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
                               y=adu / adu_err,
                               xaxis='x2',
                               yaxis='y3',
                               line=dict(color='slategrey'),
                               name='Signal-to-Noise Ratio'))

                # extrated source, sky and uncertainty
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=adu_sky,
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='firebrick'),
                               name='Sky ADU / s'))
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=adu_err,
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='orange'),
                               name='Uncertainty'))
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=adu,
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='royalblue'),
                               name='Target ADU / s'))

                # Decorative stuff
                fig.update_layout(
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(zeroline=False,
                               domain=[0.5, 1],
                               showgrid=False,
                               title='Spatial Direction / pixel'),
                    yaxis2=dict(
                        range=[
                            min(
                                np.nanmin(
                                    sigma_clip(np.log10(adu),
                                               sigma=5.,
                                               masked=False)),
                                np.nanmin(
                                    sigma_clip(np.log10(adu_err),
                                               sigma=5.,
                                               masked=False)),
                                np.nanmin(
                                    sigma_clip(np.log10(adu_sky),
                                               sigma=5.,
                                               masked=False)), 1),
                            max(np.nanmax(np.log10(adu)),
                                np.nanmax(np.log10(adu_sky)))
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
                    showlegend=True)

                if saveiframe:
                    pio.write_html(fig,
                                   'ap_extract_' + str(j) + 'html',
                                   auto_open=open_iframe)

                if display:
                    if renderer == 'default':
                        fig.show()
                    else:
                        fig.show(renderer)

                if savejsonstring:
                    return fig.to_json()

    def _create_trace_fits(self):
        # Put the reduced data in FITS format with an image header
        self.trace_hdulist = np.array([None] * len(self.spectrum_list),
                                      dtype='object')
        for i, spec in self.spectrum_list.items():
            self.trace_hdulist[i] = fits.HDUList(
                [fits.ImageHDU(spec.trace),
                 fits.ImageHDU(spec.trace_sigma)])

    def _create_adu_fits(self):
        # Put the reduced data in FITS format with an image header
        self.adu_hdulist = np.array([None] * len(self.spectrum_list),
                                    dtype='object')
        for i, spec in self.spectrum_list.items():
            self.adu_hdulist[i] = fits.HDUList([
                fits.ImageHDU(spec.adu),
                fits.ImageHDU(spec.adu_err),
                fits.ImageHDU(spec.adu_sky)
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
        for i in range(len(self.spectrum_list)):

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
        self.wavecal_hdulist = None
        self.polyfit_hdulist = None

        self.spectrum_list = {}

        self.polyval = {
            'poly': np.polynomial.polynomial.polyval,
            'leg': np.polynomial.legendre.legval,
            'cheb': np.polynomial.chebyshev.chebval
        }

    def add_arc(self, arc):
        '''
        To provide an arc image. Make sure left (small index) is blue,
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
                'aspired.ImageReduction object.')

    def add_arc_lines(self, peaks, spec_id=None):
        '''
        Provide the pixel locations of the arc lines.

        Parameters
        ----------
        peaks: list of list or list of arrays
            The pixel locations of the arc lines. Multiple traces of the arc
            can be provided as list of list or list of arrays.
        '''
        if spec_id in list(self.spectrum_list.keys()):
            if self.spectrum_list[spec_id].adu is not None:
                warnings.warn('The given spec_id is in use already, the given '
                              'peaks will overwrite the current data.')

        if spec_id is None:
            # Add to the first spec
            spec_id = 0

        self.spectrum_list[spec_id].peaks = peaks

    def add_arc_spec(self, arc_spec, spec_id=None):
        '''
        Provide the collapsed 1D spectrum/a of the arc image.

        Parameters
        ----------
        spectrum: list of list or list of arrays
            The ADU/flux of the 1D arc spectrum/a. Multiple spectrum/a
            can be provided as list of list or list of arrays.
        '''
        if spec_id in list(self.spectrum_list.keys()):
            if self.spectrum_list[spec_id].adu is not None:
                warnings.warn('The given spec_id is in use already, the given '
                              'arc_spec will overwrite the current data.')

        if spec_id is None:
            # Add to the first spec
            spec_id = 0

        self.spectrum_list[spec_id].arc_spec = arc_spec

    def add_spec(self, adu, spec_id=None, adu_err=None, adu_sky=None):
        '''
        To provide user-supplied extracted spectrum for wavelegth calibration.

        '''

        if spec_id in list(self.spectrum_list.keys()):
            if self.spectrum_list[spec_id].adu is not None:
                warnings.warn('The given spec_id is in use already, the given '
                              'adu, adu_err and adu_sky will overwrite the '
                              'current data.')

        if spec_id is None:
            # Add to the first spec
            spec_id = 0

        self.spectrum_list[spec_id].add_adu(adu, adu_err, adu_sky)

    def remove_spec(self, spec_id):
        '''
        To modify or append a spectrum with the user-supplied one, one at a
        time.
        '''
        if spec_id not in list(self.spectrum_list.keys()):
            raise ValueError('The given spec_id does not exist.')

        self.spectrum_list[spec_id].remove_adu()

    def add_trace(self, trace, trace_sigma, spec_id=None, pixel_list=None):
        '''
        To provide user-supplied trace. The trace has to be the size as the 2D
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

        if spec_id in list(self.spectrum_list.keys()):
            if self.spectrum_list[spec_id].trace is not None:
                warnings.warn(
                    'The given spec_id is in use already, the given '
                    'trace, trace_sigma and pixel_list will overwrite the '
                    'current data.')

        if spec_id is None:
            # Add to the first spec
            spec_id = 0

        self.spectrum_list[spec_id].add_trace(trace, trace_sigma, pixel_list)

    def remove_trace(self, spec_id):
        if spec_id not in list(self.spectrum_list.keys()):
            raise ValueError('The given spec_id does not exist.')

        self.spectrum_list[spec_id].remove_trace()

    def add_polyfit(self, spec_id, polyfit_type, polyfit_coeff):
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

        if spec_id in list(self.spectrum_list.keys()):
            if self.spectrum_list[spec_id].polyfit_coeff is not None:
                warnings.warn(
                    'The given spec_id is in use already, the given '
                    'polyfit_coeff and polyfit_type will overwrite the '
                    'current data.')

        if spec_id is None:
            # Add to the first spec
            spec_id = 0

        self.spectrum_list[spec_id].add_polyfit(polyfit_type, polyfit_coeff)

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
        self.spectrum_list = twodspec.spectrum_list

    def apply_twodspec_mask(self):

        if self.saxis == 0:
            self.arc = np.transpose(self.arc)

        if self.flip:
            self.arc = np.flip(self.arc)

        self.apply_spec_mask(self.spec_mask)
        self.apply_spatial_mask(self.spatial_mask)

    '''
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
    '''

    def extract_arc_spec(self,
                         spec_id=None,
                         display=False,
                         savejsonstring=False,
                         renderer='default',
                         width=1280,
                         height=720,
                         saveiframe=False,
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
        savejsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        saveiframe: boolean
            Save as an saveiframe, can work concurrently with other renderer
            apart from exporting savejsonstring.
        open_iframe: boolean
            Open the saveiframe in the default browser if set to True.
        '''
        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, all arc spectra are extracted
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if self.arc is None:
            raise ValueError(
                'arc is not provided. Please provide arc by using add_arc() '
                'or with add_twodspec() before executing find_arc_lines().')

        for i in spec_id:

            spec = self.spectrum_list[i]

            len_trace = len(spec.trace)
            trace = np.nanmean(spec.trace)
            trace_width = np.nanmean(spec.trace_sigma) * 3.

            arc_trace = self.arc[
                max(0, int(trace - trace_width -
                           1)):min(int(trace + trace_width), len_trace), :]
            arc_spec = np.nanmedian(arc_trace, axis=0)

            spec.add_arc_spec(list(arc_spec))

            # note that the display is adjusted for the chip gaps
            if saveiframe or display or savejsonstring:

                fig = go.Figure(
                    layout=dict(autosize=False, height=height, width=width))

                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=arc_spec,
                               mode='lines',
                               line=dict(color='royalblue', width=1)))

                fig.update_layout(xaxis=dict(
                    zeroline=False,
                    range=[0, len_trace],
                    title='Spectral Direction / pixel'),
                                  yaxis=dict(zeroline=False,
                                             range=[0, max(arc_spec)],
                                             title='ADU per second'),
                                  hovermode='closest',
                                  showlegend=False)

                if saveiframe:
                    pio.write_html(fig,
                                   'arc_spec_' + str(i) + 'html',
                                   auto_open=open_iframe)

                if display:
                    if renderer == 'default':
                        fig.show()
                    else:
                        fig.show(renderer)

                if savejsonstring:
                    return fig.to_json()

    def find_arc_lines(self,
                       spec_id=None,
                       background=None,
                       percentile=10.,
                       prominence=10.,
                       distance=5.,
                       refine=True,
                       refine_window_width=5,
                       display=False,
                       width=1280,
                       height=720,
                       savejsonstring=False,
                       renderer='default',
                       saveiframe=False,
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
            background sky level to the first order. Only used if background
            is None. [ADU]
        distance: float
            Minimum separation between peaks
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        savejsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        saveiframe: boolean
            Save as an saveiframe, can work concurrently with other renderer
            apart from exporting savejsonstring.
        open_iframe: boolean
            Open the saveiframe in the default browser if set to True.

        Returns
        -------
        JSON strings if savejsonstring is set to True
        '''

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, the all arc lines are found
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if self.arc is None:
            raise ValueError(
                'arc is not provided. Please provide arc by using add_arc() '
                'or with add_twodspec() before executing find_arc_lines().')

        for i in spec_id:

            arc_spec = self.spectrum_list[i].arc_spec

            if background is None:
                background = np.nanpercentile(arc_spec, percentile)

            peaks_raw, _ = signal.find_peaks(arc_spec,
                                             distance=distance,
                                             height=background,
                                             prominence=prominence)

            self.spectrum_list[i].add_background(background)

            # Fine tuning
            if refine:
                peaks_raw = refine_peaks(arc_spec,
                                         peaks_raw,
                                         window_width=int(refine_window_width))

            self.spectrum_list[i].add_peaks_raw(list(peaks_raw))

            # Adjust for the pixel mapping (e.g. chip gaps increment)
            self.spectrum_list[i].add_peaks_pixel(
                list(self.spectrum_list[i].pixel_mapping_itp(peaks_raw)))

            if saveiframe or display or savejsonstring:
                fig = go.Figure(
                    layout=dict(autosize=False, height=height, width=width))

                # show the image on the top
                fig.add_trace(
                    go.Heatmap(x=np.arange(self.arc.shape[0]),
                               y=np.arange(self.arc.shape[1]),
                               z=np.log10(self.arc),
                               colorscale="Viridis",
                               colorbar=dict(title='log(ADU / s)')))

                # note that the image is not adjusted for the chip gaps
                # peaks_raw are plotted instead of the peaks
                trace = np.nanmean(self.spectrum_list[i].trace)
                trace_width = np.nanmean(self.spectrum_list[i].trace_sigma) * 3
                for j in peaks_raw:
                    fig.add_trace(
                        go.Scatter(x=[j, j],
                                   y=[
                                       int(trace - trace_width - 1),
                                       int(trace + trace_width)
                                   ],
                                   mode='lines',
                                   line=dict(color='firebrick', width=1)))

                fig.update_layout(
                    xaxis=dict(zeroline=False,
                               range=[0, self.arc.shape[1]],
                               title='Spectral Direction / pixel'),
                    yaxis=dict(zeroline=False,
                               range=[0, self.arc.shape[0]],
                               title='Spatial Direction / pixel'),
                    hovermode='closest',
                    showlegend=False)

                if saveiframe:
                    pio.write_html(fig,
                                   'arc_lines_' + str(i) + '.html',
                                   auto_open=open_iframe)

                if display:
                    if renderer == 'default':
                        fig.show()
                    else:
                        fig.show(renderer)

                if savejsonstring:
                    return fig.to_json()

    def initialise_calibrator(self,
                              spec_id=None,
                              peaks=None,
                              num_pix=None,
                              pixel_list=None,
                              min_wavelength=3000,
                              max_wavelength=9000,
                              plotting_library='plotly',
                              log_level='info'):

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, calibrators are initialised to all
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        for i in spec_id:

            if peaks is None:
                peaks = self.spectrum_list[i].peaks_raw

            if num_pix is None:
                num_pix = self.spectrum_list[i].len_trace

            if pixel_list is None:
                pixel_list = self.spectrum_list[i].pixel_list

            self.spectrum_list[i].calibrator = Calibrator(
                peaks=peaks,
                num_pix=num_pix,
                pixel_list=pixel_list,
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                plotting_library=plotting_library,
                log_level=log_level)

    def set_known_pairs(self, spec_id=None, pix=None, wave=None):

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, the pix-wave pairs apply to all
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        assert (pix is not None) & (
            wave is not None), "Neither pix nor wave can be None."

        assert (np.shape(pix)
                == np.shape(wave)) & (np.shape(np.shape(pix)) == np.shape(
                    np.shape(wave))), "pix and wave must have the same shape."

        for i in spec_id:
            self.spectrum_list[i].calibrator.set_known_pairs(pix=pix,
                                                             wave=wave)

    def set_fit_constraints(self,
                            spec_id=None,
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
                            polyfit_type='poly'):
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
        polyfit_type : string (default: 'poly')
            One of 'poly', 'legendre' or 'chebyshev'
        '''

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        for i in spec_id:

            self.spectrum_list[i].calibrator.set_fit_constraints(
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
                fittype=polyfit_type)

            self.spectrum_list[i].add_fit_constraints(
                num_slopes, range_tolerance, fit_tolerance, polydeg,
                candidate_thresh, linearity_thresh, ransac_thresh,
                num_candidates, xbins, ybins, brute_force)

            self.spectrum_list[i].add_polyfit_type(polyfit_type)

    def load_user_atlas(self,
                        elements,
                        wavelengths,
                        spec_id=None,
                        intensities=None,
                        constrain_poly=False,
                        vacuum=False,
                        pressure=101325.,
                        temperature=273.15,
                        relative_humidity=0.):

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        for i in spec_id:
            self.spectrum_list[i].calibrator.load_user_atlas(
                elements=elements,
                wavelengths=wavelengths,
                intensities=intensities,
                constrain_poly=constrain_poly,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity)

    def add_atlas(self,
                  elements,
                  spec_id=None,
                  min_atlas_wavelength=0,
                  max_atlas_wavelength=15000,
                  min_atlas_intensity=0,
                  min_atlas_distance=0,
                  vacuum=False,
                  pressure=101325.,
                  temperature=273.15,
                  relative_humidity=0.,
                  constrain_poly=False):
        '''
            elements: string or list of string
                String or list of strings of Chemical symbol. Case insensitive.
            min_atlas_wave: float
                Minimum wavelength of the bluest arc line, NOT OF THE SPECTRUM.
            max_atlas_wave: float
                Maximum wavelength of the reddest arc line, NOT OF THE SPECTRUM.

        '''
        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        for i in spec_id:
            self.spectrum_list[i].calibrator.add_atlas(
                elements=elements,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=min_atlas_intensity,
                min_distance=min_atlas_distance,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
                constrain_poly=constrain_poly)

            self.spectrum_list[i].add_atlas_wavelength_range(
                min_atlas_wavelength, max_atlas_wavelength)

            self.spectrum_list[i].add_min_atlas_intensity(min_atlas_intensity)

            self.spectrum_list[i].add_min_atlas_distance(min_atlas_distance)

            self.spectrum_list[i].add_weather_condition(
                pressure, temperature, relative_humidity)

    def fit(self,
            spec_id=None,
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
            filename=None):
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

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        for i in spec_id:
            self.spectrum_list[i].add_fit_config(sample_size, top_n, max_tries,
                                                 coeff, linear, weighted,
                                                 filter_close)

            polyfit_coeff, rms, residual, peak_utilisation = self.spectrum_list[
                i].calibrator.fit(sample_size, top_n, max_tries, progress,
                                  coeff, linear, weighted, filter_close)

            self.spectrum_list[i].add_fit_output_rascal(
                polyfit_coeff, rms, residual, peak_utilisation)

            if display:
                self.spectrum_list[i].calibrator.plot_fit(
                    self.spectrum_list[i].arc_spec,
                    polyfit_coeff,
                    plot_atlas=True,
                    log_spectrum=False,
                    tolerance=1.0,
                    savefig=savefig,
                    filename=filename)

    def refine_fit(self,
                   spec_id=None,
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
                   filename=None):
        '''
        A wrapper function to fine tune wavelength calibration with RASCAL
        when there is already a set of good coefficienes.

        As of 14 January 2020, it supports He, Ne, Ar, Cu, Kr, Cd, Xe,
        Hg and Th from NIST:

            https://physics.nist.gov/PhysRefData/ASD/lines_form.html

        Parameters
        ----------
        polyfit_coeff: list

        n_delta: int
            The n-th lowest order coefficients to be refitted.
        refine: boolean

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
        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        for i in spec_id:

            if polyfit_coeff is None:
                polyfit_coeff = self.spectrum_list[i].polyfit_coeff

            if polydeg is None:
                polydeg = len(polyfit_coeff) - 1

            if n_delta is None:
                n_delta = len(polyfit_coeff) - 1

            polyfit_coeff_new, _, _, residual, peak_utilisation = self.spectrum_list[
                i].calibrator.match_peaks(polyfit_coeff,
                                          n_delta=n_delta,
                                          refine=refine,
                                          tolerance=tolerance,
                                          method=method,
                                          convergence=convergence,
                                          robust_refit=robust_refit,
                                          polydeg=polydeg)
            rms = np.sqrt(np.nanmean(residual**2.))

            if display:
                self.spectrum_list[i].calibrator.plot_fit(
                    self.spectrum_list[i].arc_spec,
                    polyfit_coeff_new,
                    plot_atlas=True,
                    log_spectrum=False,
                    tolerance=1.0,
                    savefig=savefig,
                    filename=filename)

            self.spectrum_list[i].add_fit_output_refine(
                polyfit_coeff_new, rms, residual, peak_utilisation)

    def apply_wavelength_calibration(self,
                                     spec_id=None,
                                     wave_start=None,
                                     wave_end=None,
                                     wave_bin=None):
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

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        for i in spec_id:

            spec = self.spectrum_list[i]

            fit_type = spec.polyfit_type

            # Adjust for pixel shift due to chip gaps
            wave = self.polyval[fit_type](np.array(spec.pixel_list),
                                          spec.polyfit_coeff)

            # compute the new equally-spaced wavelength array
            if wave_bin is None:
                wave_bin = np.nanmedian(np.ediff1d(wave))

            if wave_start is None:
                wave_start = wave[0]

            if wave_end is None:
                wave_end = wave[-1]

            wave_resampled = np.arange(wave_start, wave_end, wave_bin)

            # apply the flux calibration and resample
            adu_resampled = spectres(wave_resampled,
                                     wave,
                                     np.array(spec.adu),
                                     verbose=False)
            if spec.adu_err is not None:
                adu_err_resampled = spectres(wave_resampled,
                                             wave,
                                             np.array(spec.adu_err),
                                             verbose=False)
            if spec.adu_sky is not None:
                adu_sky_resampled = spectres(wave_resampled,
                                             wave,
                                             np.array(spec.adu_sky),
                                             verbose=False)

            spec.add_wavelength(wave)
            spec.add_wavelength_resampled(wave_bin, wave_start, wave_end,
                                          wave_resampled)
            spec.add_adu_resampled(adu_resampled, adu_err_resampled,
                                   adu_sky_resampled)

    def _create_adu_fits(self, spec_id=None):

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        self.adu_hdulist = np.array([None](max(spec_id) + 1), dtype='object')

        for i in spec_id:
            spec = self.spectrum_list[i]
            self.adu_hdulist[i] = fits.HDUList([
                fits.ImageHDU(spec.adu),
                fits.ImageHDU(spec.adu_err),
                fits.ImageHDU(spec.adu_sky)
            ])

    def _create_adu_resampled_fits(self, spec_id=None):

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        self.adu_resampled_hdulist = np.array([None] * (max(spec_id) + 1),
                                              dtype='object')

        for i in spec_id:
            spec = self.spectrum_list[i]
            self.adu_resampled_hdulist[i] = fits.HDUList([
                fits.ImageHDU(spec.adu_resampled),
                fits.ImageHDU(spec.adu_err_resampled),
                fits.ImageHDU(spec.adu_sky_resampled)
            ])

    def _create_arc_spec_fits(self, spec_id=None):

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        self.arc_spec_hdulist = np.array([None] * (max(spec_id) + 1),
                                         dtype='object')

        for i in spec_id:
            spec = self.spectrum_list[i]
            self.arc_spec_hdulist[i] = fits.HDUList(
                [fits.ImageHDU(spec.arc_spec)])

    def _create_wavecal_fits(self, spec_id=None):

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        self.wavecal_hdulist = np.array([None] * (max(spec_id) + 1),
                                        dtype='object')

        for i in spec_id:
            spec = self.spectrum_list[i]
            self.wavecal_hdulist[i] = fits.HDUList(
                [fits.ImageHDU(spec.polyfit_coeff)])

    def save_fits(self,
                  spec_id=None,
                  output='arc_spec+wavecal',
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

        if spec_id is not None:
            if spec_id not in list(self.spectrum_list.keys()):
                raise ValueError('The given spec_id does not exist.')
        else:
            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        filename = os.path.splitext(filename)[0]

        output_split = output.split('+')

        if (self.arc_spec is None) and (self.polyfit_coeff is None):
            raise ValueError('Neither arc_spec is extracted nor wavelength is '
                             'calibrated.')

        if 'arc_spec' in output_split:
            self._create_arc_spec_fits(spec_id)

        if 'adu' in output_split:
            self._create_adu_fits(spec_id)

        if 'adu_resampled' in output_split:
            self._create_adu_resampled_fits(spec_id)

        if 'wavecal' in output_split:
            self._create_wavecal_fits(spec_id)

        # Save each trace as a separate FITS file
        for i in spec_id:

            spec = self.spectrum_list[i]

            hdu_output = fits.HDUList()

            if 'arc_spec' in output_split:
                if spec.arc_spec is not None:
                    hdu_output.extend(self.arc_spec_hdulist[i])

            if 'adu' in output_split:
                if spec.adu is not None:
                    hdu_output.extend(self.adu_hdulist[i])

            if 'adu_resampled' in output_split:
                if spec.adu_resampled is not None:
                    hdu_output.extend(self.adu_resampled_hdulist[i])

            if 'wavecal' in output_split:
                if spec.polyfit_coeff is not None:
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

    def load_standard(self, target, library=None, ftype='flux', cutoff=0.4):
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
        savejsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        saveiframe: boolean
            Save as an saveiframe, can work concurrently with other renderer
            apart from exporting savejsonstring.
        open_iframe: boolean
            Open the saveiframe in the default browser if set to True.

        Returns
        -------
        JSON strings if savejsonstring is set to True
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
                if not self.silence:
                    warnings.warn(
                        'The requested standard star cannot be found in the '
                        'given library,  or the library is not specified. '
                        'ASPIRED is using ' + self.library + '.')
        else:
            # If not, search again with the first one returned from lookup.
            self.target = libraries[0]
            libraries, _ = self.lookup_standard_libraries(self.target)
            self.library = libraries[0]
            if not self.silence:
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

    def inspect_standard(self,
                         renderer='default',
                         width=1280,
                         height=720,
                         savejsonstring=False,
                         display=True,
                         saveiframe=False,
                         open_iframe=False):
        '''
        Display the standard star plot.

        Parameters
        ----------
        renderer: string
            plotly renderer options.
        savejsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        saveiframe: boolean
            Save as an saveiframe, can work concurrently with other renderer
            apart from exporting savejsonstring.
        open_iframe: boolean
            Open the saveiframe in the default browser if set to True.

        Returns
        -------
        JSON strings if savejsonstring is set to True
        '''
        fig = go.Figure(
            layout=dict(autosize=False,
                        height=height,
                        width=width,
                        updatemenus=list([
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
            title=self.library + ': ' + self.target + ' ' + self.ftype,
            xaxis_title=r'$\text{Wavelength / A}$',
            yaxis_title=
            r'$\text{Flux / ergs cm}^{-2} \text{s}^{-1} \text{A}^{-1}$',
            hovermode='closest',
            showlegend=False)

        if saveiframe:
            pio.write_html(fig, 'standard.html', auto_open=open_iframe)

        if display:
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

        if savejsonstring:
            return fig.to_json()

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

        self.spectrum_list_science = {}
        self.spectrum_list_standard = {}

    def add_spec(self,
                 adu,
                 spec_id=None,
                 adu_err=None,
                 adu_sky=None,
                 stype='standard'):
        '''
        Add spectrum (adu, adu_err & adu_sky) one at a time.
        '''

        if stype == 'standard':

            if len(self.spectrum_list_standard.keys()) == 0:
                self.spectrum_list_standard[0] = spectrum1D()

            if spec_id in list(self.spectrum_list_standard.keys()):
                if self.spectrum_list_standard[spec_id].adu is not None:
                    warnings.warn(
                        'The given spec_id is in use already, the given '
                        'trace, trace_sigma and pixel_list will overwrite the '
                        'current data.')

            if spec_id is None:

                # If spectrum_list is not empty
                if spectrum_list_standard:
                    spec_id = max(self.spectrum_list_standard.keys()) + 1
                else:
                    spec_id = 0

            self.spectrum_list_standard[spec_id].add_adu(adu, adu_err, adu_sky)

        elif stype == 'science':

            if spec_id in list(self.spectrum_list_science.keys()):
                if self.spectrum_list_science[spec_id].adu is not None:
                    warnings.warn(
                        'The given spec_id is in use already, the given '
                        'trace, trace_sigma and pixel_list will overwrite the '
                        'current data.')

            if spec_id is None:

                # If spectrum_list is not empty
                if spectrum_list_science:
                    spec_id = max(self.spectrum_list_science.keys()) + 1
                else:
                    spec_id = 0

            self.spectrum_list_science[spec_id].add_adu(adu, adu_err, adu_sky)

        else:
            if stype not in ['science', 'standard']:
                raise ValueError('Unknown stype, please choose from '
                                 '(1) science; and/or (2) standard.')

    def add_wavelength(self,
                       wave=None,
                       wave_resampled=None,
                       spec_id=None,
                       stype='standard'):
        '''
        Add wavelength, one at a time.
        '''

        if (wave is None) & (wave_resampled is None):
            raise ValueError('wave and wave_resampled cannot be None at '
                             'at the same time.')

        elif wave_resampled is None:
            wave_resampled = wave

        elif wave is None:
            wave = wave_resampled

        if stype == 'standard':

            if len(self.spectrum_list_standard.keys()) == 0:
                self.spectrum_list_standard[0] = spectrum1D(0)

            spec = self.spectrum_list_standard[0]

            spec.add_wavelength(wave)
            spec.add_wavelength_resampled(
                wave_bin=np.nanmedian(np.array(np.ediff1d(wave_resampled))),
                wave_start=wave_resampled[0],
                wave_end=wave_resampled[-1],
                wave_resampled=wave_resampled,
            )

        elif stype == 'science':

            if spec_id is not None:
                if spec_id not in list(self.spectrum_list_science.keys()):
                    warnings.warn(
                        'The spec_id provided is not in the '
                        'spectrum_list_science, new spectrum1D with the ID '
                        'is created.')
                    self.spectrum_list_science[spec_id] = spectrum1D(spec_id)
            else:
                if len(self.spectrum_list_science) == 0:
                    self.spectrum_list_science[0] = spectrum1D(0)
                spec_id = list(self.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            for i in spec_id:

                spec = self.spectrum_list_science[i]

                spec.add_wavelength(wave)
                spec.add_wavelength_resampled(
                    wave_bin=np.nanmedian(np.array(np.ediff1d(wave))),
                    wave_start=wave_resampled[0],
                    wave_end=wave_resampled[-1],
                    wave_resampled=wave_resampled,
                )

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
            # Loop through the twodspec.spectrum_list to update the
            # spectrum_list_standard
            if not self.spectrum_list_standard:
                self.spectrum_list_standard[0] = spectrum1D(spec_id=0)
            for key, value in twodspec.spectrum_list[0].__dict__.items():
                setattr(self.spectrum_list_standard[0], key, value)

        if stype == 'science':
            # Loop through the spec_id in twodspec
            for i in twodspec.spectrum_list.keys():
                # Loop through the twodspec.spectrum_list to update the
                # spectrum_list_standard
                if i not in self.spectrum_list_science:
                    self.spectrum_list_science[i] = spectrum1D(spec_id=i)
                for key, value in twodspec.spectrum_list[i].__dict__.items():
                    setattr(self.spectrum_list_science[i], key, value)

    def add_wavecal(self, wavecal, stype='standard'):

        stype_split = stype.split('+')

        if 'standard' in stype_split:
            # Loop through the spec_id in wavecal
            for i in wavecal.spectrum_list.keys():
                # Loop through the wavecal.spectrum_list to update the
                # spectrum_list_standard
                for key, value in wavecal.spectrum_list[i].__dict__.items():
                    if not self.spectrum_list_standard[0]:
                        self.spectrum_list_standard[0] = spectrum1D(spec_id=0)
                    setattr(self.spectrum_list_standard[0], key, value)

        if 'science' in stype_split:
            # Loop through the spec_id in wavecal
            for i in wavecal.spectrum_list.keys():
                # Loop through the wavecal.spectrum_list to update the
                # spectrum_list_standard
                for key, value in wavecal.spectrum_list[i].__dict__.items():
                    if not self.spectrum_list_science[i]:
                        self.spectrum_list_science[i] = spectrum1D(spec_id=i)
                    setattr(self.spectrum_list_science[i], key, value)

    def compute_sensitivity(self,
                            kind=3,
                            smooth=False,
                            slength=5,
                            sorder=3,
                            mask_range=[[6850, 6960], [7150,
                                                       7400], [7575, 7700],
                                        [8925, 9050], [9265, 9750]]):
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

        Returns
        -------
        JSON strings if savejsonstring is set to True.
        '''

        # resampling both the observed and the database standard spectra
        # in unit of flux per second. The higher resolution spectrum is
        # resampled to match the lower resolution one.
        spec = self.spectrum_list_standard[0]

        if np.nanmedian(np.array(np.ediff1d(spec.wave))) < np.nanmedian(
                np.array(np.ediff1d(self.wave_standard_true))):
            flux_standard = spectres(self.wave_standard_true,
                                     np.array(spec.wave),
                                     np.array(spec.adu),
                                     verbose=False)
            flux_standard_true = self.fluxmag_standard_true
            wave_standard_true = self.wave_standard_true
        else:
            flux_standard = spec.adu
            flux_standard_true = spectres(spec.wave,
                                          self.wave_standard_true,
                                          self.fluxmag_standard_true,
                                          verbose=False)
            wave_standard_true = spec.wave

        # Get the sensitivity curve
        sensitivity = flux_standard_true / flux_standard

        if mask_range is None:
            mask = np.isfinite(sensitivity)
        else:
            mask = np.isfinite(sensitivity)
            for m in mask_range:
                mask = mask & ((wave_standard_true < m[0]) |
                               (wave_standard_true > m[1]))

        sensitivity = sensitivity[mask]
        wave_literature = wave_standard_true[mask]
        flux_literature = flux_standard_true[mask]

        # apply a Savitzky-Golay filter to remove noise and Telluric lines
        if smooth:
            sensitivity = signal.savgol_filter(sensitivity, slength, sorder)

        # Set the smoothing parameters
        if smooth:
            spec.add_smoothing(smooth, slength, sorder)

        sensitivity_itp = itp.interp1d(wave_literature,
                                       np.log10(sensitivity),
                                       kind=kind,
                                       fill_value='extrapolate')

        spec.add_sensitivity(sensitivity)
        spec.add_literature_standard(wave_literature, flux_literature)

        # Add to each spectrum1D object
        self.add_sensitivity_itp(sensitivity_itp, stype='science+standard')

    def add_sensitivity_itp(self, sensitivity_itp, stype='standard'):

        stype_split = stype.split('+')

        if 'standard' in stype_split:
            # Add to both science and standard spectrum_list
            self.spectrum_list_standard[0].add_sensitivity_itp(
                sensitivity_itp=sensitivity_itp)

        if 'science' in stype_split:
            spec_id = list(self.spectrum_list_science.keys())

            for i in spec_id:
                # apply the flux calibration
                self.spectrum_list_science[i].add_sensitivity_itp(
                    sensitivity_itp=sensitivity_itp)

    def inspect_sensitivity(self,
                            renderer='default',
                            width=1280,
                            height=720,
                            savejsonstring=False,
                            display=True,
                            saveiframe=False,
                            open_iframe=False):
        '''
        Display the computed sensitivity curve.

        Parameters
        ----------
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        savejsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        saveiframe: boolean
            Save as an saveiframe, can work concurrently with other renderer
            apart from exporting savejsonstring.
        open_iframe: boolean
            Open the saveiframe in the default browser if set to True.

        Returns
        -------
        JSON strings if savejsonstring is set to True.
        '''
        spec = self.spectrum_list_standard[0]
        fig = go.Figure(
            layout=dict(autosize=False,
                        height=height,
                        width=width,
                        updatemenus=list([
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
            go.Scatter(x=spec.wave_literature,
                       y=spec.flux_literature,
                       line=dict(color='royalblue', width=4),
                       name='ADU / s (Observed)'))

        fig.add_trace(
            go.Scatter(x=spec.wave_literature,
                       y=spec.sensitivity,
                       yaxis='y2',
                       line=dict(color='firebrick', width=4),
                       name='Sensitivity Curve'))

        fig.add_trace(
            go.Scatter(x=spec.wave_literature,
                       y=10.**spec.sensitivity_itp(spec.wave_literature),
                       yaxis='y2',
                       line=dict(color='black', width=2),
                       name='Best-fit Sensitivity Curve'))

        if spec.smooth:
            fig.update_layout(title='SG(' + str(spec.slength) + ', ' +
                              str(spec.sorder) + ')-Smoothed ' + self.library +
                              ': ' + self.target,
                              yaxis_title='Smoothed ADU / s')
        else:
            fig.update_layout(title=self.library + ': ' + self.target,
                              yaxis_title='ADU / s')

        fig.update_layout(hovermode='closest',
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
                                                  bgcolor='rgba(0,0,0,0)'))
        if saveiframe:
            pio.write_html(fig, 'senscurve.html', auto_open=open_iframe)

        if display:
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

        if savejsonstring:
            return fig.to_json()

    def apply_flux_calibration(self, spec_id=None, stype='science+standard'):
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

            if spec_id is not None:
                if spec_id not in list(self.spectrum_list_science.keys()):
                    raise ValueError('The given spec_id does not exist.')
            else:
                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            for i in spec_id:

                spec = self.spectrum_list_science[i]

                # apply the flux calibration
                sensitivity = 10.**spec.sensitivity_itp(spec.wave)

                flux = sensitivity * spec.adu

                if spec.adu_err is not None:
                    flux_err = sensitivity * spec.adu_err
                if spec.adu_sky is not None:
                    flux_sky = sensitivity * spec.adu_sky

                flux_resampled = spectres(spec.wave_resampled,
                                          spec.wave,
                                          np.array(flux),
                                          verbose=False)

                if spec.adu_err is None:
                    flux_err_resampled = np.zeros_like(flux_resampled)
                else:
                    flux_err_resampled = spectres(spec.wave_resampled,
                                                  spec.wave,
                                                  np.array(flux_err),
                                                  verbose=False)
                if spec.adu_sky is None:
                    flux_sky_resampled = np.zeros_like(flux_resampled)
                else:
                    flux_sky_resampled = spectres(spec.wave_resampled,
                                                  spec.wave,
                                                  np.array(flux_sky),
                                                  verbose=False)

                # Only computed for diagnostic
                sensitivity_resampled = spectres(spec.wave_resampled,
                                                 spec.wave,
                                                 np.array(sensitivity),
                                                 verbose=False)

                spec.add_flux(flux, flux_err, flux_sky)
                spec.add_flux_resampled(flux_resampled, flux_err_resampled,
                                        flux_sky_resampled)
                spec.add_sensitivity(sensitivity)
                spec.add_sensitivity_resampled(sensitivity_resampled)

            self.flux_science_calibrated = True

        if 'standard' in stype_split:

            spec = self.spectrum_list_standard[0]

            # apply the flux calibration
            sensitivity = 10.**spec.sensitivity_itp(spec.wave)

            flux = sensitivity * spec.adu
            if spec.adu_err is not None:
                flux_err = sensitivity * spec.adu_err
            if spec.adu_sky is not None:
                flux_sky = sensitivity * spec.adu_sky

            flux_resampled = spectres(spec.wave_resampled,
                                      spec.wave,
                                      np.array(flux),
                                      verbose=False)
            if spec.adu_err is None:
                flux_err_resampled = np.zeros_like(flux_resampled)
            else:
                flux_err_resampled = spectres(spec.wave_resampled,
                                              spec.wave,
                                              np.array(flux_err),
                                              verbose=False)
            if spec.adu_sky is None:
                flux_sky_resampled = np.zeros_like(flux_resampled)
            else:
                flux_sky_resampled = spectres(spec.wave_resampled,
                                              spec.wave,
                                              np.array(flux_sky),
                                              verbose=False)

            # Only computed for diagnostic
            sensitivity_resampled = spectres(spec.wave_resampled,
                                             spec.wave,
                                             np.array(sensitivity),
                                             verbose=False)

            spec.add_flux(flux, flux_err, flux_sky)
            spec.add_flux_resampled(flux_resampled, flux_err_resampled,
                                    flux_sky_resampled)
            spec.add_sensitivity(sensitivity)
            spec.add_sensitivity_resampled(sensitivity_resampled)

            self.flux_standard_calibrated = True

        if ('science' not in stype_split) and ('standard' not in stype_split):
            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

    def inspect_reduced_spectrum(self,
                                 spec_id=None,
                                 stype='science+standard',
                                 wave_min=4000.,
                                 wave_max=8000.,
                                 renderer='default',
                                 width=1280,
                                 height=720,
                                 filename=None,
                                 savepng=False,
                                 savejpg=False,
                                 savesvg=False,
                                 savepdf=False,
                                 savejsonstring=False,
                                 display=True,
                                 saveiframe=False,
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
        savejsonstring: boolean
            Set to True to return json string that can be rendered by Plotly
            in any support language.
        saveiframe: boolean
            Save as an saveiframe, can work concurrently with other renderer
            apart from exporting savejsonstring.
        open_iframe: boolean
            Open the saveiframe in the default browser if set to True.

        Returns
        -------
        JSON strings if savejsonstring is set to True.
        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if spec_id is not None:
                if spec_id not in list(self.spectrum_list_science.keys()):
                    raise ValueError('The given spec_id does not exist.')
            else:
                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            for i in spec_id:

                spec = self.spectrum_list_science[i]

                if self.flux_science_calibrated:
                    wave_mask = ((spec.wave_resampled > wave_min) &
                                 (spec.wave_resampled < wave_max))

                    flux_low = np.nanpercentile(
                        np.array(spec.flux_resampled)[wave_mask], 5) / 1.5
                    flux_high = np.nanpercentile(
                        np.array(spec.flux_resampled)[wave_mask], 95) * 1.5
                    flux_mask = ((spec.flux_resampled > flux_low) &
                                 (spec.flux_resampled < flux_high))
                    flux_min = np.log10(
                        np.nanmin(spec.flux_resampled[flux_mask]))
                    flux_max = np.log10(
                        np.nanmax(spec.flux_resampled[flux_mask]))
                else:
                    warnings.warn('Flux calibration is not available.')
                    wave_mask = ((spec.wave_resampled > wave_min) &
                                 (spec.wave_resampled < wave_max))
                    flux_mask = ((spec.adu_resampled > np.nanpercentile(
                        spec.adu_resampled[wave_mask], 5) / 1.5) &
                                 (spec.adu_resampled < np.nanpercentile(
                                     spec.adu_resampled[wave_mask], 95) * 1.5))
                    flux_min = np.log10(
                        np.nanmin(spec.adu_resampled[flux_mask]))
                    flux_max = np.log10(
                        np.nanmax(spec.adu_resampled[flux_mask]))

                fig_sci = go.Figure(
                    layout=dict(autosize=False,
                                height=height,
                                width=width,
                                updatemenus=list([
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
                    fig_sci.add_trace(
                        go.Scatter(x=spec.wave_resampled,
                                   y=spec.flux_resampled,
                                   line=dict(color='royalblue'),
                                   name='Flux'))
                    if spec.flux_err is not None:
                        fig_sci.add_trace(
                            go.Scatter(x=spec.wave_resampled,
                                       y=spec.flux_err_resampled,
                                       line=dict(color='firebrick'),
                                       name='Flux Uncertainty'))
                    if spec.flux_sky is not None:
                        fig_sci.add_trace(
                            go.Scatter(x=spec.wave_resampled,
                                       y=spec.flux_sky_resampled,
                                       line=dict(color='orange'),
                                       name='Sky Flux'))
                else:
                    fig_sci.add_trace(
                        go.Scatter(x=spec.wave_resampled,
                                   y=spec.adu_resampled,
                                   line=dict(color='royalblue'),
                                   name='ADU / s'))
                    if spec.adu_err is not None:
                        fig_sci.add_trace(
                            go.Scatter(x=spec.wave_resampled,
                                       y=spec.adu_err_resampled,
                                       line=dict(color='firebrick'),
                                       name='ADU Uncertainty / s'))
                    if spec.adu_sky is not None:
                        fig_sci.add_trace(
                            go.Scatter(x=spec.wave_resampled,
                                       y=spec.adu_sky_resampled,
                                       line=dict(color='orange'),
                                       name='Sky ADU / s'))

                fig_sci.update_layout(hovermode='closest',
                                      showlegend=True,
                                      xaxis=dict(title='Wavelength / A',
                                                 range=[wave_min, wave_max]),
                                      yaxis=dict(title='Flux',
                                                 range=[flux_min, flux_max],
                                                 type='log'),
                                      legend=go.layout.Legend(
                                          x=0,
                                          y=1,
                                          traceorder="normal",
                                          font=dict(family="sans-serif",
                                                    size=12,
                                                    color="black"),
                                          bgcolor='rgba(0,0,0,0)'))

                if filename is None:
                    filename = "spectrum_" + str(i)
                else:
                    filename = os.path.splitext(filename)[0] + "_" + str(i)

                if saveiframe:
                    pio.write_html(fig_sci,
                                   filename + '.html',
                                   auto_open=open_iframe)

                if display:
                    if renderer == 'default':
                        fig_sci.show()
                    else:
                        fig_sci.show(renderer)

                if savejpg:
                    fig_sci.write_image(filename + '.jpg', format='jpg')

                if savepng:
                    fig_sci.write_image(filename + '.png', format='png')

                if savesvg:
                    fig_sci.write_image(filename + '.svg', format='svg')

                if savepdf:
                    fig_sci.write_image(filename + '.pdf', format='pdf')

                if savejsonstring:
                    return fig_sci[j].to_json()

        if 'standard' in stype_split:

            spec = self.spectrum_list_standard[0]

            if self.flux_standard_calibrated:
                wave_standard_mask = ((spec.wave_resampled > wave_min) &
                                      (spec.wave_resampled < wave_max))
                flux_standard_mask = (
                    (spec.flux_resampled > np.nanpercentile(
                        spec.flux_resampled[wave_standard_mask], 5) / 1.5) &
                    (spec.flux_resampled < np.nanpercentile(
                        spec.flux_resampled[wave_standard_mask], 95) * 1.5))
                flux_standard_min = np.log10(
                    np.nanmin(spec.flux_resampled[flux_standard_mask]))
                flux_standard_max = np.log10(
                    np.nanmax(spec.flux_resampled[flux_standard_mask]))
            else:
                warnings.warn('Flux calibration is not available.')
                wave_standard_mask = ((spec.wave_resampled > wave_min) &
                                      (spec.wave_resampled < wave_max))
                flux_standard_mask = (
                    (spec.adu_resampled > np.nanpercentile(
                        spec.adu_resampled[wave_standard_mask], 5) / 1.5) &
                    (spec.adu_standard < np.nanpercentile(
                        spec.adu_resampled[wave_standard_mask], 95) * 1.5))
                flux_standard_min = np.log10(
                    np.nanmin(spec.adu_resampled[flux_standard_mask]))
                flux_standard_max = np.log10(
                    np.nanmax(spec.adu_resampled[flux_standard_mask]))

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
                                                 autosize=False,
                                                 height=height,
                                                 width=width,
                                                 title='Log scale'))
            # show the image on the top
            if self.flux_standard_calibrated:
                fig_standard.add_trace(
                    go.Scatter(x=spec.wave_resampled,
                               y=spec.flux_resampled,
                               line=dict(color='royalblue'),
                               name='Flux'))
                if spec.flux_err_resampled is not None:
                    fig_standard.add_trace(
                        go.Scatter(x=spec.wave_resampled,
                                   y=spec.flux_err_resampled,
                                   line=dict(color='firebrick'),
                                   name='Flux Uncertainty'))
                if spec.flux_sky_resampled is not None:
                    fig_standard.add_trace(
                        go.Scatter(x=spec.wave_resampled,
                                   y=spec.flux_sky_resampled,
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
                    go.Scatter(x=spec.wave_resampled,
                               y=spec.adu_resampled,
                               line=dict(color='royalblue'),
                               name='ADU / s'))
                if spec.adu_err_resampled is not None:
                    fig_standard.add_trace(
                        go.Scatter(x=spec.wave_resampled,
                                   y=spec.adu_err_resampled,
                                   line=dict(color='firebrick'),
                                   name='ADU Uncertainty / s'))
                if spec.adu_sky_resampled is not None:
                    fig_standard.add_trace(
                        go.Scatter(x=spec.wave_resampled,
                                   y=spec.adu_sky_resampled,
                                   line=dict(color='orange'),
                                   name='Sky ADU / s'))

            fig_standard.update_layout(
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
                                        bgcolor='rgba(0,0,0,0)'))

            if filename is None:
                filename = "spectrum_standard"
            else:
                filename = os.path.splitext(filename)[0]

            if saveiframe:
                pio.write_html(fig_standard,
                               filename + '.html',
                               auto_open=open_iframe)

            if display:
                if renderer == 'default':
                    fig_standard.show(height=height, width=width)
                else:
                    fig_standard.show(renderer, height=height, width=width)

            if savejpg:
                fig_standard.write_image(filename + '.jpg', format='jpg')

            if savepng:
                fig_standard.write_image(filename + '.png', format='png')

            if savesvg:
                fig_standard.write_image(filename + '.svg', format='svg')

            if savepdf:
                fig_standard.write_image(filename + '.pdf', format='pdf')

            if savejsonstring:
                return fig_standard.to_json()

        if ('science' not in stype_split) and ('standard' not in stype_split):
            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

    def save_sensitivity_itp(self, filename='sensitivity_itp.npy'):
        np.save(filename, self.spectrum_list_standard[0].sensitivity_itp)

    def _create_adu_fits(self, spec_id=None, stype='science+standard'):

        stype_split = stype.split('+')

        # Put the reduced data in FITS format with an image header
        if 'science' in stype_split:

            if spec_id is not None:
                if spec_id not in list(self.spectrum_list_science.keys()):
                    raise ValueError('The given spec_id does not exist.')
            else:
                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            self.adu_hdulist = np.array([None] *
                                        len(self.spectrum_list_science),
                                        dtype='object')
            for i, spec in self.spectrum_list_science.items():
                self.adu_hdulist[i] = fits.HDUList([
                    fits.ImageHDU(spec.adu),
                    fits.ImageHDU(spec.adu_err),
                    fits.ImageHDU(spec.adu_sky)
                ])

        if 'standard' in stype_split:
            spec = self.spectrum_list_standard[0]
            self.adu_standard_hdulist = fits.HDUList([
                fits.ImageHDU(spec.adu),
                fits.ImageHDU(spec.adu_err),
                fits.ImageHDU(spec.adu_sky)
            ])

    def _create_adu_resampled_fits(self,
                                   spec_id=None,
                                   stype='science+standard'):

        stype_split = stype.split('+')

        # Put the reduced data in FITS format with an image header
        if 'science' in stype_split:

            if spec_id is not None:
                if spec_id not in list(self.spectrum_list_science.keys()):
                    raise ValueError('The given spec_id does not exist.')
            else:
                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            self.adu_resampled_hdulist = np.array(
                [None] * len(self.spectrum_list_science), dtype='object')
            for i, spec in self.spectrum_list_science.items():
                self.adu_resampled_hdulist[i] = fits.HDUList([
                    fits.ImageHDU(spec.adu_resampled),
                    fits.ImageHDU(spec.adu_err_resampled),
                    fits.ImageHDU(spec.adu_sky_resampled)
                ])

        if 'standard' in stype_split:
            spec = self.spectrum_list_standard[0]
            self.adu_standard_resampled_hdulist = fits.HDUList([
                fits.ImageHDU(spec.adu_resampled),
                fits.ImageHDU(spec.adu_err_resampled),
                fits.ImageHDU(spec.adu_sky_resampled)
            ])

    def _create_flux_fits(self, spec_id=None, stype='science+standard'):
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

            if spec_id is not None:
                if spec_id not in list(self.spectrum_list_science.keys()):
                    raise ValueError('The given spec_id does not exist.')
            else:
                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            self.flux_science_hdulist = np.array(
                [None] * len(self.spectrum_list_science), dtype='object')
            for i, spec in self.spectrum_list_science.items():
                # Note that wave_start is the centre of the starting bin
                flux_fits = fits.ImageHDU(spec.flux)
                flux_err_fits = fits.ImageHDU(spec.flux_err)
                flux_sky_fits = fits.ImageHDU(spec.flux_sky)
                sensitivity_fits = fits.ImageHDU(spec.sensitivity)

                self.flux_science_hdulist[i] = fits.HDUList([
                    flux_fits, flux_err_fits, flux_sky_fits, sensitivity_fits
                ])

        if 'standard' in stype_split:
            spec = self.spectrum_list_standard[0]

            # Note that wave_start is the centre of the starting bin
            flux_fits = fits.ImageHDU(spec.flux)
            flux_err_fits = fits.ImageHDU(spec.flux_err)
            flux_sky_fits = fits.ImageHDU(spec.flux_sky)
            sensitivity_fits = fits.ImageHDU(spec.sensitivity)

            self.flux_standard_hdulist = fits.HDUList(
                [flux_fits, flux_err_fits, flux_sky_fits, sensitivity_fits])

        if ('science' not in stype_split) and ('standard' not in stype_split):
            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

    def _create_flux_resampled_fits(self,
                                    spec_id=None,
                                    stype='science+standard'):
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

            if spec_id is not None:
                if spec_id not in list(self.spectrum_list_science.keys()):
                    raise ValueError('The given spec_id does not exist.')
            else:
                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            self.flux_science_resampled_hdulist = np.array(
                [None] * len(self.spectrum_list_science), dtype='object')

            for i, spec in self.spectrum_list_science.items():

                self.flux_science_resampled_hdulist[i] = fits.HDUList()

                # Note that wave_start is the centre of the starting bin
                flux_fits = fits.ImageHDU(spec.flux_resampled)
                flux_fits.header['LABEL'] = 'Flux'
                flux_fits.header['CRPIX1'] = 1.00E+00
                flux_fits.header['CDELT1'] = spec.wave_bin
                flux_fits.header['CRVAL1'] = spec.wave_start
                flux_fits.header['CTYPE1'] = 'Wavelength'
                flux_fits.header['CUNIT1'] = 'Angstroms'
                flux_fits.header['BUNIT'] = 'erg/(s*cm**2*Angstrom)'
                self.flux_science_resampled_hdulist[i].append(flux_fits)

                flux_err_fits = fits.ImageHDU(spec.flux_err_resampled)
                flux_err_fits.header['LABEL'] = 'Flux'
                flux_err_fits.header['CRPIX1'] = 1.00E+00
                flux_err_fits.header['CDELT1'] = spec.wave_bin
                flux_err_fits.header['CRVAL1'] = spec.wave_start
                flux_err_fits.header['CTYPE1'] = 'Wavelength'
                flux_err_fits.header['CUNIT1'] = 'Angstroms'
                flux_err_fits.header['BUNIT'] = 'erg/(s*cm**2*Angstrom)'
                self.flux_science_resampled_hdulist[i].append(flux_err_fits)

                flux_sky_fits = fits.ImageHDU(spec.flux_sky_resampled)
                flux_sky_fits.header['LABEL'] = 'Flux'
                flux_sky_fits.header['CRPIX1'] = 1.00E+00
                flux_sky_fits.header['CDELT1'] = spec.wave_bin
                flux_sky_fits.header['CRVAL1'] = spec.wave_start
                flux_sky_fits.header['CTYPE1'] = 'Wavelength'
                flux_sky_fits.header['CUNIT1'] = 'Angstroms'
                flux_sky_fits.header['BUNIT'] = 'erg/(s*cm**2*Angstrom)'
                self.flux_science_resampled_hdulist[i].append(flux_sky_fits)

                sensitivity_resampled_fits = fits.ImageHDU(
                    spec.sensitivity_resampled)
                sensitivity_resampled_fits.header['LABEL'] = 'Sensitivity'
                sensitivity_resampled_fits.header['CRPIX1'] = 1.00E+00
                sensitivity_resampled_fits.header['CDELT1'] = spec.wave_bin
                sensitivity_resampled_fits.header['CRVAL1'] = spec.wave_start
                sensitivity_resampled_fits.header['CTYPE1'] = 'Wavelength'
                sensitivity_resampled_fits.header['CUNIT1'] = 'Angstroms'
                sensitivity_resampled_fits.header[
                    'BUNIT'] = 'erg/(s*cm**2*Angstrom)/ADU'
                self.flux_science_resampled_hdulist[i].append(
                    sensitivity_resampled_fits)

        if 'standard' in stype_split:

            spec = self.spectrum_list_standard[0]

            self.flux_standard_resampled_hdulist = fits.HDUList()

            # Note that wave_start is the centre of the starting bin
            flux_fits = fits.ImageHDU(spec.flux_resampled)
            flux_fits.header['LABEL'] = 'Flux'
            flux_fits.header['CRPIX1'] = 1.00E+00
            flux_fits.header['CDELT1'] = spec.wave_bin
            flux_fits.header['CRVAL1'] = spec.wave_start
            flux_fits.header['CTYPE1'] = 'Wavelength'
            flux_fits.header['CUNIT1'] = 'Angstroms'
            flux_fits.header['BUNIT'] = 'erg/(s*cm**2*Angstrom)'
            self.flux_standard_resampled_hdulist.append(flux_fits)

            flux_err_fits = fits.ImageHDU(spec.flux_err_resampled)
            flux_err_fits.header['LABEL'] = 'Flux'
            flux_err_fits.header['CRPIX1'] = 1.00E+00
            flux_err_fits.header['CDELT1'] = spec.wave_bin
            flux_err_fits.header['CRVAL1'] = spec.wave_start
            flux_err_fits.header['CTYPE1'] = 'Wavelength'
            flux_err_fits.header['CUNIT1'] = 'Angstroms'
            flux_err_fits.header['BUNIT'] = 'erg/(s*cm**2*Angstrom)'
            self.flux_standard_resampled_hdulist.append(flux_err_fits)

            flux_sky_fits = fits.ImageHDU(spec.flux_sky_resampled)
            flux_sky_fits.header['LABEL'] = 'Flux'
            flux_sky_fits.header['CRPIX1'] = 1.00E+00
            flux_sky_fits.header['CDELT1'] = spec.wave_bin
            flux_sky_fits.header['CRVAL1'] = spec.wave_start
            flux_sky_fits.header['CTYPE1'] = 'Wavelength'
            flux_sky_fits.header['CUNIT1'] = 'Angstroms'
            flux_sky_fits.header['BUNIT'] = 'erg/(s*cm**2*Angstrom)'
            self.flux_standard_resampled_hdulist.append(flux_sky_fits)

            sensitivity_resampled_fits = fits.ImageHDU(
                spec.sensitivity_resampled)
            sensitivity_resampled_fits.header['LABEL'] = 'Sensitivity'
            sensitivity_resampled_fits.header['CRPIX1'] = 1.00E+00
            sensitivity_resampled_fits.header['CDELT1'] = spec.wave_bin
            sensitivity_resampled_fits.header['CRVAL1'] = spec.wave_start
            sensitivity_resampled_fits.header['CTYPE1'] = 'Wavelength'
            sensitivity_resampled_fits.header['CUNIT1'] = 'Angstroms'
            sensitivity_resampled_fits.header[
                'BUNIT'] = 'erg/(s*cm**2*Angstrom)/ADU'
            self.flux_standard_resampled_hdulist.append(
                sensitivity_resampled_fits)

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

    def add_wavelength(self,
                       wave,
                       wave_resampled=None,
                       spec_id=None,
                       stype='science'):
        self.fluxcal.add_wavelength(wave=wave,
                                    wave_resampled=wave_resampled,
                                    spec_id=spec_id,
                                    stype=stype)

    def add_spec(self,
                 adu,
                 spec_id=None,
                 adu_err=None,
                 adu_sky=None,
                 stype='science'):

        self.fluxcal.add_spec(adu=adu,
                              spec_id=spec_id,
                              adu_err=adu_err,
                              adu_sky=adu_sky,
                              stype=stype)

        if stype == 'science':
            self.wavecal_science.add_spec(adu=adu,
                                          spec_id=spec_id,
                                          adu_err=adu_err,
                                          adu_sky=adu_sky)

        if stype == 'standard':
            self.wavecal_standard.add_spec(adu=adu,
                                           spec_id=spec_id,
                                           adu_err=adu_err,
                                           adu_sky=adu_sky)

    def add_twodspec(self, twodspec, pixel_list=None, stype='science'):
        '''
        Extract the required information from the TwoDSpec object of the
        standard.

        Parameters
        ----------
        standard: TwoDSpec object
            The TwoDSpec object with the extracted standard target
        '''

        self.fluxcal.add_twodspec(twodspec=twodspec,
                                  pixel_list=pixel_list,
                                  stype=stype)

        if stype == 'science':
            self.wavecal_science.add_twodspec(twodspec=twodspec,
                                              pixel_list=pixel_list)

        if stype == 'standard':
            self.wavecal_standard.add_twodspec(twodspec=twodspec,
                                               pixel_list=pixel_list)

    def apply_twodspec_mask(self, stype='science'):

        if stype == 'science':
            self.wavecal_science.apply_twodspec_mask()
        if stype == 'standard':
            self.wavecal_standard.apply_twodspec_mask()

    def apply_spec_mask(self, spec_mask, stype='science'):
        self.apply_spec_mask(spec_mask=spec_mask)

    def apply_spatial_mask(self, spatial_mask, stype='science'):
        self.apply_spatial_mask(spatial_mask=spatial_mask)

    def add_arc_lines(self, spec_id, peaks, stype='science'):

        if stype == 'science':
            self.wavecal_science.add_arc_lines(spec_id=spec_id, peaks=peaks)
            self.arc_lines_science_imported = True
        if stype == 'standard':
            self.wavecal_standard.add_arc_lines(spec_id=spec_id, peaks=peaks)
            self.arc_lines_standard_imported = True

    def add_arc_spec(self, spec_id, arc_spec, stype='science'):

        if stype == 'science':
            self.wavecal_science.add_arc_spec(spec_id=spec_id,
                                              arc_spec=arc_spec)
            self.arc_science_imported = True
        if stype == 'standard':
            self.wavecal_standard.add_arc_spec(spec_id=spec_id,
                                               arc_spec=arc_spec)
            self.arc_standard_imported = True

    def add_arc(self, arc, stype='science'):

        if stype == 'science':
            self.wavecal_science.add_arc(arc)
            self.arc_science_imported = True
        if stype == 'standard':
            self.wavecal_standard.add_arc(arc)
            self.arc_standard_imported = True

    def add_trace(self,
                  trace,
                  trace_sigma,
                  spec_id=None,
                  pixel_list=None,
                  stype='science'):

        if stype == 'science':
            self.wavecal_science.add_trace(trace=trace,
                                           trace_sigma=trace_sigma,
                                           spec_id=spec_id,
                                           pixel_list=pixel_list)

        if stype == 'standard':
            self.wavecal_standard.add_trace(trace=trace,
                                            trace_sigma=trace_sigma,
                                            spec_id=spec_id,
                                            pixel_list=pixel_list)

    def add_polyfit(self,
                    spec_id,
                    polyfit_coeff,
                    polyfit_type=['poly'],
                    stype='science'):

        if stype == 'science':
            self.wavecal_science.add_polyfit(spec_id=spec_id,
                                             polyfit_coeff=polyfit_coeff,
                                             polyfit_type=polyfit_type)
        if stype == 'standard':
            self.wavecal_standard.add_polyfit(spec_id=spec_id,
                                              polyfit_coeff=polyfit_coeff,
                                              polyfit_type=polyfit_type)

    def extract_arc_spec(self,
                         spec_id=None,
                         display=False,
                         savejsonstring=False,
                         renderer='default',
                         width=1280,
                         height=720,
                         saveiframe=False,
                         open_iframe=False,
                         stype='science'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.extract_arc_spec(
                spec_id=spec_id,
                display=display,
                savejsonstring=savejsonstring,
                renderer=renderer,
                width=width,
                height=height,
                saveiframe=saveiframe,
                open_iframe=open_iframe)
        if 'standard' in stype_split:
            self.wavecal_standard.extract_arc_spec(
                spec_id=spec_id,
                display=display,
                savejsonstring=savejsonstring,
                renderer=renderer,
                width=width,
                height=height,
                saveiframe=saveiframe,
                open_iframe=open_iframe)

    def find_arc_lines(self,
                       spec_id=None,
                       background=None,
                       percentile=25.,
                       prominence=10.,
                       distance=5.,
                       refine=True,
                       refine_window_width=5,
                       display=False,
                       width=1280,
                       height=720,
                       savejsonstring=False,
                       renderer='default',
                       saveiframe=False,
                       open_iframe=False,
                       stype='science'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.find_arc_lines(spec_id=spec_id,
                                                background=background,
                                                percentile=percentile,
                                                prominence=prominence,
                                                distance=distance,
                                                display=display,
                                                width=width,
                                                height=height,
                                                savejsonstring=savejsonstring,
                                                renderer=renderer,
                                                saveiframe=saveiframe,
                                                open_iframe=open_iframe)
        if 'standard' in stype_split:
            self.wavecal_standard.find_arc_lines(spec_id=spec_id,
                                                 background=background,
                                                 percentile=percentile,
                                                 prominence=prominence,
                                                 distance=distance,
                                                 display=display,
                                                 width=width,
                                                 height=height,
                                                 savejsonstring=savejsonstring,
                                                 renderer=renderer,
                                                 saveiframe=saveiframe,
                                                 open_iframe=open_iframe)

    def initialise_calibrator(self,
                              spec_id=None,
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
                spec_id=spec_id,
                peaks=peaks,
                num_pix=num_pix,
                pixel_list=pixel_list,
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                plotting_library=plotting_library,
                log_level=log_level)
        if 'standard' in stype_split:
            self.wavecal_standard.initialise_calibrator(
                spec_id=spec_id,
                peaks=peaks,
                num_pix=num_pix,
                pixel_list=pixel_list,
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                plotting_library=plotting_library,
                log_level=log_level)

    def set_known_pairs(self,
                        spec_id=None,
                        pix=None,
                        wave=None,
                        idx=None,
                        stype='standard'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.set_known_pairs(spec_id=spec_id,
                                                 pix=pix,
                                                 wave=wave)
        if 'standard' in stype_split:
            self.wavecal_science.set_known_pairs(spec_id=spec_id,
                                                 pix=pix,
                                                 wave=wave)

    def set_fit_constraints(self,
                            spec_id=None,
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
                            polyfit_type='poly',
                            stype='science'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.set_fit_constraints(
                spec_id=spec_id,
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
                polyfit_type=polyfit_type)
        if 'standard' in stype_split:
            self.wavecal_standard.set_fit_constraints(
                spec_id=spec_id,
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
                polyfit_type=polyfit_type)

    def add_atlas(self,
                  elements,
                  spec_id=None,
                  min_atlas_wavelength=0,
                  max_atlas_wavelength=15000,
                  min_intensity=0,
                  min_distance=0,
                  vacuum=False,
                  pressure=101325.,
                  temperature=273.15,
                  relative_humidity=0.,
                  constrain_poly=False,
                  stype='science'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.add_atlas(
                elements=elements,
                spec_id=spec_id,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_atlas_intensity=min_intensity,
                min_atlas_distance=min_distance,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
                constrain_poly=constrain_poly)

        if 'standard' in stype_split:
            self.wavecal_standard.add_atlas(
                elements=elements,
                spec_id=None,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_atlas_intensity=min_intensity,
                min_atlas_distance=min_distance,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
                constrain_poly=constrain_poly)

    def load_user_atlas(self,
                        elements,
                        wavelengths,
                        spec_id=None,
                        intensities=None,
                        constrain_poly=False,
                        vacuum=False,
                        pressure=101325.,
                        temperature=273.15,
                        relative_humidity=0.,
                        stype='science'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.load_user_atlas(
                elements=elements,
                wavelengths=wavelengths,
                spec_id=spec_id,
                intensities=intensities,
                constrain_poly=constrain_poly,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity)

        if 'standard' in stype_split:
            self.wavecal_standard.load_user_atlas(
                elements=elements,
                wavelengths=wavelengths,
                spec_id=spec_id,
                intensities=intensities,
                constrain_poly=constrain_poly,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity)

    def fit(self,
            spec_id=None,
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
            stype='standard'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.fit(spec_id=spec_id,
                                     sample_size=sample_size,
                                     top_n=top_n,
                                     max_tries=max_tries,
                                     progress=progress,
                                     coeff=coeff,
                                     linear=linear,
                                     weighted=weighted,
                                     filter_close=filter_close,
                                     display=display,
                                     savefig=savefig,
                                     filename=filename)

        if 'standard' in stype_split:
            self.wavecal_standard.fit(spec_id=None,
                                      top_n=top_n,
                                      max_tries=max_tries,
                                      progress=progress,
                                      coeff=coeff,
                                      linear=linear,
                                      weighted=weighted,
                                      filter_close=filter_close,
                                      display=display,
                                      savefig=savefig,
                                      filename=filename)

    def refine_fit(self,
                   spec_id=None,
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
                   stype='standard'):

        stype_split = stype.split('+')

        if 'science' in stype_split:
            self.wavecal_science.refine_fit(spec_id=spec_id,
                                            polyfit_coeff=polyfit_coeff,
                                            n_delta=n_delta,
                                            refine=refine,
                                            tolerance=tolerance,
                                            method=method,
                                            convergence=convergence,
                                            robust_refit=robust_refit,
                                            polydeg=polydeg,
                                            display=display,
                                            savefig=savefig,
                                            filename=filename)
        if 'standard' in stype_split:
            self.wavecal_standard.refine_fit(spec_id=None,
                                             polyfit_coeff=polyfit_coeff,
                                             n_delta=n_delta,
                                             refine=refine,
                                             tolerance=tolerance,
                                             method=method,
                                             convergence=convergence,
                                             robust_refit=robust_refit,
                                             polydeg=polydeg,
                                             display=display,
                                             savefig=savefig,
                                             filename=filename)

    def apply_wavelength_calibration(self,
                                     spec_id=None,
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

    def load_standard(self, target, library=None, ftype='flux', cutoff=0.4):
        self.fluxcal.load_standard(target=target,
                                   library=library,
                                   ftype=ftype,
                                   cutoff=cutoff)

    def inspect_standard(self,
                         renderer='default',
                         savejsonstring=False,
                         display=True,
                         height=1280,
                         width=720,
                         saveiframe=False,
                         open_iframe=False):

        self.fluxcal.inspect_standard(renderer=renderer,
                                      savejsonstring=savejsonstring,
                                      display=display,
                                      height=height,
                                      width=width,
                                      saveiframe=saveiframe,
                                      open_iframe=open_iframe)

    def compute_sensitivity(self,
                            kind=3,
                            smooth=False,
                            slength=5,
                            sorder=3,
                            mask_range=[[6850, 6960], [7150,
                                                       7400], [7575, 7700],
                                        [8925, 9050], [9265, 9750]]):

        self.fluxcal.compute_sensitivity(kind=kind,
                                         smooth=smooth,
                                         slength=slength,
                                         sorder=sorder,
                                         mask_range=mask_range)

    def save_sensitivity_itp(self, filename='sensitivity_itp.npy'):

        self.fluxcal.save_sensitivity_itp(filename=filename)

    def add_sensitivity_itp(self, sensitivity_itp, stype='standard'):

        self.fluxcal.add_sensitivity_itp(sensitivity_itp=sensitivity_itp,
                                         stype=stype)

    def inspect_sensitivity(self,
                            renderer='default',
                            width=1280,
                            height=720,
                            savejsonstring=False,
                            display=True,
                            saveiframe=False,
                            open_iframe=False):
        self.fluxcal.inspect_sensitivity(renderer=renderer,
                                         width=width,
                                         height=height,
                                         savejsonstring=savejsonstring,
                                         display=display,
                                         saveiframe=saveiframe,
                                         open_iframe=open_iframe)

    def apply_flux_calibration(self, spec_id=None, stype='science+standard'):

        stype_split = stype.split('+')

        self.fluxcal.apply_flux_calibration(spec_id=spec_id, stype=stype)

    def inspect_reduced_spectrum(self,
                                 spec_id=None,
                                 stype='science+standard',
                                 wave_min=4000.,
                                 wave_max=8000.,
                                 renderer='default',
                                 width=1280,
                                 height=720,
                                 filename=None,
                                 savepng=False,
                                 savejpg=False,
                                 savesvg=False,
                                 savepdf=False,
                                 display=True,
                                 savejsonstring=False,
                                 saveiframe=False,
                                 open_iframe=False):

        self.fluxcal.inspect_reduced_spectrum(spec_id=spec_id,
                                              stype=stype,
                                              wave_min=wave_min,
                                              wave_max=wave_max,
                                              renderer=renderer,
                                              width=width,
                                              height=height,
                                              filename=filename,
                                              savepng=savepng,
                                              savejpg=savejpg,
                                              savesvg=savesvg,
                                              savepdf=savepdf,
                                              savejsonstring=savejsonstring,
                                              saveiframe=saveiframe,
                                              open_iframe=open_iframe)

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
                  spec_id=None,
                  output='flux+wavecal+fluxraw+adu',
                  filename='reduced',
                  stype='science',
                  to_disk=True,
                  to_memory=False,
                  overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
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
        stype: String
            Spectral type: science or standard
        to_disk: boolean
            Default is True. If True, the fits object will be saved to disk.
        to_memory: boolean
            Default is False. If True, the fits object will be returned.
        overwrite: boolean
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        output_split = output.split('+')
        stype_split = stype.split('+')

        for i in output_split:
            if i not in [
                    'flux_resampled', 'wavecal', 'flux', 'adu', 'adu_resampled'
            ]:
                raise ValueError('%s is not a valid output.' % i)

        if ('science' not in stype_split) and ('standard' not in stype_split):
            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

        if 'science' in stype_split:

            if to_memory:
                hdu_list_science = []

            if spec_id is not None:
                if spec_id not in list(
                        self.fluxcal.spectrum_list_science.keys()):
                    raise ValueError('The given spec_id does not exist.')
            else:
                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.fluxcal.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            for i in spec_id:

                # Prepare multiple extension HDU
                hdu_output = fits.HDUList()
                if 'flux_resampled' in output_split:
                    if self.fluxcal.spectrum_list_science[
                            i].flux_resampled is None:
                        warnings.warn(
                            "Spectrum is not flux calibrated and resampled.")
                    else:
                        self.fluxcal._create_flux_resampled_fits(
                            spec_id=i, stype='science')
                        hdu_output.extend(
                            self.fluxcal.flux_science_resampled_hdulist[i])

                if 'wavecal' in output_split:
                    if self.wavecal_science.spectrum_list[
                            i].polyfit_coeff is None:
                        warnings.warn("Spectrum is not wavelength calibrated.")
                    else:
                        self.wavecal_science._create_wavecal_fits(spec_id=i)
                        hdu_output.extend(
                            self.wavecal_science.wavecal_hdulist[i])

                if 'flux' in output_split:
                    if self.fluxcal.spectrum_list_science[i].flux is None:
                        warnings.warn("Spectrum is not flux calibrated.")
                    else:
                        self.fluxcal._create_flux_fits(spec_id=i,
                                                       stype='science')
                        hdu_output.extend(self.fluxcal.flux_science_hdulist[i])

                if 'adu' in output_split:
                    if self.fluxcal.spectrum_list_science[i].adu is None:
                        warnings.warn("ADU does not exist. Have you included "
                                      "a spectrum?")
                    else:
                        self.fluxcal._create_adu_fits(spec_id=i,
                                                      stype='science')
                        hdu_output.extend(self.fluxcal.adu_hdulist[i])

                if 'adu_resampled' in output_split:
                    if self.fluxcal.spectrum_list_science[
                            i].adu_resampled is None:
                        warnings.warn("Resampled ADU does not exist. Have you "
                                      "included a spectrum? Is it wavelength "
                                      "calibrated and resampled?")
                    else:
                        self.fluxcal._create_adu_resampled_fits(
                            spec_id=i, stype='science')
                        hdu_output.extend(
                            self.fluxcal.adu_resampled_hdulist[i])

                # Convert the first HDU to PrimaryHDU
                hdu_output[0] = fits.PrimaryHDU(hdu_output[0].data,
                                                hdu_output[0].header)
                hdu_output.update_extend()

                if to_disk:
                    # Save file to disk
                    hdu_output.writeto(filename + '_science_' + str(i) +
                                       '.fits',
                                       overwrite=overwrite)

                if to_memory:
                    hdu_list_science.append(hdu_output)

        if 'standard' in stype_split:
            # Prepare multiple extension HDU
            hdu_output = fits.HDUList()
            if 'flux_resampled' in output_split:
                if self.fluxcal.spectrum_list_standard[
                        0].flux_resampled is None:
                    warnings.warn(
                        "Spectrum is not flux calibrated and resampled.")
                else:
                    self.fluxcal._create_flux_resampled_fits(spec_id=spec_id,
                                                             stype='standard')
                    hdu_output.extend(
                        self.fluxcal.flux_standard_resampled_hdulist)

            if 'wavecal' in output_split:
                if self.wavecal_standard.spectrum_list[0].polyfit_coeff is None:
                    warnings.warn("Spectrum is not wavelength calibrated.")
                else:
                    self.wavecal_standard._create_wavecal_fits()
                    hdu_output.extend(self.wavecal_standard.wavecal_hdulist[0])

            if 'flux' in output_split:
                if self.fluxcal.spectrum_list_standard[0].flux is None:
                    warnings.warn("Spectrum is not flux calibrated.")
                else:
                    self.fluxcal._create_flux_fits(spec_id=spec_id,
                                                   stype='standard')
                    hdu_output.extend(self.fluxcal.flux_standard_hdulist)

            if 'adu' in output_split:
                if self.fluxcal.spectrum_list_standard[0].adu is None:
                    warnings.warn("ADU does not exist. Have you included "
                                  "a spectrum?")
                self.fluxcal._create_adu_fits(spec_id=spec_id,
                                              stype='standard')
                hdu_output.extend(self.fluxcal.adu_standard_hdulist)

            if 'adu_resampled' in output_split:
                if self.fluxcal.spectrum_list_standard[0].adu_resampled is None:
                    warnings.warn("Resampled ADU does not exist. Have you "
                                  "included a spectrum? Is it wavelength "
                                  "calibrated and resampled?")
                else:
                    self.fluxcal._create_adu_resampled_fits(spec_id=spec_id,
                                                            stype='standard')
                    hdu_output.extend(
                        self.fluxcal.adu_standard_resampled_hdulist)

            # Convert the first HDU to PrimaryHDU
            hdu_output[0] = fits.PrimaryHDU(hdu_output[0].data,
                                            hdu_output[0].header)
            hdu_output.update_extend()

            if to_disk:
                # Save file to disk
                hdu_output.writeto(filename + '_standard.fits',
                                   overwrite=overwrite)

            if to_memory:
                hdu_list_standard = hdu_output

        if to_memory:
            # return hdu list(s)
            if 'science' in stype_split and 'standard' in stype_split:
                return hdu_list_science, hdu_list_standard
            elif 'science' in stype_split:
                return hdu_list_science
            else:
                return hdu_list_standard

    def save_csv(self,
                 spec_id=None,
                 output='flux_resampled+wavecal+flux+adu+adu_resampled',
                 filename='reduced',
                 column_major=True,
                 stype='science',
                 overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            flux: 4 HDUs
                Flux, uncertainty, sky, sensitivity (bin width = per wavelength)
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            fluxraw: 4 HDUs
                Flux, uncertainty, sky, sensitivity (bin width = per pixel)
            adu: 3 HDUs
                ADU, uncertainty and sky (bin width = per pixel)
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        stype: String
            Spectral type: science or standard
        to_disk: boolean

        to_memory: boolean

        overwrite: boolean
            Default is False.

        '''

        hdu_output = self.save_fits(output=output,
                                    filename=filename,
                                    stype=stype,
                                    to_disk=False,
                                    to_memory=True,
                                    overwrite=overwrite)

        # Split the string into strings
        output_split = output.split('+')
        stype_split = stype.split('+')

        for i in output_split:
            if i not in [
                    'flux_resampled', 'wavecal', 'flux', 'adu', 'adu_resampled'
            ]:
                raise ValueError('%s is not a valid output.' % i)

        if 'science' in stype_split and 'standard' in stype_split:
            hdu_list_science, hdu_list_standard = hdu_output
        elif 'science' in stype_split:
            hdu_list_science = hdu_output
        else:
            hdu_list_standard = hdu_output

        header = {
            'flux_resampled':
            'Resampled Flux, Resampled Flux Uncertainty, Resampled Sky Flux, Resampled Sensitivity Curve',
            'wavecal': 'Polynomial coefficients for wavelength calibration',
            'flux': 'Flux, Flux Uncertainty, Sky Flux, Sensitivity Curve',
            'adu': 'ADU, ADU Uncertainty, Sky ADU',
            'adu_resampled': 'Resampled ADU, ADU Uncertainty, Sky ADU'
        }

        n_hdu = {
            'flux_resampled': 4,
            'wavecal': 1,
            'flux': 4,
            'adu': 3,
            'adu_resampled': 3
        }

        if 'science' in stype_split:

            if spec_id is not None:
                if spec_id not in list(
                        self.fluxcal.spectrum_list_science.keys()):
                    raise ValueError('The given spec_id does not exist.')
            else:
                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.fluxcal.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            for i in spec_id:
                start = 0
                # looping through the output type of each spectrum
                for j in range(len(output_split)):
                    output_type = output_split[j]
                    end = start + n_hdu[output_type]

                    output_data = np.column_stack(
                        [hdu.data for hdu in hdu_list_science[i][start:end]])

                    np.savetxt(filename + '_' + output_type + '_' + str(i) +
                               '.csv',
                               output_data,
                               delimiter=',',
                               header=header[output_type])
                    start = end

        if 'standard' in stype_split:

            start = 0
            # looping through the output type of each spectrum
            for i in output_split:
                end = start + n_hdu[i]

                output_data = np.column_stack(
                    [hdu.data for hdu in hdu_list_standard[start:end]])

                np.savetxt(filename + '_standard_' + i + '.csv',
                           output_data,
                           delimiter=',',
                           header=header[i])
                start = end
