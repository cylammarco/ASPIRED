import difflib
import json
import os
import pkg_resources
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


class _spectrum1D():
    '''
    Base class of a 1D spectral object to hold the information of each
    extracted spectrum.

    '''

    def __init__(self, spec_id):

        # spectrum ID
        self.spec_id = spec_id
        self.fits_array = []

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
        self.widthdn = None
        self.widthup = None
        self.sepdn = None
        self.sepup = None
        self.skywidthdn = None
        self.skywidthup = None
        self.background = None
        self.extraction_type = None
        self.count = None
        self.count_err = None
        self.count_sky = None
        self.var = None

        # Wavelength calibration properties
        self.arc_spec = None
        self.peaks_raw = None
        self.peaks_pixel = None

        # fit constrains
        self.calibrator = None
        self.min_atlas_wavelength = None
        self.max_atlas_wavelength = None
        self.num_slopes = None
        self.range_tolerance = None
        self.fit_tolerance = None
        self.fit_type = None
        self.fit_deg = None
        self.candidate_tolerance = None
        self.linearity_tolerance = None
        self.ransac_tolerance = None
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
        self.fit_coeff = None
        self.rms = None
        self.residual = None
        self.peak_utilisation = None

        # fitted solution
        self.wave = None
        self.wave_bin = None
        self.wave_start = None
        self.wave_end = None
        self.wave_resampled = None
        self.arc_spec_resampled = None
        self.count_resampled = None
        self.count_err_resampled = None
        self.count_sky_resampled = None

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

        self.trace_hdulist = None
        self.count_hdulist = None
        self.arc_spec_hdulist = None
        self.wavecal_hdulist = None
        self.count_resampled_hdulist = None
        self.flux_hdulist = None
        self.flux_resampled_hdulist = None
        self.wavelength_hdulist = None

        self.hdu_output = None
        self.empty_primary_hdu = True

        self.header = {
            'flux_resampled':
            'Pesampled Flux, Resampled Flux Uncertainty, Resampled Sky Flux, '
            'Resampled Sensitivity Curve',
            'count_resampled':
            'Resampled Count, Resampled Count Uncertainty, Resampled Sky Count',
            'arc_spec':
            '1D Arc Spectrum, Arc Line Position, Arc Line Effective Position',
            'wavecal':
            'Polynomial coefficients for wavelength calibration',
            'wavelength':
            'The pixel-to-wavelength mapping',
            'flux':
            'Flux, Flux Uncertainty, Sky Flux, Sensitivity Curve',
            'count':
            'Count, Count Uncertainty, Sky Count',
            'trace':
            'Pixel positions of the trace in the spatial direction'
        }

        self.n_hdu = {
            'flux_resampled': 4,
            'count_resampled': 3,
            'arc_spec': 3,
            'wavecal': 1,
            'wavelength': 1,
            'flux': 4,
            'count': 4,
            'trace': 2
        }

        self.hdu_order = {
            'trace': 0,
            'count': 1,
            'arc_spec': 2,
            'wavecal': 3,
            'wavelength': 4,
            'count_resampled': 5,
            'flux': 6,
            'flux_resampled': 7
        }

        self.hdu_content = {
            'trace': False,
            'count': False,
            'arc_spec': False,
            'wavecal': False,
            'wavelength': False,
            'count_resampled': False,
            'flux': False,
            'flux_resampled': False
        }

    def add_trace(self, trace, trace_sigma, pixel_list=None):

        assert isinstance(
            trace, (list, np.ndarray)), 'trace has to be a list or an array'
        assert isinstance(
            trace_sigma,
            (list, np.ndarray)), 'trace_sigma has to be a list or an array'
        assert len(trace_sigma) == len(trace), 'trace and trace_sigma have to '
        ' be the same size.'

        if pixel_list is None:

            pixel_list = list(np.arange(len(trace)).astype('int'))

        else:

            assert isinstance(
                pixel_list, (list, np.ndarray)), 'pixel_list has to be a list'
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

    def add_aperture(self, widthdn, widthup, sepdn, sepup, skywidthdn,
                     skywidthup):

        assert np.isfinite(widthdn), 'widthdn has to be finite.'
        assert np.isfinite(widthup), 'widthup has to be finite.'
        assert np.isfinite(sepdn), 'sepdn has to be finite.'
        assert np.isfinite(sepup), 'sepup has to be finite.'
        assert np.isfinite(skywidthdn), 'skywidthdn has to be finite.'
        assert np.isfinite(skywidthup), 'skywidthup has to be finite.'
        self.widthdn = widthdn
        self.widthup = widthup
        self.sepdn = sepdn
        self.sepup = sepup
        self.skywidthdn = skywidthdn
        self.skywidthup = skywidthup

    def add_count(self, count, count_err=None, count_sky=None):

        assert isinstance(
            count, (list, np.ndarray)), 'count has to be a list or an array'

        if count_err is not None:

            assert isinstance(
                count_err,
                (list, np.ndarray)), 'count_err has to be a list or an array'
            assert len(count_err) == len(
                count), 'count_err has to be the same size as count'

        if count_sky is not None:

            assert isinstance(
                count_sky,
                (list, np.ndarray)), 'count_sky has to be a list or an array'
            assert len(count_sky) == len(
                count), 'count_sky has to be the same size as count'

        # Only add if all assertions are passed.
        self.count = count

        if count_err is not None:

            self.count_err = count_err

        else:

            self.count_err = np.zeros_like(self.count)

        if count_sky is not None:

            self.count_sky = count_sky

        else:

            self.count_sky = np.zeros_like(self.count)

        if self.pixel_list is None:

            pixel_list = list(np.arange(len(count)).astype('int'))
            self.add_pixel_list(pixel_list)

        else:

            assert len(
                self.pixel_list) == len(count), 'count and pixel_list have '
            'to be the same size.'

    def remove_count(self):

        self.count = None
        self.count_err = None
        self.count_sky = None

    def add_variances(self, var):

        self.var = var

    def remove_variances(self):

        self.var = None

    def add_arc_spec(self, arc_spec):

        assert isinstance(arc_spec,
                          (list, np.ndarray)), 'arc_spec has to be a list'
        self.arc_spec = arc_spec

    def remove_arc_spec(self):
        self.arc_spec = None

    def add_pixel_list(self, pixel_list):

        assert isinstance(pixel_list,
                          (list, np.ndarray)), 'pixel_list has to be a list'
        self.pixel_list = pixel_list

    def remove_pixel_list(self):

        self.pixel_list = None

    def add_pixel_mapping_itp(self, pixel_mapping_itp):

        assert type(
            pixel_mapping_itp
        ) == itp.interpolate.interp1d, 'pixel_mapping_itp has to be a '
        'scipy.interpolate.interpolate.interp1d object.'
        self.pixel_mapping_itp = pixel_mapping_itp

    def remove_pixel_mapping_itp(self):

        self.pixel_mapping_itp = None

    def add_peaks_raw(self, peaks_raw):

        assert isinstance(peaks_raw,
                          (list, np.ndarray)), 'peaks_raw has to be a list'
        self.peaks_raw = peaks_raw

    def remove_peaks_raw(self):

        self.peaks_raw = None

    def add_peaks_pixel(self, peaks_pixel):

        assert isinstance(peaks_pixel,
                          (list, np.ndarray)), 'peaks_pixel has to be a list'
        self.peaks_pixel = peaks_pixel

    def remove_peaks_pixel(self):

        self.peaks_pixel = None

    def add_peaks_wave(self, peaks_wave):

        assert isinstance(peaks_wave,
                          (list, np.ndarray)), 'peaks_wave has to be a list'
        self.peaks_wave = peaks_wave

    def remove_peaks_pixel(self):

        self.peaks_wave = None

    def add_background(self, background):

        # background Count level
        assert np.isfinite(background), 'background has to be finite.'
        self.background = background

    def remove_background(self):

        self.background = None

    def add_calibrator(self, calibrator):

        assert type(
            calibrator
        ) == rascal.Calibrator, 'calibrator has to be a rascal.Calibrator '
        'object.'
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

    def add_gain(self, gain):

        assert np.isfinite(gain), 'gain has to be finite.'
        self.gain = gain

    def remove_gain(self):

        self.gain = None

    def add_readnoise(self, readnoise):

        assert np.isfinite(readnoise), 'readnoise has to be finite.'
        self.readnoise = readnoise

    def remove_readnoise(self):

        self.readnoise = None

    def add_exptime(self, exptime):

        assert np.isfinite(exptime), 'exptime has to be finite.'
        self.exptime = exptime

    def remove_exptime(self):

        self.exptime = None

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

    def add_fit_type(self, fit_type):

        assert type(fit_type) == str, 'fit_type has to be a string'
        assert fit_type in ['poly', 'leg', 'cheb'], 'fit_type must be '
        '(1) poly(nomial); (2) leg(endre); or (3) cheb(yshev).'
        self.fit_type = fit_type

    def remove_fit_type(self):

        self.fit_type = None

    def add_fit_coeff(self, fit_coeff):

        assert isinstance(fit_coeff,
                          (list, np.ndarray)), 'fit_coeff has to be a list.'
        self.fit_coeff = fit_coeff

    def remove_fit_coeff(self):

        self.fit_coeff = None

    def add_calibrator_properties(self, num_pix, pixel_list, plotting_library,
                                  log_level):

        self.num_pix = num_pix
        self.pixel_list = pixel_list
        self.plotting_library = plotting_library
        self.log_level = log_level

    def remove_calibrator_properties(self):

        self.num_pix = None
        self.pixel_list = None
        self.plotting_library = None
        self.log_level = None

    def add_hough_properties(self, num_slopes, xbins, ybins, min_wavelength,
                             max_wavelength, range_tolerance,
                             linearity_tolerance):

        self.num_slopes = num_slopes
        self.xbins = xbins
        self.ybins = ybins
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.range_tolerance = range_tolerance
        self.linearity_tolerance = linearity_tolerance

    def remove_hough_properties(self):

        self.num_slopes = None
        self.xbins = None
        self.ybins = None
        self.min_wavelength = None
        self.max_wavelength = None
        self.range_tolerance = range_tolerance
        self.linearity_tolerance = v

    def add_ransac_properties(self, sample_size, top_n_candidate, linear,
                              filter_close, ransac_tolerance,
                              candidate_weighted, hough_weight):

        self.sample_size = sample_size
        self.top_n_candidate = top_n_candidate
        self.linear = linear
        self.filter_close = filter_close
        self.ransac_tolerance = ransac_tolerance
        self.candidate_weighted = candidate_weighted
        self.hough_weight = hough_weight

    def remove_ransac_properties(self):

        self.sample_size = None
        self.top_n_candidate = None
        self.linear = None
        self.filter_close = None
        self.ransac_tolerance = None
        self.candidate_weighted = None
        self.hough_weight = None

    def add_fit_output_final(self, fit_coeff, rms, residual, peak_utilisation):

        # add assertion here
        self.fit_coeff = fit_coeff
        self.rms = rms
        self.residual = residual
        self.peak_utilisation = peak_utilisation

    def remove_fit_output_final(self):

        self.fit_coeff = None
        self.rms = None
        self.residual = None
        self.peak_utilisation = None

    def add_fit_output_rascal(self, fit_coeff, rms, residual,
                              peak_utilisation):

        # add assertion here
        self.fit_coeff_rascal = fit_coeff
        self.rms_rascal = rms
        self.residual_rascal = residual
        self.peak_utilisation_rascal = peak_utilisation
        self.add_fit_output_final(fit_coeff, rms, residual, peak_utilisation)

    def remove_fit_output_rascal(self):

        self.fit_coeff_rascal = None
        self.rms_rascal = None
        self.residual_rascal = None
        self.peak_utilisation_rascal = None

    def add_fit_output_refine(self, fit_coeff, rms, residual,
                              peak_utilisation):

        # add assertion here
        self.fit_coeff_refine = fit_coeff
        self.rms_refine = rms
        self.residual_refine = residual
        self.peak_utilisation_refine = peak_utilisation
        self.add_fit_output_final(fit_coeff, rms, residual, peak_utilisation)

    def remove_fit_output_refine(self):

        self.fit_coeff_refine = None
        self.rms_refine = None
        self.residual_refine = None
        self.peak_utilisation_refine = None

    def add_wavelength(self, wave):

        self.wave = wave
        self.wave_bin = np.nanmedian(np.array(np.ediff1d(wave)))
        self.wave_start = np.min(wave)
        self.wave_end = np.max(wave)

    def remove_wavelength(self):

        self.wave = None

    def add_wavelength_resampled(self, wave_bin, wave_start, wave_end,
                                 wave_resampled):

        self.wave_bin = wave_bin
        self.wave_start = wave_start
        self.wave_end = wave_end
        self.wave_resampled = wave_resampled

    def remove_wavelength_resampled(self):

        self.wave_bin = None
        self.wave_start = None
        self.wave_end = None
        self.wave_resampled = None

    def add_count_resampled(self, count_resampled, count_err_resampled,
                            count_sky_resampled):

        self.count_resampled = count_resampled
        self.count_err_resampled = count_err_resampled
        self.count_sky_resampled = count_sky_resampled

    def remove_count_resampled(self):

        self.count_resampled = None
        self.count_err_resampled = None
        self.count_sky_resampled = None

    def add_smoothing(self, smooth, slength, sorder):

        self.smooth = smooth
        self.slength = slength
        self.sorder = sorder

    def remove_smoothing(self):

        self.smooth = None
        self.slength = None
        self.sorder = None

    def add_sensitivity_itp(self, sensitivity_itp):

        self.sensitivity_itp = sensitivity_itp

    def remove_sensitivity_itp(self):

        self.sensitivity_itp = None

    def add_sensitivity(self, sensitivity):

        self.sensitivity = sensitivity

    def remove_sensitivity(self):

        self.sensitivity = None

    def add_literature_standard(self, wave_literature, flux_literature):

        self.wave_literature = wave_literature
        self.flux_literature = flux_literature

    def remove_literature_standard(self):

        self.wave_literature = None
        self.flux_literature = None

    def add_flux(self, flux, flux_err, flux_sky):

        self.flux = flux
        self.flux_err = flux_err
        self.flux_sky = flux_sky

    def remove_flux(self):

        self.flux = None
        self.flux_err = None
        self.flux_sky = None

    def add_flux_resampled(self, flux_resampled, flux_err_resampled,
                           flux_sky_resampled):

        self.flux_resampled = flux_resampled
        self.flux_err_resampled = flux_err_resampled
        self.flux_sky_resampled = flux_sky_resampled

    def remove_flux_resampled(self):

        self.flux_resampled = None
        self.flux_err_resampled = None
        self.flux_sky_resampled = None

    def add_sensitivity_resampled(self, sensitivity_resampled):

        self.sensitivity_resampled = sensitivity_resampled

    def remove_sensitivity_resampled(self):

        self.sensitivity_resampled = None

    def _modify_imagehdu_data(self, hdulist, idx, method, *args):

        method_to_call = getattr(hdulist[idx].data, method)
        method_to_call(*args)

    def _modify_imagehdu_header(self, hdulist, idx, method, *args):
        '''
        e.g.
        method = 'set'
        args = 'BUNIT', 'Angstroms'

        method_to_call(*args) becomes hdu[idx].header.set('BUNIT', 'Angstroms')

        '''

        method_to_call = getattr(hdulist[idx].header, method)
        method_to_call(*args)

    def modify_trace_header(self, idx, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+---------------------+
        | HDU | Data                |
        +-----+---------------------+
        |  0  | Trace (pixel)       |
        |  1  | Trace width (pixel) |
        +-----+---------------------+

        '''

        self._modify_imagehdu_header(self.trace_hdulist, idx, method, *args)

    def modify_count_header(self, idx, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+---------------------------------+
        | HDU | Data                            |
        +-----+---------------------------------+
        |  0  | Photoelectron count             |
        |  1  | Photoelectron count uncertainty |
        |  2  | Photoelectron count (sky)       |
        |  3  | Optimal (boolean)               |
        |  4  | Line spread function            |
        +-----+---------------------------------+

        '''

        self._modify_imagehdu_header(self.count_hdulist, idx, method, *args)

    def modify_count_resampled_header(self, idx, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+---------------------------------+
        | HDU | Data                            |
        +-----+---------------------------------+
        |  0  | Photoelectron count             |
        |  1  | Photoelectron count uncertainty |
        |  2  | Photoelectron count (sky)       |
        +-----+---------------------------------+

        '''

        self._modify_imagehdu_header(self.count_resampled_hdulist, idx, method,
                                     *args)

    def modify_arc_spec_header(self, idx, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+-------------------+
        | HDU | Data              |
        +-----+-------------------+
        |  0  | Arc spectrum      |
        |  1  | Peaks (pixel)     |
        |  2  | Peaks (sub-pixel) |
        +-----+-------------------+

        '''

        self._modify_imagehdu_header(self.arc_spec_hdulist, idx, method, *args)

    def modify_wavecal_header(self, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None
        wavelength fits only has one ImageHDU so the idx is always 0

        +-----+-----------------------+
        | HDU | Data                  |
        +-----+-----------------------+
        |  0  | Best fit coefficients |
        +-----+-----------------------+

        '''

        self._modify_imagehdu_header(self.wavecal_hdulist, 0, method, *args)

    def modify_flux_header(self, idx, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------+
        | HDU | Data             |
        +-----+------------------+
        |  0  | Flux             |
        |  1  | Flux uncertainty |
        |  2  | Flux (sky)       |
        |  3  | Sensitivity      |
        +-----+------------------+

        '''

        self._modify_imagehdu_header(self.flux_hdulist, idx, method, *args)

    def modify_flux_resampled_header(self, idx, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------+
        | HDU | Data             |
        +-----+------------------+
        |  0  | Flux             |
        |  1  | Flux Uncertainty |
        |  2  | Flux Sky         |
        |  3  | Sensitivity      |
        +-----+------------------+

        '''

        self._modify_imagehdu_header(self.flux_resampled_hdulist, idx, method,
                                     *args)

    def modify_wavelength_header(self, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None
        wavelength fits only has one ImageHDU so the idx is always 0

        '''

        self._modify_imagehdu_header(self.wavelength_hdulist, 0, method, *args)

    def create_trace_fits(self, force=True):

        if (self.trace_hdulist is None) or force:

            try:

                # Put the data in ImageHDUs
                trace_ImageHDU = fits.ImageHDU(self.trace)
                trace_sigma_ImageHDU = fits.ImageHDU(self.trace_sigma)

                # Create an empty HDU list and populate with ImageHDUs
                self.trace_hdulist = fits.HDUList()
                self.trace_hdulist += [trace_ImageHDU]
                self.trace_hdulist += [trace_sigma_ImageHDU]

                # Add the trace
                self.modify_trace_header(0, 'set', 'LABEL', 'Trace')
                self.modify_trace_header(0, 'set', 'CRPIX1', 1)
                self.modify_trace_header(0, 'set', 'CDELT1', 1)
                self.modify_trace_header(0, 'set', 'CRVAL1',
                                         self.pixel_list[0])
                self.modify_trace_header(0, 'set', 'CTYPE1',
                                         'Pixel (Dispersion)')
                self.modify_trace_header(0, 'set', 'CUNIT1', 'Pixel')
                self.modify_trace_header(0, 'set', 'BUNIT', 'Pixel (Spatial)')

                # Add the trace_sigma
                self.modify_trace_header(1, 'set', 'LABEL',
                                         'Trace width/sigma')
                self.modify_trace_header(1, 'set', 'CRPIX1', 1)
                self.modify_trace_header(1, 'set', 'CDELT1', 1)
                self.modify_trace_header(1, 'set', 'CRVAL1',
                                         self.pixel_list[0])
                self.modify_trace_header(1, 'set', 'CTYPE1',
                                         'Pixel (Dispersion)')
                self.modify_trace_header(1, 'set', 'CUNIT1',
                                         'Number of Pixels')
                self.modify_trace_header(1, 'set', 'BUNIT', 'Pixel (Spatial)')

            except Exception as e:

                warnings.warn(e.__doc__)
                warnings.warn(e)

                # Set it to None if the above failed
                warnings.warn('trace ImageHDU cannot be created.')
                self.trace_hdulist = None

        else:

            warnings.warn('trace ImageHDU already exists, set force to True '
                          'to create a new one.')

    def create_count_fits(self, force=True):

        if (self.count_hdulist is None) or force:

            try:

                # Put the data in ImageHDUs
                count_ImageHDU = fits.ImageHDU(self.count)
                count_err_ImageHDU = fits.ImageHDU(self.count_err)
                count_sky_ImageHDU = fits.ImageHDU(self.count_sky)
                count_weight_ImageHDU = fits.ImageHDU(self.var)

                # Create an empty HDU list and populate with ImageHDUs
                self.count_hdulist = fits.HDUList()
                self.count_hdulist += [count_ImageHDU]
                self.count_hdulist += [count_err_ImageHDU]
                self.count_hdulist += [count_sky_ImageHDU]
                self.count_hdulist += [count_weight_ImageHDU]

                # Add the count
                self.modify_count_header(0, 'set', 'WIDTHDN', self.widthdn)
                self.modify_count_header(0, 'set', 'WIDTHUP', self.widthup)
                self.modify_count_header(0, 'set', 'SEPDN', self.sepdn)
                self.modify_count_header(0, 'set', 'SEPUP', self.sepup)
                self.modify_count_header(0, 'set', 'SKYDN', self.skywidthdn)
                self.modify_count_header(0, 'set', 'SKYUP', self.skywidthup)
                self.modify_count_header(0, 'set', 'BKGRND', self.background)
                self.modify_count_header(0, 'set', 'XTYPE',
                                         self.extraction_type)
                self.modify_count_header(0, 'set', 'LABEL', 'Electron Count')
                self.modify_count_header(0, 'set', 'CRPIX1', 1)
                self.modify_count_header(0, 'set', 'CDELT1', 1)
                self.modify_count_header(0, 'set', 'CRVAL1',
                                         self.pixel_list[0])
                self.modify_count_header(0, 'set', 'CTYPE1',
                                         ' Pixel (Dispersion)')
                self.modify_count_header(0, 'set', 'CUNIT1', 'Pixel')
                self.modify_count_header(0, 'set', 'BUNIT', 'electron')

                # Add the uncertainty count
                self.modify_count_header(1, 'set', 'LABEL',
                                         'Electron Count (Uncertainty)')
                self.modify_count_header(1, 'set', 'CRPIX1', 1)
                self.modify_count_header(1, 'set', 'CDELT1', 1)
                self.modify_count_header(1, 'set', 'CRVAL1',
                                         self.pixel_list[0])
                self.modify_count_header(1, 'set', 'CTYPE1',
                                         'Pixel (Dispersion)')
                self.modify_count_header(1, 'set', 'CUNIT1', 'Pixel')
                self.modify_count_header(1, 'set', 'BUNIT', 'electron')

                # Add the sky count
                self.modify_count_header(2, 'set', 'LABEL',
                                         'Electron Count (Sky)')
                self.modify_count_header(2, 'set', 'CRPIX1', 1)
                self.modify_count_header(2, 'set', 'CDELT1', 1)
                self.modify_count_header(2, 'set', 'CRVAL1',
                                         self.pixel_list[0])
                self.modify_count_header(2, 'set', 'CTYPE1',
                                         'Pixel (Dispersion)')
                self.modify_count_header(2, 'set', 'CUNIT1', 'Pixel')
                self.modify_count_header(2, 'set', 'BUNIT', 'electron')

                # Add the extraction weights
                self.modify_count_header(3, 'set', 'LABEL',
                                         'Optimal Extraction Profile')
                if self.var is not None:
                    self.modify_count_header(3, 'set', 'CRVAL1', len(self.var))
                    self.modify_count_header(3, 'set', 'CRPIX1', 1)
                    self.modify_count_header(3, 'set', 'CDELT1', 1)
                    self.modify_count_header(3, 'set', 'CTYPE1',
                                             'Pixel (Spatial)')
                    self.modify_count_header(3, 'set', 'CUNIT1', 'Pixel')
                    self.modify_count_header(3, 'set', 'BUNIT', 'weights')
                else:
                    self.modify_count_header(
                        3, 'set', 'COMMENT',
                        'Extraction Profile is not available.')

            except Exception as e:

                warnings.warn(e.__doc__)
                warnings.warn(e)

                # Set it to None if the above failed
                warnings.warn('count ImageHDU cannot be created.')
                self.count_hdulist = None

        else:

            warnings.warn('count ImageHDU already exists, set force to True '
                          'to create a new one.')

    def create_count_resampled_fits(self, force=True):

        if (self.count_resampled_hdulist is None) or force:

            try:

                # Put the data in ImageHDUs
                count_resampled_ImageHDU = fits.ImageHDU(self.count_resampled)
                count_resampled_err_ImageHDU = fits.ImageHDU(
                    self.count_resampled_err)
                count_resampled_sky_ImageHDU = fits.ImageHDU(
                    self.count_resampled_sky)

                # Create an empty HDU list and populate with ImageHDUs
                self.count_resampled_hdulist = fits.HDUList()
                self.count_resampled_hdulist += [count_resampled_ImageHDU]
                self.count_resampled_hdulist += [count_resampled_err_ImageHDU]
                self.count_resampled_hdulist += [count_resampled_sky_ImageHDU]

                # Add the resampled count
                self.modify_count_resampled_header(0, 'set', 'LABEL',
                                                   'Resampled Electron Count')
                self.modify_count_resampled_header(0, 'set', 'CRPIX1',
                                                   1.00E+00)
                self.modify_count_resampled_header(0, 'set', 'CDELT1',
                                                   self.wave_bin)
                self.modify_count_resampled_header(0, 'set', 'CRVAL1',
                                                   self.wave_start)
                self.modify_count_resampled_header(0, 'set', 'CTYPE1',
                                                   'Wavelength')
                self.modify_count_resampled_header(0, 'set', 'CUNIT1',
                                                   'Angstroms')
                self.modify_count_resampled_header(0, 'set', 'BUNIT',
                                                   'electron')

                # Add the resampled uncertainty count
                self.modify_count_resampled_header(
                    1, 'set', 'LABEL',
                    'Resampled Electron Count (Uncertainty)')
                self.modify_count_resampled_header(1, 'set', 'CRPIX1',
                                                   1.00E+00)
                self.modify_count_resampled_header(1, 'set', 'CDELT1',
                                                   self.wave_bin)
                self.modify_count_resampled_header(1, 'set', 'CRVAL1',
                                                   self.wave_start)
                self.modify_count_resampled_header(1, 'set', 'CTYPE1',
                                                   'Wavelength')
                self.modify_count_resampled_header(1, 'set', 'CUNIT1',
                                                   'Angstroms')
                self.modify_count_resampled_header(1, 'set', 'BUNIT',
                                                   'electron')

                # Add the resampled sky count
                self.modify_count_resampled_header(
                    2, 'set', 'LABEL', 'Resampled Electron Count (Sky)')
                self.modify_count_resampled_header(2, 'set', 'CRPIX1',
                                                   1.00E+00)
                self.modify_count_resampled_header(2, 'set', 'CDELT1',
                                                   self.wave_bin)
                self.modify_count_resampled_header(2, 'set', 'CRVAL1',
                                                   self.wave_start)
                self.modify_count_resampled_header(2, 'set', 'CTYPE1',
                                                   'Wavelength')
                self.modify_count_resampled_header(2, 'set', 'CUNIT1',
                                                   'Angstroms')
                self.modify_count_resampled_header(2, 'set', 'BUNIT',
                                                   'electron')

            except Exception as e:

                warnings.warn(e.__doc__)
                warnings.warn(e)

                # Set it to None if the above failed
                warnings.warn('count_resampled ImageHDU cannot be created.')
                self.count_resampled_hdulist = None

        else:

            warnings.warn(
                'count_resampled ImageHDU already exists, set force to True '
                'to create a new one.')

    def create_arc_spec_fits(self, force=True):

        if (self.arc_spec_hdulist is None) or force:

            try:

                # Put the data in ImageHDUs
                arc_spec_ImageHDU = fits.ImageHDU(self.arc_spec)
                peaks_raw_ImageHDU = fits.ImageHDU(self.peaks_raw)
                peaks_pixel_ImageHDU = fits.ImageHDU(self.peaks_pixel)

                # Create an empty HDU list and populate with ImageHDUs
                self.arc_spec_hdulist = fits.HDUList()
                self.arc_spec_hdulist += [arc_spec_ImageHDU]
                self.arc_spec_hdulist += [peaks_raw_ImageHDU]
                self.arc_spec_hdulist += [peaks_pixel_ImageHDU]

                # Add the arc spectrum
                self.modify_arc_spec_header(0, 'set', 'LABEL',
                                            'Electron Count')
                self.modify_arc_spec_header(0, 'set', 'CRPIX1', 1)
                self.modify_arc_spec_header(0, 'set', 'CDELT1', 1)
                self.modify_arc_spec_header(0, 'set', 'CRVAL1',
                                            self.pixel_list[0])
                self.modify_arc_spec_header(0, 'set', 'CTYPE1',
                                            'Pixel (Dispersion)')
                self.modify_arc_spec_header(0, 'set', 'CUNIT1', 'Pixel')
                self.modify_arc_spec_header(0, 'set', 'BUNIT', 'electron')

                # Add the peaks in native pixel value
                self.modify_arc_spec_header(1, 'set', 'LABEL',
                                            'Peaks (Detector Pixel)')
                self.modify_arc_spec_header(1, 'set', 'BUNIT', 'Pixel')

                # Add the peaks in effective pixel value
                self.modify_arc_spec_header(2, 'set', 'LABEL',
                                            'Peaks (Effective Pixel)')
                self.modify_arc_spec_header(2, 'set', 'BUNIT', 'Pixel')

            except Exception as e:

                warnings.warn(e.__doc__)
                warnings.warn(e)

                # Set it to None if the above failed
                warnings.warn('arc_spec ImageHDU cannot be created.')
                self.arc_spec_hdulist = None

        else:

            warnings.warn(
                'arc_spec ImageHDU already exists, set force to True '
                'to create a new one.')

    def create_wavecal_fits(self, force=True):

        if (self.wavecal_hdulist is None) or force:

            try:

                # Put the data in an ImageHDU
                wavecal_ImageHDU = fits.ImageHDU(self.fit_coeff)

                # Create an empty HDU list and populate with ImageHDUs
                self.wavecal_hdulist = fits.HDUList()
                self.wavecal_hdulist += [wavecal_ImageHDU]

                # Add the wavelength calibration header info
                self.modify_wavecal_header('set', 'FTYPE', self.fit_type)
                self.modify_wavecal_header('set', 'FDEG', self.fit_deg)
                self.modify_wavecal_header('set', 'FFRMS', self.rms)
                self.modify_wavecal_header('set', 'ATLWMIN',
                                           self.min_atlas_wavelength)
                self.modify_wavecal_header('set', 'ATLWMAX',
                                           self.max_atlas_wavelength)
                self.modify_wavecal_header('set', 'NSLOPES', self.num_slopes)
                self.modify_wavecal_header('set', 'RNGTOL',
                                           self.range_tolerance)
                self.modify_wavecal_header('set', 'FITTOL', self.fit_tolerance)
                self.modify_wavecal_header('set', 'CANTHRE',
                                           self.candidate_tolerance)
                self.modify_wavecal_header('set', 'LINTHRE',
                                           self.linearity_tolerance)
                self.modify_wavecal_header('set', 'RANTHRE',
                                           self.ransac_tolerance)
                self.modify_wavecal_header('set', 'NUMCAN',
                                           self.num_candidates)
                self.modify_wavecal_header('set', 'XBINS', self.xbins)
                self.modify_wavecal_header('set', 'YBINS', self.ybins)
                self.modify_wavecal_header('set', 'BRUTE', self.brute_force)
                self.modify_wavecal_header('set', 'SAMSIZE', self.sample_size)
                self.modify_wavecal_header('set', 'TOPN', self.top_n)
                self.modify_wavecal_header('set', 'MAXTRY', self.max_tries)
                self.modify_wavecal_header('set', 'INCOEFF', self.intput_coeff)
                self.modify_wavecal_header('set', 'LINEAR', self.linear)
                self.modify_wavecal_header('set', 'W8ED', self.weighted)
                self.modify_wavecal_header('set', 'FILTER', self.filter_close)
                self.modify_wavecal_header('set', 'PUSAGE',
                                           self.peak_utilisation)
                self.modify_wavecal_header('set', 'LABEL', 'Electron Count')
                self.modify_wavecal_header('set', 'CRPIX1', 1.00E+00)
                self.modify_wavecal_header('set', 'CDELT1', self.wave_bin)
                self.modify_wavecal_header('set', 'CRVAL1', self.wave_start)
                self.modify_wavecal_header('set', 'CTYPE1', 'Wavelength')
                self.modify_wavecal_header('set', 'CUNIT1', 'Angstroms')
                self.modify_wavecal_header('set', 'BUNIT', 'electron')

            except Exception as e:

                warnings.warn(e.__doc__)
                warnings.warn(e)

                # Set it to None if the above failed
                warnings.warn('wavecal ImageHDU cannot be created.')
                self.wavecal_hdulist = None

        else:

            warnings.warn(
                'wave_cal ImageHDU already exists, set force to True '
                'to create a new one.')

    def create_flux_fits(self, force=True):

        if (self.flux_hdulist is None) or force:

            try:

                # Put the data in ImageHDUs
                flux_ImageHDU = fits.ImageHDU(self.flux)
                flux_err_ImageHDU = fits.ImageHDU(self.flux_err)
                flux_sky_ImageHDU = fits.ImageHDU(self.flux_sky)
                sensitivity_ImageHDU = fits.ImageHDU(self.sensitivity)

                # Create an empty HDU list and populate with ImageHDUs
                self.flux_hdulist = fits.HDUList()
                self.flux_hdulist += [flux_ImageHDU]
                self.flux_hdulist += [flux_err_ImageHDU]
                self.flux_hdulist += [flux_sky_ImageHDU]
                self.flux_hdulist += [sensitivity_ImageHDU]

                # Note that wave_start is the centre of the starting bin
                self.modify_flux_header(0, 'set', 'LABEL', 'Flux')
                self.modify_flux_header(0, 'set', 'CRPIX1', 1.00E+00)
                self.modify_flux_header(0, 'set', 'CDELT1', 1)
                self.modify_flux_header(0, 'set', 'CRVAL1', self.pixel_list[0])
                self.modify_flux_header(0, 'set', 'CTYPE1', 'Pixel')
                self.modify_flux_header(0, 'set', 'CUNIT1', 'Pixel')
                self.modify_flux_header(0, 'set', 'BUNIT',
                                        'erg/(s*cm**2*Angstrom)')

                self.modify_flux_header(1, 'set', 'LABEL', 'Flux')
                self.modify_flux_header(1, 'set', 'CRPIX1', 1.00E+00)
                self.modify_flux_header(1, 'set', 'CDELT1', 1)
                self.modify_flux_header(1, 'set', 'CRVAL1', self.pixel_list[0])
                self.modify_flux_header(1, 'set', 'CTYPE1', 'Pixel')
                self.modify_flux_header(1, 'set', 'CUNIT1', 'Pixel')
                self.modify_flux_header(1, 'set', 'BUNIT',
                                        'erg/(s*cm**2*Angstrom)')

                self.modify_flux_header(2, 'set', 'LABEL', 'Flux')
                self.modify_flux_header(2, 'set', 'CRPIX1', 1.00E+00)
                self.modify_flux_header(2, 'set', 'CDELT1', 1)
                self.modify_flux_header(2, 'set', 'CRVAL1', self.pixel_list[0])
                self.modify_flux_header(2, 'set', 'CTYPE1', 'Pixel')
                self.modify_flux_header(2, 'set', 'CUNIT1', 'Pixel')
                self.modify_flux_header(2, 'set', 'BUNIT',
                                        'erg/(s*cm**2*Angstrom)')

                self.modify_flux_header(3, 'set', 'LABEL', 'Sensitivity')
                self.modify_flux_header(3, 'set', 'CRPIX1', 1.00E+00)
                self.modify_flux_header(3, 'set', 'CDELT1', 1)
                self.modify_flux_header(3, 'set', 'CRVAL1', self.pixel_list[0])
                self.modify_flux_header(3, 'set', 'CTYPE1', 'Pixel')
                self.modify_flux_header(3, 'set', 'CUNIT1', 'Pixel')
                self.modify_flux_header(3, 'set', 'BUNIT',
                                        'erg/(s*cm**2*Angstrom)/Count')

            except Exception as e:

                warnings.warn(e.__doc__)
                warnings.warn(e)

                # Set it to None if the above failed
                warnings.warn('flux ImageHDU cannot be created.')
                self.flux_hdulist = None

        else:

            warnings.warn('flux ImageHDU already exists, set force to True '
                          'to create a new one.')

    def create_flux_resampled_fits(self, force=True):

        if (self.flux_resampled_hdulist is None) or force:

            try:

                # Put the data in ImageHDUs
                flux_resampled_ImageHDU = fits.ImageHDU(self.flux_resampled)
                flux_err_resampled_ImageHDU = fits.ImageHDU(
                    self.flux_err_resampled)
                flux_sky_resampled_ImageHDU = fits.ImageHDU(
                    self.flux_sky_resampled)
                sensitivity_resampled_ImageHDU = fits.ImageHDU(
                    self.sensitivity_resampled)

                # Create an empty HDU list and populate with ImageHDUs
                self.flux_resampled_hdulist = fits.HDUList()
                self.flux_resampled_hdulist += [flux_resampled_ImageHDU]
                self.flux_resampled_hdulist += [flux_err_resampled_ImageHDU]
                self.flux_resampled_hdulist += [flux_sky_resampled_ImageHDU]
                self.flux_resampled_hdulist += [sensitivity_resampled_ImageHDU]

                # Note that wave_start is the centre of the starting bin
                self.modify_flux_resampled_header(0, 'set', 'LABEL', 'Flux')
                self.modify_flux_resampled_header(0, 'set', 'CRPIX1', 1.00E+00)
                self.modify_flux_resampled_header(0, 'set', 'CDELT1',
                                                  self.wave_bin)
                self.modify_flux_resampled_header(0, 'set', 'CRVAL1',
                                                  self.wave_start)
                self.modify_flux_resampled_header(0, 'set', 'CTYPE1',
                                                  'Wavelength')
                self.modify_flux_resampled_header(0, 'set', 'CUNIT1',
                                                  'Angstroms')
                self.modify_flux_resampled_header(0, 'set', 'BUNIT',
                                                  'erg/(s*cm**2*Angstrom)')

                self.modify_flux_resampled_header(1, 'set', 'LABEL', 'Flux')
                self.modify_flux_resampled_header(1, 'set', 'CRPIX1', 1.00E+00)
                self.modify_flux_resampled_header(1, 'set', 'CDELT1',
                                                  self.wave_bin)
                self.modify_flux_resampled_header(1, 'set', 'CRVAL1',
                                                  self.wave_start)
                self.modify_flux_resampled_header(1, 'set', 'CTYPE1',
                                                  'Wavelength')
                self.modify_flux_resampled_header(1, 'set', 'CUNIT1',
                                                  'Angstroms')
                self.modify_flux_resampled_header(1, 'set', 'BUNIT',
                                                  'erg/(s*cm**2*Angstrom)')

                self.modify_flux_resampled_header(2, 'set', 'LABEL', 'Flux')
                self.modify_flux_resampled_header(2, 'set', 'CRPIX1', 1.00E+00)
                self.modify_flux_resampled_header(2, 'set', 'CDELT1',
                                                  self.wave_bin)
                self.modify_flux_resampled_header(2, 'set', 'CRVAL1',
                                                  self.wave_start)
                self.modify_flux_resampled_header(2, 'set', 'CTYPE1',
                                                  'Wavelength')
                self.modify_flux_resampled_header(2, 'set', 'CUNIT1',
                                                  'Angstroms')
                self.modify_flux_resampled_header(2, 'set', 'BUNIT',
                                                  'erg/(s*cm**2*Angstrom)')

                self.modify_flux_resampled_header(3, 'set', 'LABEL',
                                                  'Sensitivity')
                self.modify_flux_resampled_header(3, 'set', 'CRPIX1', 1.00E+00)
                self.modify_flux_resampled_header(3, 'set', 'CDELT1',
                                                  self.wave_bin)
                self.modify_flux_resampled_header(3, 'set', 'CRVAL1',
                                                  self.wave_start)
                self.modify_flux_resampled_header(3, 'set', 'CTYPE1',
                                                  'Wavelength')
                self.modify_flux_resampled_header(3, 'set', 'CUNIT1',
                                                  'Angstroms')
                self.modify_flux_resampled_header(
                    3, 'set', 'BUNIT', 'erg/(s*cm**2*Angstrom)/Count')

            except Exception as e:

                warnings.warn(e.__doc__)
                warnings.warn(e)

                # Set it to None if the above failed
                warnings.warn('flux_resampled ImageHDU cannot be created.')
                self.flux_resampled_hdulist = None

        else:

            warnings.warn(
                'flux_resampled ImageHDU already exists, set force to True '
                'to create a new one.')

    def create_wavelength_fits(self, force=True):

        if (self.wavelength_hdulist is None) or force:

            try:

                # Put the data in an ImageHDU
                wavelength_ImageHDU = fits.ImageHDU(self.wave)

                # Create an empty HDU list and populate with the ImageHDU
                self.wavelength_hdulist = fits.HDUList()
                self.wavelength_hdulist += [wavelength_ImageHDU]

                # Add the calibrated wavelength
                self.modify_wavelength_header('set', 'LABEL',
                                              'Pixel-Wavelength Mapping')
                self.modify_wavelength_header('set', 'CRPIX1', 1)
                self.modify_wavelength_header('set', 'CDELT1', 1)
                self.modify_wavelength_header('set', 'CRVAL1',
                                              self.pixel_list[0])
                self.modify_wavelength_header('set', 'CTYPE1',
                                              'Pixel (Dispersion)')
                self.modify_wavelength_header('set', 'CUNIT1', 'Pixel')
                self.modify_wavelength_header('set', 'BUNIT', 'Angstroms')

            except Exception as e:

                warnings.warn(e.__doc__)
                warnings.warn(e)

                # Set it to None if the above failed
                warnings.warn('wavelength ImageHDU cannot be created.')
                self.wavelength_hdulist = None
        else:

            warnings.warn(
                'wavelength ImageHDU already exists, set force to True '
                'to create a new one.')

    def remove_trace_fits(self):

        self.trace_hdulist = None

    def remove_count_fits(self):

        self.count_hdulist = None

    def remove_count_resampled_fits(self):

        self.count_resampled_hdulist = None

    def remove_arc_spec_fits(self):

        self.arc_spec_hdulist = None

    def remove_wavecal_fits(self):

        self.wavecal_hdulist = None

    def remove_flux_resampled_fits(self):

        self.flux_resampled_hdulist = None

    def remove_wavelength_fits(self):

        self.wavelength_hdulist = None

    def create_fits(self,
                    output,
                    force=False,
                    empty_primary_hdu=True,
                    return_hdu_list=False):
        '''
        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        return_hdu_list: boolean (default: False)
            Set to True to return the HDU List

        '''

        output_split = output.split('+')

        # Will check if the HDUs need to be (re)created inside the functions
        if 'trace' in output_split:

            self.create_trace_fits(force)

        if 'count' in output_split:

            self.create_count_fits(force)

        if 'arc_spec' in output_split:

            self.create_arc_spec_fits(force)

        if 'wavecal' in output_split:

            self.create_wavecal_fits(force)

        if 'wavelength' in output_split:

            self.create_wavelength_fits(force)

        if 'count_resampled' in output_split:

            self.create_count_resampled_fits(force)

        if 'flux' in output_split:

            self.create_flux_fits(force)

        if 'flux_resampled' in output_split:

            self.create_flux_resampled_fits(force)

        # If the requested list of HDUs is already good to go
        if set(self.hdu_content) == set(output_split):

            # If there is an empty primary HDU, but requested without
            if (self.empty_primary_hdu & (not empty_primary_hdu)):

                self.hdu_output.pop(0)
                self.empty_primary_hdu = False

            # If there is not an empty primary HDU, but requested one
            elif ((not self.empty_primary_hdu) & empty_primary_hdu):

                self.hdu_output.insert(0, fits.PrimaryHDU())
                self.empty_primary_hdu = True

            # Otherwise, the self.hdu_output does not need to be modified
            else:

                pass

        # If the requested list is different or empty, (re)create the list
        else:

            self.hdu_output = None

            # Empty list for appending HDU lists
            hdu_output = fits.HDUList()

            # If leaving the primary HDU empty
            if empty_primary_hdu:

                hdu_output.append(fits.PrimaryHDU())

            # Join the list(s)
            if 'trace' in output_split:

                hdu_output += self.trace_hdulist
                self.hdu_content['trace'] = True

            if 'count' in output_split:

                hdu_output += self.count_hdulist
                self.hdu_content['count'] = True

            if 'arc_spec' in output_split:

                hdu_output += self.arc_spec_hdulist
                self.hdu_content['arc_spec'] = True

            if 'wavecal' in output_split:

                hdu_output += self.wavecal_hdulist
                self.hdu_content['wavecal'] = True

            if 'wavelength' in output_split:

                hdu_output += self.wavelength_hdulist
                self.hdu_content['wavelength'] = True

            if 'count_resampled' in output_split:

                hdu_output += self.count_resampled_hdulist
                self.hdu_content['count_resampled'] = True

            if 'flux' in output_split:

                hdu_output += self.flux_hdulist
                self.hdu_content['flux'] = True

            if 'flux_resampled' in output_split:

                hdu_output += self.flux_resampled_hdulist
                self.hdu_content['flux_resampled'] = True

            # If the primary HDU is not chosen to be empty
            if not empty_primary_hdu:

                # Convert the first HDU to PrimaryHDU
                hdu_output[0] = fits.PrimaryHDU(hdu_output[0].data,
                                                hdu_output[0].header)
                hdu_output.update_extend()
                self.empty_primary_hdu = False

            hdu_output.update_extend()
            self.empty_primary_hdu = True
            self.hdu_output = hdu_output

        if return_hdu_list:

            return self.hdu_output

    def save_fits(self,
                  output,
                  filename,
                  force=False,
                  overwrite=False,
                  empty_primary_hdu=True):
        '''
        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        overwrite: boolean
            Default is False.
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)

        '''

        self.create_fits(output, force, empty_primary_hdu)

        # Save file to disk
        self.hdu_output.writeto(filename + '.fits', overwrite=overwrite)

    def save_csv(self, output, filename, force, overwrite):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        overwrite: boolean
            Default is False.

        '''

        self.create_fits(output, force, empty_primary_hdu=False)

        output_split = output.split('+')

        start = 0

        for output_type in self.hdu_order.keys():

            if output_type in output_split:

                end = start + self.n_hdu[output_type]
                output_data = np.column_stack(
                    [hdu.data for hdu in self.hdu_output[start:end]])

                if overwrite or (not os.path.exists(filename + '_' +
                                                    output_type + '.csv')):

                    np.savetxt(filename + '_' + output_type + '.csv',
                               output_data,
                               delimiter=',',
                               header=self.header[output_type])

                start = end


class TwoDSpec:
    '''
    This is a class for processing a 2D spectral image.

    '''
    def __init__(self, data, header=None, **kwargs):
        '''
        The constructor takes the data and the header, and the the header
        infromation will be read automatically. See set_properties()
        for the detail information of the keyword arguments.

        parameters
        ----------
        data: 2D numpy array (M x N) OR astropy.io.fits object
            2D spectral image in either format
        header: FITS header (deafult: None)
            THIS WILL OVERRIDE the header from the astropy.io.fits object
        **kwargs: keyword arguments (default: see set_properties())
            see set_properties().

        '''

        # If data provided is an numpy array
        if isinstance(data, np.ndarray):

            self.img = data
            self.header = header

        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(data, fits.hdu.image.PrimaryHDU) or isinstance(
                data, fits.hdu.image.ImageHDU):

            self.img = data.data
            self.header = data.header

        # If it is an ImageReduction object
        elif isinstance(data, ImageReduction):

            # If the data is not reduced, reduce it here. Error handling is
            # done by the ImageReduction class
            if data.image_fits is None:

                data._create_image_fits()

            self.img = data.image_fits.data
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
            self.img = fitsfile_tmp.data
            self.header = fitsfile_tmp.header
            fitsfile_tmp = None

        else:

            raise TypeError(
                'Please provide a numpy array, an ' +
                'astropy.io.fits.hdu.image.PrimaryHDU object or an ' +
                'ImageReduction object.')

        self.saxis = 1
        self.waxis = 0

        self.spatial_mask = (1, )
        self.spec_mask = (1, )
        self.flip = False
        self.cosmicray = False
        self.cosmicray_sigma = 5.

        # Default values if not supplied or cannot be automatically identified
        # from the header
        self.readnoise = 0.
        self.gain = 1.
        self.seeing = 1.
        self.exptime = 1.

        self.silence = False

        # Default keywords to be searched in the order in the list
        self.readnoise_keyword = ['RDNOISE', 'RNOISE', 'RN']
        self.gain_keyword = ['GAIN']
        self.seeing_keyword = ['SEEING', 'L1SEEING', 'ESTSEE']
        self.exptime_keyword = [
            'XPOSURE', 'EXPTIME', 'EXPOSED', 'TELAPSED', 'ELAPSED'
        ]

        self.set_properties(**kwargs)

    def set_properties(self,
                       saxis=None,
                       spatial_mask=None,
                       spec_mask=None,
                       flip=None,
                       cosmicray=None,
                       cosmicray_sigma=None,
                       readnoise=None,
                       gain=None,
                       seeing=None,
                       exptime=None,
                       silence=None):
        '''
        The read noise, detector gain, seeing and exposure time will be 
        automatically extracted from the FITS header if it conforms with the
        IAUFWG FITS standard.

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
            (Deafult is None, which will be replaced with 0.0)
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

        if saxis is not None:

            self.saxis = saxis

            if self.saxis == 1:

                self.waxis = 0

            elif self.saxis == 0:

                self.waxis = 1

            else:

                raise ValueError(
                    "saxis can only be 0 or 1, {} is given".format(saxis))

        if spatial_mask is not None:

            self.spatial_mask = spatial_mask

        if spec_mask is not None:

            self.spec_mask = spec_mask

        if flip is not None:

            self.flip = flip

        # cosmic ray rejection
        if cosmicray is not None:

            self.cosmicray = cosmicray

            self.img = detect_cosmics(self.img,
                                      sigclip=self.cosmicray_sigma,
                                      readnoise=self.readnoise,
                                      gain=self.gain,
                                      fsmode='convolve',
                                      psfmodel='gaussy',
                                      psfsize=31,
                                      psffwhm=self.seeing)[1]

        if cosmicray_sigma is not None:

            self.cosmicray_sigma = cosmicray_sigma

        # Get the Read Noise
        if readnoise is not None:

            self.readnoise = readnoise

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

        # Get the gain
        if gain is not None:

            self.gain = gain

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

            self.seeing = seeing

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

            self.exptime = exptime

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

        if silence is not None:

            self.silence = silence

        self.img = self.img / self.exptime

        # the valid y-range of the chip (i.e. spatial direction)
        if (len(self.spatial_mask) > 1):

            if self.saxis == 1:

                self.img = self.img[self.spatial_mask]

            else:

                self.img = self.img[:, self.spatial_mask]

        # the valid x-range of the chip (i.e. spectral direction)
        if (len(self.spec_mask) > 1):

            if self.saxis == 1:

                self.img = self.img[:, self.spec_mask]

            else:

                self.img = self.img[self.spec_mask]

        # get the length in the spectral and spatial directions
        self.spec_size = np.shape(self.img)[self.saxis]
        self.spatial_size = np.shape(self.img)[self.waxis]

        if self.saxis == 0:

            self.img = np.transpose(self.img)

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
        append: boolean (default: False)
            Set to False to overwrite the current list.

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
        append: boolean (default: False)
            Set to False to overwrite the current list.

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
        append: boolean (default: False)
            Set to False to overwrite the current list.

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
        append: boolean (default: False)
            Set to False to overwrite the current list.

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

    def _identify_spectra(self, f_height, display, renderer, return_jsonstring,
                          save_iframe, filename, open_iframe):
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
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

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

        if display or save_iframe or return_jsonstring:

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

            if save_iframe:

                if filename is None:

                    pio.write_html(fig,
                                   'identify_spectra.html',
                                   auto_open=open_iframe)

                else:

                    pio.write_html(fig,
                                   filename + '.html',
                                   auto_open=open_iframe)

            # display disgnostic plot
            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if return_jsonstring:

                return fig.to_json()

    def _optimal_signal(self,
                        pix,
                        xslice,
                        sky,
                        mu,
                        sigma,
                        tol=1e-6,
                        max_iter=99,
                        forced=False,
                        variances=None):
        """
        Make sure the counts are the number of photoelectrons or an equivalent
        detector unit, and not counts per second.

        Iterate to get the optimal signal. Following the algorithm on
        Horne, 1986, PASP, 98, 609 (1986PASP...98..609H).

        Parameters
        ----------
        pix: 1-d numpy array
            pixel number along the spatial direction
        xslice: 1-d numpy array
            Count along the pix, has to be the same length as pix
        sky: 1-d numpy array
            Count of the fitted sky along the pix, has to be the same
            length as pix
        mu: float
            The center of the Gaussian
        sigma: float
            The width of the Gaussian
        tol: float
            The tolerance limit for the covergence
        max_iter: int
            The maximum number of iteration in the optimal extraction
        forced: boolean
            Forced extraction with the given weights
        variances: 2-d numpy array
            The 1/weights of used for optimal extraction, has to be the
            same length as the pix.

        Returns
        -------
        signal: float
            The optimal signal.
        noise: float
            The noise associated with the optimal signal.
        suboptimal: boolean
            List indicating whether the extraction at that pixel was
            optimal or not.  True = suboptimal, False = optimal.
        var_f: float
            Weight function used in the extraction.

        """

        # step 2 - initial variance estimates
        var1 = self.readnoise + np.abs(xslice) / self.gain

        # step 4a - extract standard spectrum
        f = xslice - sky
        f[f < 0] = 0.
        f1 = np.nansum(f)

        # step 4b - variance of standard spectrum
        v1 = 1. / np.nansum(1. / var1)

        # step 5 - construct the spatial profile
        P = self._gaus(pix, 1., 0., mu, sigma)
        P /= np.nansum(P)

        f_diff = 1
        v_diff = 1
        i = 0
        suboptimal = False

        mask_cr = np.ones(len(P), dtype=bool)

        if forced:

            var_f = variances

        while (f_diff > tol) | (v_diff > tol):

            f0 = f1
            v0 = v1

            # step 6 - revise variance estimates
            if not forced:

                var_f = self.readnoise + np.abs(P * f0 + sky) / self.gain

            # step 7 - cosmic ray mask, only start considering after the
            # 2nd iteration. 1 pixel is masked at a time until convergence,
            # once the pixel is masked, it will stay masked.
            if i > 1:

                ratio = (self.cosmicray_sigma**2. * var_f) / (f - P * f0)**2.
                outlier = np.sum(ratio > 1)

                if outlier > 0:

                    mask_cr[np.argmax(ratio)] = False

            # step 8a - extract optimal signal
            f1 = np.nansum((P * f / var_f)[mask_cr]) / \
                np.nansum((P**2. / var_f)[mask_cr])
            # step 8b - variance of optimal signal
            v1 = np.nansum(P[mask_cr]) / np.nansum((P**2. / var_f)[mask_cr])

            f_diff = abs((f1 - f0) / f0)
            v_diff = abs((v1 - v0) / v0)

            i += 1

            if i == int(max_iter):

                suboptimal = True
                break

        signal = f1
        noise = np.sqrt(v1)

        return signal, noise, suboptimal, var_f

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
                 fit_deg=3,
                 ap_faint=10,
                 display=False,
                 renderer='default',
                 width=1280,
                 height=720,
                 return_jsonstring=False,
                 save_iframe=False,
                 filename=None,
                 open_iframe=False):
        '''
        Aperture tracing by first using cross-correlation then the peaks are
        fitting with a polynomial with an order of floor(nwindow, 10) with a
        minimum order of 1. Nothing is returned unless return_jsonstring of the
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
            Minimum separation between spectra (only if there are multiple
            sources on the longslit).
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
            background sky level to the first order. [Count]
        tol: float
            Maximum allowed shift between neighbouring slices, this value is
            referring to native pixel size without the application of the
            resampling or rescaling. [pix]
        fit_deg: int
            Degree of the polynomial fit of the trace.
        ap_faint: float
            The percentile toleranceold of Count aperture to be used for fitting
            the trace. Note that this percentile is of the Count, not of the
            number of subspectra.
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

        Returns
        -------
        json string if return_jsonstring is True, otherwise only an image is displayed

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
                                     np.array(pix_resampled).reshape(-1),
                                     np.array(lines).reshape(-1),
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

            # Get the median of the subspectrum and then get the Count at the
            # centre of the aperture
            ap_val = np.zeros(nwindow)

            for j in range(nwindow):

                # rounding
                idx = int(spec_idx[i][j] + 0.5)
                ap_val[j] = np.nanmedian(img_split[j], axis=1)[idx]

            # Mask out the faintest ap_faint percentile
            mask = (ap_val > np.nanpercentile(ap_val, ap_faint))

            # fit the trace
            ap_p = np.polyfit(spec_pix[mask], spec_idx[i][mask], int(fit_deg))
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

            self.spectrum_list[i] = _spectrum1D(i)
            self.spectrum_list[i].add_trace(list(ap), [ap_sigma] * len(ap))

            self.spectrum_list[i].add_gain(self.gain)
            self.spectrum_list[i].add_readnoise(self.readnoise)
            self.spectrum_list[i].add_exptime(self.exptime)

        # Plot
        if save_iframe or display or return_jsonstring:

            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width))

            fig.add_trace(
                go.Heatmap(z=np.log10(self.img),
                           zmin=self.zmin,
                           zmax=self.zmax,
                           colorscale="Viridis",
                           colorbar=dict(title='log( e- / s )')))

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

            if save_iframe:

                if filename is None:

                    pio.write_html(fig, 'ap_trace.html', auto_open=open_iframe)

                else:

                    pio.write_html(fig,
                                   filename + '.html',
                                   auto_open=open_iframe)

            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if return_jsonstring:

                return fig.to_json()

    def add_trace(self, trace, trace_sigma, spec_id=None):
        '''
        Add user-supplied trace. The trace has to have the size as the 2D
        spectral image in the spectral direction.

        Parameters
        ----------
        trace: list or numpy.ndarray (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: list or numpy.ndarray (N)
            Standard deviation of the Gaussian profile of a trace
        spec_id: int
            The ID corresponding to the spectrum1D object

        '''

        assert isinstance(spec_id, (int, list, np.ndarray)) or (
            spec_id is
            None), 'spec_id has to be an integer, None, list or array.'

        if spec_id is None:

            if len(np.shape(trace)) == 1:

                spec_id = [0]

            elif len(np.shape(trace)) == 2:

                spec_id = list(np.arange(np.shape(trace)[0]))

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if isinstance(spec_id, np.ndarray):

            spec_id = list(spec_id)

        assert isinstance(
            trace, (list, np.ndarray)), 'trace has to be a list or an array.'
        assert isinstance(
            trace_sigma,
            (list, np.ndarray)), 'trace_sigma has to be a list or an array.'
        assert len(trace) == len(trace_sigma), 'trace and trace_sigma have to '
        'be the same size.'

        for i in spec_id:

            if i in self.spectrum_list.keys():

                self.spectrum_list[i].add_trace(trace, trace_sigma)

            else:

                self.spectrum_list[i] = _spectrum1D(i)
                self.spectrum_list[i].add_trace(trace, trace_sigma)

    def remove_trace(self, spec_id):
        '''
        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object

        '''

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
                   tolerance=1e-6,
                   max_iter=99,
                   forced=False,
                   variances=None,
                   display=False,
                   renderer='default',
                   width=1280,
                   height=720,
                   return_jsonstring=False,
                   save_iframe=False,
                   filename=None,
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

        Nothing is returned unless return_jsonstring of the plotly graph is set to be
        returned. The count, count_sky and count_err are stored as properties of the
        TwoDSpec object.

        count: 1-d array
            The summed count at each column about the trace. Note: is not
            sky subtracted!
        count_err: 1-d array
            the uncertainties of the count values
        count_sky: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract

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
        spec_id: int
            The ID corresponding to the spectrum1D object
        optimal: boolean
            Set optimal extraction. (Default is True)
        tolerance: float
            The tolerance limit for the convergence of the optimal extraction
        max_iter: float
            The maximum number of iterations before optimal extraction fails
            and return to standard tophot extraction
        forced: boolean
            To perform forced optimal extraction  by using the given aperture
            profile as it is without interation, the resulting uncertainty
            will almost certainly be wrong. This is an experimental feature.
        variances: list or numpy.ndarray
            The weight function for forced extraction. It is only used if force
            is set to True.
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

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
            count_sky = np.zeros(len_trace)
            count_err = np.zeros(len_trace)
            count = np.zeros(len_trace)
            var = np.ones((len_trace, 2 * apwidth + 1))
            suboptimal = np.zeros(len_trace, dtype=bool)

            if isinstance(apwidth, int):

                # first do the aperture count
                widthdn = apwidth
                widthup = apwidth

            elif len(apwidth) == 2:
                widthdn = apwidth[0]
                widthup = apwidth[1]

            else:

                raise TypeError(
                    'apwidth can only be an int or a list of two ints')

            if isinstance(skysep, int):

                # first do the aperture count
                sepdn = skysep
                sepup = skysep

            elif len(skysep) == 2:

                sepdn = skysep[0]
                sepup = skysep[1]

            else:

                raise TypeError(
                    'skysep can only be an int or a list of two ints')

            if isinstance(skywidth, int):

                # first do the aperture count
                skywidthdn = skywidth
                skywidthup = skywidth

            elif len(skywidth) == 2:

                skywidthdn = skywidth[0]
                skywidthup = skywidth[1]

            else:

                raise TypeError(
                    'skywidth can only be an int or a list of two ints')

            for i, pos in enumerate(self.spectrum_list[j].trace):

                itrace = int(pos)
                pix_frac = pos - itrace

                # fix width if trace is too close to the edge
                if (itrace + widthup > self.spatial_size):

                    widthup = self.spatial_size - itrace - 1

                if (itrace - widthdn < 0):

                    widthdn = itrace - 1  # i.e. starting at pixel row 1

                # simply add up the total count around the trace +/- width
                xslice = self.img[itrace - widthdn:itrace + widthup + 1, i]
                count_ap = np.sum(xslice) - pix_frac * xslice[-1] - (
                    1 - pix_frac) * xslice[0]

                if (skywidthup >= 0) or (skywidthdn >= 0):

                    # get the indexes of the sky regions
                    y0 = max(itrace - widthdn - sepdn - skywidthdn, 0)
                    y1 = max(itrace - widthdn - sepdn, 0)
                    y2 = min(itrace + widthup + sepup + 1,
                             self.spatial_size - 1)
                    y3 = min(itrace + widthup + sepup + skywidthup + 1,
                             self.spatial_size - 1)
                    y = np.append(np.arange(y0, y1), np.arange(y2, y3))
                    z = self.img[y, i]

                    if (skydeg > 0):

                        # fit a polynomial to the sky in this column
                        polyfit = np.polyfit(y, z, skydeg)
                        # define the aperture in this column
                        ap = np.arange(itrace - widthdn, itrace + widthup + 1)
                        # evaluate the polynomial across the aperture, and sum
                        count_sky_slice = np.polyval(polyfit, ap)
                        count_sky[i] = np.sum(
                            count_sky_slice) - pix_frac * count_sky_slice[
                                -1] - (1 - pix_frac) * count_sky_slice[0]

                    elif (skydeg == 0):

                        count_sky[i] = np.nanmean(z) * (len(xslice) - 1)

                    else:

                        warnings.warn('skydeg cannot be negative. sky '
                                      'background is set to zero.')
                        count_sky[i] = np.zeros(len(xslice))

                else:

                    count_sky[i] = np.zeros(len(xslice))

                # if optimal extraction
                if optimal:

                    pix = np.arange(itrace - widthdn, itrace + widthup + 1)

                    # Fit the sky background
                    if (skydeg > 0):

                        sky = np.polyval(polyfit, pix)

                    else:

                        sky = np.ones(len(pix)) * np.nanmean(z)

                    if forced:

                        if np.ndim(variances) == 0:

                            if np.isfinite(variances):

                                var_i = np.ones(len(pix)) * variances

                            else:

                                var_i = np.ones(len(pix))
                                warnings.warn('Variances are set to 1.')

                        elif np.ndim(variances) == 1:
                            if len(variances) == len(pix):

                                var_i = variances

                            elif len(variances) == len_trace:

                                var_i = np.ones(len(pix)) * variances[i]

                            else:

                                var_i = np.ones(len(pix))
                                warnings.warn('Variances are set to 1.')

                        elif np.ndim(variances) == 2:

                            var_i = variances[i]

                        else:

                            var_i = np.ones(len(pix))
                            warnings.warn('Variances are set to 1.')

                    else:

                        var_i = None

                    # Get the optimal signals
                    # pix is the native pixel position
                    # pos is the trace at the native pixel position
                    count[i], count_err[i], suboptimal[i], var[
                        i] = self._optimal_signal(
                            pix=pix,
                            xslice=xslice * self.exptime,
                            sky=sky * self.exptime,
                            mu=pos,
                            sigma=self.spectrum_list[j].trace_sigma[i],
                            tol=tolerance,
                            max_iter=max_iter,
                            forced=forced,
                            variances=var_i)
                    count[i] /= self.exptime
                    count_err[i] /= self.exptime

                else:

                    #-- finally, compute the error in this pixel
                    sigB = np.nanstd(
                        z) * self.exptime  # standarddev in the background data
                    nB = len(y)  # number of bkgd pixels
                    nA = widthdn + widthup + 1  # number of aperture pixels

                    # Based on aperture phot err description by F. Masci, Caltech:
                    # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
                    # All the counts are in per second already, so need to
                    count[i] = count_ap - count_sky[i]
                    count_err[i] = np.sqrt(count[i] * self.exptime /
                                           self.gain + (nA + nA**2. / nB) *
                                           (sigB**2.)) / self.exptime

            self.spectrum_list[j].add_aperture(widthdn, widthup, sepdn, sepup,
                                               skywidthdn, skywidthup)
            self.spectrum_list[j].add_count(list(count), list(count_err),
                                            list(count_sky))
            self.spectrum_list[j].add_variances(var)
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
                        'Signal extracted is likely to be suboptimal, please '
                        'try a longer iteration, larger tolerance or revert '
                        'to top-hat extraction.')

            if save_iframe or display or return_jsonstring:

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
                        colorbar=dict(title='log( e- / s )')))

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
                               y=count / count_err,
                               xaxis='x2',
                               yaxis='y3',
                               line=dict(color='slategrey'),
                               name='Signal-to-Noise Ratio'))

                # extrated source, sky and uncertainty
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=count_sky,
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='firebrick'),
                               name='Sky count / (e- / s)'))
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=count_err,
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='orange'),
                               name='Uncertainty count / (e- / s)'))
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=count,
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='royalblue'),
                               name='Target count / (e- / s)'))

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
                                    sigma_clip(count, sigma=5., masked=False)),
                                np.nanmin(
                                    sigma_clip(count_err,
                                               sigma=5.,
                                               masked=False)),
                                np.nanmin(
                                    sigma_clip(count_sky,
                                               sigma=5.,
                                               masked=False)), 1),
                            max(np.nanmax(count), np.nanmax(count_sky))
                        ],
                        zeroline=False,
                        domain=[0, 0.5],
                        showgrid=True,
                        title='Count / s',
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

                if save_iframe:

                    if filename is None:

                        pio.write_html(fig,
                                       'ap_extract_' + str(j) + '.html',
                                       auto_open=open_iframe)

                    else:

                        pio.write_html(fig,
                                       filename + '_' + str(j) + '.html',
                                       auto_open=open_iframe)

                if display:

                    if renderer == 'default':

                        fig.show()

                    else:

                        fig.show(renderer)

                if return_jsonstring:

                    return fig.to_json()

    def save_fits(self,
                  output='trace+count',
                  filename='TwoDSpecExtracted',
                  force=False,
                  overwrite=False,
                  empty_primary_hdu=True):
        '''
        Save the reduced image to disk.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDU
                Pixel position of the trace in the spatial direction
                and the best fit gaussian line spread function sigma
            count: 3 HDUs
                Flux, uncertainty and sky (bin width = per wavelength)
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        overwrite: boolean
            Default is False.
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)

        '''

        filename = os.path.splitext(filename)[0]

        for i in output.split('+'):

            if i not in ['trace', 'count']:

                raise ValueError('%s is not a valid output.' % i)

        # Save each trace as a separate FITS file
        for i in range(len(self.spectrum_list)):

            filename_i = filename + '_' + output + '_' + str(i)

            self.spectrum_list[i].save_fits(
                output=output,
                filename=filename_i,
                force=force,
                overwrite=overwrite,
                empty_primary_hdu=empty_primary_hdu)


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
            if arc.saxis == 1:
                self.arc = arc.arc_master
            else:
                self.arc = np.transpose(arc.arc_master)
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
        spec_id: int
            The ID corresponding to the spectrum1D object

        '''

        if spec_id in list(self.spectrum_list.keys()):

            if self.spectrum_list[spec_id].count is not None:

                warnings.warn('The given spec_id is in use already, the given '
                              'peaks will overwrite the current data.')

        if spec_id is None:

            # Add to the first spec
            spec_id = 0

        if len(self.spectrum_list.keys()) == 0:

            self.spectrum_list[0] = _spectrum1D(0)

        self.spectrum_list[spec_id].peaks = peaks

    def add_arc_spec(self, arc_spec, spec_id=None):
        '''
        Provide the collapsed 1D spectrum/a of the arc image.

        Parameters
        ----------
        arc_spec: list of list or list of arrays
            The Count/flux of the 1D arc spectrum/a. Multiple spectrum/a
            can be provided as list of list or list of arrays.
        spec_id: int
            The ID corresponding to the spectrum1D object

        '''

        if spec_id in list(self.spectrum_list.keys()):

            if self.spectrum_list[spec_id].count is not None:

                warnings.warn('The given spec_id is in use already, the given '
                              'arc_spec will overwrite the current data.')

        if spec_id is None:

            # Add to the first spec
            spec_id = 0

        if len(self.spectrum_list.keys()) == 0:

            self.spectrum_list[0] = _spectrum1D(0)

        self.spectrum_list[spec_id].arc_spec = list(arc_spec)

    def add_spec(self, count, count_err=None, count_sky=None, spec_id=None):
        '''
        To provide user-supplied extracted spectrum for wavelegth calibration.

        Parameters
        ----------
        count: 1-d array
            The summed count at each column about the trace. Note: is not
            sky subtracted!
        count_err: 1-d array
            the uncertainties of the count values
        count_sky: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        spec_id: int
            The ID corresponding to the spectrum1D object

        '''

        if spec_id in list(self.spectrum_list.keys()):

            if self.spectrum_list[spec_id].count is not None:
                warnings.warn(
                    'The given spec_id is in use already, the given '
                    'count, count_err and count_sky will overwrite the '
                    'current data.')

        if spec_id is None:

            # Add to the first spec
            spec_id = 0

        if len(self.spectrum_list.keys()) == 0:

            self.spectrum_list[0] = _spectrum1D(0)

        self.spectrum_list[spec_id].add_count(count, count_err, count_sky)

    def remove_spec(self, spec_id):
        '''
        To modify or append a spectrum with the user-supplied one, one at a
        time.

        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object

        '''

        if spec_id not in list(self.spectrum_list.keys()):

            raise ValueError('The given spec_id does not exist.')

        self.spectrum_list[spec_id].remove_count()

    def add_trace(self, trace, trace_sigma, spec_id=None, pixel_list=None):
        '''
        To provide user-supplied trace. The trace has to be the size as the 2D
        spectral image in the spectral direction. Make sure the trace pixels
        are corresponding to the arc image, left (small index) is blue, right
        (large index) is red.

        Parameters
        ----------
        trace: 1D numpy array of list of 1D numpy array (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: 1D numpy array of list of 1D numpy array (N)
            Standard deviation of the Gaussian profile of a trace
        spec_id: int
            The ID corresponding to the spectrum1D object
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

        if len(self.spectrum_list.keys()) == 0:

            self.spectrum_list[0] = _spectrum1D(0)

        self.spectrum_list[spec_id].add_trace(trace, trace_sigma, pixel_list)

    def remove_trace(self, spec_id):
        '''
        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object

        '''

        if spec_id not in list(self.spectrum_list.keys()):

            raise ValueError('The given spec_id does not exist.')

        self.spectrum_list[spec_id].remove_trace()

    def add_fit_coeff(self, spec_id, fit_type, fit_coeff):
        '''
        To provide the polynomial coefficients and polynomial type for science,
        standard or both. Science stype can provide multiple traces. Standard
        stype can only accept one trace.

        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object
        fit_coeff: list or list of list
            Polynomial fit coefficients.
        fit_type: str or list of str
            Strings starting with 'poly', 'leg' or 'cheb' for polynomial,
            legendre and chebyshev fits. Case insensitive.

        '''

        if isinstance(fit_type, str):
            fit_type = [fit_type]

        if spec_id in list(self.spectrum_list.keys()):

            if self.spectrum_list[spec_id].fit_coeff is not None:

                warnings.warn('The given spec_id is in use already, the given '
                              'fit_coeff and fit_type will overwrite the '
                              'current data.')

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if spec_id is None:

            # Add to the first spec
            spec_id = list(np.arange(len(fit_type)))

        if len(self.spectrum_list.keys()) == 0:

            self.spectrum_list[0] = _spectrum1D(0)

        for i, s in enumerate(spec_id):

            if len(np.array(fit_type)) > 1:

                self.spectrum_list[s].add_fit_type(fit_type[i])
                self.spectrum_list[s].add_fit_coeff(fit_coeff[i])

            else:
                self.spectrum_list[s].add_fit_type(fit_type[0])
                self.spectrum_list[s].add_fit_coeff(fit_coeff)

    def from_twodspec(self, twodspec, pixel_list=None):
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
        pixel_list: list or numpy array
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(len(spec.trace[0])), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]

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

    def apply_twodspec_mask_to_arc(self):
        '''
        *EXPERIMENTAL*
        Apply both the spec_mask and spatial_mask that are already stroed in
        the object.

        '''

        if self.saxis == 0:

            self.arc = np.transpose(self.arc)

        if self.flip:

            self.arc = np.flip(self.arc)

        self.apply_spec_mask_to_arc(self.spec_mask)
        self.apply_spatial_mask_to_arc(self.spatial_mask)

    def apply_spec_mask_to_arc(self, spec_mask):
        '''
        *EXPERIMENTAL*
        Apply to use only the valid x-range of the chip (i.e. dispersion
        direction)

        parameters
        ----------
        spec_mask: 1D numpy array (M)
            Mask in the spectral direction, can be the indices of the pixels
            to be included (size <M) or a 1D numpy array of True/False (size M)
            (Default is (1,) i.e. keep everything)

        '''

        if (len(spec_mask) > 1):

            if self.saxis == 1:

                self.arc = self.arc[:, spec_mask]

            else:

                self.arc = self.arc[spec_mask]

    def apply_spatial_mask_to_arc(self, spatial_mask):
        '''
        *EXPERIMENTAL*
        Apply to use only the valid y-range of the chip (i.e. spatial
        direction)

        parameters
        ----------
        spatial_mask: 1D numpy array (N)
            Mask in the spatial direction, can be the indices of the pixels
            to be included (size <N) or a 1D numpy array of True/False (size N)
            (Default is (1,) i.e. keep everything)

        '''

        if (len(spatial_mask) > 1):

            if self.saxis == 1:

                self.arc = self.arc[spatial_mask]

            else:

                self.arc = self.arc[:, spatial_mask]

    def extract_arc_spec(self,
                         spec_id=None,
                         display=False,
                         return_jsonstring=False,
                         renderer='default',
                         width=1280,
                         height=720,
                         save_iframe=False,
                         filename=None,
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
        spec_id: int
            The ID corresponding to the spectrum1D object
        display: boolean
            Set to True to display disgnostic plot.
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

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
                'or with from_twodspec() before executing find_arc_lines().')

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
            if save_iframe or display or return_jsonstring:

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
                                             title='e- / s'),
                                  hovermode='closest',
                                  showlegend=False)

                if save_iframe:

                    if filename is None:

                        pio.write_html(fig,
                                       'arc_spec_' + str(i) + '.html',
                                       auto_open=open_iframe)

                    else:

                        pio.write_html(fig,
                                       filename + '_' + str(i) + '.html',
                                       auto_open=open_iframe)

                if display:

                    if renderer == 'default':

                        fig.show()

                    else:

                        fig.show(renderer)

                if return_jsonstring:

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
                       renderer='default',
                       width=1280,
                       height=720,
                       return_jsonstring=False,
                       save_iframe=False,
                       filename=None,
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
        spec_id: int
            The ID corresponding to the spectrum1D object
        background: int
            User-supplied estimated background level
        percentile: float
            The percentile of the flux to be used as the estimate of the
            background sky level to the first order. Only used if background
            is None. [Count]
        prominence: float
            The minimum prominence to be considered as a peak
        distance: float
            Minimum separation between peaks
        refine: boolean
            Set to true to fit a gaussian to get the peak at sub-pixel
            precision
        refine_window_width: boolean
            The number of pixels (on each side of the existing peaks) to be
            fitted with gaussian profiles over.
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True

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
                'or with from_twodspec() before executing find_arc_lines().')

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

            if save_iframe or display or return_jsonstring:

                fig = go.Figure(
                    layout=dict(autosize=False, height=height, width=width))

                # show the image on the top
                fig.add_trace(
                    go.Heatmap(x=np.arange(self.arc.shape[0]),
                               y=np.arange(self.arc.shape[1]),
                               z=np.log10(self.arc),
                               colorscale="Viridis",
                               colorbar=dict(title='log( e- / s )')))

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

                if save_iframe:

                    if filename is None:

                        pio.write_html(fig,
                                       'arc_lines_' + str(i) + '.html',
                                       auto_open=open_iframe)

                    else:

                        pio.write_html(fig,
                                       filename + '_' + str(i) + '.html',
                                       auto_open=open_iframe)

                if display:

                    if renderer == 'default':

                        fig.show()

                    else:

                        fig.show(renderer)

                if return_jsonstring:

                    return fig.to_json()

    def initialise_calibrator(self, spec_id=None, peaks=None, spectrum=None):
        '''
        Initialise a RASCAL calibrator.

        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object
        peaks: list (default: None)
            The pixel values of the peaks (start from zero)
        spectrum: list
            The spectral intensity as a function of pixel.

        '''

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

            if spectrum is None:

                spectrum = self.spectrum_list[i].count

            self.spectrum_list[i].calibrator = Calibrator(peaks=peaks,
                                                          spectrum=spectrum)

    def set_calibrator_properties(self,
                                  spec_id=None,
                                  num_pix=None,
                                  pixel_list=None,
                                  plotting_library='plotly',
                                  log_level='info'):
        '''
        Set the properties of the calibrator.

        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object
        num_pix: int (default: None)
            The number of pixels in the dispersion direction
        pixel_list: list or numpy array (default: None)
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(num_pix), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        plotting_library : string (default: 'matplotlib')
            Choose between matplotlib and plotly.
        log_level : string (default: 'info')
            Choose {critical, error, warning, info, debug, notset}.

        '''

        if spec_id is not None:

            if spec_id not in list(self.spectrum_list.keys()):

                raise ValueError('The given spec_id does not exist.')

        else:

            # if spec_id is None, calibrators are initialised to all
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        for i in spec_id:

            if num_pix is None:

                num_pix = self.spectrum_list[i].len_trace

            if pixel_list is None:

                pixel_list = self.spectrum_list[i].pixel_list

            self.spectrum_list[i].calibrator.set_calibrator_properties(
                num_pix=num_pix,
                pixel_list=pixel_list,
                plotting_library=plotting_library,
                log_level=log_level)

            self.spectrum_list[i].add_calibrator_properties(
                num_pix, pixel_list, plotting_library, log_level)

    def set_hough_properties(self,
                             spec_id=None,
                             num_slopes=5000,
                             xbins=500,
                             ybins=500,
                             min_wavelength=3000,
                             max_wavelength=9000,
                             range_tolerance=500,
                             linearity_tolerance=50):
        '''
        Set the properties of the hough transform.

        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object
        num_slopes: int (default: 1000)
            Number of slopes to consider during Hough transform
        xbins: int (default: 50)
            Number of bins for Hough accumulation
        ybins: int (default: 50)
            Number of bins for Hough accumulation
        min_wavelength: float (default: 3000)
            Minimum wavelength of the spectrum.
        max_wavelength: float (default: 9000)
            Maximum wavelength of the spectrum.
        range_tolerance: float (default: 500)
            Estimation of the error on the provided spectral range
            e.g. 3000-5000 with tolerance 500 will search for
            solutions that may satisfy 2500-5500
        linearity_tolerance: float (default: 100)
            A toleranceold (Ansgtroms) which defines some padding around the
            range tolerance to allow for non-linearity. This should be the
            maximum expected excursion from linearity.

        '''

        if spec_id is not None:

            if spec_id not in list(self.spectrum_list.keys()):

                raise ValueError('The given spec_id does not exist.')

        else:

            # if spec_id is None, calibrators are initialised to all
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        for i in spec_id:

            self.spectrum_list[i].calibrator.set_hough_properties(
                num_slopes=num_slopes,
                xbins=xbins,
                ybins=ybins,
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                range_tolerance=range_tolerance,
                linearity_tolerance=linearity_tolerance)

            self.spectrum_list[i].add_hough_properties(num_slopes, xbins,
                                                       ybins, min_wavelength,
                                                       max_wavelength,
                                                       range_tolerance,
                                                       linearity_tolerance)

    def set_ransac_properties(self,
                              spec_id=None,
                              sample_size=5,
                              top_n_candidate=5,
                              linear=True,
                              filter_close=False,
                              ransac_tolerance=5,
                              candidate_weighted=True,
                              hough_weight=1.0):
        '''
        Set the properties of the RANSAC process.

        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object
        sample_size: int (default: 5)
            Number of pixel-wavelength hough pairs to be used for each arc line
            being picked.
        top_n_candidate: int (default: 5)
            Top ranked lines to be fitted.
        linear: boolean (default: True)
            True to use the hough transformed gradient, otherwise, use the
            known polynomial.
        filter_close: boolean (default: False)
            Remove the pairs that are out of bounds in the hough space.
        ransac_tolerance: float (default: 1)
            The distance criteria  (Angstroms) to be considered an inlier to a
            fit. This should be close to the size of the expected residuals on
            the final fit (e.g. 1A is typical)
        candidate_weighted: boolean (default: True)
            Set to True to down-weight pairs that are far from the fit.
        hough_weight: float or None (default: 1.0)
            Set to use the hough space to weigh the fit. The theoretical
            optimal weighting is unclear. The larger the value, the heavily it
            relies on the overdensity in the hough space for a good fit.

        '''

        if spec_id is not None:

            if spec_id not in list(self.spectrum_list.keys()):

                raise ValueError('The given spec_id does not exist.')

        else:

            # if spec_id is None, calibrators are initialised to all
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        for i in spec_id:

            self.spectrum_list[i].calibrator.set_ransac_properties(
                sample_size=sample_size,
                top_n_candidate=top_n_candidate,
                linear=linear,
                filter_close=filter_close,
                ransac_tolerance=ransac_tolerance,
                candidate_weighted=candidate_weighted,
                hough_weight=hough_weight)

            self.spectrum_list[i].add_ransac_properties(
                sample_size, top_n_candidate, linear, filter_close,
                ransac_tolerance, candidate_weighted, hough_weight)

    def do_hough_transform(self, spec_id):
        '''
        Perform Hough transform on the pixel-wavelength pairs with the
        configuration set by the set_hough_properties().

        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object

        '''

        if spec_id is not None:

            if spec_id not in list(self.spectrum_list.keys()):

                raise ValueError('The given spec_id does not exist.')

        else:

            # if spec_id is None, calibrators are initialised to all
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        for i in spec_id:

            self.spectrum_list[i].calibrator.do_hough_transform()

    def set_known_pairs(self, spec_id=None, pix=None, wave=None):
        '''
        Provide manual pixel-wavelength pair(s), they will be appended to the
        list of pixel-wavelength pairs after the random sample being drawn from
        the RANSAC step, i.e. they are ALWAYS PRESENT in the fitting step. Use
        with caution because it can skew or bias the fit significantly, make
        sure the pixel value is accurate to at least 1/10 of a pixel.

        This can be used, for example, for low intensity lines at the edge of
        the spectrum.

        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object
        pix : numeric value, list or numpy 1D array (N) (default: None)
            Any pixel value, can be outside the detector chip and
            serve purely as anchor points.
        wave : numeric value, list or numpy 1D array (N) (default: None)
            The matching wavelength for each of the pix.

        '''

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

    def load_user_atlas(self,
                        elements,
                        wavelengths,
                        intensities=None,
                        constrain_poly=False,
                        vacuum=False,
                        pressure=101325.,
                        temperature=273.15,
                        relative_humidity=0.,
                        spec_id=None):
        '''
        *Remove* all the arc lines loaded to the Calibrator and then use the user
        supplied arc lines instead.

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements : list
            Element (required). Preferably a standard (i.e. periodic table)
            name for convenience with built-in atlases
        wavelengths : list
            Wavelength to add (Angstrom)
        intensities : list
            Relative line intensities
        constrain_poly : boolean
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: boolean
            Set to true to convert the input wavelength to air-wavelengths
            based on the given pressure, temperature and humidity.
        pressure: float
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float
            In percentage.
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object

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
                  min_atlas_wavelength=1000.,
                  max_atlas_wavelength=30000.,
                  min_intensity=10.,
                  min_distance=10.,
                  candidate_tolerance=10.,
                  constrain_poly=False,
                  vacuum=False,
                  pressure=101325.,
                  temperature=273.15,
                  relative_humidity=0.,
                  spec_id=None):
        '''
        Adds an atlas of arc lines to the calibrator, given an element.

        Arc lines are taken from a general list of NIST lines and can be filtered
        using the minimum relative intensity (note this may not be accurate due to
        instrumental effects such as detector response, dichroics, etc) and
        minimum line separation.

        Lines are filtered first by relative intensity, then by separation. This
        is to improve robustness in the case where there is a strong line very
        close to a weak line (which is within the separation limit).

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements: string or list of strings
            Chemical symbol, case insensitive
        min_atlas_wavelength: float (default: None)
            Minimum wavelength of the arc lines.
        max_atlas_wavelength: float (default: None)
            Maximum wavelength of the arc lines.
        min_intensity: float (default: None)
            Minimum intensity of the arc lines. Refer to NIST for the intensity.
        min_distance: float (default: None)
            Minimum separation between neighbouring arc lines.
        candidate_tolerance: float (default: 10)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: boolean
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: boolean
            Set to True if the light path from the arc lamb to the detector
            plane is entirely in vacuum.
        pressure: float
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float
            In percentage.
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object

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
                min_intensity=min_intensity,
                min_distance=min_distance,
                candidate_tolerance=candidate_tolerance,
                constrain_poly=constrain_poly,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity)

            self.spectrum_list[i].add_atlas_wavelength_range(
                min_atlas_wavelength, max_atlas_wavelength)

            self.spectrum_list[i].add_min_atlas_intensity(min_intensity)

            self.spectrum_list[i].add_min_atlas_distance(min_distance)

            self.spectrum_list[i].add_weather_condition(
                pressure, temperature, relative_humidity)

    def fit(self,
            spec_id=None,
            max_tries=5000,
            fit_deg=4,
            fit_coeff=None,
            fit_tolerance=10.,
            fit_type='poly',
            brute_force=False,
            progress=True,
            display=False,
            savefig=False,
            filename=None):
        '''
        A wrapper function to perform wavelength calibration with RASCAL. As of
        14 January 2020, it supports He, Ne, Ar, Cu, Kr, Cd, Xe, Hg and Th from
        `NIST <https://physics.nist.gov/PhysRefData/ASD/lines_form.html>`_.

        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object.
        max_tries: int
            Number of trials of polynomial fitting.
        fit_deg: int (default: 4)
            The degree of the polynomial to be fitted.
        fit_coeff: list (default: None)
            *NOT CURRENTLY USED*
            Set the baseline of the least square fit. If no fits outform this
            set of polynomial coefficients, this will be used as the best fit.
        fit_tolerance: float (default: 10)
            Sets a tolerance on whether a fit found by RANSAC is considered
            acceptable
        fit_type: string (default: 'poly')
            One of 'poly', 'legendre' or 'chebyshev'
        brute_force: boolean (default: False)
            Set to True to try all possible combination in the given parameter
            space
        progress: boolean
            Set to show the progress using tdqm (if imported).
        display: boolean
            Set to show diagnostic plot.
        savefig: string
            Set to save figure.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.

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

            fit_coeff, rms, residual, peak_utilisation = self.spectrum_list[
                i].calibrator.fit(max_tries=max_tries,
                                  fit_deg=fit_deg,
                                  fit_coeff=None,
                                  fit_tolerance=fit_tolerance,
                                  fit_type=fit_type,
                                  brute_force=brute_force,
                                  progress=progress)

            self.spectrum_list[i].add_fit_type(fit_type)

            self.spectrum_list[i].add_fit_output_rascal(
                fit_coeff, rms, residual, peak_utilisation)

            if display:

                self.spectrum_list[i].calibrator.plot_fit(
                    self.spectrum_list[i].arc_spec,
                    fit_coeff=fit_coeff,
                    plot_atlas=True,
                    log_spectrum=False,
                    tolerance=1.0,
                    savefig=savefig,
                    filename=filename)

    def refine_fit(self,
                   spec_id=None,
                   fit_coeff=None,
                   n_delta=None,
                   refine=True,
                   tolerance=10.,
                   method='Nelder-Mead',
                   convergence=1e-6,
                   robust_refit=True,
                   fit_deg=None,
                   display=False,
                   savefig=False,
                   filename=None):
        '''
        *EXPERIMENTAL*
        A wrapper function to fine tune wavelength calibration with RASCAL
        when there is already a set of good coefficienes.

        Refine the polynomial fit coefficients. Recommended to use in it
        multiple calls to first refine the lowest order and gradually increase
        the order of coefficients to be included for refinement. This is be
        achieved by providing delta in the length matching the number of the
        lowest degrees to be refined.

        Set refine to True to improve on the polynomial solution.

        Set robust_refit to True to fit all the detected peaks with the
        given polynomial solution for a fit using maximal information, with
        the degree of polynomial = fit_deg.

        Set both refine and robust_refit to False will return the list of
        arc lines are well fitted by the current solution within the
        tolerance limit provided.

        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object
        fit_coeff : list
            List of polynomial fit coefficients.
        n_delta : int (default: None)
            The number of the highest polynomial order to be adjusted
        refine : boolean (default: True)
            Set to True to refine solution.
        tolerance : float (default: 10.)
            Absolute difference between fit and model in the unit of nm.
        method : string (default: 'Nelder-Mead')
            scipy.optimize.minimize method.
        convergence : float (default: 1e-6)
            scipy.optimize.minimize tol.
        robust_refit : boolean (default: True)
            Set to True to fit all the detected peaks with the given polynomial
            solution.
        fit_deg : int (default: length of the input coefficients)
            Order of polynomial fit with all the detected peaks.
        display: boolean
            Set to show diagnostic plot.
        savefig: string
            Set to save figure.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.

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

            if fit_coeff is None:

                fit_coeff = self.spectrum_list[i].fit_coeff

            if fit_deg is None:

                fit_deg = len(fit_coeff) - 1

            if n_delta is None:

                n_delta = len(fit_coeff) - 1

            fit_coeff_new, _, _, residual, peak_utilisation = self.spectrum_list[
                i].calibrator.match_peaks(fit_coeff,
                                          n_delta=n_delta,
                                          refine=refine,
                                          tolerance=tolerance,
                                          method=method,
                                          convergence=convergence,
                                          robust_refit=robust_refit,
                                          fit_deg=fit_deg)
            rms = np.sqrt(np.nanmean(residual**2.))

            if display:

                self.spectrum_list[i].calibrator.plot_fit(
                    self.spectrum_list[i].arc_spec,
                    fit_coeff_new,
                    plot_atlas=True,
                    log_spectrum=False,
                    tolerance=1.0,
                    savefig=savefig,
                    filename=filename)

            self.spectrum_list[i].add_fit_output_refine(
                fit_coeff_new, rms, residual, peak_utilisation)

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
        spec_id: int
            The ID corresponding to the spectrum1D object
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

            fit_type = spec.fit_type

            # Adjust for pixel shift due to chip gaps
            wave = self.polyval[fit_type](np.array(spec.pixel_list),
                                          spec.fit_coeff).reshape(-1)

            # compute the new equally-spaced wavelength array
            if wave_bin is None:

                wave_bin = np.nanmedian(np.ediff1d(wave))

            if wave_start is None:

                wave_start = wave[0]

            if wave_end is None:

                wave_end = wave[-1]

            wave_resampled = np.arange(wave_start, wave_end, wave_bin)

            # apply the flux calibration and resample
            count_resampled = spectres(np.array(wave_resampled).reshape(-1),
                                       np.array(wave).reshape(-1),
                                       np.array(spec.count).reshape(-1),
                                       verbose=False)

            if spec.count_err is not None:

                count_err_resampled = spectres(
                    np.array(wave_resampled).reshape(-1),
                    np.array(wave).reshape(-1),
                    np.array(spec.count_err).reshape(-1),
                    verbose=False)

            if spec.count_sky is not None:

                count_sky_resampled = spectres(
                    np.array(wave_resampled).reshape(-1),
                    np.array(wave).reshape(-1),
                    np.array(spec.count_sky).reshape(-1),
                    verbose=False)

            spec.add_wavelength(wave)
            spec.add_wavelength_resampled(wave_bin, wave_start, wave_end,
                                          wave_resampled)
            spec.add_count_resampled(count_resampled, count_err_resampled,
                                     count_sky_resampled)

    def create_fits(self,
                    spec_id=None,
                    output='arc_spec+wavecal+count_resampled',
                    force=False,
                    empty_primary_hdu=True,
                    return_id=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        return_id: boolean (default: False)
            Set to True to return the set of spec_id

        '''

        # Split the string into strings
        for i in output.split('+'):

            if i not in [
                    'trace', 'count', 'arc_spec', 'wavecal', 'wavelegth',
                    'count_resampled'
            ]:

                raise ValueError('%s is not a valid output.' % i)

        if spec_id is not None:

            if spec_id not in list(self.spectrum_list.keys()):

                raise ValueError('The given spec_id does not exist.')

        else:

            # if spec_id is None, contraints are applied to all calibrators
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        for i in spec_id:

            self.spectrum_list[i].create_fits(
                output=output,
                force=force,
                empty_primary_hdu=empty_primary_hdu)

        if return_id:

            return spec_id

    def save_fits(self,
                  spec_id=None,
                  output='arc_spec+wavecal+count_resampled',
                  filename='wavecal',
                  force=False,
                  empty_primary_hdu=True,
                  overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        overwrite: boolean
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # This create the FITS and do all the checks. The save_fits() below
        # will not re-create the FITS, and a warning will be given which
        # can be ignored.
        spec_id = self.create_fits(spec_id=spec_id,
                                   output=output,
                                   empty_primary_hdu=empty_primary_hdu,
                                   force=force,
                                   return_id=True)

        for i in spec_id:

            filename_i = filename + '_' + str(i)

            self.spectrum_list[i].save_fits(
                output=output,
                filename=filename_i,
                force=False,
                overwrite=overwrite,
                empty_primary_hdu=empty_primary_hdu)

    def save_csv(self,
                 spec_id=None,
                 output='arc_spec+wavecal+count_resampled',
                 filename='wavecal',
                 force=False,
                 overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        overwrite: boolean
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # This create the FITS and do all the checks. The save_fits() below
        # will not re-create the FITS, and a warning will be given which
        # can be ignored.
        spec_id = self.create_fits(spec_id=spec_id,
                                   output=output,
                                   empty_primary_hdu=False,
                                   force=force,
                                   return_id=True)

        for i in spec_id:

            filename_i = filename + '_' + str(i)

            self.spectrum_list[i].save_csv(output=output,
                                           filename=filename_i,
                                           force=Flase,
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
            open(
                pkg_resources.resource_filename(
                    'aspired', 'standards/lib_to_uname.json')))
        self.uname_to_lib = json.load(
            open(
                pkg_resources.resource_filename(
                    'aspired', 'standards/uname_to_lib.json')))

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

        Parameters
        ----------
        target: str
            Name of the standard star
        cutoff: float (default: 0.4)
            The similarity toleranceold [0 (completely different) - 1 (identical)]

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


        Parameters
        ----------
        target: string
            Name of the standard star
        library: string
            Name of the library of standard star
        ftype: string
            'flux' or 'mag'
        cutoff: float
            The toleranceold for the word similarity in the range of [0, 1].

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
                         display=True,
                         renderer='default',
                         width=1280,
                         height=720,
                         return_jsonstring=False,
                         save_iframe=False,
                         filename=None,
                         open_iframe=False):
        '''
        Display the standard star plot.

        Parameters
        ----------
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

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

        if save_iframe:

            if filename is None:

                pio.write_html(fig, 'standard.html', auto_open=open_iframe)

            else:

                pio.write_html(fig, filename + '.html', auto_open=open_iframe)

        if display:

            if renderer == 'default':

                fig.show()

            else:

                fig.show(renderer)

        if return_jsonstring:

            return fig.to_json()


class FluxCalibration(StandardLibrary):
    def __init__(self, silence=False):
        '''
        Initialise a FluxCalibration object.

        Parameters
        ----------
        silence: boolean
            Set to True to suppress all verbose warnings.

        '''

        # Load the dictionary
        super().__init__()
        self.science_imported = False
        self.standard_imported = False
        self.flux_science_calibrated = False
        self.flux_standard_calibrated = False

        self.spectrum_list_science = {}
        self.spectrum_list_standard = {}

    def add_spec(self,
                 count,
                 spec_id=None,
                 count_err=None,
                 count_sky=None,
                 stype='science+standard'):
        '''
        Add spectrum (count, count_err & count_sky) one at a time.

        Parameters
        ----------
        count: 1-d array
            The summed count at each column about the trace.
        spec_id: int
            The ID corresponding to the spectrum1D object
        count_err: 1-d array
            the uncertainties of the count values
        count_sky: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'standard' in stype_split:

            if len(self.spectrum_list_standard.keys()) == 0:

                self.spectrum_list_standard[0] = _spectrum1D(0)

            if spec_id in list(self.spectrum_list_standard.keys()):

                if self.spectrum_list_standard[spec_id].count is not None:
                    warnings.warn(
                        'The given spec_id is in use already, the given '
                        'trace, trace_sigma and pixel_list will overwrite the '
                        'current data.')

            if spec_id is None:

                # If spectrum_list is not empty
                if not self.spectrum_list_standard:

                    spec_id = max(self.spectrum_list_standard.keys()) + 1

                else:

                    spec_id = 0

            self.spectrum_list_standard[spec_id].add_count(
                count, count_err, count_sky)

        elif 'science' in stype_split:

            if len(self.spectrum_list_science.keys()) == 0:

                self.spectrum_list_science[0] = _spectrum1D(0)

            if spec_id in list(self.spectrum_list_science.keys()):

                if self.spectrum_list_science[spec_id].count is not None:
                    warnings.warn(
                        'The given spec_id is in use already, the given '
                        'trace, trace_sigma and pixel_list will overwrite the '
                        'current data.')

            if spec_id is None:

                # If spectrum_list is not empty
                if not self.spectrum_list_science:

                    spec_id = max(self.spectrum_list_science.keys()) + 1

                else:

                    spec_id = 0

            self.spectrum_list_science[spec_id].add_count(
                count, count_err, count_sky)

        else:

            if stype not in ['science', 'standard']:

                raise ValueError('Unknown stype, please choose from '
                                 '(1) science; and/or (2) standard.')

    def add_wavelength(self,
                       wave=None,
                       wave_resampled=None,
                       spec_id=None,
                       stype='science+standard'):
        '''
        Add wavelength (must be same length as the spectrum), one at a time.

        Parameters
        ----------
        wave : numeric value, list or numpy 1D array (N) (default: None)
            The wavelength of each pixels of the spectrum.
        wave_resampled:
            The resampled wavelength of each pixels of the spectrum.
        spec_id: int
            The ID corresponding to the spectrum1D object
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if (wave is None) & (wave_resampled is None):
            raise ValueError('wave and wave_resampled cannot be None at '
                             'at the same time.')

        elif wave_resampled is None:
            wave_resampled = wave

        elif wave is None:
            wave = wave_resampled

        stype_split = stype.split('+')

        if 'standard' in stype_split:

            if len(self.spectrum_list_standard.keys()) == 0:
                self.spectrum_list_standard[0] = _spectrum1D(0)

            spec = self.spectrum_list_standard[0]

            spec.add_wavelength(wave)
            spec.add_wavelength_resampled(
                wave_bin=np.nanmedian(np.array(np.ediff1d(wave_resampled))),
                wave_start=wave_resampled[0],
                wave_end=wave_resampled[-1],
                wave_resampled=wave_resampled,
            )

        elif 'science' in stype_split:

            if spec_id is not None:
                if spec_id not in list(self.spectrum_list_science.keys()):
                    warnings.warn(
                        'The spec_id provided is not in the '
                        'spectrum_list_science, new _spectrum1D with the ID '
                        'is created.')
                    self.spectrum_list_science[spec_id] = _spectrum1D(spec_id)
            else:
                if not self.spectrum_list_science:
                    self.spectrum_list_science[0] = _spectrum1D(0)
                spec_id = list(self.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            for i, s in enumerate(spec_id):

                spec = self.spectrum_list_science[s]

                if (len(np.shape(np.array(wave))) == 2):

                    spec.add_wavelength(wave[i])

                else:

                    spec.add_wavelength(wave)

                if (len(np.shape(np.array(wave))) == 2):

                    spec.add_wavelength_resampled(
                        wave_bin=np.nanmedian(
                            np.array(np.ediff1d(wave_resampled[i]))),
                        wave_start=wave_resampled[i][0],
                        wave_end=wave_resampled[i][-1],
                        wave_resampled=wave_resampled[i],
                    )

                else:

                    spec.add_wavelength_resampled(
                        wave_bin=np.nanmedian(
                            np.array(np.ediff1d(wave_resampled))),
                        wave_start=wave_resampled[0],
                        wave_end=wave_resampled[-1],
                        wave_resampled=wave_resampled,
                    )

    def from_twodspec(self,
                      twodspec,
                      pixel_list=None,
                      stype='science+standard'):
        '''
        To add a TwoDSpec object or numpy array to provide the traces, line
        spread function of the traces, optionally the pixel values
        correcponding to the traces.

        Science type twodspec can contain more than 1 spectrum.

        Parameters
        ----------
        twodspec: TwoDSpec object
            TwoDSpec of the science/standard image containin the trace(s)
        pixel_list: list or numpy array
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(len(spec.trace[0])), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'standard' in stype_split:

            # Loop through the twodspec.spectrum_list to update the
            # spectrum_list_standard
            if not self.spectrum_list_standard:

                self.spectrum_list_standard[0] = _spectrum1D(spec_id=0)

            for key, value in twodspec.spectrum_list[0].__dict__.items():

                setattr(self.spectrum_list_standard[0], key, value)

        if 'science' in stype_split:

            # Loop through the spec_id in twodspec
            for i in twodspec.spectrum_list.keys():

                # Loop through the twodspec.spectrum_list to update the
                # spectrum_list_standard
                if i not in self.spectrum_list_science:

                    self.spectrum_list_science[i] = _spectrum1D(spec_id=i)

                for key, value in twodspec.spectrum_list[i].__dict__.items():

                    setattr(self.spectrum_list_science[i], key, value)

    def add_wavecal(self, wavecal, stype='science+standard'):
        '''
        Copy the spectrum_list from the WavelengthCalibration object to here.

        Parameters
        ----------
        wavecal: WavelengthPolyFit object
            The WavelengthPolyFit object for the science target, flux will
            not be calibrated if this is not provided.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'standard' in stype_split:

            # Loop through the wavecal.spectrum_list to update the
            # spectrum_list_standard
            for key, value in wavecal.spectrum_list[0].__dict__.items():

                if not self.spectrum_list_standard[0]:

                    self.spectrum_list_standard[0] = _spectrum1D(spec_id=0)

                setattr(self.spectrum_list_standard[0], key, value)

        if 'science' in stype_split:

            # Loop through the spec_id in wavecal
            for i in wavecal.spectrum_list.keys():

                # Loop through the wavecal.spectrum_list to update the
                # spectrum_list_science
                for key, value in wavecal.spectrum_list[i].__dict__.items():

                    if not self.spectrum_list_science[i]:

                        self.spectrum_list_science[i] = _spectrum1D(spec_id=i)

                    setattr(self.spectrum_list_science[i], key, value)

    def compute_sensitivity(self,
                            kind=3,
                            smooth=False,
                            slength=5,
                            sorder=3,
                            mask_range=[[6850, 6960], [7575, 7700],
                                        [8925, 9050], [9265, 9750]],
                            mask_fit_order=1,
                            mask_fit_size=1):
        '''
        The sensitivity curve is computed by dividing the true values by the
        wavelength calibrated standard spectrum, which is resampled with the
        spectres.spectres(). The curve is then interpolated with a cubic spline
        by default and is stored as a scipy interp1d object.

        A Savitzky-Golay filter is available for smoothing before the
        interpolation but it is not used by default.

        6850 - 6960, 7575 - 7700, 8925 - 9050 and 9265 - 9750 A are masked by
        default.

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
        mask_fit_order: int
            Order of polynomial to be fitted over the masked regions
        mask_fit_size: int
            Number of "pixels" to be fitted on each side of the masked regions.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        '''

        # resampling both the observed and the database standard spectra
        # in unit of flux per second. The higher resolution spectrum is
        # resampled to match the lower resolution one.
        spec = self.spectrum_list_standard[0]

        if np.nanmedian(np.array(np.ediff1d(spec.wave))) < np.nanmedian(
                np.array(np.ediff1d(self.wave_standard_true))):

            flux_standard = spectres(np.array(
                self.wave_standard_true).reshape(-1),
                                     np.array(spec.wave).reshape(-1),
                                     np.array(spec.count).reshape(-1),
                                     verbose=False)
            flux_standard_true = self.fluxmag_standard_true
            wave_standard_true = self.wave_standard_true

        else:

            flux_standard = spec.count
            flux_standard_true = spectres(
                np.array(spec.wave).reshape(-1),
                np.array(self.wave_standard_true).reshape(-1),
                np.array(self.fluxmag_standard_true).reshape(-1),
                verbose=False)
            wave_standard_true = spec.wave

        # Get the sensitivity curve
        sensitivity = flux_standard_true / flux_standard
        sensitivity_masked = sensitivity.copy()

        if mask_range is not None:

            for m in mask_range:

                # If the mask is partially outside the spectrum, ignore
                if (m[0] < min(wave_standard_true)) or (
                        m[1] > max(wave_standard_true)):

                    continue

                # Get the indices for the two sides of the masking region
                left_end = int(max(np.where(wave_standard_true <= m[0])[0]))
                left_start = int(left_end - mask_fit_size)
                right_start = int(min(np.where(wave_standard_true >= m[1])[0]))
                right_end = int(right_start + mask_fit_size)

                # Get the wavelengths of the two sides
                wave_temp = np.concatenate(
                    (wave_standard_true[left_start:left_end],
                     wave_standard_true[right_start:right_end]))

                # Get the sensitivity of the two sides
                sensitivity_temp = np.concatenate(
                    (sensitivity[left_start:left_end],
                     sensitivity[right_start:right_end]))

                # Fit the polynomial across the masked region
                coeff = np.polynomial.polynomial.polyfit(
                    wave_temp, sensitivity_temp, mask_fit_order)

                # Replace the snsitivity values with the fitted values
                sensitivity_masked[
                    left_end:right_start] = np.polynomial.polynomial.polyval(
                        wave_standard_true[left_end:right_start], coeff)

        mask = np.isfinite(sensitivity_masked)
        sensitivity_masked = sensitivity_masked[mask]
        wave_standard_masked = wave_standard_true[mask]
        flux_standard_masked = flux_standard_true[mask]

        # apply a Savitzky-Golay filter to remove noise and Telluric lines
        if smooth:

            sensitivity_masked = signal.savgol_filter(sensitivity_masked,
                                                      slength, sorder)
            # Set the smoothing parameters
            spec.add_smoothing(smooth, slength, sorder)

        sensitivity_itp = itp.interp1d(wave_standard_masked,
                                       np.log10(sensitivity_masked),
                                       kind=kind,
                                       fill_value='extrapolate')

        spec.add_sensitivity(sensitivity_masked)
        spec.add_literature_standard(wave_standard_masked,
                                     flux_standard_masked)

        # Add to each _spectrum1D object
        self.add_sensitivity_itp(sensitivity_itp, stype='science+standard')

    def add_sensitivity_itp(self, sensitivity_itp, stype='science+standard'):
        '''
        parameters
        ----------
        sensitivity_itp: str
            Interpolated sensivity curve object.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

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
                            display=True,
                            renderer='default',
                            width=1280,
                            height=720,
                            return_jsonstring=False,
                            save_iframe=False,
                            filename=None,
                            open_iframe=False):
        '''
        Display the computed sensitivity curve.

        Parameters
        ----------
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

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
                       name='Count / s (Observed)'))

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
                              yaxis_title='Smoothed Count / s')

        else:

            fig.update_layout(title=self.library + ': ' + self.target,
                              yaxis_title='Count / s')

        fig.update_layout(hovermode='closest',
                          showlegend=True,
                          xaxis_title=r'$\text{Wavelength / A}$',
                          yaxis=dict(title='Count / s'),
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

        if save_iframe:

            if filename is None:

                pio.write_html(fig, 'senscurve.html', auto_open=open_iframe)

            else:

                pio.write_html(fig, filename + '.html', auto_open=open_iframe)

        if display:

            if renderer == 'default':

                fig.show()

            else:

                fig.show(renderer)

        if return_jsonstring:

            return fig.to_json()

    def apply_flux_calibration(self, spec_id=None, stype='science+standard'):
        '''
        Apply the computed sensitivity curve. And resample the spectra to
        match the highest resolution (the smallest wavelength bin) part of the
        spectrum.

        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object
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

                flux = sensitivity * spec.count

                if spec.count_err is not None:

                    flux_err = sensitivity * spec.count_err

                if spec.count_sky is not None:

                    flux_sky = sensitivity * spec.count_sky

                flux_resampled = spectres(np.array(
                    spec.wave_resampled).reshape(-1),
                                          np.array(spec.wave).reshape(-1),
                                          np.array(flux).reshape(-1),
                                          verbose=False)

                if spec.count_err is None:

                    flux_err_resampled = np.zeros_like(flux_resampled)

                else:

                    flux_err_resampled = spectres(
                        np.array(spec.wave_resampled).reshape(-1),
                        np.array(spec.wave).reshape(-1),
                        np.array(flux_err).reshape(-1),
                        verbose=False)

                if spec.count_sky is None:

                    flux_sky_resampled = np.zeros_like(flux_resampled)

                else:

                    flux_sky_resampled = spectres(
                        np.array(spec.wave_resampled).reshape(-1),
                        np.array(spec.wave).reshape(-1),
                        np.array(flux_sky).reshape(-1),
                        verbose=False)

                # Only computed for diagnostic
                sensitivity_resampled = spectres(
                    np.array(spec.wave_resampled).reshape(-1),
                    np.array(spec.wave).reshape(-1),
                    np.array(sensitivity).reshape(-1),
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

            flux = sensitivity * spec.count

            if spec.count_err is not None:

                flux_err = sensitivity * spec.count_err

            if spec.count_sky is not None:

                flux_sky = sensitivity * spec.count_sky

            flux_resampled = spectres(np.array(
                spec.wave_resampled).reshape(-1),
                                      np.array(spec.wave).reshape(-1),
                                      np.array(flux).reshape(-1),
                                      verbose=False)

            if spec.count_err is None:

                flux_err_resampled = np.zeros_like(flux_resampled)

            else:

                flux_err_resampled = spectres(np.array(
                    spec.wave_resampled).reshape(-1),
                                              np.array(spec.wave).reshape(-1),
                                              np.array(flux_err).reshape(-1),
                                              verbose=False)

            if spec.count_sky is None:

                flux_sky_resampled = np.zeros_like(flux_resampled)

            else:

                flux_sky_resampled = spectres(np.array(
                    spec.wave_resampled).reshape(-1),
                                              np.array(spec.wave).reshape(-1),
                                              np.array(flux_sky).reshape(-1),
                                              verbose=False)

            # Only computed for diagnostic
            sensitivity_resampled = spectres(np.array(
                spec.wave_resampled).reshape(-1),
                                             np.array(spec.wave).reshape(-1),
                                             np.array(sensitivity).reshape(-1),
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
                                 display=True,
                                 renderer='default',
                                 width=1280,
                                 height=720,
                                 filename=None,
                                 save_png=False,
                                 save_jpg=False,
                                 save_svg=False,
                                 save_pdf=False,
                                 return_jsonstring=False,
                                 save_iframe=False,
                                 open_iframe=False):
        '''
        Display the reduced spectra.

        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        wave_min: float
            Minimum wavelength to display
        wave_max: float
            Maximum wavelength to display
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        save_png: boolean
            Save an png image of the Plotly plot
        save_jpg: boolean
            Save an png image of the Plotly plot
        save_svg: boolean
            Save an png image of the Plotly plot
        save_pdf: boolean
            Save a pdf of the Plotly plot
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

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

                    wave_mask = (
                        (np.array(spec.wave_resampled).reshape(-1) > wave_min)
                        &
                        (np.array(spec.wave_resampled).reshape(-1) < wave_max))

                    flux_low = np.nanpercentile(
                        np.array(spec.flux_resampled).reshape(-1)[wave_mask],
                        5) / 1.5
                    flux_high = np.nanpercentile(
                        np.array(spec.flux_resampled).reshape(-1)[wave_mask],
                        95) * 1.5
                    flux_mask = (
                        (np.array(spec.flux_resampled).reshape(-1) > flux_low)
                        &
                        (np.array(spec.flux_resampled).reshape(-1) < flux_high)
                    )
                    flux_min = np.log10(
                        np.nanmin(
                            np.array(
                                spec.flux_resampled).reshape(-1)[flux_mask]))
                    flux_max = np.log10(
                        np.nanmax(
                            np.array(
                                spec.flux_resampled).reshape(-1)[flux_mask]))

                else:

                    warnings.warn('Flux calibration is not available.')
                    wave_mask = (
                        (np.array(spec.wave_resampled).reshape(-1) > wave_min)
                        &
                        (np.array(spec.wave_resampled).reshape(-1) < wave_max))
                    flux_mask = ((np.array(
                        spec.count_resampled).reshape(-1) > np.nanpercentile(
                            np.array(spec.count_resampled).reshape(-1)
                            [wave_mask], 5) / 1.5) &
                                 (np.array(spec.count_resampled).reshape(-1) <
                                  np.nanpercentile(
                                      np.array(spec.count_resampled).reshape(
                                          -1)[wave_mask], 95) * 1.5))
                    flux_min = np.log10(
                        np.nanmin(
                            np.array(
                                spec.count_resampled).reshape(-1)[flux_mask]))
                    flux_max = np.log10(
                        np.nanmax(
                            np.array(
                                spec.count_resampled).reshape(-1)[flux_mask]))

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
                                   y=spec.count_resampled,
                                   line=dict(color='royalblue'),
                                   name='Count / (e- / s)'))

                    if spec.count_err is not None:

                        fig_sci.add_trace(
                            go.Scatter(x=spec.wave_resampled,
                                       y=spec.count_err_resampled,
                                       line=dict(color='firebrick'),
                                       name='Count Uncertainty / (e- / s)'))

                    if spec.count_sky is not None:

                        fig_sci.add_trace(
                            go.Scatter(x=spec.wave_resampled,
                                       y=spec.count_sky_resampled,
                                       line=dict(color='orange'),
                                       name=r'Sky Count / (e- / s)'))

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

                    filename_output = "spectrum_" + str(i)

                else:

                    filename_output = os.path.splitext(
                        filename)[0] + "_" + str(i)

                if save_iframe:

                    pio.write_html(fig_sci,
                                   filename_output + '.html',
                                   auto_open=open_iframe)

                if display:

                    if renderer == 'default':

                        fig_sci.show()

                    else:

                        fig_sci.show(renderer)

                if save_jpg:

                    fig_sci.write_image(filename_output + '.jpg', format='jpg')

                if save_png:

                    fig_sci.write_image(filename_output + '.png', format='png')

                if save_svg:

                    fig_sci.write_image(filename_output + '.svg', format='svg')

                if save_pdf:

                    fig_sci.write_image(filename_output + '.pdf', format='pdf')

                if return_jsonstring:

                    return fig_sci[j].to_json()

        if 'standard' in stype_split:

            spec = self.spectrum_list_standard[0]

            if self.flux_standard_calibrated:

                wave_standard_mask = (
                    (np.array(spec.wave_resampled).reshape(-1) > wave_min) &
                    (np.array(spec.wave_resampled).reshape(-1) < wave_max))
                flux_standard_mask = (
                    (np.array(spec.flux_resampled).reshape(-1) >
                     np.nanpercentile(
                         np.array(spec.flux_resampled).reshape(-1)
                         [wave_standard_mask], 5) / 1.5) &
                    (np.array(spec.flux_resampled).reshape(-1) <
                     np.nanpercentile(
                         np.array(spec.flux_resampled).reshape(-1)
                         [wave_standard_mask], 95) * 1.5))
                flux_standard_min = np.log10(
                    np.nanmin(
                        np.array(spec.flux_resampled).reshape(-1)
                        [flux_standard_mask]))
                flux_standard_max = np.log10(
                    np.nanmax(
                        np.array(spec.flux_resampled).reshape(-1)
                        [flux_standard_mask]))

            else:

                warnings.warn('Flux calibration is not available.')
                wave_standard_mask = (
                    (np.array(spec.wave_resampled).reshape(-1) > wave_min) &
                    (np.array(spec.wave_resampled).reshape(-1) < wave_max))
                flux_standard_mask = (
                    (np.array(spec.count_resampled).reshape(-1) >
                     np.nanpercentile(
                         np.array(spec.count_resampled).reshape(-1)
                         [wave_standard_mask], 5) / 1.5) &
                    (np.array(spec.count_standard).reshape(-1) <
                     np.nanpercentile(
                         np.array(spec.count_resampled).reshape(-1)
                         [wave_standard_mask], 95) * 1.5))
                flux_standard_min = np.log10(
                    np.nanmin(
                        np.array(spec.count_resampled).reshape(-1)
                        [flux_standard_mask]))
                flux_standard_max = np.log10(
                    np.nanmax(
                        np.array(spec.count_resampled).reshape(-1)
                        [flux_standard_mask]))

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
                               y=spec.count_resampled,
                               line=dict(color='royalblue'),
                               name='Counts / (e- / s)'))

                if spec.count_err_resampled is not None:

                    fig_standard.add_trace(
                        go.Scatter(x=spec.wave_resampled,
                                   y=spec.count_err_resampled,
                                   line=dict(color='firebrick'),
                                   name='Counts Uncertainty / (e- / s)'))

                if spec.count_sky_resampled is not None:

                    fig_standard.add_trace(
                        go.Scatter(x=spec.wave_resampled,
                                   y=spec.count_sky_resampled,
                                   line=dict(color='orange'),
                                   name='Sky Counts / (e- / s)'))

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

                filename_output = "spectrum_standard"

            else:

                filename_output = os.path.splitext(filename)[0]

            if save_iframe:

                pio.write_html(fig_standard,
                               filename_output + '.html',
                               auto_open=open_iframe)

            if display:

                if renderer == 'default':

                    fig_standard.show(height=height, width=width)

                else:

                    fig_standard.show(renderer, height=height, width=width)

            if save_jpg:

                fig_standard.write_image(filename_output + '.jpg',
                                         format='jpg')

            if save_png:

                fig_standard.write_image(filename_output + '.png',
                                         format='png')

            if save_svg:

                fig_standard.write_image(filename_output + '.svg',
                                         format='svg')

            if save_pdf:

                fig_standard.write_image(filename_output + '.pdf',
                                         format='pdf')

            if return_jsonstring:

                return fig_standard.to_json()

        if ('science' not in stype_split) and ('standard' not in stype_split):

            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

    def save_sensitivity_itp(self, filename='sensitivity_itp.npy'):
        '''
        Parameters
        ----------
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.

        '''

        np.save(filename, self.spectrum_list_standard[0].sensitivity_itp)

    def create_fits(self,
                    spec_id=None,
                    output='count+count_resampled+flux+flux_resampled',
                    stype='science+standard',
                    force=False,
                    empty_primary_hdu=True,
                    return_id=False):
        '''
        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        return_id: boolean (default: False)
            Set to True to return the set of spec_id.

        '''

        # Split the string into strings
        output_split = output.split('+')
        stype_split = stype.split('+')

        for i in output_split:

            if i not in [
                    'count', 'count_resampled', 'wavelength', 'flux',
                    'flux_resampled'
            ]:

                raise ValueError('%s is not a valid output.' % i)

        if ('science' not in stype_split) and ('standard' not in stype_split):

            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

        if 'science' in stype_split:

            if self.flux_science_calibrated:

                if spec_id is not None:

                    if spec_id not in list(self.spectrum_list_science.keys()):

                        raise ValueError('The given spec_id does not exist.')
                else:

                    # if spec_id is None, contraints are applied to all calibrators
                    spec_id = list(self.spectrum_list_science.keys())

            else:

                raise Error('Flux is not calibrated.')

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            for i in spec_id:

                if self.flux_science_calibrated:

                    # If flux is calibrated
                    self.spectrum_list_science[i].create_fits(
                        output=output,
                        force=force,
                        empty_primary_hdu=empty_primary_hdu)

                else:

                    raise Error('Flux is not calibrated.')

            if return_id:

                return spec_id

        if 'standard' in stype_split:

            if self.flux_standard_calibrated:

                self.spectrum_list_standard[0].create_fits(
                    output=output,
                    force=force,
                    empty_primary_hdu=empty_primary_hdu)

            else:

                raise Error('Flux is not calibrated.')

    def save_fits(self,
                  spec_id=None,
                  output='count+count_resampled+flux+flux_resampled',
                  filename='fluxcal',
                  stype='science+standard',
                  empty_primary_hdu=True,
                  force=False,
                  overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            wavelength: 1 HDU
                Wavelength of each pixel
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        overwrite: boolean
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        if 'science' in stype_split:

            # Create the FITS here to go through all the checks, the save_fits()
            # below does not re-create the FITS. A warning will be given, but it
            # can be ignored.
            spec_id = self.create_fits(spec_id=spec_id,
                                       output=output,
                                       stype='science',
                                       empty_primary_hdu=empty_primary_hdu,
                                       force=force,
                                       return_id=True)

            for i in spec_id:

                filename_i = filename + '_science_' + str(i)

                self.spectrum_list_science[i].save_fits(
                    output=output,
                    filename=filename_i,
                    force=False,
                    overwrite=overwrite,
                    empty_primary_hdu=empty_primary_hdu)

        if 'standard' in stype_split:

            # Create the FITS here to go through all the checks, the save_fits()
            # below does not re-create the FITS. A warning will be given, but it
            # can be ignored.
            self.create_fits(spec_id=[0],
                             output=output,
                             stype='standard',
                             empty_primary_hdu=empty_primary_hdu,
                             force=force)

            self.spectrum_list_standard[0].save_fits(
                output=output,
                filename=filename + '_standard',
                force=False,
                overwrite=overwrite,
                empty_primary_hdu=empty_primary_hdu)

    def save_csv(self,
                 spec_id=None,
                 output='count+count_resampled+flux+flux_resampled',
                 filename='fluxcal',
                 stype='science+standard',
                 overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            wavelength: 1 HDU
                Wavelength of each pixel
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        overwrite: boolean
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            # Create the FITS here to go through all the checks, the save_fits()
            # below does not re-create the FITS. A warning will be given, but it
            # can be ignored.
            spec_id = self.create_fits(spec_id=spec_id,
                                       output=output,
                                       stype='science',
                                       empty_primary_hdu=empty_primary_hdu,
                                       force=force,
                                       return_id=True)

            for i in spec_id:

                filename_i = filename + '_science_' + str(i)

                # If flux is calibrated
                if self.flux_science_calibrated:

                    self.spectrum_list_science[i].save_csv(output=output,
                                                           filename=filename_i,
                                                           force=False,
                                                           overwrite=overwrite)
                else:

                    raise Error('Flux is not calibrated.')

        if 'standard' in stype_split:

            # Create the FITS here to go through all the checks, the save_fits()
            # below does not re-create the FITS. A warning will be given, but it
            # can be ignored.
            self.create_fits(spec_id=[0],
                             output=output,
                             stype='standard',
                             empty_primary_hdu=empty_primary_hdu,
                             force=force)

            # If flux is calibrated
            if self.flux_standard_calibrated:

                self.spectrum_list_standard[0].save_csv(output=output,
                                                        filename=filename +
                                                        '_standard',
                                                        force=False,
                                                        overwrite=overwrite)
            else:

                raise Error('Standard Flux is not calibrated.')


class OneDSpec():
    def __init__(self, silence=False):
        '''
        This class applies the wavelength calibrations and compute & apply the
        flux calibration to the extracted 1D spectra. The standard TwoDSpec
        object is not required for data reduction, but the flux calibrated
        standard observation will not be available for diagnostic.

        Parameters
        ----------
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

        self.wavelength_science_calibrated = False
        self.wavelength_standard_calibrated = False

        self.flux_science_calibrated = False
        self.flux_standard_calibrated = False

    def add_fluxcalibration(self, fluxcal):
        '''
        Provide the pre-calibrated FluxCalibration object.

        Parameters
        ----------
        fluxcal: FluxCalibration object
            The true mag/flux values.

        '''

        try:

            self.fluxcal = fluxcal
            self.flux_imported = True

        except:

            raise TypeError('Please provide a valid StandardFlux.')

    def add_wavelengthcalibration(self, wavecal, stype):
        '''
        Provide the pre-calibrated WavelengthCalibration object.

        Parameters
        ----------
        wavecal: WavelengthPolyFit object
            The WavelengthPolyFit object for the science target, flux will
            not be calibrated if this is not provided.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        for s in stype_split:

            self.fluxcal.add_wavecal(wavecal, s)

            if s == 'science':

                wavecal_science_imported = True

            if s == 'standard':

                wavecal_standard_imported = True

    def add_wavelength(self,
                       wave,
                       wave_resampled=None,
                       spec_id=None,
                       stype='science+standard'):
        '''
        Parameters
        ----------
        wave : numeric value, list or numpy 1D array (N) (default: None)
            The wavelength of each pixels of the spectrum.
        wave_resampled:
            The resampled wavelength of each pixels of the spectrum.
        spec_id: int
            The ID corresponding to the spectrum1D object
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        self.fluxcal.add_wavelength(wave=wave,
                                    wave_resampled=wave_resampled,
                                    spec_id=spec_id,
                                    stype=stype)

    def add_spec(self,
                 count,
                 spec_id=None,
                 count_err=None,
                 count_sky=None,
                 stype='science+standard'):
        '''
        Parameters
        ----------
        count: 1-d array
            The summed count at each column about the trace. Note: is not
            sky subtracted!
        spec_id: int
            The ID corresponding to the spectrum1D object
        count_err: 1-d array
            the uncertainties of the count values
        count_sky: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        self.fluxcal.add_spec(count=count,
                              spec_id=spec_id,
                              count_err=count_err,
                              count_sky=count_sky,
                              stype=stype)

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.add_spec(count=count,
                                          spec_id=spec_id,
                                          count_err=count_err,
                                          count_sky=count_sky)

        if 'standard' in stype_split:

            self.wavecal_standard.add_spec(count=count,
                                           spec_id=spec_id,
                                           count_err=count_err,
                                           count_sky=count_sky)

    def from_twodspec(self,
                      twodspec,
                      pixel_list=None,
                      stype='science+standard'):
        '''
        Extract the required information from the TwoDSpec object of the
        standard.

        Parameters
        ----------
        twodspec: TwoDSpec object
            TwoDSpec of the science/standard image containin the trace(s) and
            trace_sigma(s).
        pixel_list: list or numpy array
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(len(spec.trace[0])), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        self.fluxcal.from_twodspec(twodspec=twodspec,
                                   pixel_list=pixel_list,
                                   stype=stype)

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.from_twodspec(twodspec=twodspec,
                                               pixel_list=pixel_list)

        if 'standard' in stype_split:

            self.wavecal_standard.from_twodspec(twodspec=twodspec,
                                                pixel_list=pixel_list)

    def apply_twodspec_mask_to_arc(self, stype='science+standard'):
        '''
        *EXPERIMENTAL*
        Apply both the spec_mask and spatial_mask that are already stroed in
        the object.

        parameters
        ----------
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.apply_twodspec_mask_to_arc()

        if 'standard' in stype_split:

            self.wavecal_standard.apply_twodspec_mask_to_arc()

    def apply_spec_mask_to_arc(self, spec_mask, stype='science+standard'):
        '''
        *EXPERIMENTAL*
        Apply to use only the valid x-range of the chip (i.e. dispersion
        direction)

        parameters
        ----------
        spec_mask: 1D numpy array (M)
            Mask in the spectral direction, can be the indices of the pixels
            to be included (size <M) or a 1D numpy array of True/False (size M)
            (Default is (1,) i.e. keep everything)
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        self.apply_spec_mask_to_arc(spec_mask=spec_mask)

    def apply_spatial_mask_to_arc(self, spatial_mask, stype='science+standard'):
        '''
        *EXPERIMENTAL*
        Apply to use only the valid y-range of the chip (i.e. spatial
        direction)

        parameters
        ----------
        spatial_mask: 1D numpy array (N)
            Mask in the spatial direction, can be the indices of the pixels
            to be included (size <N) or a 1D numpy array of True/False (size N)
            (Default is (1,) i.e. keep everything)
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        self.apply_spatial_mask_to_arc(spatial_mask=spatial_mask)

    def add_arc_lines(self, spec_id, peaks, stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object
        peaks: list of list or list of arrays
            The pixel locations of the arc lines. Multiple traces of the arc
            can be provided as list of list or list of arrays.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.add_arc_lines(spec_id=spec_id, peaks=peaks)
            self.arc_lines_science_imported = True

        if 'standard' in stype_split:

            self.wavecal_standard.add_arc_lines(spec_id=spec_id, peaks=peaks)
            self.arc_lines_standard_imported = True

    def add_arc_spec(self, spec_id, arc_spec, stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object
        arc_spec: 1-d array
            The count of the summed 1D arc spec
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.add_arc_spec(spec_id=spec_id,
                                              arc_spec=arc_spec)
            self.arc_science_imported = True

        if 'standard' in stype_split:

            self.wavecal_standard.add_arc_spec(spec_id=spec_id,
                                               arc_spec=arc_spec)
            self.arc_standard_imported = True

    def add_arc(self, arc, stype='science+standard'):
        '''
        arc: 2D numpy array (M x N) OR astropy.io.fits object
            2D spectral image in either format
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.add_arc(arc)
            self.arc_science_imported = True

        if 'standard' in stype_split:

            self.wavecal_standard.add_arc(arc)
            self.arc_standard_imported = True

    def add_trace(self,
                  trace,
                  trace_sigma,
                  spec_id=None,
                  pixel_list=None,
                  stype='science+standard'):
        '''
        Parameters
        ----------
        trace: list or numpy.ndarray (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: list or numpy.ndarray (N)
            Standard deviation of the Gaussian profile of a trace
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        pixel_list: list or numpy.narray
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(len(spec.trace[0])), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.add_trace(trace=trace,
                                           trace_sigma=trace_sigma,
                                           spec_id=spec_id,
                                           pixel_list=pixel_list)

        if 'standard' in stype_split:

            self.wavecal_standard.add_trace(trace=trace,
                                            trace_sigma=trace_sigma,
                                            spec_id=spec_id,
                                            pixel_list=pixel_list)

    def add_fit_coeff(self,
                      fit_coeff,
                      fit_type=['poly'],
                      spec_id=None,
                      stype='science+standard'):
        '''
        Parameters
        ----------
        fit_coeff: list or list of list
            Polynomial fit coefficients.
        fit_type: str or list of str
            Strings starting with 'poly', 'leg' or 'cheb' for polynomial,
            legendre and chebyshev fits. Case insensitive.
        spec_id: int or None
            The ID corresponding to the spectrum1D object
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.add_fit_coeff(spec_id=spec_id,
                                               fit_coeff=fit_coeff,
                                               fit_type=fit_type)

        if 'standard' in stype_split:

            self.wavecal_standard.add_fit_coeff(spec_id=spec_id,
                                                fit_coeff=fit_coeff,
                                                fit_type=fit_type)

    def extract_arc_spec(self,
                         spec_id=None,
                         display=False,
                         return_jsonstring=False,
                         renderer='default',
                         width=1280,
                         height=720,
                         save_iframe=False,
                         filename=None,
                         open_iframe=False,
                         stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        display: boolean (default: False)
            Set to True to display disgnostic plot.
        return_jsonstring: boolean (default: False)
            set to True to return json string that can be rendered by Plotly
            in any support language.
        renderer: string (default: 'default')
            plotly renderer options.
        width: int/float (default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (default: 720)
            Number of pixels in the vertical direction of the outputs
        save_iframe: boolean (default: False)
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str or None (default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean (default: False)
            Open the save_iframe in the default browser if set to True.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.extract_arc_spec(
                spec_id=spec_id,
                display=display,
                return_jsonstring=return_jsonstring,
                renderer=renderer,
                width=width,
                height=height,
                save_iframe=save_iframe,
                filename=filename,
                open_iframe=open_iframe)

        if 'standard' in stype_split:

            self.wavecal_standard.extract_arc_spec(
                spec_id=spec_id,
                display=display,
                return_jsonstring=return_jsonstring,
                renderer=renderer,
                width=width,
                height=height,
                save_iframe=save_iframe,
                filename=filename,
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
                       return_jsonstring=False,
                       renderer='default',
                       save_iframe=False,
                       filename=None,
                       open_iframe=False,
                       stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        background: int or None (default: None)
            User-supplied estimated background level
        percentile: float (default: 25.)
            The percentile of the flux to be used as the estimate of the
            background sky level to the first order. Only used if background
            is None. [Count]
        prominence: float (default: 10.)
            The minimum prominence to be considered as a peak
        distance: float (default: 5.)
            Minimum separation between peaks
        refine: boolean (default: True)
            Set to true to fit a gaussian to get the peak at sub-pixel
            precision
        refine_window_width: int or float (default: 5)
            The number of pixels (on each side of the existing peaks) to be
            fitted with gaussian profiles over.
        display: boolean (default: False)
            Set to True to display disgnostic plot.
        width: int/float (default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (default: 720)
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean (default: False)
            set to True to return json string that can be rendered by Plotly
            in any support language.
        renderer: string (default: 'default')
            plotly renderer options.
        save_iframe: boolean (default: False)
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str or None (default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean (default: False)
            Open the save_iframe in the default browser if set to True.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        Returns
        -------
        JSON strings if return_jsonstring is set to True

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.find_arc_lines(
                spec_id=spec_id,
                background=background,
                percentile=percentile,
                prominence=prominence,
                distance=distance,
                display=display,
                width=width,
                height=height,
                return_jsonstring=return_jsonstring,
                renderer=renderer,
                save_iframe=save_iframe,
                filename=filename,
                open_iframe=open_iframe)

        if 'standard' in stype_split:

            self.wavecal_standard.find_arc_lines(
                spec_id=spec_id,
                background=background,
                percentile=percentile,
                prominence=prominence,
                distance=distance,
                display=display,
                width=width,
                height=height,
                return_jsonstring=return_jsonstring,
                renderer=renderer,
                save_iframe=save_iframe,
                filename=filename,
                open_iframe=open_iframe)

    def initialise_calibrator(self,
                              spec_id=None,
                              peaks=None,
                              spectrum=None,
                              stype='science+standard'):
        '''
        If the peaks were found with find_arc_lines(), peaks and num_pix can
        be None.

        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        peaks: list, numpy.ndarray or None (default: None)
            The pixel values of the peaks (start from zero)
        spectrum: list, numpy.ndarray or None (default: None)
            The spectral intensity as a function of pixel.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.initialise_calibrator(spec_id=spec_id,
                                                       peaks=peaks,
                                                       spectrum=spectrum)

        if 'standard' in stype_split:

            self.wavecal_standard.initialise_calibrator(spec_id=spec_id,
                                                        peaks=peaks,
                                                        spectrum=spectrum)

    def set_calibrator_properties(self,
                                  spec_id=None,
                                  num_pix=None,
                                  pixel_list=None,
                                  plotting_library='plotly',
                                  log_level='info',
                                  stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        num_pix: int (default: None)
            The number of pixels in the dispersion direction
        pixel_list: list or numpy array (default: None)
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(num_pix), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        plotting_library : string (default: 'plotly')
            Choose between matplotlib and plotly.
        log_level : string (default: 'info')
            Choose {critical, error, warning, info, debug, notset}.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.set_calibrator_properties(
                spec_id=spec_id,
                num_pix=num_pix,
                pixel_list=pixel_list,
                plotting_library=plotting_library,
                log_level=log_level)

        if 'standard' in stype_split:

            self.wavecal_standard.set_calibrator_properties(
                spec_id=spec_id,
                num_pix=num_pix,
                pixel_list=pixel_list,
                plotting_library=plotting_library,
                log_level=log_level)

    def set_hough_properties(self,
                             spec_id=None,
                             num_slopes=5000,
                             xbins=500,
                             ybins=500,
                             min_wavelength=3000,
                             max_wavelength=9000,
                             range_tolerance=500,
                             linearity_tolerance=50,
                             stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object
        num_slopes: int (default: 1000)
            Number of slopes to consider during Hough transform
        xbins: int (default: 50)
            Number of bins for Hough accumulation
        ybins: int (default: 50)
            Number of bins for Hough accumulation
        min_wavelength: float (default: 3000)
            Minimum wavelength of the spectrum.
        max_wavelength: float (default: 9000)
            Maximum wavelength of the spectrum.
        range_tolerance: float (default: 500)
            Estimation of the error on the provided spectral range
            e.g. 3000-5000 with tolerance 500 will search for
            solutions that may satisfy 2500-5500
        linearity_tolerance: float (default: 100)
            A toleranceold (Ansgtroms) which defines some padding around the
            range tolerance to allow for non-linearity. This should be the
            maximum expected excursion from linearity.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.set_hough_properties(
                spec_id=spec_id,
                num_slopes=num_slopes,
                xbins=xbins,
                ybins=ybins,
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                range_tolerance=range_tolerance,
                linearity_tolerance=linearity_tolerance)

        if 'standard' in stype_split:

            self.wavecal_standard.set_hough_properties(
                spec_id=spec_id,
                num_slopes=num_slopes,
                xbins=xbins,
                ybins=ybins,
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                range_tolerance=range_tolerance,
                linearity_tolerance=linearity_tolerance)

    def set_ransac_properties(self,
                              spec_id=None,
                              sample_size=5,
                              top_n_candidate=5,
                              linear=True,
                              filter_close=False,
                              ransac_tolerance=5,
                              candidate_weighted=True,
                              hough_weight=1.0,
                              stype='science+standard'):
        '''
        Configure the Calibrator. This may require some manual twiddling before
        the calibrator can work efficiently. However, in theory, a large
        max_tries in fit() should provide a good solution in the expense of
        performance (minutes instead of seconds).

        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object
        sample_size: int (default: 5)
            Number of pixel-wavelength hough pairs to be used for each arc line
            being picked.
        top_n_candidate: int (default: 5)
            Top ranked lines to be fitted.
        linear: boolean (default: True)
            True to use the hough transformed gradient, otherwise, use the
            known polynomial.
        filter_close: boolean (default: False)
            Remove the pairs that are out of bounds in the hough space.
        ransac_tolerance: float (default: 1)
            The distance criteria  (Angstroms) to be considered an inlier to a
            fit. This should be close to the size of the expected residuals on
            the final fit (e.g. 1A is typical)
        candidate_weighted: boolean (default: True)
            Set to True to down-weight pairs that are far from the fit.
        hough_weight: float or None (default: 1.0)
            Set to use the hough space to weigh the fit. The theoretical
            optimal weighting is unclear. The larger the value, the heavily it
            relies on the overdensity in the hough space for a good fit.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.set_ransac_properties(
                spec_id=spec_id,
                sample_size=sample_size,
                top_n_candidate=top_n_candidate,
                linear=linear,
                filter_close=filter_close,
                ransac_tolerance=ransac_tolerance,
                candidate_weighted=candidate_weighted,
                hough_weight=hough_weight)

        if 'standard' in stype_split:

            self.wavecal_standard.set_ransac_properties(
                spec_id=spec_id,
                sample_size=sample_size,
                top_n_candidate=top_n_candidate,
                linear=linear,
                filter_close=filter_close,
                ransac_tolerance=ransac_tolerance,
                candidate_weighted=candidate_weighted,
                hough_weight=hough_weight)

    def set_known_pairs(self,
                        spec_id=None,
                        pix=None,
                        wave=None,
                        stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object
        pix : numeric value, list or numpy 1D array (N) (default: None)
            Any pixel value, can be outside the detector chip and
            serve purely as anchor points.
        wave : numeric value, list or numpy 1D array (N) (default: None)
            The matching wavelength for each of the pix.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.set_known_pairs(spec_id=spec_id,
                                                 pix=pix,
                                                 wave=wave)

        if 'standard' in stype_split:

            self.wavecal_standard.set_known_pairs(spec_id=spec_id,
                                                  pix=pix,
                                                  wave=wave)

    def load_user_atlas(self,
                        elements,
                        wavelengths,
                        spec_id=None,
                        intensities=None,
                        candidate_tolerance=10.,
                        constrain_poly=False,
                        vacuum=False,
                        pressure=101325.,
                        temperature=273.15,
                        relative_humidity=0.,
                        stype='science+standard'):
        '''
        *Remove* all the arc lines loaded to the Calibrator and then use the user
        supplied arc lines instead.

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements : list
            Element (required). Preferably a standard (i.e. periodic table)
            name for convenience with built-in atlases
        wavelengths : list or None (default: None)
            Wavelength to add (Angstrom)
        intensities : list or None (default: None)
            Relative line intensities
        candidate_tolerance: float (default: 10.)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: boolean (default: False)
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: boolean (default: False)
            Set to true to convert the input wavelength to air-wavelengths
            based on the given pressure, temperature and humidity.
        pressure: float (default: 101325)
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float (default: 273.15)
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float (default: 0)
            In percentage.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

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

    def add_atlas(self,
                  elements,
                  spec_id=None,
                  min_atlas_wavelength=1000.,
                  max_atlas_wavelength=30000.,
                  min_intensity=10.,
                  min_distance=10.,
                  candidate_tolerance=10.,
                  constrain_poly=False,
                  vacuum=False,
                  pressure=101325.,
                  temperature=273.15,
                  relative_humidity=0,
                  stype='science+standard'):
        '''
        Parameters
        ----------
        elements: string or list of strings
            Chemical symbol, case insensitive
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object
        min_atlas_wavelength: float (default: None)
            Minimum wavelength of the arc lines.
        max_atlas_wavelength: float (default: None)
            Maximum wavelength of the arc lines.
        min_intensity: float (default: None)
            Minimum intensity of the arc lines. Refer to NIST for the intensity.
        min_distance: float (default: None)
            Minimum separation between neighbouring arc lines.
        candidate_tolerance: float (default: 10)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: boolean
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: boolean (default: False)
            Set to true to convert the input wavelength to air-wavelengths
            based on the given pressure, temperature and humidity.
        pressure: float (default: 101325)
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float (default: 273.15)
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float (default: 0)
            In percentage.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.add_atlas(
                elements=elements,
                spec_id=spec_id,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=min_intensity,
                min_distance=min_distance,
                candidate_tolerance=candidate_tolerance,
                constrain_poly=constrain_poly,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity)

        if 'standard' in stype_split:

            self.wavecal_standard.add_atlas(
                elements=elements,
                spec_id=spec_id,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=min_intensity,
                min_distance=min_distance,
                candidate_tolerance=candidate_tolerance,
                constrain_poly=constrain_poly,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity)

    def do_hough_transform(self, spec_id=None, stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int (default: None)
            The ID corresponding to the spectrum1D object
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.do_hough_transform(spec_id=spec_id)

        if 'standard' in stype_split:

            self.wavecal_standard.do_hough_transform(spec_id=spec_id)

    def fit(self,
            spec_id=None,
            max_tries=5000,
            fit_deg=4,
            fit_coeff=None,
            fit_tolerance=10.,
            fit_type='poly',
            brute_force=False,
            progress=True,
            display=False,
            savefig=False,
            filename=None,
            stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        max_tries: int (default: 5000)
            Number of trials of polynomial fitting.
        fit_deg: int (default: 4)
            The degree of the polynomial to be fitted.
        fit_coeff: list (default: None)
            Set the baseline of the least square fit. If no fits outform this
            set of polynomial coefficients, this will be used as the best fit.
        fit_tolerance: float (default: 10)
            Sets a tolerance on whether a fit found by RANSAC is considered
            acceptable
        fit_type: string (default: 'poly')
            One of 'poly', 'legendre' or 'chebyshev'
        brute_force: boolean (default: False)
            Set to True to try all possible combination in the given parameter
            space
        progress: boolean (default: False)
            Set to show the progress using tdqm (if imported).
        display: boolean (default: False)
            Set to show diagnostic plot.
        savefig: boolean (default: False)
            Set to save figure.
        filename: str or None (default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.fit(spec_id=spec_id,
                                     max_tries=max_tries,
                                     fit_deg=fit_deg,
                                     fit_coeff=fit_coeff,
                                     fit_tolerance=fit_tolerance,
                                     fit_type=fit_type,
                                     brute_force=brute_force,
                                     progress=progress,
                                     display=display,
                                     savefig=savefig,
                                     filename=filename)

        if 'standard' in stype_split:

            self.wavecal_standard.fit(spec_id=spec_id,
                                      max_tries=max_tries,
                                      fit_deg=fit_deg,
                                      fit_coeff=fit_coeff,
                                      fit_tolerance=fit_tolerance,
                                      fit_type=fit_type,
                                      brute_force=brute_force,
                                      progress=progress,
                                      display=display,
                                      savefig=savefig,
                                      filename=filename)

    def refine_fit(self,
                   spec_id=None,
                   fit_coeff=None,
                   n_delta=None,
                   refine=True,
                   tolerance=10.,
                   method='Nelder-Mead',
                   convergence=1e-6,
                   robust_refit=True,
                   fit_deg=None,
                   display=False,
                   savefig=False,
                   filename=None,
                   stype='science+standard'):
        '''
        * EXPERIMENTAL *

        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        fit_coeff: list or None (default: None)
            List of polynomial fit coefficients.
        n_delta: int (default: None)
            The number of the highest polynomial order to be adjusted
        refine: boolean (default: True)
            Set to True to refine solution.
        tolerance : float (default: 10.)
            Absolute difference between fit and model in the unit of nm.
        method: string (default: 'Nelder-Mead')
            scipy.optimize.minimize method.
        convergence: float (default: 1e-6)
            scipy.optimize.minimize tol.
        robust_refit: boolean (default: True)
            Set to True to fit all the detected peaks with the given polynomial
            solution.
        fit_deg: int (default: length of the input coefficients - 1)
            Order of polynomial fit with all the detected peaks.
        display: boolean (default: False)
            Set to show diagnostic plot.
        savefig: boolean (default: False)
            Set to save figure.
        filename: str or None (default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.refine_fit(spec_id=spec_id,
                                            fit_coeff=fit_coeff,
                                            n_delta=n_delta,
                                            refine=refine,
                                            tolerance=tolerance,
                                            method=method,
                                            convergence=convergence,
                                            robust_refit=robust_refit,
                                            fit_deg=fit_deg,
                                            display=display,
                                            savefig=savefig,
                                            filename=filename)

        if 'standard' in stype_split:

            self.wavecal_standard.refine_fit(spec_id=None,
                                             fit_coeff=fit_coeff,
                                             n_delta=n_delta,
                                             refine=refine,
                                             tolerance=tolerance,
                                             method=method,
                                             convergence=convergence,
                                             robust_refit=robust_refit,
                                             fit_deg=fit_deg,
                                             display=display,
                                             savefig=savefig,
                                             filename=filename)

    def apply_wavelength_calibration(self,
                                     spec_id=None,
                                     wave_start=None,
                                     wave_end=None,
                                     wave_bin=None,
                                     stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        wave_start: float or None (default: None)
            Provide the minimum wavelength for resampling.
        wave_end: float or None (default: None)
            Provide the maximum wavelength for resampling
        wave_bin: float or None (default: None)
            Provide the resampling bin size
        stype: string or None (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            self.wavecal_science.apply_wavelength_calibration(
                wave_start, wave_end, wave_bin)
            self.fluxcal.add_wavecal(self.wavecal_science, stype='science')
            self.wavelength_science_calibrated = True

        if 'standard' in stype_split:

            self.wavecal_standard.apply_wavelength_calibration(
                wave_start, wave_end, wave_bin)
            self.fluxcal.add_wavecal(self.wavecal_standard, stype='standard')
            self.wavelength_standard_calibrated = True

    def lookup_standard_libraries(self, target, cutoff=0.4):
        '''
        Parameters
        ----------
        target: str
            Name of the standard star
        cutoff: float (default: 0.4)
            The similarity toleranceold [0=completely different, 1=identical]

        '''

        self.fluxcal.lookup_standard_libraries(target, cutoff)

    def load_standard(self, target, library=None, ftype='flux', cutoff=0.4):
        '''
        Parameters
        ----------
        target: string
            Name of the standard star
        library: string
            Name of the library of standard star
        ftype: string
            'flux' or 'mag'
        cutoff: float
            The toleranceold for the word similarity in the range of [0, 1].

        '''

        self.fluxcal.load_standard(target=target,
                                   library=library,
                                   ftype=ftype,
                                   cutoff=cutoff)

    def inspect_standard(self,
                         display=True,
                         renderer='default',
                         width=1280,
                         height=720,
                         return_jsonstring=False,
                         save_iframe=False,
                         filename=None,
                         open_iframe=False):
        '''
        Parameters
        ----------
        display: boolean (default: True)
            Set to True to display disgnostic plot.
        renderer: string (default: 'default')
            plotly renderer options.
        width: int/float (default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (default: 720)
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean (default: False)
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean (default: False)
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str or None (default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean (default: False)
            Open the save_iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        '''

        self.fluxcal.inspect_standard(renderer=renderer,
                                      return_jsonstring=return_jsonstring,
                                      display=display,
                                      height=height,
                                      width=width,
                                      save_iframe=save_iframe,
                                      filename=filename,
                                      open_iframe=open_iframe)

    def compute_sensitivity(self,
                            kind=3,
                            smooth=False,
                            slength=5,
                            sorder=3,
                            mask_range=[[6850, 6960], [7575, 7700],
                                        [8925, 9050], [9265, 9750]],
                            mask_fit_order=1,
                            mask_fit_size=1):
        '''
        Parameters
        ----------
        kind: string or integer [1,2,3,4,5 only] (default: 3)
            interpolation kind is one of [‘linear’, ‘nearest’, ‘zero’,
             ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’]
        smooth: boolean (default: False)
            set to smooth the input spectrum with scipy.signal.savgol_filter
        slength: int (default: 5)
            SG-filter window size
        sorder: int (default: 3)
            SG-filter polynomial order
        mask_range: None or list of list
            (default: 6850-6960, 7575-7700, 8925-9050, 9265-9750)
            Masking out regions not suitable for fitting the sensitivity curve.
            None for no mask. List of list has the pattern
            [[min1, max1], [min2, max2],...]
        mask_fit_order: int (default: 1)
            Order of polynomial to be fitted over the masked regions
        mask_fit_size: int (default: 1)
            Number of "pixels" to be fitted on each side of the masked regions.

        '''

        self.fluxcal.compute_sensitivity(kind=kind,
                                         smooth=smooth,
                                         slength=slength,
                                         sorder=sorder,
                                         mask_range=mask_range,
                                         mask_fit_order=mask_fit_order,
                                         mask_fit_size=mask_fit_size)

    def save_sensitivity_itp(self, filename='sensitivity_itp.npy'):
        '''
        Parameters
        ----------
        filename: str
            Filename for the output interpolated sensivity curve.

        '''

        self.fluxcal.save_sensitivity_itp(filename=filename)

    def add_sensitivity_itp(self, sensitivity_itp, stype='science+standard'):
        '''
        Parameters
        ----------
        sensitivity_itp: str
            Interpolated sensivity curve object.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        self.fluxcal.add_sensitivity_itp(sensitivity_itp=sensitivity_itp,
                                         stype=stype)

    def inspect_sensitivity(self,
                            display=True,
                            renderer='default',
                            width=1280,
                            height=720,
                            return_jsonstring=False,
                            save_iframe=False,
                            filename=None,
                            open_iframe=False):
        '''
        Parameters
        ----------
        display: boolean (default: True)
            Set to True to display disgnostic plot.
        renderer: string (default: 'default')
            plotly renderer options.
        width: int/float (default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (default: 720)
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean (default: False)
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean (default: False)
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str or None (default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean (default: False)
            Open the save_iframe in the default browser if set to True.

        '''

        self.fluxcal.inspect_sensitivity(renderer=renderer,
                                         width=width,
                                         height=height,
                                         return_jsonstring=return_jsonstring,
                                         display=display,
                                         save_iframe=save_iframe,
                                         filename=filename,
                                         open_iframe=open_iframe)

    def apply_flux_calibration(self, spec_id=None, stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        self.fluxcal.apply_flux_calibration(spec_id=spec_id, stype=stype)

        if 'science' in stype_split:

            self.flux_science_calibrated = True

        if 'standard' in stype_split:

            self.flux_standard_calibrated = True

    def inspect_reduced_spectrum(self,
                                 spec_id=None,
                                 stype='science+standard',
                                 wave_min=4000.,
                                 wave_max=8000.,
                                 renderer='default',
                                 width=1280,
                                 height=720,
                                 filename=None,
                                 save_png=False,
                                 save_jpg=False,
                                 save_svg=False,
                                 save_pdf=False,
                                 display=True,
                                 return_jsonstring=False,
                                 save_iframe=False,
                                 open_iframe=False):
        '''
        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        wave_min: float (default: 4000.)
            Minimum wavelength to display
        wave_max: float (default: 8000.)
            Maximum wavelength to display
        renderer: string (default: 'default')
            plotly renderer options.
        width: int/float (default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (default: 720)
            Number of pixels in the vertical direction of the outputs
        filename: str or None (default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        save_png: boolean (default: False)
            Save an png image of the Plotly plot
        save_jpg: boolean (default: False)
            Save an png image of the Plotly plot
        save_svg: boolean (default: False)
            Save an png image of the Plotly plot
        save_pdf: boolean (default: False)
            Save a pdf of the Plotly plot
        display: boolean (default: True)
            Set to True to display disgnostic plot.
        return_jsonstring: boolean (default: False)
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean (default: False)
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        open_iframe: boolean (default: False)
            Open the save_iframe in the default browser if set to True.

        '''

        self.fluxcal.inspect_reduced_spectrum(
            spec_id=spec_id,
            stype=stype,
            wave_min=wave_min,
            wave_max=wave_max,
            renderer=renderer,
            width=width,
            height=height,
            filename=filename,
            save_png=save_png,
            save_jpg=save_jpg,
            save_svg=save_svg,
            save_pdf=save_pdf,
            return_jsonstring=return_jsonstring,
            save_iframe=save_iframe,
            open_iframe=open_iframe)

    def create_fits(self,
                    spec_id=None,
                    output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                    stype='science+standard',
                    force=False,
                    empty_primary_hdu=True,
                    return_id=False):
        '''
        Create a HDU list, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        output: String
            (default: 'arc_spec+wavecal+wavelength+flux+flux_resampled')
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        return_id: boolean (default: False)
            Set to True to return the set of spec_id

        '''

        # Split the string into strings
        stype_split = stype.split('+')

        for i in output.split('+'):

            if i not in [
                    'trace', 'count', 'arc_spec', 'wavecal', 'wavelegth',
                    'count_resampled', 'flux', 'flux_resampled'
            ]:

                raise ValueError('%s is not a valid output.' % i)

        if ('science' not in stype_split) and ('standard' not in stype_split):

            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

        if 'science' in stype_split:

            if self.flux_science_calibrated:

                if spec_id is not None:

                    if spec_id not in list(
                            self.fluxcal.spectrum_list_science.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, contraints are applied to all calibrators
                    spec_id = list(self.fluxcal.spectrum_list_science.keys())

            elif self.wavelength_science_calibrated:

                # Note that wavecal ONLY has sepctrum_list, it is not science
                # and standard specified.
                if spec_id is not None:

                    if spec_id not in list(
                            self.wavecal_science.spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, contraints are applied to all calibrators
                    spec_id = list(self.wavecal_science.spectrum_list.keys())

            else:

                raise Error('Neither wavelength nor flux is calibrated.')

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            for i in spec_id:

                # If flux is calibrated
                if self.flux_science_calibrated:

                    self.fluxcal.spectrum_list_science[i].create_fits(
                        output=output,
                        force=force,
                        empty_primary_hdu=empty_primary_hdu)

                # If flux is not calibrated, but wavelength is calibrated
                # Note that wavecal ONLY has sepctrum_list, it is not science
                # and standard specified.
                elif self.wavelength_science_calibrated:

                    self.wavecal_science.spectrum_list[i].create_fits(
                        output=output,
                        force=force,
                        empty_primary_hdu=empty_primary_hdu)

                # Should be trapped above so this line should never be run

                else:

                    raise Error('Neither wavelength nor flux is calibrated.')

            if return_id:

                return spec_id

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.flux_standard_calibrated:

                self.fluxcal.spectrum_list_standard[0].create_fits(
                    output=output,
                    force=force,
                    empty_primary_hdu=empty_primary_hdu)

            # If flux is not calibrated, but wavelength is calibrated
            # Note that wavecal ONLY has sepctrum_list, it is not science
            # and standard specified.
            elif self.wavelength_standard_calibrated:

                self.wavecal_standard.spectrum_list[0].create_fits(
                    output=output,
                    force=force,
                    empty_primary_hdu=empty_primary_hdu)

            # Should be trapped above so this line should never be run
            else:

                raise Error('Neither wavelength nor flux is calibrated.')

    def save_fits(self,
                  spec_id=None,
                  output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                  filename='reduced',
                  stype='science+standard',
                  force=False,
                  empty_primary_hdu=True,
                  overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        output: String
            (default: 'arc_spec+wavecal+wavelength+flux+flux_resampled')
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        filename: String (default: 'reduced')
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        overwrite: boolean (default: False)
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            spec_id = self.create_fits(spec_id=spec_id,
                                       output=output,
                                       stype='science',
                                       empty_primary_hdu=empty_primary_hdu,
                                       force=force,
                                       return_id=True)

            for i in spec_id:

                filename_i = filename + '_science_' + str(i)

                # If flux is calibrated
                if self.flux_science_calibrated:

                    self.fluxcal.spectrum_list_science[i].save_fits(
                        output=output,
                        filename=filename_i,
                        force=False,
                        overwrite=overwrite,
                        empty_primary_hdu=empty_primary_hdu)

                # If flux is not calibrated, and weather or not the wavelength
                # is calibrated.
                else:

                    self.wavecal_science.spectrum_list[i].save_fits(
                        output=output,
                        filename=filename_i,
                        force=False,
                        overwrite=overwrite,
                        empty_primary_hdu=empty_primary_hdu)

        if 'standard' in stype_split:

            self.create_fits(spec_id=[0],
                             output=output,
                             stype='standard',
                             empty_primary_hdu=empty_primary_hdu,
                             force=force)

            # If flux is calibrated
            if self.flux_standard_calibrated:

                self.fluxcal.spectrum_list_standard[0].save_fits(
                    output=output,
                    filename=filename + '_standard',
                    force=force,
                    overwrite=overwrite,
                    empty_primary_hdu=empty_primary_hdu)

            # If flux is not calibrated, and weather or not the wavelength
            # is calibrated.
            else:

                self.wavecal_standard.spectrum_list[0].save_fits(
                    output=output,
                    filename=filename + '_standard',
                    force=force,
                    overwrite=overwrite,
                    empty_primary_hdu=empty_primary_hdu)

    def save_csv(self,
                 spec_id=None,
                 output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                 filename='reduced',
                 stype='science+standard',
                 force=False,
                 overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int or None (default: None)
            The ID corresponding to the spectrum1D object
        output: String
            (default: 'arc_spec+wavecal+wavelength+flux+flux_resampled')
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 5 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        filename: String (default: 'reduced')
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        force: boolean (default: False)
            Set to True to force recreating the HDU 
        overwrite: boolean (default: False)
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            spec_id = self.create_fits(spec_id=spec_id,
                                       output=output,
                                       stype='science',
                                       empty_primary_hdu=False,
                                       force=force,
                                       return_id=True)

            for i in spec_id:

                filename_i = filename + '_science_' + str(i)

                # If flux is calibrated
                if self.flux_science_calibrated:

                    self.fluxcal.spectrum_list_science[i].save_csv(
                        output=output,
                        filename=filename_i,
                        force=force,
                        overwrite=overwrite)

                # If flux is not calibrated, and weather or not the wavelength
                # is calibrated.
                else:

                    self.wavecal_science.spectrum_list[i].save_csv(
                        output=output,
                        filename=filename_i,
                        force=force,
                        overwrite=overwrite)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.flux_standard_calibrated:

                self.fluxcal.spectrum_list_standard[0].save_csv(
                    output=output,
                    filename=filename + '_standard',
                    force=force,
                    overwrite=overwrite)

            # If flux is not calibrated, and weather or not the wavelength
            # is calibrated.
            else:

                self.wavecal_standard.spectrum_list[0].save_csv(
                    output=output,
                    filename=filename + '_standard',
                    force=force,
                    overwrite=overwrite)
