from astropy.io import fits
from rascal.calibrator import Calibrator
from scipy import interpolate as itp
import warnings

import os
import numpy as np


class Spectrum1D():
    '''
    Base class of a 1D spectral object to hold the information of each
    extracted spectrum.

    '''
    def __init__(self, spec_id=None):

        # spectrum ID
        if spec_id is None:
            self.spec_id = 0
        elif type(spec_id) == int:
            self.spec_id = spec_id
        else:
            raise ValueError(
                'spec_id has to be of type int, {} is given.'.format(
                    type(spec_id)))

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
        self.sensitivity_func = None
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
            'Resampled Count, Resampled Count Uncertainty, '
            'Resampled Sky Count',
            'arc_spec':
            '1D Arc Spectrum, Arc Line Position, Arc Line Effective Position',
            'wavecal':
            'Polynomial coefficients for wavelength calibration',
            'wavelength':
            'The pixel-to-wavelength mapping',
            'flux':
            'Flux, Flux Uncertainty, Sky Flux',
            'sensitivity':
            'Sensitivity Curve',
            'weight_map':
            'Weight map of the extration (variance)',
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
            'sensitivity': 1,
            'flux': 3,
            'weight_map': 1,
            'count': 3,
            'trace': 2
        }

        self.hdu_order = {
            'trace': 0,
            'count': 1,
            'weight_map': 2,
            'arc_spec': 3,
            'wavecal': 4,
            'wavelength': 5,
            'count_resampled': 6,
            'sensitivity': 7,
            'flux': 8,
            'flux_resampled': 9
        }

        self.hdu_content = {
            'trace': False,
            'count': False,
            'weight_map': False,
            'arc_spec': False,
            'wavecal': False,
            'wavelength': False,
            'count_resampled': False,
            'sensitivity': False,
            'flux': False,
            'flux_resampled': False
        }

    def merge(self, spectrum1D, overwrite=False):
        '''
        This function copies all the info from the supplied spectrum1D to
        this one.

        '''

        for attr, value in self.__dict__.items():

            if attr == 'spec_id':

                if getattr(spectrum1D, attr) != 0:

                    setattr(self, attr, value)

            else:

                if getattr(self, attr) is None or []:

                    setattr(self, attr, getattr(spectrum1D, attr))

                else:

                    if overwrite:

                        setattr(self, attr, getattr(spectrum1D, attr))

                    else:

                        # if not overwrite, do nothing
                        pass

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

    def remove_peaks_wave(self):

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
        ) == Calibrator, 'calibrator has to be a rascal.Calibrator '
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
        self.range_tolerance = None
        self.linearity_tolerance = None

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

    def add_sensitivity_func(self, sensitivity_func):

        self.sensitivity_func = sensitivity_func

    def remove_sensitivity_func(self):

        self.sensitivity_func = None

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
        +-----+---------------------------------+

        '''

        self._modify_imagehdu_header(self.count_hdulist, idx, method, *args)

    def modify_weight_map_header(self, idx, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+----------------------+
        | HDU | Data                 |
        +-----+----------------------+
        |  0  | Line spread function |
        +-----+----------------------+

        '''

        self._modify_imagehdu_header(self.weight_map_hdulist, idx, method,
                                     *args)

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

    def modify_sensitivity_header(self, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+-------------+
        | HDU | Data        |
        +-----+-------------+
        |  0  | Sensitivity |
        +-----+-------------+

        '''

        self._modify_imagehdu_header(self.sensitivity_hdulist, 0, method,
                                     *args)

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
        +-----+------------------+

        '''

        self._modify_imagehdu_header(self.flux_hdulist, idx, method, *args)

    def modify_sensitivity_resampled_header(self, method, *args):
        '''
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+-----------------------+
        | HDU | Data                  |
        +-----+-----------------------+
        |  0  | Sensitivity_resampled |
        +-----+-----------------------+

        '''

        self._modify_imagehdu_header(self.sensitivity_resampled_hdulist, 0,
                                     method, *args)

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

    def create_trace_fits(self):

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
            self.modify_trace_header(0, 'set', 'CRVAL1', self.pixel_list[0])
            self.modify_trace_header(0, 'set', 'CTYPE1', 'Pixel (Dispersion)')
            self.modify_trace_header(0, 'set', 'CUNIT1', 'Pixel')
            self.modify_trace_header(0, 'set', 'BUNIT', 'Pixel (Spatial)')

            # Add the trace_sigma
            self.modify_trace_header(1, 'set', 'LABEL', 'Trace width/sigma')
            self.modify_trace_header(1, 'set', 'CRPIX1', 1)
            self.modify_trace_header(1, 'set', 'CDELT1', 1)
            self.modify_trace_header(1, 'set', 'CRVAL1', self.pixel_list[0])
            self.modify_trace_header(1, 'set', 'CTYPE1', 'Pixel (Dispersion)')
            self.modify_trace_header(1, 'set', 'CUNIT1', 'Number of Pixels')
            self.modify_trace_header(1, 'set', 'BUNIT', 'Pixel (Spatial)')

        except Exception as e:

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('trace ImageHDU cannot be created.')
            self.trace_hdulist = None

    def create_count_fits(self):

        try:

            # Put the data in ImageHDUs
            count_ImageHDU = fits.ImageHDU(self.count)
            count_err_ImageHDU = fits.ImageHDU(self.count_err)
            count_sky_ImageHDU = fits.ImageHDU(self.count_sky)

            # Create an empty HDU list and populate with ImageHDUs
            self.count_hdulist = fits.HDUList()
            self.count_hdulist += [count_ImageHDU]
            self.count_hdulist += [count_err_ImageHDU]
            self.count_hdulist += [count_sky_ImageHDU]

            # Add the count
            self.modify_count_header(0, 'set', 'WIDTHDN', self.widthdn)
            self.modify_count_header(0, 'set', 'WIDTHUP', self.widthup)
            self.modify_count_header(0, 'set', 'SEPDN', self.sepdn)
            self.modify_count_header(0, 'set', 'SEPUP', self.sepup)
            self.modify_count_header(0, 'set', 'SKYDN', self.skywidthdn)
            self.modify_count_header(0, 'set', 'SKYUP', self.skywidthup)
            self.modify_count_header(0, 'set', 'BKGRND', self.background)
            self.modify_count_header(0, 'set', 'XTYPE', self.extraction_type)
            self.modify_count_header(0, 'set', 'LABEL', 'Electron Count')
            self.modify_count_header(0, 'set', 'CRPIX1', 1)
            self.modify_count_header(0, 'set', 'CDELT1', 1)
            self.modify_count_header(0, 'set', 'CRVAL1', self.pixel_list[0])
            self.modify_count_header(0, 'set', 'CTYPE1', ' Pixel (Dispersion)')
            self.modify_count_header(0, 'set', 'CUNIT1', 'Pixel')
            self.modify_count_header(0, 'set', 'BUNIT', 'electron')

            # Add the uncertainty count
            self.modify_count_header(1, 'set', 'LABEL',
                                     'Electron Count (Uncertainty)')
            self.modify_count_header(1, 'set', 'CRPIX1', 1)
            self.modify_count_header(1, 'set', 'CDELT1', 1)
            self.modify_count_header(1, 'set', 'CRVAL1', self.pixel_list[0])
            self.modify_count_header(1, 'set', 'CTYPE1', 'Pixel (Dispersion)')
            self.modify_count_header(1, 'set', 'CUNIT1', 'Pixel')
            self.modify_count_header(1, 'set', 'BUNIT', 'electron')

            # Add the sky count
            self.modify_count_header(2, 'set', 'LABEL', 'Electron Count (Sky)')
            self.modify_count_header(2, 'set', 'CRPIX1', 1)
            self.modify_count_header(2, 'set', 'CDELT1', 1)
            self.modify_count_header(2, 'set', 'CRVAL1', self.pixel_list[0])
            self.modify_count_header(2, 'set', 'CTYPE1', 'Pixel (Dispersion)')
            self.modify_count_header(2, 'set', 'CUNIT1', 'Pixel')
            self.modify_count_header(2, 'set', 'BUNIT', 'electron')

        except Exception as e:

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('count ImageHDU cannot be created.')
            self.count_hdulist = None

    def create_weight_map_fits(self):

        try:

            # Put the data in ImageHDUs
            weight_map_ImageHDU = fits.ImageHDU(self.var)

            # Create an empty HDU list and populate with ImageHDUs
            self.weight_map_hdulist = fits.HDUList()
            self.weight_map_hdulist += [weight_map_ImageHDU]

            # Add the extraction weights
            self.modify_weight_map_header(0, 'set', 'LABEL',
                                          'Optimal Extraction Profile')
            if self.var is not None:
                self.modify_weight_map_header(0, 'set', 'CRVAL1',
                                              len(self.var))
                self.modify_weight_map_header(0, 'set', 'CRPIX1', 1)
                self.modify_weight_map_header(0, 'set', 'CDELT1', 1)
                self.modify_weight_map_header(0, 'set', 'CTYPE1',
                                              'Pixel (Spatial)')
                self.modify_weight_map_header(0, 'set', 'CUNIT1', 'Pixel')
                self.modify_weight_map_header(0, 'set', 'BUNIT', 'weights')
            else:
                self.modify_weight_map_header(
                    0, 'set', 'COMMENT',
                    'Extraction Profile is not available.')

        except Exception as e:

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('A weight map ImageHDU cannot be created.')
            self.weight_map_hdulist = None

    def create_count_resampled_fits(self):

        try:

            # Put the data in ImageHDUs
            count_resampled_ImageHDU = fits.ImageHDU(self.count_resampled)
            count_err_resampled_ImageHDU = fits.ImageHDU(
                self.count_err_resampled)
            count_sky_resampled_ImageHDU = fits.ImageHDU(
                self.count_sky_resampled)

            # Create an empty HDU list and populate with ImageHDUs
            self.count_resampled_hdulist = fits.HDUList()
            self.count_resampled_hdulist += [count_resampled_ImageHDU]
            self.count_resampled_hdulist += [count_err_resampled_ImageHDU]
            self.count_resampled_hdulist += [count_sky_resampled_ImageHDU]

            # Add the resampled count
            self.modify_count_resampled_header(0, 'set', 'LABEL',
                                               'Resampled Electron Count')
            self.modify_count_resampled_header(0, 'set', 'CRPIX1', 1.00E+00)
            self.modify_count_resampled_header(0, 'set', 'CDELT1',
                                               self.wave_bin)
            self.modify_count_resampled_header(0, 'set', 'CRVAL1',
                                               self.wave_start)
            self.modify_count_resampled_header(0, 'set', 'CTYPE1',
                                               'Wavelength')
            self.modify_count_resampled_header(0, 'set', 'CUNIT1', 'Angstroms')
            self.modify_count_resampled_header(0, 'set', 'BUNIT', 'electron')

            # Add the resampled uncertainty count
            self.modify_count_resampled_header(
                1, 'set', 'LABEL', 'Resampled Electron Count (Uncertainty)')
            self.modify_count_resampled_header(1, 'set', 'CRPIX1', 1.00E+00)
            self.modify_count_resampled_header(1, 'set', 'CDELT1',
                                               self.wave_bin)
            self.modify_count_resampled_header(1, 'set', 'CRVAL1',
                                               self.wave_start)
            self.modify_count_resampled_header(1, 'set', 'CTYPE1',
                                               'Wavelength')
            self.modify_count_resampled_header(1, 'set', 'CUNIT1', 'Angstroms')
            self.modify_count_resampled_header(1, 'set', 'BUNIT', 'electron')

            # Add the resampled sky count
            self.modify_count_resampled_header(
                2, 'set', 'LABEL', 'Resampled Electron Count (Sky)')
            self.modify_count_resampled_header(2, 'set', 'CRPIX1', 1.00E+00)
            self.modify_count_resampled_header(2, 'set', 'CDELT1',
                                               self.wave_bin)
            self.modify_count_resampled_header(2, 'set', 'CRVAL1',
                                               self.wave_start)
            self.modify_count_resampled_header(2, 'set', 'CTYPE1',
                                               'Wavelength')
            self.modify_count_resampled_header(2, 'set', 'CUNIT1', 'Angstroms')
            self.modify_count_resampled_header(2, 'set', 'BUNIT', 'electron')

        except Exception as e:

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('count_resampled ImageHDU cannot be created.')
            self.count_resampled_hdulist = None

    def create_arc_spec_fits(self):

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
            self.modify_arc_spec_header(0, 'set', 'LABEL', 'Electron Count')
            self.modify_arc_spec_header(0, 'set', 'CRPIX1', 1)
            self.modify_arc_spec_header(0, 'set', 'CDELT1', 1)
            self.modify_arc_spec_header(0, 'set', 'CRVAL1', self.pixel_list[0])
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

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('arc_spec ImageHDU cannot be created.')
            self.arc_spec_hdulist = None

    def create_wavecal_fits(self):

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
            self.modify_wavecal_header('set', 'RNGTOL', self.range_tolerance)
            self.modify_wavecal_header('set', 'FITTOL', self.fit_tolerance)
            self.modify_wavecal_header('set', 'CANTHRE',
                                       self.candidate_tolerance)
            self.modify_wavecal_header('set', 'LINTHRE',
                                       self.linearity_tolerance)
            self.modify_wavecal_header('set', 'RANTHRE', self.ransac_tolerance)
            self.modify_wavecal_header('set', 'NUMCAN', self.num_candidates)
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
            self.modify_wavecal_header('set', 'PUSAGE', self.peak_utilisation)
            self.modify_wavecal_header('set', 'LABEL', 'Electron Count')
            self.modify_wavecal_header('set', 'CRPIX1', 1.00E+00)
            self.modify_wavecal_header('set', 'CDELT1', self.wave_bin)
            self.modify_wavecal_header('set', 'CRVAL1', self.wave_start)
            self.modify_wavecal_header('set', 'CTYPE1', 'Wavelength')
            self.modify_wavecal_header('set', 'CUNIT1', 'Angstroms')
            self.modify_wavecal_header('set', 'BUNIT', 'electron')

        except Exception as e:

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('wavecal ImageHDU cannot be created.')
            self.wavecal_hdulist = None

    def create_sensitivity_fits(self):

        try:

            # Put the data in ImageHDUs
            sensitivity_ImageHDU = fits.ImageHDU(self.sensitivity)

            # Create an empty HDU list and populate with ImageHDUs
            self.sensitivity_hdulist = fits.HDUList()
            self.sensitivity_hdulist += [sensitivity_ImageHDU]

            self.modify_sensitivity_header('set', 'LABEL', 'Sensitivity')
            self.modify_sensitivity_header('set', 'CRPIX1', 1.00E+00)
            self.modify_sensitivity_header('set', 'CDELT1', 1)
            self.modify_sensitivity_header('set', 'CRVAL1', self.pixel_list[0])
            self.modify_sensitivity_header('set', 'CTYPE1', 'Pixel')
            self.modify_sensitivity_header('set', 'CUNIT1', 'Pixel')
            self.modify_sensitivity_header('set', 'BUNIT',
                                           'erg/(s*cm**2*Angstrom)/Count')

        except Exception as e:

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('sensitivity ImageHDU cannot be created.')
            self.sensitivity_hdulist = None

    def create_flux_fits(self):

        try:

            # Put the data in ImageHDUs
            flux_ImageHDU = fits.ImageHDU(self.flux)
            flux_err_ImageHDU = fits.ImageHDU(self.flux_err)
            flux_sky_ImageHDU = fits.ImageHDU(self.flux_sky)

            # Create an empty HDU list and populate with ImageHDUs
            self.flux_hdulist = fits.HDUList()
            self.flux_hdulist += [flux_ImageHDU]
            self.flux_hdulist += [flux_err_ImageHDU]
            self.flux_hdulist += [flux_sky_ImageHDU]

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

        except Exception as e:

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('flux ImageHDU cannot be created.')
            self.flux_hdulist = None

    def create_sensitivity_resampled_fits(self):

        try:

            # Put the data in ImageHDUs
            sensitivity_resampled_ImageHDU = fits.ImageHDU(
                self.sensitivity_resampled)

            # Create an empty HDU list and populate with ImageHDUs
            self.sensitivity_resampled_hdulist = fits.HDUList()
            self.sensitivity_resampled_hdulist += [
                sensitivity_resampled_ImageHDU
            ]

            self.modify_sensitivity_resampled_header('set', 'LABEL',
                                                     'Sensitivity')
            self.modify_sensitivity_resampled_header('set', 'CRPIX1', 1.00E+00)
            self.modify_sensitivity_resampled_header('set', 'CDELT1', 1)
            self.modify_sensitivity_resampled_header('set', 'CRVAL1',
                                                     self.pixel_list[0])
            self.modify_sensitivity_resampled_header('set', 'CTYPE1', 'Pixel')
            self.modify_sensitivity_resampled_header('set', 'CUNIT1', 'Pixel')
            self.modify_sensitivity_resampled_header(
                'set', 'BUNIT', 'erg/(s*cm**2*Angstrom)/Count')

        except Exception as e:

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('sensitivity ImageHDU cannot be created.')
            self.sensitivity_resampled_hdulist = None

    def create_flux_resampled_fits(self):

        try:

            # Put the data in ImageHDUs
            flux_resampled_ImageHDU = fits.ImageHDU(self.flux_resampled)
            flux_err_resampled_ImageHDU = fits.ImageHDU(
                self.flux_err_resampled)
            flux_sky_resampled_ImageHDU = fits.ImageHDU(
                self.flux_sky_resampled)

            # Create an empty HDU list and populate with ImageHDUs
            self.flux_resampled_hdulist = fits.HDUList()
            self.flux_resampled_hdulist += [flux_resampled_ImageHDU]
            self.flux_resampled_hdulist += [flux_err_resampled_ImageHDU]
            self.flux_resampled_hdulist += [flux_sky_resampled_ImageHDU]

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_resampled_header(0, 'set', 'LABEL', 'Flux')
            self.modify_flux_resampled_header(0, 'set', 'CRPIX1', 1.00E+00)
            self.modify_flux_resampled_header(0, 'set', 'CDELT1',
                                              self.wave_bin)
            self.modify_flux_resampled_header(0, 'set', 'CRVAL1',
                                              self.wave_start)
            self.modify_flux_resampled_header(0, 'set', 'CTYPE1', 'Wavelength')
            self.modify_flux_resampled_header(0, 'set', 'CUNIT1', 'Angstroms')
            self.modify_flux_resampled_header(0, 'set', 'BUNIT',
                                              'erg/(s*cm**2*Angstrom)')

            self.modify_flux_resampled_header(1, 'set', 'LABEL', 'Flux')
            self.modify_flux_resampled_header(1, 'set', 'CRPIX1', 1.00E+00)
            self.modify_flux_resampled_header(1, 'set', 'CDELT1',
                                              self.wave_bin)
            self.modify_flux_resampled_header(1, 'set', 'CRVAL1',
                                              self.wave_start)
            self.modify_flux_resampled_header(1, 'set', 'CTYPE1', 'Wavelength')
            self.modify_flux_resampled_header(1, 'set', 'CUNIT1', 'Angstroms')
            self.modify_flux_resampled_header(1, 'set', 'BUNIT',
                                              'erg/(s*cm**2*Angstrom)')

            self.modify_flux_resampled_header(2, 'set', 'LABEL', 'Flux')
            self.modify_flux_resampled_header(2, 'set', 'CRPIX1', 1.00E+00)
            self.modify_flux_resampled_header(2, 'set', 'CDELT1',
                                              self.wave_bin)
            self.modify_flux_resampled_header(2, 'set', 'CRVAL1',
                                              self.wave_start)
            self.modify_flux_resampled_header(2, 'set', 'CTYPE1', 'Wavelength')
            self.modify_flux_resampled_header(2, 'set', 'CUNIT1', 'Angstroms')
            self.modify_flux_resampled_header(2, 'set', 'BUNIT',
                                              'erg/(s*cm**2*Angstrom)')

        except Exception as e:

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('flux_resampled ImageHDU cannot be created.')
            self.flux_resampled_hdulist = None

    def create_wavelength_fits(self):

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
            self.modify_wavelength_header('set', 'CRVAL1', self.pixel_list[0])
            self.modify_wavelength_header('set', 'CTYPE1',
                                          'Pixel (Dispersion)')
            self.modify_wavelength_header('set', 'CUNIT1', 'Pixel')
            self.modify_wavelength_header('set', 'BUNIT', 'Angstroms')

        except Exception as e:

            warnings.warn(str(e))

            # Set it to None if the above failed
            warnings.warn('wavelength ImageHDU cannot be created.')
            self.wavelength_hdulist = None

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
            count: 3 HDUs
                Count, uncertainty, and sky (pixel)
            weight_map: 1 HDU
                Weight (pixel)
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
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        return_hdu_list: boolean (default: False)
            Set to True to return the HDU List

        '''

        output_split = output.split('+')

        # Will check if the HDUs need to be (re)created inside the functions
        if 'trace' in output_split:

            self.create_trace_fits()

        if 'count' in output_split:

            self.create_count_fits()

        if 'weight_map' in output_split:

            self.create_count_fits()

        if 'arc_spec' in output_split:

            self.create_arc_spec_fits()

        if 'wavecal' in output_split:

            self.create_wavecal_fits()

        if 'wavelength' in output_split:

            self.create_wavelength_fits()

        if 'count_resampled' in output_split:

            self.create_count_resampled_fits()

        if 'sensitivity' in output_split:

            self.create_sensitivity_fits()

        if 'flux' in output_split:

            self.create_flux_fits()

        if 'sensitivity_resampled' in output_split:

            self.create_sensitivity_resampled_fits()

        if 'flux_resampled' in output_split:

            self.create_flux_resampled_fits()

        # If the requested list of HDUs is already good to go
        if set([k for k, v in self.hdu_content.items()
                if v]) == set(output_split):

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
            for k, v in self.hdu_content.items():
                self.hdu_content[k] = False

            # Empty list for appending HDU lists
            hdu_output = fits.HDUList()

            # If leaving the primary HDU empty
            if empty_primary_hdu:

                hdu_output.append(fits.PrimaryHDU())

            # Join the list(s)
            if 'trace' in output_split and not self.hdu_content['trace']:

                hdu_output += self.trace_hdulist
                self.hdu_content['trace'] = True

            if 'count' in output_split and not self.hdu_content['count']:

                hdu_output += self.count_hdulist
                self.hdu_content['count'] = True

            if 'weight_map' in output_split and not self.hdu_content[
                    'weight_map']:

                hdu_output += self.count_hdulist
                self.hdu_content['weight_map'] = True

            if 'arc_spec' in output_split and not self.hdu_content['arc_spec']:

                hdu_output += self.arc_spec_hdulist
                self.hdu_content['arc_spec'] = True

            if 'wavecal' in output_split and not self.hdu_content['wavecal']:

                hdu_output += self.wavecal_hdulist
                self.hdu_content['wavecal'] = True

            if 'wavelength' in output_split and not self.hdu_content[
                    'wavelength']:

                hdu_output += self.wavelength_hdulist
                self.hdu_content['wavelength'] = True

            if 'count_resampled' in output_split and not self.hdu_content[
                    'count_resampled']:

                hdu_output += self.count_resampled_hdulist
                self.hdu_content['count_resampled'] = True

            if 'sensitivity' in output_split and not self.hdu_content[
                    'sensitivity']:

                hdu_output += self.sensitivity_hdulist
                self.hdu_content['sensitivity'] = True

            if 'flux' in output_split and not self.hdu_content['flux']:

                hdu_output += self.flux_hdulist
                self.hdu_content['flux'] = True

            if 'sensitivity_resampled' in output_split and not\
                    self.hdu_content['sensitivity_resampled']:

                hdu_output += self.sensitivity_resampled_hdulist
                self.hdu_content['sensitivity_resampled'] = True

            if 'flux_resampled' in output_split and not self.hdu_content[
                    'flux_resampled']:

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
            count: 3 HDUs
                Count, uncertainty, and sky (pixel)
            weight_map: 1 HDU
                Weight (pixel)
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
        overwrite: boolean
            Default is False.
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)

        '''

        self.create_fits(output, empty_primary_hdu)

        # Save file to disk
        self.hdu_output.writeto(filename + '.fits', overwrite=overwrite)

    def save_csv(self, output, filename, overwrite):
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
            count: 4 HDUs
                Count, uncertainty, sky, and optimal flag (pixel)
            weight_map: 1 HDU
                Weight (pixel)
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
        overwrite: boolean
            Default is False.

        '''

        self.create_fits(output, empty_primary_hdu=False)

        output_split = output.split('+')

        start = 0

        for output_type in self.hdu_order.keys():

            if output_type in output_split:

                end = start + self.n_hdu[output_type]

                if output_type != 'arc_spec':

                    output_data = np.column_stack(
                        [hdu.data for hdu in self.hdu_output[start:end]])

                    if overwrite or (not os.path.exists(filename + '_' +
                                                        output_type + '.csv')):

                        np.savetxt(filename + '_' + output_type + '.csv',
                                   output_data,
                                   delimiter=',',
                                   header=self.header[output_type])

                else:

                    output_data_arc_spec = self.hdu_output[start].data
                    output_data_arc_peaks = np.column_stack(
                        [hdu.data for hdu in self.hdu_output[start + 1:end]])

                    if overwrite or (
                            not os.path.exists(filename + '_arc_spec.csv')):

                        np.savetxt(filename + '_arc_spec.csv',
                                   output_data_arc_spec,
                                   delimiter=',',
                                   header=self.header[output_type])

                    if overwrite or (
                            not os.path.exists(filename + '_arc_peaks.csv')):

                        np.savetxt(filename + '_arc_peaks.csv',
                                   output_data_arc_peaks,
                                   delimiter=',',
                                   header=self.header[output_type])

                start = end