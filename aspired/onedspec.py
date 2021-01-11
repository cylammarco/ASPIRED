import copy
import numpy as np
import os
from plotly import graph_objects as go
from plotly import io as pio
from spectres import spectres

from .wavelengthcalibration import WavelengthCalibration
from .fluxcalibration import FluxCalibration
from .spectrum1D import Spectrum1D


class OneDSpec():
    def __init__(self, verbose=True):
        '''
        This class applies the wavelength calibrations and compute & apply the
        flux calibration to the extracted 1D spectra. The standard TwoDSpec
        object is not required for data reduction, but the flux calibrated
        standard observation will not be available for diagnostic.

        Parameters
        ----------
        verbose: boolean
            Set to True to suppress all verbose warnings.

        '''

        self.science_imported = False
        self.standard_imported = False
        self.arc_science_imported = False
        self.arc_standard_imported = False
        self.arc_lines_science_imported = False
        self.arc_lines_standard_imported = False

        self.science_wavecal = [WavelengthCalibration(verbose)]
        self.standard_wavecal = WavelengthCalibration(verbose)
        self.fluxcal = FluxCalibration(verbose)

        self.science_wavelength_calibrated = False
        self.standard_wavelength_calibrated = False

        self.science_flux_calibrated = False
        self.standard_flux_calibrated = False

        self.science_spectrum_list = {0: Spectrum1D(0)}
        self.standard_spectrum_list = {0: Spectrum1D(0)}

        # Link them up
        self.science_wavecal[0].from_spectrum1D(self.science_spectrum_list[0])
        self.standard_wavecal.from_spectrum1D(self.standard_spectrum_list[0])
        self.fluxcal.from_spectrum1D(self.standard_spectrum_list[0])

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

        except Exception as e:

            raise TypeError(
                'Please provide a valid FluxCalibration(): {}'.format(e))

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

            if s == 'science':

                self.science_wavecal = wavecal

            if s == 'standard':

                self.standard_wavecal = wavecal

    def add_wavelength(self, wave, spec_id=None, stype='science+standard'):
        '''
        Parameters
        ----------
        wave : numeric value, list or numpy 1D array (N) (default: None)
            The wavelength of each pixels of the spectrum.
        spec_id: int
            The ID corresponding to the spectrum1D object
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].add_wavelength(wave=wave)

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_spectrum_list[0].add_wavelength(wave=wave)

    def add_wavelength_resampled(self,
                                 wave_resampled,
                                 spec_id=None,
                                 stype='science+standard'):
        '''
        Parameters
        ----------
        wave_resampled:
            The wavelength of the resampled spectrum.
        spec_id: int
            The ID corresponding to the spectrum1D object
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].add_wavelength_resampled(
                        wave_resampled=wave_resampled)

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_spectrum_list[0].add_wavelength_resampled(
                    wave_resampled=wave_resampled)

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

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if spec_id is not None:

                if spec_id not in list(self.science_spectrum_list.keys()):

                    raise ValueError('The given spec_id does not exist.')

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            for i in spec_id:

                self.science_spectrum_list[i].add_count(count=count,
                                                        count_err=count_err,
                                                        count_sky=count_sky)

            self.science_imported = True

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].add_count(count=count,
                                                     count_err=count_err,
                                                     count_sky=count_sky)

            self.standard_imported = True

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].add_arc_lines(peaks=peaks)

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_spectrum_list[0].add_arc_lines(peaks=peaks)

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].add_arc_spec(
                        arc_spec=arc_spec)

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_spectrum_list[0].add_arc_spec(arc_spec=arc_spec)

    def add_arc(self, arc, spec_id=None, stype='science+standard'):
        '''
        arc: 2D numpy array (M x N) OR astropy.io.fits object
            2D spectral image in either format
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].add_arc(arc=arc)

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_spectrum_list[0].add_arc(arc=arc)

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].add_trace(
                        trace=trace,
                        trace_sigma=trace_sigma,
                        pixel_list=pixel_list)

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_spectrum_list[0].add_trace(
                    trace=trace,
                    trace_sigma=trace_sigma,
                    pixel_list=pixel_list)

    def add_fit_coeff(self,
                      fit_coeff,
                      fit_type='poly',
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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].add_fit_coeff(
                        fit_coeff=fit_coeff, fit_type=fit_type)

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_spectrum_list[0].add_fit_coeff(
                    fit_coeff=fit_coeff, fit_type=fit_type)

    def from_twodspec(self,
                      twodspec,
                      spec_id=None,
                      deep_copy=False,
                      stype='science+standard'):
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
            TwoDSpec of the science image containin the trace(s) and
            trace_sigma(s).
        deep_copy: boolean
            Set to true to clone the spectrum_list from twodspec.
        stype: string (default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if deep_copy:

                self.science_spectrum_list = copy.deepcopy(
                    twodspec.spectrum_list)

            else:

                self.science_spectrum_list = twodspec.spectrum_list

            self.science_wavecal =\
                [copy.deepcopy(self.science_wavecal[0]) for i in range(
                    len(self.science_spectrum_list))]

            if spec_id is not None:

                if spec_id not in list(self.science_spectrum_list.keys()):

                    raise ValueError('The given spec_id does not exist.')

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            # reference the spectrum1D to the WavelengthCalibration
            for i in spec_id:

                # By reference
                self.science_wavecal[i].from_spectrum1D(
                    self.science_spectrum_list[i])

            self.science_imported = True

        if 'standard' in stype_split:

            if deep_copy:

                self.standard_spectrum_list = copy.deepcopy(
                    twodspec.spectrum_list)

            else:

                self.standard_spectrum_list = twodspec.spectrum_list

            # By reference
            self.standard_wavecal.from_spectrum1D(
                self.standard_spectrum_list[0])
            self.fluxcal.from_spectrum1D(self.standard_spectrum_list[0])

            self.standard_imported = True

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_wavecal[i].find_arc_lines(
                        background=background,
                        percentile=percentile,
                        prominence=prominence,
                        distance=distance,
                        refine=refine,
                        refine_window_width=refine_window_width,
                        display=display,
                        renderer=renderer,
                        width=width,
                        height=height,
                        return_jsonstring=return_jsonstring,
                        save_iframe=save_iframe,
                        filename=filename,
                        open_iframe=open_iframe)

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_wavecal.find_arc_lines(
                    background=background,
                    percentile=percentile,
                    prominence=prominence,
                    distance=distance,
                    refine=refine,
                    refine_window_width=refine_window_width,
                    display=display,
                    renderer=renderer,
                    width=width,
                    height=height,
                    return_jsonstring=return_jsonstring,
                    save_iframe=save_iframe,
                    filename=filename,
                    open_iframe=open_iframe)

            else:

                raise RuntimeError('Standard spectrum/a are not imported.')

    def initialise_calibrator(self,
                              spec_id=None,
                              peaks=None,
                              arc_spec=None,
                              stype='science+standard'):
        '''
        If the peaks were found with find_arc_lines(), peaks and spectrum can
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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    if peaks is None:

                        peaks = self.science_spectrum_list[i].peaks

                    if arc_spec is None:

                        arc_spec = self.science_spectrum_list[i].count

                    self.science_wavecal[i].from_spectrum1D(
                        self.science_spectrum_list[i])
                    self.science_wavecal[i].initialise_calibrator(
                        peaks=peaks, arc_spec=arc_spec)
                    self.science_wavecal[i].set_hough_properties()
                    self.science_wavecal[i].set_ransac_properties()

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                if peaks is None:

                    peaks = self.standard_spectrum_list[0].peaks

                if arc_spec is None:

                    arc_spec = self.standard_spectrum_list[0].count

                self.standard_wavecal.from_spectrum1D(
                    self.standard_spectrum_list[0])
                self.standard_wavecal.initialise_calibrator(peaks=peaks,
                                                            arc_spec=arc_spec)
                self.standard_wavecal.set_hough_properties()
                self.standard_wavecal.set_ransac_properties()

            else:

                raise RuntimeError('Standard spectrum is not imported.')

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_wavcal[i].set_calibrator_properties(
                        num_pix=num_pix,
                        pixel_list=pixel_list,
                        plotting_library=plotting_library,
                        log_level=log_level)

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_wavecal.set_calibrator_properties(
                    num_pix=num_pix,
                    pixel_list=pixel_list,
                    plotting_library=plotting_library,
                    log_level=log_level)

            else:

                raise RuntimeError('Standard spectrum/a are not imported.')

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_wavecal[i].set_hough_properties(
                        num_slopes=num_slopes,
                        xbins=xbins,
                        ybins=ybins,
                        min_wavelength=min_wavelength,
                        max_wavelength=max_wavelength,
                        range_tolerance=range_tolerance,
                        linearity_tolerance=linearity_tolerance)

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_wavecal.set_hough_properties(
                    num_slopes=num_slopes,
                    xbins=xbins,
                    ybins=ybins,
                    min_wavelength=min_wavelength,
                    max_wavelength=max_wavelength,
                    range_tolerance=range_tolerance,
                    linearity_tolerance=linearity_tolerance)

            else:

                raise RuntimeError('Standard spectrum/a are not imported.')

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_wavecal[i].set_ransac_properties(
                        sample_size=sample_size,
                        top_n_candidate=top_n_candidate,
                        linear=linear,
                        filter_close=filter_close,
                        ransac_tolerance=ransac_tolerance,
                        candidate_weighted=candidate_weighted,
                        hough_weight=hough_weight)

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_wavecal.set_ransac_properties(
                    sample_size=sample_size,
                    top_n_candidate=top_n_candidate,
                    linear=linear,
                    filter_close=filter_close,
                    ransac_tolerance=ransac_tolerance,
                    candidate_weighted=candidate_weighted,
                    hough_weight=hough_weight)

            else:

                raise RuntimeError('Standard spectrum/a are not imported.')

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_wavecal[i].set_known_pairs(pix=pix, wave=wave)

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_wavecal.set_known_pairs(pix=pix, wave=wave)

            else:

                raise RuntimeError('Standard spectrum/a are not imported.')

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
        *Remove* all the arc lines loaded to the Calibrator and then use the
        user supplied arc lines instead.

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_wavecal[i].load_user_atlas(
                        elements=elements,
                        wavelengths=wavelengths,
                        intensities=intensities,
                        candidate_tolerance=candidate_tolerance,
                        constrain_poly=constrain_poly,
                        vacuum=vacuum,
                        pressure=pressure,
                        temperature=temperature,
                        relative_humidity=relative_humidity)

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_wavecal.load_user_atlas(
                    elements=elements,
                    wavelengths=wavelengths,
                    intensities=intensities,
                    candidate_tolerance=candidate_tolerance,
                    constrain_poly=constrain_poly,
                    vacuum=vacuum,
                    pressure=pressure,
                    temperature=temperature,
                    relative_humidity=relative_humidity)

            else:

                raise RuntimeError('Standard spectrum/a are not imported.')

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
            Minimum intensity of the arc lines. Refer to NIST for the
            intensity.
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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_wavecal[i].add_atlas(
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

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_wavecal.add_atlas(
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

            else:

                raise RuntimeError('Standard spectrum/a are not imported.')

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_wavecal[i].do_hough_transform()

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_wavecal.do_hough_transform()

            else:

                raise RuntimeError('Standard spectrum/a are not imported.')

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_wavecal[i].fit(max_tries=max_tries,
                                                fit_deg=fit_deg,
                                                fit_coeff=fit_coeff,
                                                fit_tolerance=fit_tolerance,
                                                fit_type=fit_type,
                                                brute_force=brute_force,
                                                progress=progress,
                                                display=display,
                                                savefig=savefig,
                                                filename=filename)

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_wavecal.fit(max_tries=max_tries,
                                          fit_deg=fit_deg,
                                          fit_coeff=fit_coeff,
                                          fit_tolerance=fit_tolerance,
                                          fit_type=fit_type,
                                          brute_force=brute_force,
                                          progress=progress,
                                          display=display,
                                          savefig=savefig,
                                          filename=filename)

            else:

                raise RuntimeError('Standard spectrum/a are not imported.')

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_wavecal[i].refine_fit(
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

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_wavecal.refine_fit(fit_coeff=fit_coeff,
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

            else:

                raise RuntimeError('Standard spectrum/a are not imported.')

    def apply_wavelength_calibration(self,
                                     spec_id=None,
                                     wave_start=None,
                                     wave_end=None,
                                     wave_bin=None,
                                     stype='science+standard'):
        '''
        Apply the wavelength calibration. Because the polynomial fit can run
        away at the two ends, the default minimum and maximum are limited to
        1,000 and 12,000 A, respectively. This can be overridden by providing
        user's choice of wave_start and wave_end.

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

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, contraints are applied to all
                    #  calibrators
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    spec = self.science_spectrum_list[i]

                    # Adjust for pixel shift due to chip gaps
                    wave = self.science_wavecal[i].polyval[spec.fit_type](np.array(spec.pixel_list),
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
                    count_resampled = spectres(
                        np.array(wave_resampled).reshape(-1),
                        np.array(wave).reshape(-1),
                        np.array(spec.count).reshape(-1),
                        verbose=True)

                    if spec.count_err is not None:

                        count_err_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.count_err).reshape(-1),
                            verbose=True)

                    if spec.count_sky is not None:

                        count_sky_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.count_sky).reshape(-1),
                            verbose=True)

                    spec.add_wavelength(wave)
                    spec.add_wavelength_resampled(wave_resampled)
                    spec.add_count_resampled(count_resampled,
                                             count_err_resampled,
                                             count_sky_resampled)

                self.science_wavelength_calibrated = True

            else:

                raise RuntimeError('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_imported:

                spec = self.standard_spectrum_list[0]

                # Adjust for pixel shift due to chip gaps
                wave = self.standard_wavecal.polyval[spec.fit_type](np.array(spec.pixel_list),
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
                count_resampled = spectres(
                    np.array(wave_resampled).reshape(-1),
                    np.array(wave).reshape(-1),
                    np.array(spec.count).reshape(-1),
                    verbose=True)

                if spec.count_err is not None:

                    count_err_resampled = spectres(
                        np.array(wave_resampled).reshape(-1),
                        np.array(wave).reshape(-1),
                        np.array(spec.count_err).reshape(-1),
                        verbose=True)

                if spec.count_sky is not None:

                    count_sky_resampled = spectres(
                        np.array(wave_resampled).reshape(-1),
                        np.array(wave).reshape(-1),
                        np.array(spec.count_sky).reshape(-1),
                        verbose=True)

                spec.add_wavelength(wave)
                spec.add_wavelength_resampled(wave_resampled)
                spec.add_count_resampled(count_resampled, count_err_resampled,
                                         count_sky_resampled)

                self.standard_wavelength_calibrated = True

            else:

                raise RuntimeError('Standard spectrum is not imported.')

            self.standard_wavelength_calibrated = True

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
                            k=3,
                            smooth=False,
                            method='interpolate',
                            slength=5,
                            sorder=3,
                            mask_range=[[6850, 6960], [7575, 7700],
                                        [8925, 9050], [9265, 9750]],
                            mask_fit_order=1,
                            mask_fit_size=1):
        '''
        Parameters
        ----------
        k: integer [1,2,3,4,5 only]
            The order of the spline.
        smooth: boolean (default: False)
            set to smooth the input spectrum with scipy.signal.savgol_filter
        weighted: boolean (Default: True)
            Set to weight the interpolation/polynomial fit by the
            inverse uncertainty.
        method: str (Default: interpolate)
            This should be either 'interpolate' of 'polynomial'. Note that the
            polynomial is computed from the interpolated function. The
            default is interpolate because it is much more stable at the
            wavelength limits of a spectrum in an automated system.
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

        self.fluxcal.compute_sensitivity(k=k,
                                         smooth=smooth,
                                         method=method,
                                         slength=slength,
                                         sorder=sorder,
                                         mask_range=mask_range,
                                         mask_fit_order=mask_fit_order,
                                         mask_fit_size=mask_fit_size)

    def save_sensitivity_func(self, filename='sensitivity_func.npy'):
        '''
        Parameters
        ----------
        filename: str
            Filename for the output interpolated sensivity curve.

        '''

        self.fluxcal.save_sensitivity_func(filename=filename)

    def add_sensitivity_func(self, sensitivity_func):
        '''
        Parameters
        ----------
        sensitivity_func: str
            Interpolated sensivity curve object.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        self.fluxcal.add_sensitivity_func(sensitivity_func=sensitivity_func)

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

        if 'science' in stype_split:

            if spec_id is not None:

                if spec_id not in list(self.science_spectrum_list.keys()):

                    raise ValueError('The given spec_id does not exist.')

            else:

                # if spec_id is None, contraints are applied to all
                #  calibrators
                spec_id = list(self.science_spectrum_list.keys())

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            for i in spec_id:

                self.fluxcal.apply_flux_calibration(
                    target_spectrum1D=self.science_spectrum_list[i])

            self.science_flux_calibrated = True

        if 'standard' in stype_split:

            self.fluxcal.apply_flux_calibration(
                target_spectrum1D=self.standard_spectrum_list[0])

            self.standard_flux_calibrated = True

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

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if spec_id is not None:

                if spec_id not in list(self.science_spectrum_list.keys()):

                    raise ValueError('The given spec_id does not exist.')

            else:

                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.science_spectrum_list.keys())

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            for i in spec_id:

                spec = self.science_spectrum_list[i]

                if self.science_flux_calibrated:

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

                if self.science_flux_calibrated:

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

                    return fig_sci[i].to_json()

        if 'standard' in stype_split:

            spec = self.standard_spectrum_list[0]

            if self.standard_flux_calibrated:

                standard_wave_mask = (
                    (np.array(spec.wave_resampled).reshape(-1) > wave_min) &
                    (np.array(spec.wave_resampled).reshape(-1) < wave_max))
                standard_flux_mask = (
                    (np.array(spec.flux_resampled).reshape(-1) >
                     np.nanpercentile(
                         np.array(spec.flux_resampled).reshape(-1)
                         [standard_wave_mask], 5) / 1.5) &
                    (np.array(spec.flux_resampled).reshape(-1) <
                     np.nanpercentile(
                         np.array(spec.flux_resampled).reshape(-1)
                         [standard_wave_mask], 95) * 1.5))
                standard_flux_min = np.log10(
                    np.nanmin(
                        np.array(spec.flux_resampled).reshape(-1)
                        [standard_flux_mask]))
                standard_flux_max = np.log10(
                    np.nanmax(
                        np.array(spec.flux_resampled).reshape(-1)
                        [standard_flux_mask]))

            else:

                standard_wave_mask = (
                    (np.array(spec.wave_resampled).reshape(-1) > wave_min) &
                    (np.array(spec.wave_resampled).reshape(-1) < wave_max))
                standard_flux_mask = (np.array(
                    spec.count_resampled).reshape(-1) > np.nanpercentile(
                        np.array(spec.count_resampled).reshape(-1)
                        [standard_wave_mask], 5) / 1.5)
                standard_flux_min = np.log10(
                    np.nanmin(
                        np.array(spec.count_resampled).reshape(-1)
                        [standard_flux_mask]))
                standard_flux_max = np.log10(
                    np.nanmax(
                        np.array(spec.count_resampled).reshape(-1)
                        [standard_flux_mask]))

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
            if self.standard_flux_calibrated:

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

                if self.fluxcal.standard_fluxmag_true is not None:

                    fig_standard.add_trace(
                        go.Scatter(x=self.fluxcal.standard_wave_true,
                                   y=self.fluxcal.standard_fluxmag_true,
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
                           range=[standard_flux_min, standard_flux_max],
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

    def create_fits(self,
                    spec_id=None,
                    output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                    stype='science+standard',
                    recreate=True,
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
            count: 4 HDUs
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
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        return_id: boolean (default: False)
            Set to True to return the set of spec_id

        '''

        # Split the string into strings
        stype_split = stype.split('+')
        output_split = output.split('+')

        for i in output_split:

            if i not in [
                    'trace', 'count', 'weight_map', 'arc_spec', 'wavecal',
                    'wavelength', 'count_resampled', 'flux', 'flux_resampled'
            ]:

                raise ValueError('%s is not a valid output.' % i)

        if ('science' not in stype_split) and ('standard' not in stype_split):

            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, contraints are applied to all
                    #  calibrators
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].create_fits(
                        output=output,
                        recreate=recreate,
                        empty_primary_hdu=empty_primary_hdu)

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_spectrum_list[0].create_fits(
                    output=output,
                    recreate=recreate,
                    empty_primary_hdu=empty_primary_hdu)

    def modify_trace_header(self,
                            idx,
                            method,
                            *args,
                            spec_id=None,
                            stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].modify_trace_header(
                        idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].modify_trace_header(
                    idx, method, *args)

    def modify_count_header(self,
                            idx,
                            method,
                            *args,
                            spec_id=None,
                            stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].modify_count_header(
                        idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].modify_count_header(
                    idx, method, *args)

    def modify_weight_map_header(self,
                                 idx,
                                 method,
                                 *args,
                                 spec_id=None,
                                 stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].modify_weight_map_header(
                        idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].modify_weight_map_header(
                    idx, method, *args)

    def modify_count_resampled_header(self,
                                      idx,
                                      method,
                                      *args,
                                      spec_id=None,
                                      stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[
                        i].modify_count_resampled_header(idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].modify_count_resampled_header(
                    idx, method, *args)

    def modify_arc_spec_header(self,
                               idx,
                               method,
                               *args,
                               spec_id=None,
                               stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].modify_arc_spec_header(
                        idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].modify_arc_spec_header(
                    idx, method, *args)

    def modify_wavecal_header(self,
                              idx,
                              method,
                              *args,
                              spec_id=None,
                              stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].modify_wavecal_header(
                        idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].modify_wavecal_header(
                    idx, method, *args)

    def modify_wavelength_header(self,
                                 idx,
                                 method,
                                 *args,
                                 spec_id=None,
                                 stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].modify_wavelength_header(
                        idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].modify_wavelength_header(
                    idx, method, *args)

    def modify_sensitivity_header(self,
                                  idx,
                                  method,
                                  *args,
                                  spec_id=None,
                                  stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].modify_sensitivity_header(
                        idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].modify_sensitivity_header(
                    idx, method, *args)

    def modify_flux_header(self,
                           idx,
                           method,
                           *args,
                           spec_id=None,
                           stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].modify_flux_header(
                        idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].modify_flux_header(
                    idx, method, *args)

    def modify_sensitibity_resampled_header(self,
                                            idx,
                                            method,
                                            *args,
                                            spec_id=None,
                                            stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[
                        i].modify_sensitivity_resampled_header(
                            idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[
                    0].modify_sensitivity_resampled_header(idx, method, *args)

    def modify_flux_resampled_header(self,
                                     idx,
                                     method,
                                     *args,
                                     spec_id=None,
                                     stype='science+standard'):

        # Split the string into strings

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    self.science_spectrum_list[i].modify_flux_resampled_header(
                        idx, method, *args)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].modify_flux_resampled_header(
                    idx, method, *args)

    def save_fits(self,
                  spec_id=None,
                  output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                  filename='reduced',
                  stype='science+standard',
                  recreate=False,
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
            count: 4 HDUs
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
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        overwrite: boolean (default: False)
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        stype_split = stype.split('+')
        output_split = output.split('+')

        for i in output_split:

            if i not in [
                    'trace', 'count', 'weight_map', 'arc_spec', 'wavecal',
                    'wavelength', 'count_resampled', 'flux', 'flux_resampled'
            ]:

                raise ValueError('%s is not a valid output.' % i)

        if ('science' not in stype_split) and ('standard' not in stype_split):

            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, contraints are applied to all
                    #  calibrators
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    filename_i = filename + '_science_' + str(i)

                    self.science_spectrum_list[i].save_fits(
                        output=output,
                        filename=filename_i,
                        overwrite=overwrite,
                        recreate=recreate,
                        empty_primary_hdu=empty_primary_hdu)

        if 'standard' in stype_split:

            if self.standard_imported:

                self.standard_spectrum_list[0].save_fits(
                    output=output,
                    filename=filename + '_standard',
                    overwrite=overwrite,
                    recreate=recreate,
                    empty_primary_hdu=empty_primary_hdu)

    def save_csv(self,
                 spec_id=None,
                 output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                 filename='reduced',
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
            (default: 'arc_spec+wavecal+wavelength+flux+flux_resampled')
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 4 HDUs
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
        overwrite: boolean (default: False)
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        stype_split = stype.split('+')
        output_split = output.split('+')

        for i in output_split:

            if i not in [
                    'trace', 'count', 'weight_map', 'arc_spec', 'wavecal',
                    'wavelength', 'count_resampled', 'flux', 'flux_resampled'
            ]:

                raise ValueError('%s is not a valid output.' % i)

        if ('science' not in stype_split) and ('standard' not in stype_split):

            raise ValueError('Unknown stype, please choose from (1) science; '
                             'and/or (2) standard. use + as delimiter.')

        if 'science' in stype_split:

            if self.science_imported:

                if spec_id is not None:

                    if spec_id not in list(self.science_spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, contraints are applied to all
                    #  calibrators
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    filename_i = filename + '_science_' + str(i)

                    self.science_spectrum_list[i].save_csv(output=output,
                                                           filename=filename_i,
                                                           overwrite=overwrite)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.standard_imported:

                self.standard_spectrum_list[0].save_csv(output=output,
                                                        filename=filename +
                                                        '_standard',
                                                        overwrite=overwrite)
