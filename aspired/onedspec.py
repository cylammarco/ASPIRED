import os
import warnings

from .wavelengthcalibration import WavelengthCalibration
from .fluxcalibration import FluxCalibration


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
        self.sensitivity = Sensitivity(silence)

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

        except Exception as e:

            raise TypeError(
                'Please provide a valid StandardFlux: {}'.format(e))

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

                self.wavecal_science_imported = True

            if s == 'standard':

                self.wavecal_standard_imported = True

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
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''
        """

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
        """

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

    def apply_spatial_mask_to_arc(self,
                                  spatial_mask,
                                  stype='science+standard'):
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

            for spectrum_i in self.spectrum_list:
                arc_spec = spectrum_i.arc_spec

                peaks_raw = self.wavecal_science.find_arc_lines(
                    arc_spec=arc_spec,
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

                spectrum_i.add_background(background)
                spectrum_i.add_peaks_raw(list(peaks_raw))
                spectrum_i.add_peaks_pixel(
                    list(spectrum_i.pixel_mapping_itp(peaks_raw)))

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

            self.wavecal_science.load_user_atlas(
                elements=elements,
                wavelengths=wavelengths,
                spec_id=spec_id,
                intensities=intensities,
                candidate_tolerance=candidate_tolerance,
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
                candidate_tolerance=candidate_tolerance,
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
        """

        def apply_wavelength_calibration(self,
                                        spec_id=None,
                                        wave_start=None,
                                        wave_end=None,
                                        wave_bin=None):

            if spec_id is not None:

                if spec_id not in list(self.spectrum_list.keys()):

                    raise ValueError('The given spec_id does not exist.')

            else:

                # if spec_id is None, contraints are applied to all
                #  calibrators
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
                count_resampled = spectres(
                    np.array(wave_resampled).reshape(-1),
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

        """

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

    def add_sensitivity_func(self, sensitivity_func, stype='science+standard'):
        '''
        Parameters
        ----------
        sensitivity_func: str
            Interpolated sensivity curve object.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        self.fluxcal.add_sensitivity_func(sensitivity_func=sensitivity_func,
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
            display=display,
            return_jsonstring=return_jsonstring,
            save_iframe=save_iframe,
            open_iframe=open_iframe)

    def create_fits(self,
                    spec_id=None,
                    output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                    stype='science+standard',
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

            if self.flux_science_calibrated:

                if spec_id is not None:

                    if spec_id not in list(
                            self.fluxcal.spectrum_list_science.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, contraints are applied to all
                    # calibrators
                    spec_id = list(self.fluxcal.spectrum_list_science.keys())

            elif self.wavelength_science_calibrated:

                # Note that wavecal ONLY has sepctrum_list, it is not science
                # and standard specified.
                if spec_id is not None:

                    if spec_id not in list(
                            self.wavecal_science.spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, contraints are applied to all
                    # calibrators
                    spec_id = list(self.wavecal_science.spectrum_list.keys())

            else:

                try:
                    print('loading from fluxcal')
                    print(self.fluxcal.spectrum_list_science.keys())

                    if spec_id is not None:
                        print('spec_id is not None.')
                        if spec_id not in list(
                                self.fluxcal.spectrum_list_science.keys()):

                            raise ValueError(
                                'The given spec_id does not exist.')

                    else:
                        print('spec_id is None.')

                        # if spec_id is None, contraints are applied to all
                        # calibrators
                        spec_id = list(
                            self.fluxcal.spectrum_list_science.keys())

                except Exception as e:

                    warnings.warn(str(e))

                    if spec_id is not None:

                        if spec_id not in list(
                                self.wavecal_science.spectrum_list.keys()):

                            raise ValueError(
                                'The given spec_id does not exist.')

                    else:

                        # if spec_id is None, contraints are applied to all
                        # calibrators
                        spec_id = list(
                            self.wavecal_science.spectrum_list.keys())

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            for i in spec_id:

                # If flux is calibrated
                if self.flux_science_calibrated:

                    self.fluxcal.spectrum_list_science[i].create_fits(
                        output=output, empty_primary_hdu=empty_primary_hdu)

                # If flux is not calibrated, but wavelength is calibrated
                # Note that wavecal ONLY has sepctrum_list, it is not science
                # and standard specified.
                elif self.wavelength_science_calibrated:

                    self.wavecal_science.spectrum_list[i].create_fits(
                        output=output, empty_primary_hdu=empty_primary_hdu)

                else:

                    # This is when exporting before any calibration, we
                    # can't be sure whether there is data in the fluxcal
                    # or in the wavecal.
                    try:

                        self.fluxcal.spectrum_list_science[i].create_fits(
                            output=output, empty_primary_hdu=empty_primary_hdu)

                    except Exception as e:

                        warnings.warn(str(e))

                        self.wavecal_science.spectrum_list[i].create_fits(
                            output=output, empty_primary_hdu=empty_primary_hdu)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.flux_standard_calibrated:

                self.fluxcal.spectrum_list_standard[0].create_fits(
                    output=output, empty_primary_hdu=empty_primary_hdu)

            # If flux is not calibrated, but wavelength is calibrated
            # Note that wavecal ONLY has sepctrum_list, it is not science
            # and standard specified.
            elif self.wavelength_standard_calibrated:

                self.wavecal_standard.spectrum_list[0].create_fits(
                    output=output, empty_primary_hdu=empty_primary_hdu)

            else:

                # This is when exporting before any calibration, we
                # can't be sure whether there is data in the fluxcal
                # or in the wavecal.
                try:

                    self.fluxcal.spectrum_list_standard[0].create_fits(
                        output=output, empty_primary_hdu=empty_primary_hdu)

                except Exception as e:

                    warnings.warn(str(e))

                    self.wavecal_standard.spectrum_list[0].create_fits(
                        output=output, empty_primary_hdu=empty_primary_hdu)

        if return_id:

            return spec_id

    def modify_trace_header(self,
                            idx,
                            method,
                            spec_id=None,
                            stype='science+standard',
                            *args):

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.flux_science_calibrated:

                if spec_id is not None:

                    if spec_id not in list(
                            self.fluxcal.spectrum_list_science.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, contraints are applied to all
                    # calibrators
                    spec_id = list(self.fluxcal.spectrum_list_science.keys())

            elif self.wavelength_science_calibrated:

                # Note that wavecal ONLY has sepctrum_list, it is not science
                # and standard specified.
                if spec_id is not None:

                    if spec_id not in list(
                            self.wavecal_science.spectrum_list.keys()):

                        raise ValueError('The given spec_id does not exist.')

                else:

                    # if spec_id is None, contraints are applied to all
                    # calibrators
                    spec_id = list(self.wavecal_science.spectrum_list.keys())

            else:

                raise ValueError('Trace or Count cannot be found. Neither '
                                 'wavelength nor flux is calibrated.')

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            for i in spec_id:

                # If flux is calibrated
                if self.flux_science_calibrated:

                    self.fluxcal.spectrum_list_science[i].modify_trace_header(
                        idx, method, *args)

                # If flux is not calibrated, but wavelength is calibrated
                # Note that wavecal ONLY has sepctrum_list, it is not science
                # and standard specified.
                elif self.wavelength_science_calibrated:

                    self.wavecal_science.spectrum_list[i].modify_trace_header(
                        idx, method, *args)

                # Should be trapped above so this line should never be run
                else:

                    raise RuntimeError(
                        'This should not happen, please submit an issue.')

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.flux_standard_calibrated:

                self.fluxcal.spectrum_list_standard[0].modify_trace_header(
                    idx, method, *args)

            # If flux is not calibrated, but wavelength is calibrated
            # Note that wavecal ONLY has sepctrum_list, it is not science
            # and standard specified.
            elif self.wavelength_standard_calibrated:

                self.wavecal_standard.spectrum_list[0].modify_trace_header(
                    idx, method, *args)

            # Should be trapped above so this line should never be run
            else:

                raise RuntimeError(
                    'This should not happen, please submit an issue.')

    def save_fits(self,
                  spec_id=None,
                  output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                  filename='reduced',
                  stype='science+standard',
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

        if 'science' in stype_split:

            spec_id = self.create_fits(spec_id=spec_id,
                                       output=output,
                                       stype='science',
                                       empty_primary_hdu=empty_primary_hdu,
                                       return_id=True)

            for i in spec_id:

                filename_i = filename + '_science_' + str(i)

                # If flux is calibrated
                if self.flux_science_calibrated:

                    self.fluxcal.spectrum_list_science[i].save_fits(
                        output=output,
                        filename=filename_i,
                        overwrite=overwrite,
                        empty_primary_hdu=empty_primary_hdu)

                # If flux is not calibrated, and weather or not the wavelength
                # is calibrated.
                elif self.wavelength_science_calibrated:

                    self.wavecal_science.spectrum_list[i].save_fits(
                        output=output,
                        filename=filename_i,
                        overwrite=overwrite,
                        empty_primary_hdu=empty_primary_hdu)

                # This is probably saving trace or count before flux and/or
                # wavelength calibration.
                else:

                    try:

                        self.fluxcal.spectrum_list_science[i].save_fits(
                            output=output,
                            filename=filename_i,
                            overwrite=overwrite,
                            empty_primary_hdu=empty_primary_hdu)

                    except Exception as e:

                        warnings.warn(str(e))

                        self.wavecal_science.spectrum_list[i].save_fits(
                            output=output,
                            filename=filename_i,
                            overwrite=overwrite,
                            empty_primary_hdu=empty_primary_hdu)

        if 'standard' in stype_split:

            self.create_fits(spec_id=[0],
                             output=output,
                             stype='standard',
                             empty_primary_hdu=empty_primary_hdu)

            # If flux is calibrated
            if self.flux_standard_calibrated:

                self.fluxcal.spectrum_list_standard[0].save_fits(
                    output=output,
                    filename=filename + '_standard',
                    overwrite=overwrite,
                    empty_primary_hdu=empty_primary_hdu)

            # If flux is not calibrated, and weather or not the wavelength
            # is calibrated.
            elif self.wavelength_standard_calibrated:

                self.wavecal_standard.spectrum_list[0].save_fits(
                    output=output,
                    filename=filename + '_standard',
                    overwrite=overwrite,
                    empty_primary_hdu=empty_primary_hdu)

            # This is probably saving trace or count before flux and/or
            # wavelength calibration.
            else:

                try:

                    self.fluxcal.spectrum_list_standard[0].save_fits(
                        output=output,
                        filename=filename + '_standard',
                        overwrite=overwrite,
                        empty_primary_hdu=empty_primary_hdu)

                except Exception as e:

                    warnings.warn(str(e))

                    self.wavecal_standard.spectrum_list[0].save_fits(
                        output=output,
                        filename=filename + '_standard',
                        overwrite=overwrite,
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

        if 'science' in stype_split:

            spec_id = self.create_fits(spec_id=spec_id,
                                       output=output,
                                       stype='science',
                                       empty_primary_hdu=False,
                                       return_id=True)

            for i in spec_id:

                filename_i = filename + '_science_' + str(i)

                # If flux is calibrated
                if self.flux_science_calibrated:

                    self.fluxcal.spectrum_list_science[i].save_csv(
                        output=output,
                        filename=filename_i,
                        overwrite=overwrite)

                # If flux is not calibrated, and weather or not the wavelength
                # is calibrated.
                else:

                    self.wavecal_science.spectrum_list[i].save_csv(
                        output=output,
                        filename=filename_i,
                        overwrite=overwrite)

        if 'standard' in stype_split:

            # If flux is calibrated
            if self.flux_standard_calibrated:

                self.fluxcal.spectrum_list_standard[0].save_csv(
                    output=output,
                    filename=filename + '_standard',
                    overwrite=overwrite)

            # If flux is not calibrated, and weather or not the wavelength
            # is calibrated.
            else:

                self.wavecal_standard.spectrum_list[0].save_csv(
                    output=output,
                    filename=filename + '_standard',
                    overwrite=overwrite)
