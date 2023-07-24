# -*- coding: utf-8 -*-
import datetime
import logging
import os
from typing import Callable, Union

import astropy
import numpy as np
from astropy.io import fits
from rascal.calibrator import Calibrator
from scipy import interpolate as itp

__all__ = ["SpectrumOneD"]


class SpectrumOneD:
    """
    Base class of a 1D spectral object to hold the information of each
    extracted spectrum and the raw headers if was provided during the
    data reduction. The FITS or CSV file output are all done here.

    """

    def __init__(
        self,
        spec_id: Union[np.ndarray, list, int] = None,
        verbose: bool = True,
        logger_name: str = "SpectrumOneD",
        log_level: str = "INFO",
        log_file_folder: str = "default",
        log_file_name: str = None,
    ):
        """
        Initialise the object with a logger.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object. Note that this
            ID is unique in each reduction only.
        verbose: boolean (Default: True)
            Set to False to suppress all verbose warnings, except for
            critical failure.
        logger_name: str (Default: SpectrumOneD)
            This will set the name of the logger, if the name is used already,
            it will reference to the existing logger. This will be the
            first part of the default log file name unless log_file_name is
            provided.
        log_level: str (Default: 'INFO')
            Four levels of logging are available, in decreasing order of
            information and increasing order of severity: (1) DEBUG, (2) INFO,
            (3) WARNING, (4) ERROR and (5) CRITICAL. WARNING means that
            there is suboptimal operations in some parts of that step. ERROR
            means that the requested operation cannot be performed, but the
            software can handle it by either using the default setting or
            skipping the operation. CRITICAL means that the requested
            operation cannot be resolved without human interaction, this is
            most usually coming from missing data.
        log_file_folder: None or str (Default: "default")
            Folder in which the file is save, set to default to save to the
            current path.
        log_file_name: None or str (Default: None)
            File name of the log, set to None to self.logger.warning to screen
            only.

        """

        # Set-up logger
        self.logger = logging.getLogger(logger_name)
        if (log_level.lower() == "critical") or (not verbose):
            self.logger.setLevel(logging.CRITICAL)
            self.log_level = "critical"
        elif log_level.lower() == "error":
            self.logger.setLevel(logging.ERROR)
            self.log_level = "error"
        elif log_level.lower() == "warning":
            self.logger.setLevel(logging.WARNING)
            self.log_level = "warning"
        elif log_level.lower() == "info":
            self.logger.setLevel(logging.INFO)
            self.log_level = "info"
        elif log_level.lower() == "debug":
            self.logger.setLevel(logging.DEBUG)
            self.log_level = "debug"
        else:
            raise ValueError("Unknonw logging level.")

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
        )

        if log_file_name is None:
            # Only print log to screen
            self.handler = logging.StreamHandler()
        else:
            if log_file_name == "default":
                t_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                log_file_name = "{logger_name}_{t_str}.log"
            # Save log to file
            if log_file_folder == "default":
                log_file_folder = ""

            self.handler = logging.FileHandler(
                os.path.join(log_file_folder, log_file_name), "a+"
            )

        self.handler.setFormatter(formatter)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(self.handler)

        # spectrum ID
        if spec_id is None:
            self.spec_id = 0
        elif isinstance(spec_id, int):
            self.spec_id = spec_id
        else:
            error_msg = (
                "spec_id has to be of type int, "
                + "{} is given.".format(type(spec_id))
            )
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        # Reduction Meta-data
        self.time_of_reduction = datetime.datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S"
        )

        # Raw image headers
        self.spectrum_header = None
        self.arc_header = None
        self.standard_header = None

        # Detector properties
        self.gain = None
        self.readnoise = None
        self.exptime = None
        self.seeing = None

        # Observing Condition
        self.relative_humidity = None
        self.pressure = None
        self.temperature = None
        self.airmass = None

        # Trace properties
        self.trace = None
        self.trace_sigma = None
        self.len_trace = None
        self.effective_pixel = None
        self.pixel_mapping_itp = None
        self.widthdn = None
        self.widthup = None
        self.sepdn = None
        self.sepup = None
        self.skywidthdn = None
        self.skywidthup = None
        self.extraction_type = None
        self.count = None
        self.count_err = None
        self.count_sky = None
        self.var = None
        self.profile = None
        self.profile_func = None

        # Wavelength calibration properties
        self.arc_spec = None
        self.peaks = None
        self.peaks_refined = None

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
        self.matched_peaks = None
        self.matched_atlas = None
        self.rms = None
        self.residual = None
        self.peak_utilisation = None
        self.atlas_utilisation = None

        # fit output
        self.fit_coeff_rascal = None
        self.matched_peaks_rascal = None
        self.matched_atlas_rascal = None
        self.rms_rascal = None
        self.residual_rascal = None
        self.peak_utilisation_rascal = None
        self.atlas_utilisation_rascal = None

        # fit output
        self.fit_coeff_refine = None
        self.matched_peaks_refine = None
        self.matched_atlas_refine = None
        self.rms_refine = None
        self.residual_refine = None
        self.peak_utilisation_refine = None
        self.atlas_utilisation_refine = None

        # fitted solution
        self.wave = None
        self.wave_bin = None
        self.wave_start = None
        self.wave_end = None
        self.wave_resampled = None
        self.count_resampled = None
        self.count_err_resampled = None
        self.count_sky_resampled = None

        # Fluxes
        self.flux = None
        self.flux_err = None
        self.flux_sky = None
        self.flux_atm_ext_corrected = None
        self.flux_err_atm_ext_corrected = None
        self.flux_sky_atm_ext_corrected = None
        self.flux_telluric_corrected = None
        self.flux_err_telluric_corrected = None
        self.flux_sky_telluric_corrected = None
        self.flux_atm_ext_telluric_corrected = None
        self.flux_err_atm_ext_telluric_corrected = None
        self.flux_sky_atm_ext_telluric_corrected = None
        self.flux_resampled = None
        self.flux_err_resampled = None
        self.flux_sky_resampled = None
        self.flux_resampled_atm_ext_corrected = None
        self.flux_err_resampled_atm_ext_corrected = None
        self.flux_sky_resampled_atm_ext_corrected = None
        self.flux_resampled_telluric_corrected = None
        self.flux_err_resampled_telluric_corrected = None
        self.flux_sky_resampled_telluric_corrected = None
        self.flux_resampled_atm_ext_telluric_corrected = None
        self.flux_err_resampled_atm_ext_telluric_corrected = None
        self.flux_sky_resampled_atm_ext_telluric_corrected = None

        # Continuum
        self.count_continuum = None
        self.count_resampled_continuum = None
        self.flux_continuum = None
        self.flux_resampled_continuum = None

        # Sensitivity curve smoothing properties
        self.smooth = None
        self.lowess_kwargs = None

        # standard star
        self.library = None
        self.target = None

        # Tellurics
        self.telluric_func = None
        self.telluric_profile = None
        self.telluric_profile_resampled = None
        self.telluric_factor = 1.0
        self.telluric_nudge_factor = 1.0

        # Sensitivity curve properties
        self.sensitivity = None
        self.sensitivity_resampled = None
        self.sensitivity_func = None
        self.wave_literature = None
        self.flux_literature = None

        # Atmospheric extinction properties
        self.atm_ext = None
        self.atm_ext_resampled = None

        # HDU lists for output
        self.trace_hdulist = None
        self.count_hdulist = None
        self.weight_map_hdulist = None
        self.arc_spec_hdulist = None
        self.arc_lines_hdulist = None
        self.wavecal_hdulist = None
        self.wavelength_hdulist = None
        self.wavelength_resampled_hdulist = None
        self.count_resampled_hdulist = None
        self.sensitivity_hdulist = None
        self.flux_hdulist = None
        self.atm_ext_hdulist = None
        self.flux_atm_ext_corrected_hdulist = None
        self.telluric_profile_hdulist = None
        self.flux_telluric_corrected_hdulist = None
        self.flux_atm_ext_telluric_corrected_hdulist = None
        self.sensitivity_resampled_hdulist = None
        self.flux_resampled_hdulist = None
        self.atm_ext_resampled_hdulist = None
        self.flux_resampled_atm_ext_corrected_hdulist = None
        self.telluric_profile_resampled_hdulist = None
        self.flux_resampled_telluric_corrected_hdulist = None
        self.flux_resampled_atm_ext_telluric_corrected_hdulist = None

        # FITS output properties
        self.hdu_output = None
        self.empty_primary_hdu = True

        self.hdu_name = {
            1: "trace",
            2: "count",
            3: "weight_map",
            4: "arc_spec",
            5: "arc_lines",
            6: "wavecal_coefficients",
            7: "wavelength",
            8: "wavelength_resampled",
            9: "count_resampled",
            10: "sensitivity",
            11: "flux",
            12: "atm_ext",
            13: "flux_atm_ext_corrected",
            14: "telluric_profile",
            15: "flux_telluric_corrected",
            16: "flux_atm_ext_telluric_corrected",
            17: "sensitivity_resampled",
            18: "flux_resampled",
            19: "atm_ext_resampled",
            20: "flux_resampled_atm_ext_corrected",
            21: "telluric_profile_resampled",
            22: "flux_resampled_telluric_corrected",
            23: "flux_resampled_atm_ext_telluric_corrected",
        }

        self.hdu_derscription = {
            1: "Pixel positions of the trace in the spatial direction, "
            + "Width of the trace",
            2: "Count, Count uncertainty, Sky count",
            3: "Weight map of the extration (variance)",
            4: "1D Arc spectrum",
            5: "Arc line position, Arc line effective position",
            6: "Polynomial coefficients for wavelength calibration",
            7: "The pixel-to-wavelength mapping",
            8: "The pixel-to-wavelength mapping in the resampled coordiates",
            9: "Resampled count, Resampled count uncertainty, "
            + "Resampled sky count",
            10: "Sensitivity curve",
            11: "Flux, Flux uncertainty, Sky flux",
            12: "Atmospheric extinction correction factor",
            13: "Flux (atmospheric extinction corrected), "
            + "Flux uncertainty (atmospheric extinction corrected), "
            + "Sky flux (atmospheric extinction corrected)",
            14: "Telluric absorption profile",
            15: "Flux (telluric absorption corrected), "
            + "Flux Uncertainty (telluric absorption corrected), "
            + "Sky Flux (telluric absorption corrected)",
            16: "Flux (atmospheric extinction and telluric absorption "
            + "corrected), Flux Uncertainty (atmospheric extinction and "
            + "telluric absorption corrected), Sky flux (atmospheric "
            + "extinction and telluric absorption corrected)",
            17: "Resampled sensitivity curve",
            18: "Resampled flux, Resampled flux uncertainty, "
            + "Resampled sky flux",
            19: "Atmospheric extinction correction factor (resampled)",
            20: "Flux (resampled, atmospheric extinction), "
            + "Flux uncertainty (resampled, atmospheric extinction), "
            + "Sky flux (resampled, atmospheric extinction)",
            21: "Telluric absorption profile (resampled)",
            22: "Flux (resampled, telluric absorption corrected), "
            + "Flux uncertainty (resampled, telluric absorption corrected), "
            + "Sky flux (resampled, telluric absorption corrected)",
            23: "Flux (atmospheric extinction and telluric absorption "
            + "corrected), Flux uncertainty (atmospheric extinction and "
            + "telluric absorption corrected), Sky flux (atmospheric "
            + "extinction and telluric absorption corrected)",
        }

        self.ext_name = {
            1: ["trace", "trace_sigma"],
            2: ["count", "count_err", "count_sky"],
            3: ["weight_map"],
            4: ["arc_spec"],
            5: ["peaks", "peaks_refined"],
            6: ["polynomials"],
            7: ["wavelength"],
            8: ["wavelength_resampled"],
            9: [
                "count_resampled",
                "count_err_resampled",
                "count_sky_resampled",
            ],
            10: ["sensitivity"],
            11: ["flux", "flux_err", "flux_sky"],
            12: ["atm_ext"],
            13: [
                "flux_atm_ext_corrected",
                "flux_err_atm_ext_corrected",
                "flux_sky_atm_ext_corrected",
            ],
            14: ["telluric_profile"],
            15: [
                "flux_telluric_corrected",
                "flux_err_telluric_corrected",
                "flux_sky_telluric_corrected",
            ],
            16: [
                "flux_atm_ext_telluric_corrected",
                "flux_err_atm_ext_telluric_corrected",
                "flux_sky_atm_ext_telluric_corrected",
            ],
            17: ["sensitivity_resampled"],
            18: ["flux_resampled", "flux_err_resampled", "flux_sky_resampled"],
            19: ["atm_ext_resampled"],
            20: [
                "flux_resampled_atm_ext_corrected",
                "flux_err_resampled_atm_ext_corrected",
                "flux_sky_resampled_atm_ext_corrected",
            ],
            21: ["telluric_profile_resampled"],
            22: [
                "flux_resampled_telluric_corrected",
                "flux_err_resampled_telluric_corrected",
                "flux_sky_resampled_telluric_corrected",
            ],
            23: [
                "flux_resampled_atm_ext_telluric_corrected",
                "flux_err_resampled_atm_ext_telluric_corrected",
                "flux_sky_resampled_atm_ext_telluric_corrected",
            ],
        }

        self.header = {}
        for d1, d2 in zip(
            self.hdu_name.values(), self.hdu_derscription.values()
        ):
            self.header[d1] = d2

        # The order in which HDUs are arranged
        self.hdu_order = {v: k for k, v in self.hdu_name.items()}

        # The HDU availability
        self.hdu_content = dict.fromkeys(self.hdu_order, False)

        # Set the counters of HDUs
        number_of_hdus = {
            1: 2,
            2: 3,
            3: 1,
            4: 1,
            5: 2,
            6: 1,
            7: 1,
            8: 1,
            9: 3,
            10: 1,
            11: 3,
            12: 1,
            13: 3,
            14: 1,
            15: 3,
            16: 3,
            17: 1,
            18: 3,
            19: 1,
            20: 3,
            21: 1,
            22: 3,
            23: 3,
        }
        self.n_hdu = {}
        for d1, d2 in zip(self.hdu_name.values(), number_of_hdus.values()):
            self.n_hdu[d1] = d2

    def merge(self, spectrum_oned, overwrite: bool = False):
        """
        This function copies all the info from the supplied spectrum_oned to
        this one, including the spec_id.

        Parameters
        ----------
        spectrum_oned: SpectrumOneD object
            The source SpectrumOneD to be deep copied over.
        overwrite: boolean (Default: False)
            Set to True to overwrite all the data in this SpectrumOneD.

        """

        for attr, value in self.__dict__.items():
            if attr == "spec_id":
                if getattr(spectrum_oned, attr) != 0:
                    setattr(self, attr, value)

            if getattr(self, attr) is None or []:
                setattr(self, attr, getattr(spectrum_oned, attr))

            if overwrite:
                setattr(self, attr, getattr(spectrum_oned, attr))

            else:
                # if not overwrite, do nothing
                pass

    def add_spectrum_header(
        self, header: Union[fits.Header, list, np.ndarray]
    ):
        """
        Add a header for the spectrum. Typically put the header of the
        raw 2D spectral image of the science target(s) here. Some
        automated operations rely on reading from the header.

        Parameters
        ----------
        header: astropy.io.fits.Header object

        """

        if header is not None:
            if isinstance(header, fits.Header):
                self.spectrum_header = header
                self.logger.info("spectrum_header is stored.")
            elif isinstance(header[0], fits.Header):
                self.spectrum_header = header[0]
                self.logger.info("spectrum_header is stored.")
            else:
                self.logger.error(
                    "Unsupported type of header is provided: {}. "
                    "Only astropy.io.fits.Header or such in a list or array "
                    "can beaccepted.".format(type(header))
                )

    def remove_spectrum_header(self):
        """
        Remove the FITS header of the target.

        """

        self.spectrum_header = None

    def add_standard_header(
        self, header: Union[fits.Header, list, np.ndarray]
    ):
        """
        Add a header for the standard star that it is flux calibrated
        against. The, if opted for, automatic atmospheric extinction
        correction is applied based on the airmass found in this header.
        Typically put the header of the raw 2D spectral image of the
        standard star here. Some automated operations rely on reading
        from the header.

        Parameters
        ----------
        header: astropy.io.fits.Header object

        """

        if header is not None:
            if isinstance(header, fits.Header):
                self.standard_header = header
                self.logger.info("standard_header is stored.")
            elif isinstance(header[0], fits.Header):
                self.standard_header = header[0]
                self.logger.info("standard_header is stored.")
            else:
                self.logger.error(
                    "Unsupported type of header is provided: {}. "
                    "Only astropy.io.fits.Header or such in a list or array "
                    "can be accepted.".format(type(header))
                )

    def remove_standard_header(self):
        """
        Remove the FITS header of the standard.

        """

        self.standard_header = None

    def add_arc_header(self, header: Union[fits.Header, list, np.ndarray]):
        """
        Add a header for the arc that it is wavelength calibrated
        against. Typically put the header of the raw 2D spectral image
        of the arc here. Some automated operations rely on reading from
        the header.

        Parameters
        ----------
        header: astropy.io.fits.Header object

        """

        if header is not None:
            if isinstance(header, fits.Header):
                self.arc_header = header
                self.logger.info("arc_header is stored.")
            elif isinstance(header[0], fits.Header):
                self.arc_header = header[0]
                self.logger.info("arc_header is stored.")
            else:
                self.logger.error(
                    "Unsupported type of header is provided: {}. "
                    "Only astropy.io.fits.Header or such in a list or array "
                    "can be accepted.".format(type(header))
                )

    def remove_arc_header(self):
        """
        Remove the arc_header.

        """

        self.arc_header = None

    def add_trace(
        self,
        trace: Union[list, np.ndarray],
        trace_sigma: Union[list, np.ndarray],
        effective_pixel: Union[list, np.ndarray] = None,
    ):
        """
        Add the trace of the spectrum.

        Parameters
        ----------
        trace: list or numpy.ndarray (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: list or numpy.ndarray (N)
            Standard deviation of the Gaussian profile of a trace
        effective_pixel: list or numpy array (Default: None)
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range (num_pix), for example, in the case of accounting for chip
            gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]

        """

        assert isinstance(
            trace, (list, np.ndarray)
        ), "trace has to be a list or a numpy array"
        assert isinstance(
            trace_sigma, (list, np.ndarray)
        ), "trace_sigma has to be a list or a numpy array"
        assert len(trace_sigma) == len(trace), "trace and trace_sigma have to "
        " be the same size."

        if effective_pixel is None:
            effective_pixel = list(np.arange(len(trace)).astype("int"))

        else:
            assert isinstance(
                effective_pixel, (list, np.ndarray)
            ), "effective_pixel has to be a list or a numpy array"
            assert len(effective_pixel) == len(
                trace
            ), "trace and effective_pixel have "
            "to be the same size."

        pixel_mapping_itp = itp.interp1d(
            np.arange(len(trace)),
            effective_pixel,
            kind="cubic",
            fill_value="extrapolate",
        )

        # Only add if all assertions are passed.
        self.trace = trace
        self.trace_sigma = trace_sigma
        self.len_trace = len(trace)
        self.add_effective_pixel(effective_pixel)
        self.add_pixel_mapping_itp(pixel_mapping_itp)

    def remove_trace(self):
        """
        Remove the trace, trace_sigma and len_trace, also remove the effective_pixel
        and the interpolation function that maps the pixel values to the
        effective pixel values.

        """

        self.trace = None
        self.trace_sigma = None
        self.len_trace = None
        self.remove_effective_pixel()
        self.remove_pixel_mapping_itp()

    def add_aperture(
        self,
        widthdn: int,
        widthup: int,
        sepdn: int,
        sepup: int,
        skywidthdn: int,
        skywidthup: int,
    ):
        """
        The size of the aperture in which the spectrum is extracted. This is
        merely the limit where extraction is performed, it does not hold the
        information of the weighting::

                        .................................   ^
            Sky         .................................   |   skywidthup
                        .................................   v
                        .................................     ^
                        .................................     | sepup
                        .................................     v
                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ^
                        ---------------------------------   |   widthup
            Spectrum    =================================   v ^
                        ---------------------------------     | widthdn
                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     v
                        .................................   ^
                        .................................   |   sepdn
                        .................................   v
                        .................................     ^
            Sky         .................................     | skywidthdn
                        .................................     v

        Parameters
        ----------
        widthdn: real positive number
            The aperture size on the bottom side of the spectrum.
        widthup: real positive number
            The aperture size on the top side of the spectrum.
        sepdn: real positive number
            The gap between the spectrum and the sky region on the bottom
            side of the spectrum.
        sepup: real positive number
            The gap between the spectrum and the sky region on the top
            side of the spectrum.
        skywidthdn: real positive number
            The sky region on the bottom side of the spectrum.
        skywidthup: real positive number
            The sky region on the top side of the spectrum.

        """

        assert np.isfinite(widthdn), "widthdn has to be finite."
        assert np.isfinite(widthup), "widthup has to be finite."
        assert np.isfinite(sepdn), "sepdn has to be finite."
        assert np.isfinite(sepup), "sepup has to be finite."
        assert np.isfinite(skywidthdn), "skywidthdn has to be finite."
        assert np.isfinite(skywidthup), "skywidthup has to be finite."
        self.widthdn = widthdn
        self.widthup = widthup
        self.sepdn = sepdn
        self.sepup = sepup
        self.skywidthdn = skywidthdn
        self.skywidthup = skywidthup

    def add_count(
        self,
        count: Union[list, np.ndarray],
        count_err: Union[list, np.ndarray] = None,
        count_sky: Union[list, np.ndarray] = None,
    ):
        """
        Add the photoelectron counts and the associated optional
        uncertainty and sky counts.

        Parameters
        ----------
        count: 1-d array
            The summed count at each column along the trace.
        count_err: 1-d array
            the uncertainties of the count values
        count_sky: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract

        """

        # Check data type
        assert isinstance(
            count, (list, np.ndarray)
        ), "count has to be a list or a numpy array"

        if count_err is not None:
            assert isinstance(
                count_err, (list, np.ndarray)
            ), "count_err has to be a list or a numpy array"
            assert len(count_err) == len(
                count
            ), "count_err has to be the same size as count"

        if count_sky is not None:
            assert isinstance(
                count_sky, (list, np.ndarray)
            ), "count_sky has to be a list or a numpy array"
            assert len(count_sky) == len(
                count
            ), "count_sky has to be the same size as count"

        self.count = count

        # Only add if they are provided
        if count_err is not None:
            self.count_err = count_err

        else:
            self.count_err = np.zeros_like(self.count)

        if count_sky is not None:
            self.count_sky = count_sky

        else:
            self.count_sky = np.zeros_like(self.count)

        if self.effective_pixel is None:
            effective_pixel = list(np.arange(len(count)).astype("int"))
            self.add_effective_pixel(effective_pixel)

        else:
            assert len(self.effective_pixel) == len(count), (
                "count and effective_pixel have " + "to be the same size."
            )

    def remove_count(self):
        """
        Remove the count, count_err and count_sky.

        """

        self.count = None
        self.count_err = None
        self.count_sky = None

    def add_variances(self, var: Union[list, np.ndarray]):
        """
        Add the weight map of the extraction.

        Parameters
        ----------
        var: 2-d array
            The weigh map of the input image for extraction.

        """

        self.var = var

    def remove_variances(self):
        """
        Remove the variances.

        """

        self.var = None

    def add_line_spread_profile_upsampled(
        self, line_spread_profile_upsampled: astropy.modeling.Fittable1DModel
    ):
        """
        Add the empirical line spread profile as measured from the upsampled image.

        Parameters
        ----------
        profile_func: a fitted astropy.modeling.Fittable1DModel
            The fitted trace profile.
        """

        self.line_spread_profile_upsampled = line_spread_profile_upsampled

    def remove_line_spread_profile_upsampled(self):
        """
        Remove the fitted trace profile.
        """

        self.line_spread_profile_upsampled = None

    def add_line_spread_profile(
        self, line_spread_profile: astropy.modeling.Fittable1DModel
    ):
        """
        Add the empirical line spread profile as measured.

        Parameters
        ----------
        profile_func: a fitted astropy.modeling.Fittable1DModel
            The fitted trace profile.
        """

        self.line_spread_profile = line_spread_profile

    def remove_line_spread_profile(self):
        """
        Remove the fitted trace profile.
        """

        self.line_spread_profile = None

    def add_profile_func(self, profile_func: Callable):
        """
        Add the fitted trace profile.

        Parameters
        ----------
        profile_func: a fitted astropy.modeling.Fittable1DModel
            The fitted trace profile.
        """

        self.profile_func = profile_func

    def remove_profile_func(self):
        """
        Remove the fitted trace profile.
        """

        self.profile_func = None

    def add_profile(self, profile: Union[list, np.ndarray]):
        """
        Add the extraction profile (generated from the profile_func).

        Parameters
        ----------
        profile: 1-d array
            The weight function of the extraction profile.

        """

        self.profile = profile

    def remove_profile(self):
        """
        Remove the extraction profile.

        """

        self.profile = None

    def add_arc_spec(self, arc_spec: Union[list, np.ndarray]):
        """
        Add the extracted 1D spectrum of the arc.

        Parameters
        ----------
        arc_spec: 1-d array
            The photoelectron count of the spectrum of the arc lamp.

        """

        assert isinstance(
            arc_spec, (list, np.ndarray)
        ), "arc_spec has to be a list or a numpy array"
        self.arc_spec = arc_spec

    def remove_arc_spec(self):
        """
        Remove the 1D spectrum of the arc.

        """
        self.arc_spec = None

    def add_effective_pixel(self, effective_pixel: Union[list, np.ndarray]):
        """
        Add the pixel list, which contain the effective pixel values that
        has accounted for chip gaps and non-integer pixel value.

        Parameters
        ----------
        effective_pixel: 1-d array
            The effective position of the pixel.

        """

        assert isinstance(
            effective_pixel, (list, np.ndarray)
        ), "effective_pixel has to be a list or a numpy array"
        self.effective_pixel = effective_pixel

    def remove_effective_pixel(self):
        """
        Remove the pixel list.

        """

        self.effective_pixel = None

    def add_pixel_mapping_itp(self, pixel_mapping_itp: Callable):
        """
        Add the interpolated callable function that convert raw pixel
        values into effective pixel values.

        Parameters
        ----------
        pixel_mapping_itp: callable function
            The mapping function for raw-effective pixel position.

        """

        assert callable(
            pixel_mapping_itp
        ), "pixel_mapping_itp has to be a Callable function."
        self.pixel_mapping_itp = pixel_mapping_itp

    def remove_pixel_mapping_itp(self):
        """
        Remove the interpolation function that maps the pixel values to the
        effective pixel value.

        """

        self.pixel_mapping_itp = None

    def add_peaks(self, peaks: Union[list, np.ndarray]):
        """
        Add the pixel values (int) where arc lines are located.

        Parameters
        ----------
        peaks: 1-d array
            The pixel value of the identified peaks.

        """

        assert isinstance(
            peaks, (list, np.ndarray)
        ), "peaks has to be a list or a numpy array"
        self.peaks = peaks

    def remove_peaks(self):
        """
        Remove the list of peaks.

        """

        self.peaks = None

    def add_peaks_refined(self, peaks_refined: Union[list, np.ndarray]):
        """
        Add the refined pixel values (float) where arc lines are located.

        Parameters
        ----------
        peaks: 1-d array
            The pixel value of the refined peak positions.

        """

        assert isinstance(
            peaks_refined, (list, np.ndarray)
        ), "peaks_refined has to be a list or a numpy array"
        self.peaks_refined = peaks_refined

    def remove_peaks_refined(self):
        """
        Remove the list of refined peaks.

        """

        self.peaks_refined = None

    def add_peaks_wave(self, peaks_wave: Union[list, np.ndarray]):
        """
        Add the wavelength (Angstrom) of the arc lines.

        Parameters
        ----------
        peaks: 1-d array
            The wavelength value of the peaks.

        """

        assert isinstance(
            peaks_wave, (list, np.ndarray)
        ), "peaks_wave has to be a list or a numpy array"
        self.peaks_wave = peaks_wave

    def remove_peaks_wave(self):
        """
        Remove the list of wavelengths of the arc lines.

        """

        self.peaks_wave = None

    def add_calibrator(self, calibrator: Calibrator):
        """
        Add a RASCAL Calibrator object.

        Parameters
        ----------
        calibrator: rascal.Calibrator()
            A RASCAL Calibrator object.

        """

        assert isinstance(calibrator, Calibrator)

        self.calibrator = calibrator

    def remove_calibrator(self):
        """
        Remove the Calibrator.

        """

        self.calibrator = None

    def add_atlas_wavelength_range(
        self, min_atlas_wavelength: float, max_atlas_wavelength: float
    ):
        """
        Add the allowed range of wavelength calibration.

        Parameters
        ----------
        min_atlas_wavelength: float
            The minimum wavelength of the atlas.
        max_atlas_wavelength: float
            The maximum wavelength of the atlas.

        """

        assert np.isfinite(
            min_atlas_wavelength
        ), "min_atlas_wavelength has to be finite."
        assert np.isfinite(
            max_atlas_wavelength
        ), "max_atlas_wavelength has to be finite."
        self.min_atlas_wavelength = min_atlas_wavelength
        self.max_atlas_wavelength = max_atlas_wavelength

    def remove_atlas_wavelength_range(self):
        """
        Remove the atlas wavelength range.

        """

        self.min_atlas_wavelength = None
        self.max_atlas_wavelength = None

    def add_min_atlas_intensity(self, min_atlas_intensity: float):
        """
        Add the minimum allowed line intensity (theoretical NIST value)
        that were used for wavelength calibration.

        Parameters
        ----------
        min_atlas_intensity: float
            The minimum line strength used to get the atlas.

        """

        assert np.isfinite(
            min_atlas_intensity
        ), "min_atlas_intensity has to be finite."
        self.min_atlas_intensity = min_atlas_intensity

    def remove_min_atlas_intensity(self):
        """
        Remove the minimum atlas intensity.

        """

        self.min_atlas_intensity = None

    def add_min_atlas_distance(self, min_atlas_distance: float):
        """
        Add the minimum allowed line distance (only consider lines that
        passed the minimum intensity threshold) that were used for
        wavelength calibration.

        Parameters
        ----------
        min_atlas_distance: float
            Minimum wavelength separation between neighbouring lines.

        """

        assert np.isfinite(
            min_atlas_distance
        ), "min_atlas_distance has to be finite."
        self.min_atlas_distance = min_atlas_distance

    def remove_min_atlas_distance(self):
        """
        Remove the minimum distance between lines to be accepted.

        """

        self.min_atlas_distance = None

    def add_gain(self, gain: float):
        """
        Add the gain value of the spectral image. This value can be
        different from the one in the header as this can be overwritten
        by an user input, while the header value is raw.

        Parameters
        ----------
        gain: float
            The gain value of the detector.

        """

        assert np.isfinite(gain), "gain has to be finite."
        self.gain = gain

    def remove_gain(self):
        """
        Remove the gain value.

        """

        self.gain = None

    def add_readnoise(self, readnoise: float):
        """
        Add the readnoise value of the spectral image. This value can be
        different from the one in the header as this can be overwritten
        by an user input, while the header value is raw.

        Parameters
        ----------
        readnoise: float
            The read noise value of the detector.

        """

        assert np.isfinite(readnoise), "readnoise has to be finite."
        self.readnoise = readnoise

    def remove_readnoise(self):
        """
        Remove the readnoise value.

        """

        self.readnoise = None

    def add_exptime(self, exptime: float):
        """
        Add the exposure time of the spectral image. This value can be
        different from the one in the header as this can be overwritten
        by an user input, while the header value is raw.

        Parameters
        ----------
        exptime: float
            The exposure time of the input image.

        """

        assert np.isfinite(exptime), "exptime has to be finite."
        self.exptime = exptime

    def remove_exptime(self):
        """
        Remove the exptime value.

        """

        self.exptime = None

    def add_airmass(self, airmass: float):
        """
        Add the airmass when the observation was carried out. This value
        can be different from the one in the header as this can be
        overwritten by an user input, while the header value is raw.

        Parameters
        ----------
        airmass: float
            The effective airmass during the exposure.

        """

        assert np.isfinite(airmass), "airmass has to be finite."
        self.airmass = airmass

    def remove_airmass(self):
        """
        Remove the airmass value.

        """

        self.airmass = None

    def add_seeing(self, seeing: float):
        """
        Add the seeing when the observation was carried out. This value
        can be different from the one in the header as this can be
        overwritten by an user input, while the header value is raw.

        Parameters
        ----------
        seeing: float
            The effective seeing of the observation.

        """

        assert np.isfinite(seeing), "airmass has to be finite."
        self.seeing = seeing

    def remove_seeing(self):
        """
        Remove the seeing value.

        """

        self.seeing = None

    def add_weather_condition(
        self, pressure: float, temperature: float, relative_humidity: float
    ):
        """
        Add the pressure, temperature and relative humidity when the spectral
        image was taken. These value can be different from the ones in the
        header as this can be overwritten by an user input, while the header
        value is raw.

        Parameters
        ----------
        pressure: float
            The air pressure during the observation (in pascal).
        temperature: float
            The temperature during the observation (in Kelvin).
        relative_hhumidity: float
            The relative humidity during the observation (between 0 and 1).

        """

        assert np.isfinite(pressure), "pressure has to be finite."
        assert np.isfinite(temperature), "temperature has to be finite."
        assert np.isfinite(
            relative_humidity
        ), "relative_humidity has to be finite."
        self.pressure = pressure
        self.temperature = temperature
        self.relative_humidity = relative_humidity

    def remove_weather_condition(self):
        """
        Remove the Pressure, Temperature and Humidity.

        """

        self.pressure = None
        self.temperature = None
        self.relative_humidity = None

    def add_fit_type(self, fit_type: str):
        """
        Add the kind of polynomial used for fitting the pixel-to-wavelength
        function.

        Parameters
        ----------
        fit_type: str
            The type of polynomial for fitting the pixel-wavelength function.

        """

        assert type(fit_type) == str, "fit_type has to be a string"
        assert fit_type in ["poly", "leg", "cheb"], "fit_type must be "
        "(1) poly(nomial); (2) leg(endre); or (3) cheb(yshev)."
        self.fit_type = fit_type

    def remove_fit_type(self):
        """
        Remove the chosen type of polynomial.

        """

        self.fit_type = None

    def add_fit_coeff(self, fit_coeff: Union[list, np.ndarray]):
        """
        Add the polynomial co-efficients of the pixel-to-wavelength
        function. Note that this overwrites the wavelength calibrated
        fit coefficients.

        Parameters
        ----------
        fit_coeff: list or 1-d array
            The set of coefficients of the pixel-wavelength function.

        """

        assert isinstance(
            fit_coeff, (list, np.ndarray)
        ), "fit_coeff has to be a list or a numpy array."
        self.fit_coeff = fit_coeff

    def remove_fit_coeff(self):
        """
        Remove the polynomial co-efficients of the pixel-to-wavelength
        function.

        """

        self.fit_coeff = None

    def add_calibrator_properties(
        self,
        num_pix: int,
        effective_pixel: Union[list, np.ndarray],
        plotting_library: str,
    ):
        """
        Add the properties of the RASCAL Calibrator.

        Parameters
        ----------
        num_pix: int
            The number of pixels in the dispersion direction
        effective_pixel: list or numpy array
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(num_pix), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        plotting_library : string
            Choose between matplotlib and plotly.

        """

        self.num_pix = num_pix
        self.plotting_library = plotting_library
        self.effective_pixel = effective_pixel

    def remove_calibrator_properties(self):
        """
        Remove the properties of the RASCAL Calibrator as recorded in ASPIRED.
        This does NOT remove the calibrator properites INSIDE the calibrator.

        """

        self.num_pix = None
        self.effective_pixel = None
        self.plotting_library = None

    def add_hough_properties(
        self,
        num_slopes: int,
        xbins: int,
        ybins: int,
        min_wavelength: float,
        max_wavelength: float,
        range_tolerance: float,
        linearity_tolerance: float,
    ):
        """
        Add the hough transform configuration of the RASCAL Calibrator.

        Parameters
        ----------
        num_slopes: int
            Number of slopes to consider during Hough transform
        xbins: int
            Number of bins for Hough accumulation
        ybins: int
            Number of bins for Hough accumulation
        min_wavelength: float
            Minimum wavelength of the spectrum.
        max_wavelength: float
            Maximum wavelength of the spectrum.
        range_tolerance: float
            Estimation of the error on the provided spectral range
            e.g. 3000-5000 with tolerance 500 will search for
            solutions that may satisfy 2500-5500
        linearity_tolerance: float
            A toleranceold (Ansgtroms) which defines some padding around the
            range tolerance to allow for non-linearity. This should be the
            maximum expected excursion from linearity.

        """

        self.num_slopes = num_slopes
        self.xbins = xbins
        self.ybins = ybins
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.range_tolerance = range_tolerance
        self.linearity_tolerance = linearity_tolerance

    def remove_hough_properties(self):
        """
        Remove the hough transform configuration of the RASCAL Calibrator.
        This does NOT remove the hough transform configuration INSIDE the
        calibrator.

        """

        self.num_slopes = None
        self.xbins = None
        self.ybins = None
        self.min_wavelength = None
        self.max_wavelength = None
        self.range_tolerance = None
        self.linearity_tolerance = None

    def add_ransac_properties(
        self,
        sample_size: int,
        top_n_candidate: int,
        linear: bool,
        filter_close: bool,
        ransac_tolerance: float,
        candidate_weighted: bool,
        hough_weight: float,
        minimum_matches: int,
        minimum_peak_utilisation: float,
        minimum_fit_error: float,
    ):
        """
        Add the RANSAC properties of the RASCAL Calibrator.

        Parameters
        ----------
        sample_size: int
            Number of pixel-wavelength hough pairs to be used for each arc line
            being picked.
        top_n_candidate: int
            Top ranked lines to be fitted.
        linear: bool
            True to use the hough transformed gradient, otherwise, use the
            known polynomial.
        filter_close: bool
            Remove the pairs that are out of bounds in the hough space.
        ransac_tolerance: float
            The distance criteria  (Angstroms) to be considered an inlier to a
            fit. This should be close to the size of the expected residuals on
            the final fit (e.g. 1A is typical)
        candidate_weighted: bool
            Set to True to down-weight pairs that are far from the fit.
        hough_weight: float or None
            Set to use the hough space to weigh the fit. The theoretical
            optimal weighting is unclear. The larger the value, the heavily it
            relies on the overdensity in the hough space for a good fit.
        minimum_matches: int
            Minimum number of matches to accept the solution.
        minimum_peak_utilisation: float
            Minimum percentage of peaks used in the solution.
        minimum_fit_error: float
            Minimum fitting error. Probably only useful in simulated noiseless
            case.

        """

        self.sample_size = sample_size
        self.top_n_candidate = top_n_candidate
        self.linear = linear
        self.filter_close = filter_close
        self.ransac_tolerance = ransac_tolerance
        self.candidate_weighted = candidate_weighted
        self.hough_weight = hough_weight
        self.minimum_matches = minimum_matches
        self.minimum_peak_utilisation = minimum_peak_utilisation
        self.minimum_fit_error = minimum_fit_error

    def remove_ransac_properties(self):
        """
        Remove the RANSAC properties of the RASCAL Calibrator.

        """

        self.sample_size = None
        self.top_n_candidate = None
        self.linear = None
        self.filter_close = None
        self.ransac_tolerance = None
        self.candidate_weighted = None
        self.hough_weight = None
        self.minimum_matches = None
        self.minimum_peak_utilisation = None
        self.minimum_fit_error = None

    def add_fit_output_final(
        self,
        fit_coeff: Union[list, np.ndarray],
        matched_peaks: Union[list, np.ndarray],
        matched_atlas: Union[list, np.ndarray],
        rms: float,
        residual: Union[list, np.ndarray],
        peak_utilisation: float,
        atlas_utilisation: float,
    ):
        """
        Add the final accepted polynomial solution.

        Parameters
        ----------
        fit_coeff: list
            Set the baseline of the least square fit. If no fits outform this
            set of polynomial coefficients, this will be used as the best fit.
        matched_peaks: list
            List of matched peaks
        matched_atlas: list
            List of matched atlas lines
        rms: float
            The root-mean-squared of the fit
        residual: list
            The residual of each fitted peak
        peak_utilisation: float
            The fraction of the input peaks used in the fit
        atlas_utilisation: float
            The fraction of the input atlas used in the fit

        """

        # add assertion here
        self.fit_coeff = fit_coeff
        self.matched_peaks = matched_peaks
        self.matched_atlas = matched_atlas
        self.rms = rms
        self.residual = residual
        self.peak_utilisation = peak_utilisation
        self.atlas_utilisation = atlas_utilisation

    def remove_fit_output_final(self):
        """
        Remove the accepted polynomial solultion.
        """

        self.fit_coeff = None
        self.matched_peaks = None
        self.matched_atlas = None
        self.rms = None
        self.residual = None
        self.peak_utilisation = None
        self.atlas_utilisation = None

    def add_fit_output_rascal(
        self,
        fit_coeff: Union[list, np.ndarray],
        matched_peaks: Union[list, np.ndarray],
        matched_atlas: Union[list, np.ndarray],
        rms: float,
        residual: Union[list, np.ndarray],
        peak_utilisation: float,
        atlas_utilisation: float,
    ):
        """
        Add the RASCAL polynomial solution.

        Parameters
        ----------
        fit_coeff: list
            Set the baseline of the least square fit. If no fits outform this
            set of polynomial coefficients, this will be used as the best fit.
        matched_peaks: list
            List of matched peaks
        matched_atlas: list
            List of matched atlas lines
        rms: float
            The root-mean-squared of the fit
        residual: list
            The residual of each fitted peak
        peak_utilisation: float
            The fraction of the input peaks used in the fit
        atlas_utilisation: float
            The fraction of the input atlas used in the fit

        """

        # add assertion here
        self.fit_coeff_rascal = fit_coeff
        self.matched_peaks_rascal = matched_peaks
        self.matched_atlas_rascal = matched_atlas
        self.rms_rascal = rms
        self.residual_rascal = residual
        self.peak_utilisation_rascal = peak_utilisation
        self.atlas_utilisation_rascal = atlas_utilisation
        self.add_fit_output_final(
            fit_coeff,
            matched_peaks,
            matched_atlas,
            rms,
            residual,
            peak_utilisation,
            atlas_utilisation,
        )

    def remove_fit_output_rascal(self):
        """
        Remove the RASCAL polynomial solution.

        """

        self.fit_coeff_rascal = None
        self.matched_peaks_rascal = None
        self.matched_atlas_rascal = None
        self.rms_rascal = None
        self.residual_rascal = None
        self.peak_utilisation_rascal = None
        self.atlas_utilisation_rascal = None

    def add_fit_output_refine(
        self,
        fit_coeff: Union[list, np.ndarray],
        matched_peaks: Union[list, np.ndarray],
        matched_atlas: Union[list, np.ndarray],
        rms: float,
        residual: Union[list, np.ndarray],
        peak_utilisation: float,
        atlas_utilisation: float,
    ):
        """
        Add the refined RASCAL polynomial solution.

        Parameters
        ----------
        fit_coeff: list
            Set the baseline of the least square fit. If no fits outform this
            set of polynomial coefficients, this will be used as the best fit.
        matched_peaks: list
            List of matched peaks
        matched_atlas: list
            List of matched atlas lines
        rms: float
            The root-mean-squared of the fit
        residual: list
            The residual of each fitted peak
        peak_utilisation: float
            The fraction of the input peaks used in the fit
        atlas_utilisation: float
            The fraction of the input atlas used in the fit

        """

        # add assertion here
        self.fit_coeff_refine = fit_coeff
        self.matched_peaks_refine = matched_peaks
        self.matched_atlas_refine = matched_atlas
        self.rms_refine = rms
        self.residual_refine = residual
        self.peak_utilisation_refine = peak_utilisation
        self.atlas_utilisation_refine = atlas_utilisation
        self.add_fit_output_final(
            fit_coeff,
            matched_peaks,
            matched_atlas,
            rms,
            residual,
            peak_utilisation,
            atlas_utilisation,
        )

    def remove_fit_output_refine(self):
        """
        Remove the refined RASCAL polynomial solution.

        """

        self.fit_coeff_refine = None
        self.matched_peaks_refine = None
        self.matched_atlas_refine = None
        self.rms_refine = None
        self.residual_refine = None
        self.peak_utilisation_refine = None
        self.atlas_utilisation_refine = None

    def add_wavelength(self, wave: Union[list, np.ndarray]):
        """
        Add the wavelength of each effective pixel.

        Parameters
        ----------
        wave: list or 1d-array
            The wavelength values at each effective pixel.

        """

        self.wave = wave
        # Note that the native pixels have varing bin size.
        self.wave_bin = np.nanmedian(np.array(np.ediff1d(wave)))
        self.wave_start = np.min(wave)
        self.wave_end = np.max(wave)

    def remove_wavelength(self):
        """
        Remove the wavelength of each effective pixel.

        """

        self.wave = None

    def add_wavelength_resampled(
        self, wave_resampled: Union[list, np.ndarray]
    ):
        """
        Add the wavelength of the resampled spectrum which has an evenly
        distributed wavelength spacing.

        Parameters
        ----------
        wave: list or 1d-array
            The resampled wavelength values.

        """

        # We assume that the resampled spectrum has fixed bin size
        self.wave_bin = np.nanmedian(np.array(np.ediff1d(wave_resampled)))
        self.wave_start = np.min(wave_resampled)
        self.wave_end = np.max(wave_resampled)
        self.wave_resampled = wave_resampled

    def remove_wavelength_resampled(self):
        """
        Add the resampled wavelength.

        """

        self.wave_bin = None
        self.wave_start = None
        self.wave_end = None
        self.wave_resampled = None

    def add_count_resampled(
        self,
        count_resampled: Union[list, np.ndarray],
        count_err_resampled: Union[list, np.ndarray],
        count_sky_resampled: Union[list, np.ndarray],
    ):
        """
        Add the photoelectron counts of the resampled spectrum which has
        an evenly distributed wavelength spacing.

        Parameters
        ----------
        count_resampled: list or 1d-array
            The resampled photoelectron count.
        count_err_resampled: list or 1d-array
            The uncertainty of the resampled photoelectron count.
        count_sky_resampled: list or 1d-array
            The background sky level of the resampled photoelectron count.

        """

        self.count_resampled = count_resampled
        self.count_err_resampled = count_err_resampled
        self.count_sky_resampled = count_sky_resampled

    def remove_count_resampled(self):
        """
        Remove the photoelectron counts of the resampled spectrum.

        """

        self.count_resampled = None
        self.count_err_resampled = None
        self.count_sky_resampled = None

    def add_standard_star(self, library: str, target: str):
        """
        Add the name of the standard star and its source.

        Parameters
        ----------
        library: str
            The name of the colelction of standard stars.
        target: str
            The name of the standard star.

        """

        self.library = library
        self.target = target

    def remove_standard_star(self):
        """
        Remove the name of the standard star and its source.

        """

        self.library = None
        self.target = None

    def add_smoothing(self, smooth: bool, **kwargs: dict):
        """
        Add the SG smoothing parameters for computing the sensitivity curve.

        Parameters
        ----------
        smooth: bool
            Indicate if smoothing was applied.
        kwargs: dict
            Other keyword arguments passed to the lowess smoothing function

        """

        self.smooth = smooth
        self.lowess_kwargs = kwargs

    def remove_smoothing(self):
        """
        Remove the smoothing parameters for computing the sensitivity curve.

        """

        self.smooth = None
        self.slength = None
        self.sorder = None

    def add_sensitivity_func(self, sensitivity_func: Callable):
        """
        Add the callable function of the sensitivity curve.

        Parameters
        ----------
        sensitivity_func: callable function
            The sensitivity curve as a function of wavelength.

        """

        self.sensitivity_func = sensitivity_func

    def remove_sensitivity_func(self):
        """
        Remove the sensitivity curve function.

        """

        self.sensitivity_func = None

    def add_sensitivity(self, sensitivity: Union[list, np.ndarray]):
        """
        Add the sensitivity values for each pixel (the list from dividing
        the literature standard by the photoelectron count).

        Parameters
        ----------
        sensitivity: list or 1-d array
            The sensitivity at the effective pixel.

        """

        self.sensitivity = sensitivity

    def remove_sensitivity(self):
        """
        Remove the sensitivity values.

        """

        self.sensitivity = None

    def add_sensitivity_resampled(
        self, sensitivity_resampled: Union[list, np.ndarray]
    ):
        """
        Add the sensitivity after the spectrum is resampled.

        Parameters
        ----------
        sensitivity: list or 1-d array
            The sensitivity in the reasmpled space.

        """

        self.sensitivity_resampled = sensitivity_resampled

    def remove_sensitivity_resampled(self):
        """
        Remove the sensitivity after the spectrum is resampled.

        """

        self.sensitivity_resampled = None

    def add_literature_standard(
        self,
        wave_literature: Union[list, np.ndarray],
        flux_literature: Union[list, np.ndarray],
    ):
        """
        Add the literature wavelength and flux values of the standard star
        used for calibration.

        Parameters
        ----------
        wave_literature: list or 1-d array
            The wavelength values of the literature standard used.
        flux_literature: list or 1-d array
            The flux values of the literature standard used.

        """

        self.wave_literature = wave_literature
        self.flux_literature = flux_literature

    def remove_literature_standard(self):
        """
        Remove the literature wavelength and flux values of the standard
        star used for calibration

        """

        self.wave_literature = None
        self.flux_literature = None

    def add_count_continuum(self, count_continuum: Union[list, np.ndarray]):
        """
        Add the continuum count value (should be the same size as count).

        Parameters
        ----------
        count_continuum: list or 1-d array
            The photoelectron count of the continuum.

        """

        self.count_continuum = count_continuum

    def remove_count_continuum(self):
        """
        Remove the continuum count values.

        """

        self.count_continuum = None

    def add_count_resampled_continuum(
        self, count_resampled_continuum: Union[list, np.ndarray]
    ):
        """
        Add the continuum count_resampled value (should be the same size as
        count_resampled).

        Parameters
        ----------
        count_resampled_continuum: list or 1-d array
            The photoelectron count of the continuum at the resampled
            wavelength.

        """

        self.count_resampled_continuum = count_resampled_continuum

    def remove_count_resampled_continuum(self):
        """
        Remove the continuum count_resampled values.

        """

        self.count_resampled_continuum = None

    def add_flux_continuum(self, flux_continuum: Union[list, np.ndarray]):
        """
        Add the continuum flux value (should be the same size as flux).

        Parameters
        ----------
        flux_continuum: list or 1-d array
            The flux of the continuum.

        """

        self.flux_continuum = flux_continuum

    def remove_flux_continuum(self):
        """
        Remove the continuum flux values.

        """

        self.flux_continuum = None

    def add_telluric_func(self, telluric_func: Callable):
        """
        Add the Telluric interpolated function.

        Parameters
        ----------
        telluric: Callable function
            A callable function to interpolate the telluric features [0, 1]

        """

        self.telluric_func = telluric_func

    def remove_telluric_func(self):
        """
        Remove the Telluric interpolated function.

        """

        self.telluric_func = None

    def add_telluric_profile(self, telluric_profile: Union[list, np.ndarray]):
        """
        Add the Telluric profile - relative intensity at each pixel.

        Parameters
        ----------
        telluric_profile: list or numpy.ndarray
            The relative intensity of the telluric absorption strength [0, 1]

        """

        self.telluric_profile = telluric_profile

    def remove_telluric_profile(self):
        """
        Remove the Telluric profile.

        """

        self.telluric_profile = None

    def add_telluric_factor(self, telluric_factor: float):
        """
        Add the Telluric factor.

        Parameters
        ----------
        telluric_factor: float
            The multiplier to the telluric profile that minimises the sum
            of the deviation of corrected spectrum from the continuum
            spectrum.

        """

        self.telluric_factor = telluric_factor

    def remove_telluric_factor(self):
        """
        Remove the Telluric factor.

        """

        self.telluric_factor = None

    def add_telluric_nudge_factor(self, telluric_nudge_factor: float):
        """
        Add the Telluric nudge factor.

        Parameters
        ----------
        telluric_nudge_factor: float
            The multiplier to the telluric profile that minimises the sum
            of the deviation of corrected spectrum from the continuum
            spectrum.

        """

        self.telluric_nudge_factor = telluric_nudge_factor

    def remove_telluric_nudge_factor(self):
        """
        Remove the Telluric nudge factor.

        """

        self.telluric_nudge_factor = None

    def add_flux(
        self,
        flux: Union[list, np.ndarray],
        flux_err: Union[list, np.ndarray],
        flux_sky: Union[list, np.ndarray],
    ):
        """
        Add the flux and the associated uncertainty and sky background
        in the raw pixel sampling.

        Parameters
        ----------
        flux: list or numpy.ndarray
            The flux at each extracted column of pixels.
        flux_err: list or numpy.ndarray
            The uncertainty in flux at each extracted column of pixels.
        flux_sky: list or numpy.ndarray
            The flux of the background sky at each extracted column of pixels.

        """

        self.flux = flux
        self.flux_err = flux_err
        self.flux_sky = flux_sky

    def remove_flux(self):
        """
        Remove the flux, uncertainty of flux, and background sky flux.

        """

        self.flux = None
        self.flux_err = None
        self.flux_sky = None

    def add_atm_ext(self, atm_ext: Union[list, np.ndarray]):
        """
        Add the atmospheric extinction correction factor in the native
        wavelengths.

        Parameters
        ----------
        atm_ext: list or numpy.ndarray
            The atmospheric absorption corrected flux at each extracted
            column of pixels.

        """

        self.atm_ext = atm_ext

    def remove_atm_ext(self):
        """
        Remove the atmospheric extinction correction factor in the native
        wavelengthss.

        """

        self.atm_ext = None

    def add_flux_atm_ext_corrected(
        self,
        flux: Union[list, np.ndarray],
        flux_err: Union[list, np.ndarray],
        flux_sky: Union[list, np.ndarray],
    ):
        """
        Add the atmospheric extinction corrected flux and the associated
        uncertainty and sky background in the raw pixel sampling.

        Parameters
        ----------
        flux: list or numpy.ndarray
            The atmospheric absorption corrected flux at each extracted
            column of pixels.
        flux_err: list or numpy.ndarray
            The atmospheric absorption corrected uncertainty in flux at each
            extracted column of pixels.
        flux_sky: list or numpy.ndarray
            The atmospheric absorption corrected flux of the background sky
            at each extracted column of pixels.

        """

        self.flux_atm_ext_corrected = flux
        self.flux_err_atm_ext_corrected = flux_err
        self.flux_sky_atm_ext_corrected = flux_sky

    def remove_flux_atm_ext_corrected(self):
        """
        Remove the atmospheric absorption corrected flux, uncertainty of flux,
        and background sky flux in the raw pixel sampling.

        """

        self.flux_atm_ext_corrected = None
        self.flux_err_atm_ext_corrected = None
        self.flux_sky_atm_ext_corrected = None

    def add_flux_telluric_corrected(
        self,
        flux: Union[list, np.ndarray],
        flux_err: Union[list, np.ndarray],
        flux_sky: Union[list, np.ndarray],
    ):
        """
        Add the telluric flux and the associated uncertainty and sky background
        in the raw pixel sampling.

        Parameters
        ----------
        flux: list or numpy.ndarray
            The atmospheric absorption corrected flux at each extracted
            column of pixels.
        flux_err: list or numpy.ndarray
            The atmospheric absorption corrected uncertainty in flux at each
            extracted column of pixels.
        flux_sky: list or numpy.ndarray
            The atmospheric absorption corrected flux of the background sky
            at each extracted column of pixels.

        """

        self.flux_telluric_corrected = flux
        self.flux_err_telluric_corrected = flux_err
        self.flux_sky_telluric_corrected = flux_sky

    def remove_flux_telluric_corrected(self):
        """
        Remove the telluric corrected flux, uncertainty of flux,
        and background sky flux in the raw pixel sampling.

        """

        self.flux_telluric_corrected = None
        self.flux_err_telluric_corrected = None
        self.flux_sky_telluric_corrected = None

    def add_flux_atm_ext_telluric_corrected(
        self,
        flux: Union[list, np.ndarray],
        flux_err: Union[list, np.ndarray],
        flux_sky: Union[list, np.ndarray],
    ):
        """
        Add the atmospheric extinction and telluric corrected flux and the
        associated uncertainty and sky background in the raw pixel sampling.

        Parameters
        ----------
        flux: list or numpy.ndarray
            The atmospheric absorption corrected flux at each extracted
            column of pixels.
        flux_err: list or numpy.ndarray
            The atmospheric absorption corrected uncertainty in flux at each
            extracted column of pixels.
        flux_sky: list or numpy.ndarray
            The atmospheric absorption corrected flux of the background sky
            at each extracted column of pixels.

        """

        self.flux_atm_ext_telluric_corrected = flux
        self.flux_err_atm_ext_telluric_corrected = flux_err
        self.flux_sky_atm_ext_telluric_corrected = flux_sky

    def remove_flux_atm_ext_telluric_corrected(self):
        """
        Remove the atmospheric absorption and telluric corrected flux,
        uncertainty of flux, and background sky flux in the raw pixel
        sampling.

        """

        self.flux_atm_ext_telluric_corrected = None
        self.flux_err_atm_ext_telluric_corrected = None
        self.flux_sky_atm_ext_telluric_corrected = None

    def add_flux_resampled(
        self,
        flux_resampled: Union[list, np.ndarray],
        flux_err_resampled: Union[list, np.ndarray],
        flux_sky_resampled: Union[list, np.ndarray],
    ):
        """
        Add the flux and the associated uncertainty and sky background
        of the resampled spectrum.

        Parameters
        ----------
        flux: list or numpy.ndarray
            The flux at the resampled wavelenth.
        flux_err: list or numpy.ndarray
            The uncertainty in flux at the resampled wavelenth.
        flux_sky: list or numpy.ndarray
            The flux of the background sky at the resampled wavelenth.

        """

        self.flux_resampled = flux_resampled
        self.flux_err_resampled = flux_err_resampled
        self.flux_sky_resampled = flux_sky_resampled

    def remove_flux_resampled(self):
        """
        Remove the flux, uncertainty of flux, and background sky flux
        of the resampled spectrum.

        """

        self.flux_resampled = None
        self.flux_err_resampled = None
        self.flux_sky_resampled = None

    def add_atm_ext_resampled(
        self, atm_ext_resampled: Union[list, np.ndarray]
    ):
        """
        Add the atmospheric extinction correction factor in the resampled
        wavelengths.

        Parameters
        ----------
        atm_ext_resampled: list or numpy.ndarray
            The atmospheric absorption corrected flux at the resampled
            wavelengths.

        """

        self.atm_ext_resampled = atm_ext_resampled

    def remove_atm_ext_resampled(self):
        """
        Remove the atmospheric extinction correction factor in the resampled
        wavelengths.

        """

        self.atm_ext_resampled = None

    def add_flux_resampled_atm_ext_corrected(
        self,
        flux: Union[list, np.ndarray],
        flux_err: Union[list, np.ndarray],
        flux_sky: Union[list, np.ndarray],
    ):
        """
        Add the flux and the associated uncertainty and sky background
        of the resampled spectrumg.

        Parameters
        ----------
        flux: list or numpy.ndarray
            The atmospheric absorption corrected flux at each extracted
            column of pixels.
        flux_err: list or numpy.ndarray
            The atmospheric absorption corrected uncertainty in flux at each
            extracted column of pixels.
        flux_sky: list or numpy.ndarray
            The atmospheric absorption corrected flux of the background sky
            at each extracted column of pixels.

        """

        self.flux_resampled_atm_ext_corrected = flux
        self.flux_err_resampled_atm_ext_corrected = flux_err
        self.flux_sky_resampled_atm_ext_corrected = flux_sky

    def remove_flux_resampled_atm_ext_corrected(self):
        """
        Remove the atmospheric absorption corrected flux, uncertainty of flux,
        and background sky flux of the resampled spectrum.

        """

        self.flux_resampled_atm_ext_corrected = None
        self.flux_err_resampled_atm_ext_corrected = None
        self.flux_sky_resampled_atm_ext_corrected = None

    def add_telluric_profile_resampled(
        self, telluric_profile_resampled: Union[list, np.ndarray]
    ):
        """
        Add the telluric absorption profile in the resampled wavelengths.

        Parameters
        ----------
        telluric_profile_resampled: list or numpy.ndarray
            The atmospheric absorption corrected flux at each extracted
            column of pixels.

        """

        self.telluric_profile_resampled = telluric_profile_resampled

    def remove_telluric_profile_resampled(self):
        """
        Remove the telluric absorption profile in the resampled wavelengthss.

        """

        self.telluric_profile_resampled = None

    def add_flux_resampled_telluric_corrected(
        self,
        flux: Union[list, np.ndarray],
        flux_err: Union[list, np.ndarray],
        flux_sky: Union[list, np.ndarray],
    ):
        """
        Add the flux and the associated uncertainty and sky background
        of the resampled spectrum.

        Parameters
        ----------
        flux: list or numpy.ndarray
            The telluric corrected flux at each extracted
            column of pixels.
        flux_err: list or numpy.ndarray
            The telluric corrected uncertainty in flux at each
            extracted column of pixels.
        flux_sky: list or numpy.ndarray
            The telluric corrected flux of the background sky
            at each extracted column of pixels.

        """

        self.flux_resampled_telluric_corrected = flux
        self.flux_err_resampled_telluric_corrected = flux_err
        self.flux_sky_resampled_telluric_corrected = flux_sky

    def remove_flux_resampled_telluric_corrected(self):
        """
        Remove the atmospheric absorption corrected flux, uncertainty of flux,
        and background sky flux of the resampled spectrum.

        """

        self.flux_resampled_telluric_corrected = None
        self.flux_err_resampled_telluric_corrected = None
        self.flux_sky_resampled_telluric_corrected = None

    def add_flux_resampled_atm_ext_telluric_corrected(
        self,
        flux: Union[list, np.ndarray],
        flux_err: Union[list, np.ndarray],
        flux_sky: Union[list, np.ndarray],
    ):
        """
        Add the flux and the associated uncertainty and sky background
        of the resampled spectrum.

        Parameters
        ----------
        flux: list or numpy.ndarray
            The atmospheric absorption and telluric corrected flux at each
            extracted column of pixels.
        flux_err: list or numpy.ndarray
            The atmospheric absorption and telluric corrected uncertainty in
            flux at each extracted column of pixels.
        flux_sky: list or numpy.ndarray
            The atmospheric absorption and telluric corrected flux of the
            background sky at each extracted column of pixels.

        """

        self.flux_resampled_atm_ext_telluric_corrected = flux
        self.flux_err_resampled_atm_ext_telluric_corrected = flux_err
        self.flux_sky_resampled_atm_ext_telluric_corrected = flux_sky

    def remove_flux_resampled_atm_ext_telluric_corrected(self):
        """
        Remove the atmospheric absorption and telluric corrected flux,
        uncertainty of flux, and background sky flux of the resampled
        spectrum.

        """

        self.flux_resampled_atm_ext_telluric_corrected = None
        self.flux_err_resampled_atm_ext_telluric_corrected = None
        self.flux_sky_resampled_atm_ext_telluric_corrected = None

    def _modify_imagehdu_data(
        self, hdulist: list, idx: int, method: str, *args: str
    ):
        """
        Wrapper function to modify the data of an ImageHDU object.

        """

        method_to_call = getattr(hdulist[idx].data, method)
        method_to_call(*args)

    def _modify_imagehdu_header(
        self, hdulist: list, idx: int, method: str, *args: str
    ):
        """
        Wrapper function to modify the header of an ImageHDU object.

        e.g.
        method = 'set'
        args = 'BUNIT', 'Angstroms'

        method_to_call(*args) becomes hdu[idx].header.set('BUNIT', 'Angstroms')

        """

        method_to_call = getattr(hdulist[idx].header, method)
        method_to_call(*args)

    def modify_trace_header(self, idx: int, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+---------------------+
        | HDU | Data                |
        +-----+---------------------+
        |  0  | Trace (pixel)       |
        |  1  | Trace width (pixel) |
        +-----+---------------------+

        """

        self._modify_imagehdu_header(self.trace_hdulist, idx, method, *args)

    def modify_count_header(self, idx: int, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+---------------------------------+
        | HDU | Data                            |
        +-----+---------------------------------+
        |  0  | Photoelectron count             |
        |  1  | Photoelectron count uncertainty |
        |  2  | Photoelectron count (sky)       |
        +-----+---------------------------------+

        """

        self._modify_imagehdu_header(self.count_hdulist, idx, method, *args)

    def modify_weight_map_header(self, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+----------+
        | HDU | Data     |
        +-----+----------+
        |  0  | Variance |
        +-----+----------+

        """

        self._modify_imagehdu_header(self.weight_map_hdulist, 0, method, *args)

    def modify_arc_spec_header(self, idx: int, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+-------------------+
        | HDU | Data              |
        +-----+-------------------+
        |  0  | Arc spectrum      |
        +-----+-------------------+

        """

        self._modify_imagehdu_header(self.arc_spec_hdulist, idx, method, *args)

    def modify_arc_lines_header(self, idx: int, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+-------------------+
        | HDU | Data              |
        +-----+-------------------+
        |  0  | Peaks (pixel)     |
        |  1  | Peaks (sub-pixel) |
        +-----+-------------------+

        """

        self._modify_imagehdu_header(
            self.arc_lines_hdulist, idx, method, *args
        )

    def modify_wavecal_header(self, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None
        wavelength fits only has one ImageHDU so the idx is always 0

        +-----+-----------------------+
        | HDU | Data                  |
        +-----+-----------------------+
        |  0  | Best fit coefficients |
        +-----+-----------------------+

        """

        self._modify_imagehdu_header(self.wavecal_hdulist, 0, method, *args)

    def modify_wavelength_header(self, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None
        wavelength fits only has one ImageHDU so the idx is always 0

        +-----+------------------------------------------+
        | HDU | Data                                     |
        +-----+------------------------------------------+
        |  0  | Wavelength value at each extracted pixel |
        +-----+------------------------------------------+

        """

        self._modify_imagehdu_header(self.wavelength_hdulist, 0, method, *args)

    def modify_wavelength_resampled_header(self, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None
        wavelength fits only has one ImageHDU so the idx is always 0

        +-----+---------------------------------------------+
        | HDU | Data                                        |
        +-----+---------------------------------------------+
        |  0  | Wavelength value at each resampled position |
        +-----+---------------------------------------------+

        """

        self._modify_imagehdu_header(
            self.wavelength_resampled_hdulist, 0, method, *args
        )

    def modify_count_resampled_header(self, idx: int, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+---------------------------------+
        | HDU | Data                            |
        +-----+---------------------------------+
        |  0  | Photoelectron count             |
        |  1  | Photoelectron count uncertainty |
        |  2  | Photoelectron count (sky)       |
        +-----+---------------------------------+

        """

        self._modify_imagehdu_header(
            self.count_resampled_hdulist, idx, method, *args
        )

    def modify_sensitivity_header(self, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+-------------+
        | HDU | Data        |
        +-----+-------------+
        |  0  | Sensitivity |
        +-----+-------------+

        """

        self._modify_imagehdu_header(
            self.sensitivity_hdulist, 0, method, *args
        )

    def modify_flux_header(self, idx: int, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------+
        | HDU | Data             |
        +-----+------------------+
        |  0  | Flux             |
        |  1  | Flux uncertainty |
        |  2  | Flux (sky)       |
        +-----+------------------+

        """

        self._modify_imagehdu_header(self.flux_hdulist, idx, method, *args)

    def modify_atm_ext_header(self, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------------+
        | HDU | Data                   |
        +-----+------------------------+
        |  0  | Atmospheric Extinction |
        +-----+------------------------+

        """

        self._modify_imagehdu_header(self.atm_ext_hdulist, 0, method, *args)

    def modify_flux_atm_ext_corrected_header(
        self, idx: int, method: str, *args: str
    ):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------+
        | HDU | Data             |
        +-----+------------------+
        |  0  | Flux             |
        |  1  | Flux uncertainty |
        |  2  | Flux (sky)       |
        +-----+------------------+

        """

        self._modify_imagehdu_header(
            self.flux_atm_ext_corrected_hdulist, idx, method, *args
        )

    def modify_telluric_profile_header(self, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+-----------------------------+
        | HDU | Data                        |
        +-----+-----------------------------+
        |  0  | Telluric Absorption Profile |
        +-----+-----------------------------+

        """

        self._modify_imagehdu_header(
            self.telluric_profile_hdulist, 0, method, *args
        )

    def modify_flux_telluric_corrected_header(
        self, idx: int, method: str, *args: str
    ):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------+
        | HDU | Data             |
        +-----+------------------+
        |  0  | Flux             |
        |  1  | Flux uncertainty |
        |  2  | Flux (sky)       |
        +-----+------------------+

        """

        self._modify_imagehdu_header(
            self.flux_telluric_corrected_hdulist, idx, method, *args
        )

    def modify_flux_atm_ext_telluric_corrected_header(
        self, idx: int, method: str, *args: str
    ):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------+
        | HDU | Data             |
        +-----+------------------+
        |  0  | Flux             |
        |  1  | Flux uncertainty |
        |  2  | Flux (sky)       |
        +-----+------------------+

        """

        self._modify_imagehdu_header(
            self.flux_atm_ext_telluric_corrected_hdulist, idx, method, *args
        )

    def modify_sensitivity_resampled_header(self, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+-----------------------+
        | HDU | Data                  |
        +-----+-----------------------+
        |  0  | Sensitivity_resampled |
        +-----+-----------------------+

        """

        self._modify_imagehdu_header(
            self.sensitivity_resampled_hdulist, 0, method, *args
        )

    def modify_flux_resampled_header(self, idx: int, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------+
        | HDU | Data             |
        +-----+------------------+
        |  0  | Flux             |
        |  1  | Flux Uncertainty |
        |  2  | Flux Sky         |
        +-----+------------------+

        """

        self._modify_imagehdu_header(
            self.flux_resampled_hdulist, idx, method, *args
        )

    def modify_atm_ext_resampled_header(self, method: str, *args: str):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------------+
        | HDU | Data                   |
        +-----+------------------------+
        |  0  | Atmospheric Extinction |
        +-----+------------------------+

        """

        self._modify_imagehdu_header(
            self.atm_ext_resampled_hdulist, 0, method, *args
        )

    def modify_flux_resampled_atm_ext_corrected_header(
        self, idx: int, method: str, *args: str
    ):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------+
        | HDU | Data             |
        +-----+------------------+
        |  0  | Flux             |
        |  1  | Flux uncertainty |
        |  2  | Flux (sky)       |
        +-----+------------------+

        """

        self._modify_imagehdu_header(
            self.flux_resampled_atm_ext_corrected_hdulist, idx, method, *args
        )

    def modify_telluric_profile_resampled_header(
        self, method: str, *args: str
    ):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+-----------------------------+
        | HDU | Data                        |
        +-----+-----------------------------+
        |  0  | Telluric Absorption Profile |
        +-----+-----------------------------+

        """

        self._modify_imagehdu_header(
            self.telluric_profile_resampled_hdulist, 0, method, *args
        )

    def modify_flux_resampled_telluric_corrected_header(
        self, idx: int, method: str, *args: str
    ):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------+
        | HDU | Data             |
        +-----+------------------+
        |  0  | Flux             |
        |  1  | Flux uncertainty |
        |  2  | Flux (sky)       |
        +-----+------------------+

        """

        self._modify_imagehdu_header(
            self.flux_resampled_telluric_corrected_hdulist, idx, method, *args
        )

    def modify_flux_resampled_atm_ext_telluric_corrected_header(
        self, idx: int, method: str, *args: str
    ):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+------------------+
        | HDU | Data             |
        +-----+------------------+
        |  0  | Flux             |
        |  1  | Flux uncertainty |
        |  2  | Flux (sky)       |
        +-----+------------------+

        """

        self._modify_imagehdu_header(
            self.flux_resampled_atm_ext_telluric_corrected_hdulist,
            idx,
            method,
            *args,
        )

    def create_trace_fits(self):
        """
        Create an ImageHDU for the trace.

        """

        try:
            # Use the header of the spectrum
            if self.spectrum_header is not None:
                trace_ImageHDU = fits.ImageHDU(
                    self.trace, header=self.spectrum_header
                )
                trace_sigma_ImageHDU = fits.ImageHDU(
                    self.trace_sigma, header=self.spectrum_header
                )
            else:
                trace_ImageHDU = fits.ImageHDU(self.trace)
                trace_sigma_ImageHDU = fits.ImageHDU(self.trace_sigma)

            # Create an empty HDU list and populate with ImageHDUs
            self.trace_hdulist = fits.HDUList()
            self.trace_hdulist += [trace_ImageHDU]
            self.trace_hdulist += [trace_sigma_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["trace"]]

            # Add the trace
            self.modify_trace_header(0, "set", "EXTNAME", hdu_names[0])
            self.modify_trace_header(0, "set", "LABEL", "Trace")
            self.modify_trace_header(0, "set", "CRPIX1", 1)
            self.modify_trace_header(0, "set", "CDELT1", 1)
            self.modify_trace_header(
                0, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_trace_header(0, "set", "CTYPE1", "Pixel (Dispersion)")
            self.modify_trace_header(0, "set", "CUNIT1", "Pixel")
            self.modify_trace_header(0, "set", "BUNIT", "Pixel (Spatial)")

            # Add the trace_sigma
            self.modify_trace_header(1, "set", "EXTNAME", hdu_names[1])
            self.modify_trace_header(1, "set", "LABEL", "Trace width")
            self.modify_trace_header(1, "set", "CRPIX1", 1)
            self.modify_trace_header(1, "set", "CDELT1", 1)
            self.modify_trace_header(
                1, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_trace_header(1, "set", "CTYPE1", "Pixel (Dispersion)")
            self.modify_trace_header(1, "set", "CUNIT1", "Number of Pixels")
            self.modify_trace_header(1, "set", "BUNIT", "Pixel (Spatial)")

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("trace ImageHDU cannot be created.")
            self.trace_hdulist = None

    def create_count_fits(self):
        """
        Create an ImageHDU for the extracted spectrum in photoelectron counts.

        """

        try:
            # Use the header of the spectrum
            if self.spectrum_header is not None:
                count_ImageHDU = fits.ImageHDU(
                    self.count, header=self.spectrum_header
                )
                count_err_ImageHDU = fits.ImageHDU(
                    self.count_err, header=self.spectrum_header
                )
                count_sky_ImageHDU = fits.ImageHDU(
                    self.count_sky, header=self.spectrum_header
                )
            else:
                count_ImageHDU = fits.ImageHDU(self.count)
                count_err_ImageHDU = fits.ImageHDU(self.count_err)
                count_sky_ImageHDU = fits.ImageHDU(self.count_sky)

            # Create an empty HDU list and populate with ImageHDUs
            self.count_hdulist = fits.HDUList()
            self.count_hdulist += [count_ImageHDU]
            self.count_hdulist += [count_err_ImageHDU]
            self.count_hdulist += [count_sky_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["count"]]

            # Add the count
            self.modify_count_header(0, "set", "WIDTHDN", self.widthdn)
            self.modify_count_header(0, "set", "WIDTHUP", self.widthup)
            self.modify_count_header(0, "set", "SEPDN", self.sepdn)
            self.modify_count_header(0, "set", "SEPUP", self.sepup)
            self.modify_count_header(0, "set", "SKYDN", self.skywidthdn)
            self.modify_count_header(0, "set", "SKYUP", self.skywidthup)
            self.modify_count_header(0, "set", "XTYPE", self.extraction_type)
            self.modify_count_header(0, "set", "EXTNAME", hdu_names[0])
            self.modify_count_header(0, "set", "LABEL", "Electron count")
            self.modify_count_header(0, "set", "CRPIX1", 1)
            self.modify_count_header(0, "set", "CDELT1", 1)
            self.modify_count_header(
                0, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_count_header(0, "set", "CTYPE1", " Pixel (Dispersion)")
            self.modify_count_header(0, "set", "CUNIT1", "Pixel")
            self.modify_count_header(0, "set", "BUNIT", "electron")
            self.modify_count_header(0, "set", "XPOSURE", self.exptime)
            self.modify_count_header(0, "set", "GAIN", self.gain)
            self.modify_count_header(0, "set", "RNOISE", self.readnoise)
            self.modify_count_header(0, "set", "SEEING", self.seeing)
            self.modify_count_header(0, "set", "AIRMASS", self.airmass)

            # Add the uncertainty count
            self.modify_count_header(1, "set", "EXTNAME", hdu_names[1])
            self.modify_count_header(
                1, "set", "LABEL", "Electron count (Uncertainty)"
            )
            self.modify_count_header(1, "set", "CRPIX1", 1)
            self.modify_count_header(1, "set", "CDELT1", 1)
            self.modify_count_header(
                1, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_count_header(1, "set", "CTYPE1", "Pixel (Dispersion)")
            self.modify_count_header(1, "set", "CUNIT1", "Pixel")
            self.modify_count_header(1, "set", "BUNIT", "electron")

            # Add the sky count
            self.modify_count_header(2, "set", "EXTNAME", hdu_names[2])
            self.modify_count_header(2, "set", "LABEL", "Electron count (Sky)")
            self.modify_count_header(2, "set", "CRPIX1", 1)
            self.modify_count_header(2, "set", "CDELT1", 1)
            self.modify_count_header(
                2, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_count_header(2, "set", "CTYPE1", "Pixel (Dispersion)")
            self.modify_count_header(2, "set", "CUNIT1", "Pixel")
            self.modify_count_header(2, "set", "BUNIT", "electron")

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("count ImageHDU cannot be created.")
            self.count_hdulist = None

    def create_weight_map_fits(self):
        """
        Create an ImageHDU for the extraction profile weight function.

        """

        try:
            # Use the header of the spectrum
            if self.spectrum_header is not None:
                weight_map_ImageHDU = fits.ImageHDU(
                    self.var, header=self.spectrum_header
                )
            else:
                weight_map_ImageHDU = fits.ImageHDU(self.var)

            # Create an empty HDU list and populate with ImageHDUs
            self.weight_map_hdulist = fits.HDUList()
            self.weight_map_hdulist += [weight_map_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["weight_map"]]

            # Add the extraction weights
            self.modify_weight_map_header("set", "EXTNAME", hdu_names[0])
            self.modify_weight_map_header(
                "set", "LABEL", "Optimal extraction profile"
            )
            if self.var is not None:
                self.modify_weight_map_header("set", "CRVAL1", len(self.var))
                self.modify_weight_map_header("set", "CRPIX1", 1)
                self.modify_weight_map_header("set", "CDELT1", 1)
                self.modify_weight_map_header(
                    "set", "CTYPE1", "Pixel (Spatial)"
                )
                self.modify_weight_map_header("set", "CUNIT1", "Pixel")
                self.modify_weight_map_header("set", "BUNIT", "weights")
            else:
                self.modify_weight_map_header(
                    "set", "COMMENT", "Extraction Profile is not available."
                )

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("A weight map ImageHDU cannot be created.")
            self.weight_map_hdulist = None

    def create_count_resampled_fits(self):
        """
        Create an ImageHDU for the extracted spectrum in photoelectron count
        at the resampled wavelengths.

        """

        try:
            # Use the header of the spectrum
            if self.spectrum_header is not None:
                count_resampled_ImageHDU = fits.ImageHDU(
                    self.count_resampled, header=self.spectrum_header
                )
                count_err_resampled_ImageHDU = fits.ImageHDU(
                    self.count_err_resampled, header=self.spectrum_header
                )
                count_sky_resampled_ImageHDU = fits.ImageHDU(
                    self.count_sky_resampled, header=self.spectrum_header
                )
            else:
                count_resampled_ImageHDU = fits.ImageHDU(self.count_resampled)
                count_err_resampled_ImageHDU = fits.ImageHDU(
                    self.count_err_resampled
                )
                count_sky_resampled_ImageHDU = fits.ImageHDU(
                    self.count_sky_resampled
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.count_resampled_hdulist = fits.HDUList()
            self.count_resampled_hdulist += [count_resampled_ImageHDU]
            self.count_resampled_hdulist += [count_err_resampled_ImageHDU]
            self.count_resampled_hdulist += [count_sky_resampled_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["count_resampled"]]

            # Add the resampled count
            self.modify_count_resampled_header(
                0, "set", "EXTNAME", hdu_names[0]
            )
            self.modify_count_resampled_header(
                0, "set", "LABEL", "Resampled electron count"
            )
            self.modify_count_resampled_header(0, "set", "CRPIX1", 1.00e00)
            self.modify_count_resampled_header(
                0, "set", "CDELT1", self.wave_bin
            )
            self.modify_count_resampled_header(
                0, "set", "CRVAL1", self.wave_start
            )
            self.modify_count_resampled_header(
                0, "set", "CTYPE1", "Wavelength"
            )
            self.modify_count_resampled_header(0, "set", "CUNIT1", "Angstroms")
            self.modify_count_resampled_header(0, "set", "BUNIT", "electron")
            self.modify_count_resampled_header(0, "set", "BUNIT", "electron")
            self.modify_count_resampled_header(
                0, "set", "XPOSURE", self.exptime
            )
            self.modify_count_resampled_header(0, "set", "GAIN", self.gain)
            self.modify_count_resampled_header(
                0, "set", "RNOISE", self.readnoise
            )
            self.modify_count_resampled_header(0, "set", "SEEING", self.seeing)
            self.modify_count_resampled_header(
                0, "set", "AIRMASS", self.airmass
            )

            # Add the resampled uncertainty count
            self.modify_count_resampled_header(
                1, "set", "EXTNAME", hdu_names[1]
            )
            self.modify_count_resampled_header(
                1, "set", "LABEL", "Resampled electron count (Uncertainty)"
            )
            self.modify_count_resampled_header(1, "set", "CRPIX1", 1.00e00)
            self.modify_count_resampled_header(
                1, "set", "CDELT1", self.wave_bin
            )
            self.modify_count_resampled_header(
                1, "set", "CRVAL1", self.wave_start
            )
            self.modify_count_resampled_header(
                1, "set", "CTYPE1", "Wavelength"
            )
            self.modify_count_resampled_header(1, "set", "CUNIT1", "Angstroms")
            self.modify_count_resampled_header(1, "set", "BUNIT", "electron")

            # Add the resampled sky count
            self.modify_count_resampled_header(
                2, "set", "EXTNAME", hdu_names[2]
            )
            self.modify_count_resampled_header(
                2, "set", "LABEL", "Resampled electron count (Sky)"
            )
            self.modify_count_resampled_header(2, "set", "CRPIX1", 1.00e00)
            self.modify_count_resampled_header(
                2, "set", "CDELT1", self.wave_bin
            )
            self.modify_count_resampled_header(
                2, "set", "CRVAL1", self.wave_start
            )
            self.modify_count_resampled_header(
                2, "set", "CTYPE1", "Wavelength"
            )
            self.modify_count_resampled_header(2, "set", "CUNIT1", "Angstroms")
            self.modify_count_resampled_header(2, "set", "BUNIT", "electron")

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("count_resampled ImageHDU cannot be created.")
            self.count_resampled_hdulist = None

    def create_arc_spec_fits(self):
        """
        Create an ImageHDU for the spectrum of the arc lamp.

        """

        try:
            # Use the header of the arc
            if self.arc_header is not None:
                arc_spec_ImageHDU = fits.ImageHDU(
                    self.arc_spec, header=self.arc_header
                )
            else:
                arc_spec_ImageHDU = fits.ImageHDU(self.arc_spec)

            # Create an empty HDU list and populate with ImageHDUs
            self.arc_spec_hdulist = fits.HDUList()
            self.arc_spec_hdulist += [arc_spec_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["arc_spec"]]

            # Add the arc spectrum
            self.modify_arc_spec_header(0, "set", "EXTNAME", hdu_names[0])
            self.modify_arc_spec_header(
                0, "set", "LABEL", "Electron count (Arc)"
            )
            self.modify_arc_spec_header(0, "set", "CRPIX1", 1)
            self.modify_arc_spec_header(0, "set", "CDELT1", 1)
            self.modify_arc_spec_header(
                0, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_arc_spec_header(
                0, "set", "CTYPE1", "Pixel (Dispersion)"
            )
            self.modify_arc_spec_header(0, "set", "CUNIT1", "Pixel")
            self.modify_arc_spec_header(0, "set", "BUNIT", "electron")

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("arc_spec ImageHDU cannot be created.")
            self.arc_spec_hdulist = None

    def create_arc_lines_fits(self):
        """
        Create an ImageHDU for the spectrum of the arc lamp.

        """

        try:
            # Use the header of the arc
            if self.arc_header is not None:
                peaks_ImageHDU = fits.ImageHDU(
                    self.peaks, header=self.arc_header
                )
                peaks_refined_ImageHDU = fits.ImageHDU(
                    self.peaks_refined, header=self.arc_header
                )
            else:
                peaks_ImageHDU = fits.ImageHDU(self.peaks)
                peaks_refined_ImageHDU = fits.ImageHDU(self.peaks_refined)

            # Create an empty HDU list and populate with ImageHDUs
            self.arc_lines_hdulist = fits.HDUList()
            self.arc_lines_hdulist += [peaks_ImageHDU]
            self.arc_lines_hdulist += [peaks_refined_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["arc_lines"]]

            # Add the peaks in native pixel value
            self.modify_arc_lines_header(0, "set", "EXTNAME", hdu_names[0])
            self.modify_arc_lines_header(
                0, "set", "LABEL", "Peaks (Detector pixel)"
            )
            self.modify_arc_lines_header(0, "set", "BUNIT", "Pixel")

            # Add the peaks in effective pixel value
            self.modify_arc_lines_header(1, "set", "EXTNAME", hdu_names[1])
            self.modify_arc_lines_header(
                1, "set", "LABEL", "Peaks (Effective pixel)"
            )
            self.modify_arc_lines_header(1, "set", "BUNIT", "Pixel")

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("arc_lines ImageHDU cannot be created.")
            self.arc_lines_hdulist = None

    def create_wavecal_fits(self):
        """
        Create an ImageHDU for the polynomial coeffcients of the
        pixel-wavelength mapping function.

        """

        try:
            # Use the header of the arc
            if self.arc_header is not None:
                wavecal_ImageHDU = fits.ImageHDU(
                    self.fit_coeff, header=self.arc_header
                )
            else:
                wavecal_ImageHDU = fits.ImageHDU(self.fit_coeff)

            # Create an empty HDU list and populate with ImageHDUs
            self.wavecal_hdulist = fits.HDUList()
            self.wavecal_hdulist += [wavecal_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["wavecal_coefficients"]]

            # Add the wavelength calibration header info
            self.modify_wavecal_header("set", "FTYPE", self.fit_type)
            self.modify_wavecal_header("set", "FDEG", self.fit_deg)
            self.modify_wavecal_header("set", "FFRMS", self.rms)
            self.modify_wavecal_header(
                "set", "ATLWMIN", self.min_atlas_wavelength
            )
            self.modify_wavecal_header(
                "set", "ATLWMAX", self.max_atlas_wavelength
            )
            self.modify_wavecal_header("set", "NSLOPES", self.num_slopes)
            self.modify_wavecal_header("set", "RNGTOL", self.range_tolerance)
            self.modify_wavecal_header("set", "FITTOL", self.fit_tolerance)
            self.modify_wavecal_header(
                "set", "CANTHRE", self.candidate_tolerance
            )
            self.modify_wavecal_header(
                "set", "LINTHRE", self.linearity_tolerance
            )
            self.modify_wavecal_header("set", "RANTHRE", self.ransac_tolerance)
            self.modify_wavecal_header("set", "NUMCAN", self.num_candidates)
            self.modify_wavecal_header("set", "XBINS", self.xbins)
            self.modify_wavecal_header("set", "YBINS", self.ybins)
            self.modify_wavecal_header("set", "BRUTE", self.brute_force)
            self.modify_wavecal_header("set", "SAMSIZE", self.sample_size)
            self.modify_wavecal_header("set", "TOPN", self.top_n)
            self.modify_wavecal_header("set", "MAXTRY", self.max_tries)
            self.modify_wavecal_header("set", "INCOEFF", self.intput_coeff)
            self.modify_wavecal_header("set", "LINEAR", self.linear)
            self.modify_wavecal_header("set", "W8ED", self.weighted)
            self.modify_wavecal_header("set", "FILTER", self.filter_close)
            self.modify_wavecal_header("set", "PUSAGE", self.peak_utilisation)
            self.modify_wavecal_header("set", "EXTNAME", hdu_names[0])
            self.modify_wavecal_header(
                "set", "LABEL", "Wavelength calibration coefficients"
            )
            self.modify_wavecal_header("set", "CRPIX1", 1.00e00)
            self.modify_wavecal_header("set", "CDELT1", self.wave_bin)
            self.modify_wavecal_header("set", "CRVAL1", self.wave_start)
            self.modify_wavecal_header("set", "CTYPE1", "Wavelength")
            self.modify_wavecal_header("set", "CUNIT1", "Angstroms")
            self.modify_wavecal_header("set", "BUNIT", "electron")

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("wavecal ImageHDU cannot be created.")
            self.wavecal_hdulist = None

    def create_wavelength_fits(self):
        """
        Create an ImageHDU for the wavelength at each of the native pixel.

        """

        try:
            # Put the data in an ImageHDU

            # Use the header of the arc
            if self.arc_header is not None:
                wavelength_ImageHDU = fits.ImageHDU(
                    self.wave, header=self.arc_header
                )
            else:
                wavelength_ImageHDU = fits.ImageHDU(self.wave)

            # Create an empty HDU list and populate with the ImageHDU
            self.wavelength_hdulist = fits.HDUList()
            self.wavelength_hdulist += [wavelength_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["wavelength"]]

            # Add the calibrated wavelength
            self.modify_wavelength_header("set", "EXTNAME", hdu_names[0])
            self.modify_wavelength_header(
                "set", "LABEL", "Pixel-wavelength mapping"
            )
            self.modify_wavelength_header("set", "CRPIX1", 1)
            self.modify_wavelength_header("set", "CDELT1", 1)
            self.modify_wavelength_header(
                "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_wavelength_header(
                "set", "CTYPE1", "Pixel (Dispersion)"
            )
            self.modify_wavelength_header("set", "CUNIT1", "Pixel")
            self.modify_wavelength_header("set", "BUNIT", "Angstroms")

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("wavelength ImageHDU cannot be created.")
            self.wavelength_hdulist = None

    def create_wavelength_resampled_fits(self):
        """
        Create an ImageHDU for the wavelength at each resampled position.

        """

        try:
            # Put the data in an ImageHDU

            # Use the header of the arc
            if self.arc_header is not None:
                wavelength_resampled_ImageHDU = fits.ImageHDU(
                    self.wave_resampled, header=self.arc_header
                )
            else:
                wavelength_resampled_ImageHDU = fits.ImageHDU(
                    self.wave_resampled
                )

            # Create an empty HDU list and populate with the ImageHDU
            self.wavelength_resampled_hdulist = fits.HDUList()
            self.wavelength_resampled_hdulist += [
                wavelength_resampled_ImageHDU
            ]

            hdu_names = self.ext_name[self.hdu_order["wavelength_resampled"]]

            # Add the calibrated wavelength
            self.modify_wavelength_resampled_header(
                "set", "EXTNAME", hdu_names[0]
            )
            self.modify_wavelength_resampled_header(
                "set", "LABEL", "Wavelength at resampled position"
            )
            self.modify_wavelength_resampled_header("set", "CRPIX1", 1)
            self.modify_wavelength_resampled_header("set", "CDELT1", 1)
            self.modify_wavelength_resampled_header(
                "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_wavelength_resampled_header(
                "set", "CTYPE1", "Pixel (Dispersion)"
            )
            self.modify_wavelength_resampled_header("set", "CUNIT1", "Pixel")
            self.modify_wavelength_resampled_header(
                "set", "BUNIT", "Angstroms"
            )

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning(
                "wavelength_resampled ImageHDU cannot be created."
            )
            self.wavelength_resampled_hdulist = None

    def create_sensitivity_fits(self):
        """
        Create an ImageHDU for the sensitivity at each of the native pixel.

        """

        try:
            # Put the data in ImageHDUs

            # Use the header of the standard
            if self.standard_header is not None:
                sensitivity_ImageHDU = fits.ImageHDU(
                    self.sensitivity, header=self.standard_header
                )
            else:
                sensitivity_ImageHDU = fits.ImageHDU(self.sensitivity)

            # Create an empty HDU list and populate with ImageHDUs
            self.sensitivity_hdulist = fits.HDUList()
            self.sensitivity_hdulist += [sensitivity_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["sensitivity"]]

            self.modify_sensitivity_header("set", "EXTNAME", hdu_names[0])
            self.modify_sensitivity_header("set", "LABEL", "Sensitivity")
            self.modify_sensitivity_header("set", "CRPIX1", 1.00e00)
            self.modify_sensitivity_header("set", "CDELT1", 1)
            self.modify_sensitivity_header(
                "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_sensitivity_header("set", "CTYPE1", "Pixel")
            self.modify_sensitivity_header("set", "CUNIT1", "Pixel")
            self.modify_sensitivity_header(
                "set", "BUNIT", "erg/(s*cm**2*Angstrom)/Count"
            )

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("sensitivity ImageHDU cannot be created.")
            self.sensitivity_hdulist = None

    def create_flux_fits(self):
        """
        Create an ImageHDU for the flux calibrated spectrum at each native
        pixel.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data in ImageHDUs
                flux_ImageHDU = fits.ImageHDU(self.flux, header=header)
                flux_err_ImageHDU = fits.ImageHDU(self.flux_err, header=header)
                flux_sky_ImageHDU = fits.ImageHDU(self.flux_sky, header=header)

            else:
                flux_ImageHDU = fits.ImageHDU(self.flux)
                flux_err_ImageHDU = fits.ImageHDU(self.flux_err)
                flux_sky_ImageHDU = fits.ImageHDU(self.flux_sky)

            # Create an empty HDU list and populate with ImageHDUs
            self.flux_hdulist = fits.HDUList()
            self.flux_hdulist += [flux_ImageHDU]
            self.flux_hdulist += [flux_err_ImageHDU]
            self.flux_hdulist += [flux_sky_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["flux"]]

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_header(0, "set", "EXTNAME", hdu_names[0])
            self.modify_flux_header(0, "set", "LABEL", "Flux")
            self.modify_flux_header(0, "set", "CRPIX1", 1.00e00)
            self.modify_flux_header(0, "set", "CDELT1", 1)
            self.modify_flux_header(
                0, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_header(0, "set", "CTYPE1", "Pixel")
            self.modify_flux_header(0, "set", "CUNIT1", "Pixel")
            self.modify_flux_header(
                0, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )
            self.modify_flux_header(0, "set", "XPOSURE", self.exptime)
            self.modify_flux_header(0, "set", "GAIN", self.gain)
            self.modify_flux_header(0, "set", "RNOISE", self.readnoise)
            self.modify_flux_header(0, "set", "SEEING", self.seeing)
            self.modify_flux_header(0, "set", "AIRMASS", self.airmass)

            self.modify_flux_header(1, "set", "EXTNAME", hdu_names[1])
            self.modify_flux_header(1, "set", "LABEL", "Flux (Uncertainty)")
            self.modify_flux_header(1, "set", "CRPIX1", 1.00e00)
            self.modify_flux_header(1, "set", "CDELT1", 1)
            self.modify_flux_header(
                1, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_header(1, "set", "CTYPE1", "Pixel")
            self.modify_flux_header(1, "set", "CUNIT1", "Pixel")
            self.modify_flux_header(
                1, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_header(2, "set", "EXTNAME", hdu_names[2])
            self.modify_flux_header(2, "set", "LABEL", "Flux (Sky)")
            self.modify_flux_header(2, "set", "CRPIX1", 1.00e00)
            self.modify_flux_header(2, "set", "CDELT1", 1)
            self.modify_flux_header(
                2, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_header(2, "set", "CTYPE1", "Pixel")
            self.modify_flux_header(2, "set", "CUNIT1", "Pixel")
            self.modify_flux_header(
                2, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("flux ImageHDU cannot be created.")
            self.flux_hdulist = None

    def create_atm_ext_fits(self):
        """
        Create an ImageHDU for the atmospheric extinction corrected flux
        calibrated spectrum at the resampled wavelength.

        """

        try:
            header = None

            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data and header in ImageHDUs
                atm_ext_ImageHDU = fits.ImageHDU(self.atm_ext, header=header)
            else:
                # Put the data in ImageHDUs
                atm_ext_ImageHDU = fits.ImageHDU(self.atm_ext)

            # Create an empty HDU list and populate with ImageHDUs
            self.atm_ext_hdulist = fits.HDUList()
            self.atm_ext_hdulist += [atm_ext_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["atm_ext"]]

            # Note that wave_start is the centre of the starting bin
            self.modify_atm_ext_header("set", "EXTNAME", hdu_names[0])
            self.modify_atm_ext_header(
                "set", "LABEL", "Atmopheric extinction correction factor"
            )
            self.modify_atm_ext_header("set", "CRPIX1", 1.00e00)
            self.modify_atm_ext_header("set", "CDELT1", self.wave_bin)
            self.modify_atm_ext_header("set", "CRVAL1", self.wave_start)
            self.modify_atm_ext_header("set", "CTYPE1", "Wavelength")
            self.modify_atm_ext_header("set", "CUNIT1", " ")
            self.modify_atm_ext_header("set", "BUNIT", " ")

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning("atm_ext ImageHDU cannot be created.")
            self.atm_ext_hdulist = None

    def create_flux_atm_ext_corrected_fits(self):
        """
        Create an ImageHDU for the atmospheric extinction corrected flux
        calibrated spectrum at each native pixel.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data in ImageHDUs
                flux_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_atm_ext_corrected, header=header
                )
                flux_err_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_err_atm_ext_corrected, header=header
                )
                flux_sky_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_sky_atm_ext_corrected, header=header
                )

            else:
                flux_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_atm_ext_corrected
                )
                flux_err_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_err_atm_ext_corrected
                )
                flux_sky_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_sky_atm_ext_corrected
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.flux_atm_ext_corrected_hdulist = fits.HDUList()
            self.flux_atm_ext_corrected_hdulist += [
                flux_atm_ext_corrected_ImageHDU
            ]
            self.flux_atm_ext_corrected_hdulist += [
                flux_err_atm_ext_corrected_ImageHDU
            ]
            self.flux_atm_ext_corrected_hdulist += [
                flux_sky_atm_ext_corrected_ImageHDU
            ]

            hdu_names = self.ext_name[self.hdu_order["flux_atm_ext_corrected"]]

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_atm_ext_corrected_header(
                0, "set", "EXTNAME", hdu_names[0]
            )
            self.modify_flux_atm_ext_corrected_header(
                0, "set", "LABEL", "Flux atmospheric extinction corrected"
            )
            self.modify_flux_atm_ext_corrected_header(
                0, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_atm_ext_corrected_header(0, "set", "CDELT1", 1)
            self.modify_flux_atm_ext_corrected_header(
                0, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_atm_ext_corrected_header(
                0, "set", "CTYPE1", "Pixel"
            )
            self.modify_flux_atm_ext_corrected_header(
                0, "set", "CUNIT1", "Pixel"
            )
            self.modify_flux_atm_ext_corrected_header(
                0, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_atm_ext_corrected_header(
                1,
                "set",
                "EXTNAME",
                hdu_names[1],
            )
            self.modify_flux_atm_ext_corrected_header(
                1,
                "set",
                "LABEL",
                "Flux atmospheric extinction corrected (Uncertainty)",
            )
            self.modify_flux_atm_ext_corrected_header(
                1, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_atm_ext_corrected_header(1, "set", "CDELT1", 1)
            self.modify_flux_atm_ext_corrected_header(
                1, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_atm_ext_corrected_header(
                1, "set", "CTYPE1", "Pixel"
            )
            self.modify_flux_atm_ext_corrected_header(
                1, "set", "CUNIT1", "Pixel"
            )
            self.modify_flux_atm_ext_corrected_header(
                1, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_atm_ext_corrected_header(
                2,
                "set",
                "EXTNAME",
                hdu_names[2],
            )
            self.modify_flux_atm_ext_corrected_header(
                2,
                "set",
                "LABEL",
                "Flux atmospheric extinction corrected (Sky)",
            )
            self.modify_flux_atm_ext_corrected_header(
                2, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_atm_ext_corrected_header(2, "set", "CDELT1", 1)
            self.modify_flux_atm_ext_corrected_header(
                2, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_atm_ext_corrected_header(
                2, "set", "CTYPE1", "Pixel"
            )
            self.modify_flux_atm_ext_corrected_header(
                2, "set", "CUNIT1", "Pixel"
            )
            self.modify_flux_atm_ext_corrected_header(
                2, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning(
                "flux_atm_ext_corrected ImageHDU cannot be created."
            )
            self.flux_atm_ext_corrected_hdulist = None

    def create_telluric_profile_fits(self):
        """
        Create an ImageHDU for the Telluric absorption profile at resampled
        wavelength.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data and header in ImageHDUs
                telluric_profile_ImageHDU = fits.ImageHDU(
                    self.telluric_profile, header=header
                )
            else:
                # Put the data in ImageHDUs
                telluric_profile_ImageHDU = fits.ImageHDU(
                    self.telluric_profile
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.telluric_profile_hdulist = fits.HDUList()
            self.telluric_profile_hdulist += [telluric_profile_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["telluric_profile"]]

            # Note that wave_start is the centre of the starting bin
            self.modify_telluric_profile_header("set", "EXTNAME", hdu_names[0])
            self.modify_telluric_profile_header(
                "set", "LABEL", "Telluric absorption profile"
            )
            self.modify_telluric_profile_header("set", "CRPIX1", 1.00e00)
            self.modify_telluric_profile_header("set", "CDELT1", self.wave_bin)
            self.modify_telluric_profile_header(
                "set", "CRVAL1", self.wave_start
            )
            self.modify_telluric_profile_header("set", "CTYPE1", "Wavelength")
            self.modify_telluric_profile_header("set", "CUNIT1", "")
            self.modify_telluric_profile_header("set", "BUNIT", "")

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning("telluric_profile ImageHDU cannot be created.")
            self.telluric_profile_hdulist = None

    def create_flux_telluric_corrected_fits(self):
        """
        Create an ImageHDU for the telluric corrected flux calibrated
        spectrum at each native pixel.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data in ImageHDUs
                flux_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_telluric_corrected, header=header
                )
                flux_err_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_err_telluric_corrected, header=header
                )
                flux_sky_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_sky_telluric_corrected, header=header
                )

            else:
                flux_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_telluric_corrected
                )
                flux_err_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_err_telluric_corrected
                )
                flux_sky_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_sky_telluric_corrected
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.flux_telluric_corrected_hdulist = fits.HDUList()
            self.flux_telluric_corrected_hdulist += [
                flux_telluric_corrected_ImageHDU
            ]
            self.flux_telluric_corrected_hdulist += [
                flux_err_telluric_corrected_ImageHDU
            ]
            self.flux_telluric_corrected_hdulist += [
                flux_sky_telluric_corrected_ImageHDU
            ]

            hdu_names = self.ext_name[
                self.hdu_order["flux_telluric_corrected"]
            ]

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_telluric_corrected_header(
                0, "set", "EXTNAME", hdu_names[0]
            )
            self.modify_flux_telluric_corrected_header(
                0, "set", "LABEL", "Flux telluric corrected"
            )
            self.modify_flux_telluric_corrected_header(
                0, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_telluric_corrected_header(0, "set", "CDELT1", 1)
            self.modify_flux_telluric_corrected_header(
                0, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_telluric_corrected_header(
                0, "set", "CTYPE1", "Pixel"
            )
            self.modify_flux_telluric_corrected_header(
                0, "set", "CUNIT1", "Pixel"
            )
            self.modify_flux_telluric_corrected_header(
                0, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_telluric_corrected_header(
                1, "set", "EXTNAME", hdu_names[1]
            )
            self.modify_flux_telluric_corrected_header(
                1, "set", "LABEL", "Flux telluric correct (Uncertainty)"
            )
            self.modify_flux_telluric_corrected_header(
                1, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_telluric_corrected_header(1, "set", "CDELT1", 1)
            self.modify_flux_telluric_corrected_header(
                1, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_telluric_corrected_header(
                1, "set", "CTYPE1", "Pixel"
            )
            self.modify_flux_telluric_corrected_header(
                1, "set", "CUNIT1", "Pixel"
            )
            self.modify_flux_telluric_corrected_header(
                1, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_telluric_corrected_header(
                2, "set", "EXTNAME", hdu_names[2]
            )
            self.modify_flux_telluric_corrected_header(
                2, "set", "LABEL", "Flux telluric corrected (Sky)"
            )
            self.modify_flux_telluric_corrected_header(
                2, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_telluric_corrected_header(2, "set", "CDELT1", 1)
            self.modify_flux_telluric_corrected_header(
                2, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_telluric_corrected_header(
                2, "set", "CTYPE1", "Pixel"
            )
            self.modify_flux_telluric_corrected_header(
                2, "set", "CUNIT1", "Pixel"
            )
            self.modify_flux_telluric_corrected_header(
                2, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning(
                "flux_telluric_corrected ImageHDU cannot be created."
            )
            self.flux_telluric_corrected_hdulist = None

    def create_flux_atm_ext_telluric_corrected_fits(self):
        """
        Create an ImageHDU for the atmospheric extinction corrected flux
        calibrated spectrum at each native pixel.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data in ImageHDUs
                flux_atm_ext_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_atm_ext_telluric_corrected, header=header
                )
                flux_err_atm_ext_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_err_atm_ext_telluric_corrected, header=header
                )
                flux_sky_atm_ext_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_sky_atm_ext_telluric_corrected, header=header
                )

            else:
                flux_atm_ext_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_atm_ext_telluric_corrected
                )
                flux_err_atm_ext_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_err_atm_ext_telluric_corrected
                )
                flux_sky_atm_ext_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_sky_atm_ext_telluric_corrected
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.flux_atm_ext_telluric_corrected_hdulist = fits.HDUList()
            self.flux_atm_ext_telluric_corrected_hdulist += [
                flux_atm_ext_telluric_corrected_ImageHDU
            ]
            self.flux_atm_ext_telluric_corrected_hdulist += [
                flux_err_atm_ext_telluric_corrected_ImageHDU
            ]
            self.flux_atm_ext_telluric_corrected_hdulist += [
                flux_sky_atm_ext_telluric_corrected_ImageHDU
            ]

            hdu_names = self.ext_name[
                self.hdu_order["flux_atm_ext_telluric_corrected"]
            ]

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_atm_ext_telluric_corrected_header(
                0,
                "set",
                "EXTNAME",
                hdu_names[0],
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0,
                "set",
                "LABEL",
                "Flux atmospheric extinction telluric corrected",
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "CDELT1", 1
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "CTYPE1", "Pixel"
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "CUNIT1", "Pixel"
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "XPOSURE", self.exptime
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "GAIN", self.gain
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "RNOISE", self.readnoise
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "SEEING", self.seeing
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                0, "set", "AIRMASS", self.airmass
            )

            self.modify_flux_atm_ext_telluric_corrected_header(
                1,
                "set",
                "EXTNAME",
                hdu_names[1],
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                1,
                "set",
                "LABEL",
                "Flux atmospheric extinction telluric corrected (Uncertainty)",
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                1, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                1, "set", "CDELT1", 1
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                1, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                1, "set", "CTYPE1", "Pixel"
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                1, "set", "CUNIT1", "Pixel"
            )
            self.modify_flux_atm_ext_corrected_header(
                1, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_atm_ext_corrected_header(
                2,
                "set",
                "EXTNAME",
                hdu_names[2],
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                2,
                "set",
                "LABEL",
                "Flux atmospheric extinction telluric corrected (Sky)",
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                2, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                2, "set", "CDELT1", 1
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                2, "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                2, "set", "CTYPE1", "Pixel"
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                2, "set", "CUNIT1", "Pixel"
            )
            self.modify_flux_atm_ext_telluric_corrected_header(
                2, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning(
                "flux_atm_ext_telluric_corrected ImageHDU cannot be created."
            )
            self.flux_atm_ext_telluric_corrected_hdulist = None

    def create_sensitivity_resampled_fits(self):
        """
        Create an ImageHDU for the sensitivity at the resampled wavelength.

        """

        try:
            # Put the data in ImageHDUs
            sensitivity_resampled_ImageHDU = fits.ImageHDU(
                self.sensitivity_resampled
            )

            # Create an empty HDU list and populate with ImageHDUs
            self.sensitivity_resampled_hdulist = fits.HDUList()
            self.sensitivity_resampled_hdulist += [
                sensitivity_resampled_ImageHDU
            ]

            hdu_names = self.ext_name[self.hdu_order["sensitivity_resampled"]]

            self.modify_sensitivity_resampled_header(
                "set", "EXTNAME", hdu_names[0]
            )
            self.modify_sensitivity_resampled_header(
                "set", "LABEL", "Sensitivity"
            )
            self.modify_sensitivity_resampled_header("set", "CRPIX1", 1.00e00)
            self.modify_sensitivity_resampled_header("set", "CDELT1", 1)
            self.modify_sensitivity_resampled_header(
                "set", "CRVAL1", self.effective_pixel[0]
            )
            self.modify_sensitivity_resampled_header("set", "CTYPE1", "Pixel")
            self.modify_sensitivity_resampled_header("set", "CUNIT1", "Pixel")
            self.modify_sensitivity_resampled_header(
                "set", "BUNIT", "erg/(s*cm**2*Angstrom)/Count"
            )

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("sensitivity ImageHDU cannot be created.")
            self.sensitivity_resampled_hdulist = None

    def create_flux_resampled_fits(self):
        """
        Create an ImageHDU for the flux calibrated spectrum at the resampled
        wavelength.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data in ImageHDUs
                flux_resampled_ImageHDU = fits.ImageHDU(
                    self.flux_resampled, header=header
                )
                flux_err_resampled_ImageHDU = fits.ImageHDU(
                    self.flux_err_resampled, header=header
                )
                flux_sky_resampled_ImageHDU = fits.ImageHDU(
                    self.flux_sky_resampled, header=header
                )
            else:
                # Put the data in ImageHDUs
                flux_resampled_ImageHDU = fits.ImageHDU(self.flux_resampled)
                flux_err_resampled_ImageHDU = fits.ImageHDU(
                    self.flux_err_resampled
                )
                flux_sky_resampled_ImageHDU = fits.ImageHDU(
                    self.flux_sky_resampled
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.flux_resampled_hdulist = fits.HDUList()
            self.flux_resampled_hdulist += [flux_resampled_ImageHDU]
            self.flux_resampled_hdulist += [flux_err_resampled_ImageHDU]
            self.flux_resampled_hdulist += [flux_sky_resampled_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["flux_resampled"]]

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_resampled_header(
                0, "set", "EXTNAME", hdu_names[0]
            )
            self.modify_flux_resampled_header(
                0, "set", "LABEL", "Flux resampled"
            )
            self.modify_flux_resampled_header(0, "set", "CRPIX1", 1.00e00)
            self.modify_flux_resampled_header(
                0, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_header(
                0, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_header(0, "set", "CTYPE1", "Wavelength")
            self.modify_flux_resampled_header(0, "set", "CUNIT1", "Angstroms")
            self.modify_flux_resampled_header(
                0, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )
            self.modify_flux_resampled_header(
                0, "set", "XPOSURE", self.exptime
            )
            self.modify_flux_resampled_header(0, "set", "GAIN", self.gain)
            self.modify_flux_resampled_header(
                0, "set", "RNOISE", self.readnoise
            )
            self.modify_flux_resampled_header(0, "set", "SEEING", self.seeing)
            self.modify_flux_resampled_header(
                0, "set", "AIRMASS", self.airmass
            )

            self.modify_flux_resampled_header(
                1, "set", "EXTNAME", hdu_names[1]
            )
            self.modify_flux_resampled_header(
                1, "set", "LABEL", "Flux resampled (Uncertainty)"
            )
            self.modify_flux_resampled_header(1, "set", "CRPIX1", 1.00e00)
            self.modify_flux_resampled_header(
                1, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_header(
                1, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_header(1, "set", "CTYPE1", "Wavelength")
            self.modify_flux_resampled_header(1, "set", "CUNIT1", "Angstroms")
            self.modify_flux_resampled_header(
                1, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_resampled_header(
                2, "set", "EXTNAME", hdu_names[2]
            )
            self.modify_flux_resampled_header(
                2, "set", "LABEL", "Flux resampled (Sky)"
            )
            self.modify_flux_resampled_header(2, "set", "CRPIX1", 1.00e00)
            self.modify_flux_resampled_header(
                2, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_header(
                2, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_header(2, "set", "CTYPE1", "Wavelength")
            self.modify_flux_resampled_header(2, "set", "CUNIT1", "Angstroms")
            self.modify_flux_resampled_header(
                2, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

        except Exception as e:
            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("flux_resampled ImageHDU cannot be created.")
            self.flux_resampled_hdulist = None

    def create_atm_ext_resampled_fits(self):
        """
        Create an ImageHDU for the atmospheric extinction corrected flux
        calibrated spectrum at the resampled wavelength.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data and header in ImageHDUs
                atm_ext_resampled_ImageHDU = fits.ImageHDU(
                    self.atm_ext_resampled, header=header
                )
            else:
                # Put the data in ImageHDUs
                atm_ext_resampled_ImageHDU = fits.ImageHDU(
                    self.atm_ext_resampled
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.atm_ext_resampled_hdulist = fits.HDUList()
            self.atm_ext_resampled_hdulist += [atm_ext_resampled_ImageHDU]

            hdu_names = self.ext_name[self.hdu_order["atm_ext_resampled"]]

            # Note that wave_start is the centre of the starting bin
            self.modify_atm_ext_resampled_header(
                "set", "EXTNAME", hdu_names[0]
            )
            self.modify_atm_ext_resampled_header(
                "set", "LABEL", "Atmopheric extinction correction factor"
            )
            self.modify_atm_ext_resampled_header("set", "CRPIX1", 1.00e00)
            self.modify_atm_ext_resampled_header(
                "set", "CDELT1", self.wave_bin
            )
            self.modify_atm_ext_resampled_header(
                "set", "CRVAL1", self.wave_start
            )
            self.modify_atm_ext_resampled_header("set", "CTYPE1", "Wavelength")
            self.modify_atm_ext_resampled_header("set", "CUNIT1", "")
            self.modify_atm_ext_resampled_header("set", "BUNIT", "")

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning(
                "atm_ext_resampled ImageHDU cannot be created."
            )
            self.atm_ext_resampled_hdulist = None

    def create_flux_resampled_atm_ext_corrected_fits(self):
        """
        Create an ImageHDU for the atmospheric extinction corrected flux
        calibrated spectrum at the resampled wavelength.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data and header in ImageHDUs
                flux_resampled_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_resampled_atm_ext_corrected, header=header
                )
                flux_err_resampled_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_err_resampled_atm_ext_corrected, header=header
                )
                flux_sky_resampled_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_sky_resampled_atm_ext_corrected, header=header
                )
            else:
                # Put the data in ImageHDUs
                flux_resampled_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_resampled_atm_ext_corrected
                )
                flux_err_resampled_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_err_resampled_atm_ext_corrected
                )
                flux_sky_resampled_atm_ext_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_sky_resampled_atm_ext_corrected
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.flux_resampled_atm_ext_corrected_hdulist = fits.HDUList()
            self.flux_resampled_atm_ext_corrected_hdulist += [
                flux_resampled_atm_ext_corrected_ImageHDU
            ]
            self.flux_resampled_atm_ext_corrected_hdulist += [
                flux_err_resampled_atm_ext_corrected_ImageHDU
            ]
            self.flux_resampled_atm_ext_corrected_hdulist += [
                flux_sky_resampled_atm_ext_corrected_ImageHDU
            ]

            hdu_names = self.ext_name[
                self.hdu_order["flux_resampled_atm_ext_corrected"]
            ]

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_resampled_atm_ext_corrected_header(
                0,
                "set",
                "EXTNAME",
                hdu_names[0],
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0,
                "set",
                "LABEL",
                "Flux resampled atmospheric extinction corrected",
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "CTYPE1", "Wavelength"
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "CUNIT1", "Angstroms"
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "XPOSURE", self.exptime
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "GAIN", self.gain
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "RNOISE", self.readnoise
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "SEEING", self.seeing
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                0, "set", "AIRMASS", self.airmass
            )

            self.modify_flux_resampled_atm_ext_corrected_header(
                1,
                "set",
                "EXTNAME",
                hdu_names[1],
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                1,
                "set",
                "LABEL",
                "Flux resampled atmospheric extinction corrected (Uncertainty)",
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                1, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                1, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                1, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                1, "set", "CTYPE1", "Wavelength"
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                1, "set", "CUNIT1", "Angstroms"
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                1, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_resampled_atm_ext_corrected_header(
                2,
                "set",
                "EXTNAME",
                hdu_names[2],
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                2,
                "set",
                "LABEL",
                "Flux resampled atmospheric extinction corrected (Sky)",
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                2, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                2, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                2, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                2, "set", "CTYPE1", "Wavelength"
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                2, "set", "CUNIT1", "Angstroms"
            )
            self.modify_flux_resampled_atm_ext_corrected_header(
                2, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning(
                "flux_resampled_atm_ext_corrected ImageHDU cannot be created."
            )
            self.flux_resampled_atm_ext_corrected_hdulist = None

    def create_telluric_profile_resampled_fits(self):
        """
        Create an ImageHDU for the Telluric absorption profile at resampled
        wavelength.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data and header in ImageHDUs
                telluric_profile_resampled_ImageHDU = fits.ImageHDU(
                    self.telluric_profile_resampled, header=header
                )
            else:
                # Put the data in ImageHDUs
                telluric_profile_resampled_ImageHDU = fits.ImageHDU(
                    self.telluric_profile_resampled
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.telluric_profile_resampled_hdulist = fits.HDUList()
            self.telluric_profile_resampled_hdulist += [
                telluric_profile_resampled_ImageHDU
            ]

            hdu_names = self.ext_name[
                self.hdu_order["telluric_profile_resampled"]
            ]

            # Note that wave_start is the centre of the starting bin
            self.modify_telluric_profile_resampled_header(
                "set", "EXTNAME", hdu_names[0]
            )
            self.modify_telluric_profile_resampled_header(
                "set", "LABEL", "Telluric absorption profile"
            )
            self.modify_telluric_profile_resampled_header(
                "set", "CRPIX1", 1.00e00
            )
            self.modify_telluric_profile_resampled_header(
                "set", "CDELT1", self.wave_bin
            )
            self.modify_telluric_profile_resampled_header(
                "set", "CRVAL1", self.wave_start
            )
            self.modify_telluric_profile_resampled_header(
                "set", "CTYPE1", "Wavelength"
            )
            self.modify_telluric_profile_resampled_header("set", "CUNIT1", "")
            self.modify_telluric_profile_resampled_header("set", "BUNIT", "")

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning(
                "telluric_profile_resampled ImageHDU cannot be created."
            )
            self.telluric_profile_resampled_hdulist = None

    def create_flux_resampled_telluric_corrected_fits(self):
        """
        Create an ImageHDU for the atmospheric extinction corrected flux
        calibrated spectrum at the resampled wavelength.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data and header in ImageHDUs
                flux_resampled_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_resampled_telluric_corrected, header=header
                )
                flux_err_resampled_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_err_resampled_telluric_corrected, header=header
                )
                flux_sky_resampled_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_sky_resampled_telluric_corrected, header=header
                )
            else:
                # Put the data in ImageHDUs
                flux_resampled_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_resampled_telluric_corrected
                )
                flux_err_resampled_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_err_resampled_telluric_corrected
                )
                flux_sky_resampled_telluric_corrected_ImageHDU = fits.ImageHDU(
                    self.flux_sky_resampled_telluric_corrected
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.flux_resampled_telluric_corrected_hdulist = fits.HDUList()
            self.flux_resampled_telluric_corrected_hdulist += [
                flux_resampled_telluric_corrected_ImageHDU
            ]
            self.flux_resampled_telluric_corrected_hdulist += [
                flux_err_resampled_telluric_corrected_ImageHDU
            ]
            self.flux_resampled_telluric_corrected_hdulist += [
                flux_sky_resampled_telluric_corrected_ImageHDU
            ]

            hdu_names = self.ext_name[
                self.hdu_order["flux_resampled_telluric_corrected"]
            ]

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "EXTNAME", hdu_names[0]
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "LABEL", "Flux resampled telluric corrected"
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "CTYPE1", "Wavelength"
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "CUNIT1", "Angstroms"
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "XPOSURE", self.exptime
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "GAIN", self.gain
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "RNOISE", self.readnoise
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "SEEING", self.seeing
            )
            self.modify_flux_resampled_telluric_corrected_header(
                0, "set", "AIRMASS", self.airmass
            )

            self.modify_flux_resampled_telluric_corrected_header(
                1,
                "set",
                "EXTNAME",
                hdu_names[1],
            )
            self.modify_flux_resampled_telluric_corrected_header(
                1,
                "set",
                "LABEL",
                "Flux resampled telluric corrected (Uncertainty)",
            )
            self.modify_flux_resampled_telluric_corrected_header(
                1, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_resampled_telluric_corrected_header(
                1, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_telluric_corrected_header(
                1, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_telluric_corrected_header(
                1, "set", "CTYPE1", "Wavelength"
            )
            self.modify_flux_resampled_telluric_corrected_header(
                1, "set", "CUNIT1", "Angstroms"
            )
            self.modify_flux_resampled_telluric_corrected_header(
                1, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_resampled_telluric_corrected_header(
                2, "set", "EXTNAME", hdu_names[2]
            )
            self.modify_flux_resampled_telluric_corrected_header(
                2, "set", "LABEL", "Flux resampled telluric corrected (Sky)"
            )
            self.modify_flux_resampled_telluric_corrected_header(
                2, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_resampled_telluric_corrected_header(
                2, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_telluric_corrected_header(
                2, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_telluric_corrected_header(
                2, "set", "CTYPE1", "Wavelength"
            )
            self.modify_flux_resampled_telluric_corrected_header(
                2, "set", "CUNIT1", "Angstroms"
            )
            self.modify_flux_resampled_telluric_corrected_header(
                2, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning(
                "flux_resampled_telluric_corrected ImageHDU cannot be created."
            )
            self.flux_resampled_telluric_corrected_hdulist = None

    def create_flux_resampled_atm_ext_telluric_corrected_fits(self):
        """
        Create an ImageHDU for the atmospheric extinction corrected flux
        calibrated spectrum at the resampled wavelength.

        """

        try:
            header = None

            # Use the header of the standard
            if self.spectrum_header is not None:
                header = self.spectrum_header

            if header is not None:
                # Put the data and header in ImageHDUs
                flux_resampled_atm_ext_telluric_corrected_ImageHDU = (
                    fits.ImageHDU(
                        self.flux_resampled_atm_ext_telluric_corrected,
                        header=header,
                    )
                )
                flux_err_resampled_atm_ext_telluric_corrected_ImageHDU = (
                    fits.ImageHDU(
                        self.flux_err_resampled_atm_ext_telluric_corrected,
                        header=header,
                    )
                )
                flux_sky_resampled_atm_ext_telluric_corrected_ImageHDU = (
                    fits.ImageHDU(
                        self.flux_sky_resampled_atm_ext_telluric_corrected,
                        header=header,
                    )
                )
            else:
                # Put the data in ImageHDUs
                flux_resampled_atm_ext_telluric_corrected_ImageHDU = (
                    fits.ImageHDU(
                        self.flux_resampled_atm_ext_telluric_corrected
                    )
                )
                flux_err_resampled_atm_ext_telluric_corrected_ImageHDU = (
                    fits.ImageHDU(
                        self.flux_err_resampled_atm_ext_telluric_corrected
                    )
                )
                flux_sky_resampled_atm_ext_telluric_corrected_ImageHDU = (
                    fits.ImageHDU(
                        self.flux_sky_resampled_atm_ext_telluric_corrected
                    )
                )

            # Create an empty HDU list and populate with ImageHDUs
            self.flux_resampled_atm_ext_telluric_corrected_hdulist = (
                fits.HDUList()
            )
            self.flux_resampled_atm_ext_telluric_corrected_hdulist += [
                flux_resampled_atm_ext_telluric_corrected_ImageHDU
            ]
            self.flux_resampled_atm_ext_telluric_corrected_hdulist += [
                flux_err_resampled_atm_ext_telluric_corrected_ImageHDU
            ]
            self.flux_resampled_atm_ext_telluric_corrected_hdulist += [
                flux_sky_resampled_atm_ext_telluric_corrected_ImageHDU
            ]

            hdu_names = self.ext_name[
                self.hdu_order["flux_resampled_atm_ext_telluric_corrected"]
            ]

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0,
                "set",
                "EXTNAME",
                hdu_names[0],
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0,
                "set",
                "LABEL",
                "Flux resampled atmospheric extinction telluric corrected",
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "CTYPE1", "Wavelength"
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "CUNIT1", "Angstroms"
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "XPOSURE", self.exptime
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "GAIN", self.gain
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "RNOISE", self.readnoise
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "SEEING", self.seeing
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                0, "set", "AIRMASS", self.airmass
            )

            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                1,
                "set",
                "EXTNAME",
                hdu_names[1],
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                1,
                "set",
                "LABEL",
                (
                    "Flux resampled atmospheric extinction telluric corrected"
                    " (Uncertainty)"
                ),
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                1, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                1, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                1, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                1, "set", "CTYPE1", "Wavelength"
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                1, "set", "CUNIT1", "Angstroms"
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                1, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                2,
                "set",
                "EXTNAME",
                hdu_names[2],
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                2,
                "set",
                "LABEL",
                (
                    "Flux resampled atmospheric extinction telluric corrected"
                    " (Sky)"
                ),
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                2, "set", "CRPIX1", 1.00e00
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                2, "set", "CDELT1", self.wave_bin
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                2, "set", "CRVAL1", self.wave_start
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                2, "set", "CTYPE1", "Wavelength"
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                2, "set", "CUNIT1", "Angstroms"
            )
            self.modify_flux_resampled_atm_ext_telluric_corrected_header(
                2, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

        except Exception as e:
            self.logger.warning(str(e))

            # Set it to None if the above failed
            self.logger.warning(
                "flux_resampled_atm_ext_telluric_corrected ImageHDU cannot "
                "be created."
            )
            self.flux_resampled_atm_ext_telluric_corrected_hdulist = None

    def remove_trace_fits(self):
        """
        Remove the trace FITS HDUList.

        """

        self.trace_hdulist = None

    def remove_count_fits(self):
        """
        Remove the count FITS HDUList.

        """

        self.count_hdulist = None

    def remove_count_resampled_fits(self):
        """
        Remove the count_resampled FITS HDUList.

        """

        self.count_resampled_hdulist = None

    def remove_arc_spec_fits(self):
        """
        Remove the arc_spec FITS HDUList.

        """

        self.arc_spec_hdulist = None

    def remove_wavecal_fits(self):
        """
        Remove the wavecal FITS HDUList.

        """

        self.wavecal_hdulist = None

    def remove_wavelength_fits(self):
        """
        Remove the wavelength FITS HDUList.

        """

        self.wavelength_hdulist = None

    def remove_weight_map_fits(self):
        """
        Remove the weight_map FITS HDUList.

        """

        self.weight_map_hdulist = None

    def remove_flux_fits(self):
        """
        Remove the flux FITS HDUList.

        """

        self.flux_hdulist = None

    def remove_atm_ext_fits(self):
        """
        Remove the atm_ext FITS HDUList.

        """

        self.atm_ext_hdulist = None

    def remove_flux_atm_ext_corrected_fits(self):
        """
        Remove the flux_resampled_atm_ext_corrected FITS HDUList.

        """

        self.flux_atm_ext_corrected_hdulist = None

    def remove_telluric_profile_fits(self):
        """
        Remove the telluric_profile FITS HDUList.

        """

        self.telluric_profile_hdulist = None

    def remove_flux_telluric_corrected_fits(self):
        """
        Remove the flux_resampled_telluric_corrected FITS HDUList.

        """

        self.flux_telluric_corrected_hdulist = None

    def remove_flux_atm_ext_telluric_corrected_fits(self):
        """
        Remove the flux_resampled_atm_ext_telluric_corrected FITS HDUList.

        """

        self.flux_atm_ext_telluric_corrected_hdulist = None

    def remove_wavelength_resampled_fits(self):
        """
        Remove the resampled wavelength FITS HDUList.

        """

        self.wavelength_resampled_hdulist = None

    def remove_sensitivity_resampled_fits(self):
        """
        Remove the sensitivity_resampled FITS HDUList.

        """

        self.sensitivity_resampled_hdulist = None

    def remove_flux_resampled_fits(self):
        """
        Remove the flux_resampled FITS HDUList.

        """

        self.flux_resampled_hdulist = None

    def remove_atm_ext_resampled_fits(self):
        """
        Remove the atm_ext_resampled FITS HDUList.

        """

        self.atm_ext_resampled_hdulist = None

    def remove_flux_resampled_atm_ext_corrected_fits(self):
        """
        Remove the flux_resampled_atm_ext_corrected FITS HDUList.

        """

        self.flux_resampled_atm_ext_corrected_hdulist = None

    def remove_telluric_profile_resampled_fits(self):
        """
        Remove the telluric_profile_resampled FITS HDUList.

        """

        self.telluric_profile_resampled_hdulist = None

    def remove_flux_resampled_telluric_corrected_fits(self):
        """
        Remove the flux_resampled_telluric_corrected FITS HDUList.

        """

        self.flux_resampled_telluric_corrected_hdulist = None

    def remove_flux_resampled_atm_ext_telluric_corrected_fits(self):
        """
        Remove the flux_resampled_atm_ext_telluric_corrected FITS HDUList.

        """

        self.flux_resampled_atm_ext_telluric_corrected_hdulist = None

    def create_fits(
        self,
        output: str = "*",
        recreate: bool = True,
        empty_primary_hdu: bool = True,
        return_hdu_list: bool = False,
    ):
        """
        Create a HDU list, with a choice of any combination of the
        data, see below the 'output' parameters for details.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+":

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 3 HDUs
                    Count, uncertainty, and sky (pixel)
                weight_map: 1 HDU
                    Weight (pixel)
                arc_spec: 1 HDU
                    1D arc spectrum
                arc_lines: 2 HDUs
                    arc line position (pixel), and arc
                    line effective position (pixel)
                wavecal_coefficients: 1 HDU
                    Polynomial coefficients for wavelength calibration
                wavelength: 1 HDU
                    Wavelength of each pixel
                wavelength_resampled: 1 HDU
                    Wavelength of each resampled position
                count_resampled: 3 HDUs
                    Resampled Count, uncertainty, and sky (wavelength)
                sensitivity: 1 HDU
                    Sensitivity (pixel)
                flux: 3 HDUs
                    Flux, uncertainty, and sky (pixel)
                atm_ext: 1 HDU
                    Atmospheric extinction correction factor
                flux_atm_ext_corrected: 3 HDUs
                    Atmospheric extinction corrected flux, uncertainty, and
                    sky (pixel)
                telluric_profile: 1 HDU
                    Telluric absorption profile
                flux_telluric_corrected: 3 HDUs
                    Telluric corrected flux, uncertainty, and
                    sky (pixel)
                flux_atm_ext_telluric_corrected: 3 HDUs
                    Atmospheric extinction and telluric corrected flux,
                    uncertainty, and sky (pixel)
                sensitivity_resampled: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 4 HDUs
                    Flux, uncertainty, and sky (wavelength)
                atm_ext_resampled: 1 HDU
                    Atmospheric extinction correction factor
                flux_resampled_atm_ext_corrected: 3 HDUs
                    Atmospheric extinction corrected flux, uncertainty, and
                    sky (wavelength)
                telluric_profile_resampled: 1 HDU
                    Telluric absorption profile
                flux_resampled_telluic_corrected: 3 HDUs
                    Telluric corrected flux, uncertainty, and
                    sky (wavelength)
                flux_resampled_atm_ext_telluric_corrected: 3 HDUs
                    Atmospheric extinction and telluric  corrected flux,
                    uncertainty, and sky (wavelength)

        recreate: boolean (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: boolean (Default: True)
            Set to True to leave the Primary HDU blank.
        return_hdu_list: boolean (Default: False)
            Set to True to return the HDU List.

        """

        if output == "*":
            output = (
                "trace+count+weight_map+arc_spec+arc_lines+"
                "wavecal_coefficients+wavelength+wavelength_resampled+"
                "count_resampled+sensitivity+flux+atm_ext+"
                "flux_atm_ext_corrected+telluric_profile+"
                "flux_telluric_corrected+flux_atm_ext_telluric_corrected+"
                "sensitivity_resampled+flux_resampled+atm_ext_resampled+"
                "flux_resampled_atm_ext_corrected+"
                "telluric_profile_resampled+"
                "flux_resampled_telluic_corrected+"
                "flux_resampled_atm_ext_telluric_corrected"
            )

        self.logger.info(f"{output} is read.")

        output_split = output.split("+")
        self.logger.info(f"HDUs of {output_split} are to be created.")

        # If to recreate the FITS, set all contents to False
        if recreate:
            for k, v in self.hdu_content.items():
                self.hdu_content[k] = False

            self.logger.info("HDUList is cleared.")

        # If the requested list of HDUs is already good to go
        if set([k for k, v in self.hdu_content.items() if v]) == set(
            output_split
        ):
            self.logger.info("HDUList is ready to go.")

            # If there is an empty primary HDU, but requested without
            if self.empty_primary_hdu & (not empty_primary_hdu):
                self.hdu_output.pop(0)
                self.empty_primary_hdu = False

            # If there is not an empty primary HDU, but requested one
            elif (not self.empty_primary_hdu) & empty_primary_hdu:
                self.hdu_output.insert(0, fits.PrimaryHDU())
                self.empty_primary_hdu = True

            # Otherwise, the self.hdu_output does not need to be modified
            else:
                pass

        # If the requested list is different or empty, (re)create the list
        else:
            self.hdu_output = None

            self.logger.info("Populating the HDUList now.")

            # Empty list for appending HDU lists
            hdu_output = fits.HDUList()

            # If leaving the primary HDU empty
            if empty_primary_hdu:
                hdu_output.append(fits.PrimaryHDU())

            # Join the list(s)
            if "trace" in output_split:
                if not self.hdu_content["trace"]:
                    self.logger.info("Creating trace HDU now.")
                    self.create_trace_fits()
                    self.logger.info("Created trace HDU.")

                if (self.trace_hdulist is not None) and (
                    self.trace is not None
                ):
                    hdu_output += self.trace_hdulist
                    self.hdu_content["trace"] = True

            if "count" in output_split:
                if not self.hdu_content["count"]:
                    self.logger.info("Creating count HDU now.")
                    self.create_count_fits()
                    self.logger.info("Created trace HDU.")

                if (self.count_hdulist is not None) and (
                    self.count is not None
                ):
                    hdu_output += self.count_hdulist
                    self.hdu_content["count"] = True

            if "weight_map" in output_split:
                if not self.hdu_content["weight_map"]:
                    self.logger.info("Creating weight_map HDU now.")
                    self.create_weight_map_fits()
                    self.logger.info("Created weight_map HDU.")

                if (self.weight_map_hdulist is not None) and (
                    self.var is not None
                ):
                    hdu_output += self.weight_map_hdulist
                    self.hdu_content["weight_map"] = True

            if "arc_spec" in output_split:
                if not self.hdu_content["arc_spec"]:
                    self.logger.info("Creating arc_spec HDU now.")
                    self.create_arc_spec_fits()
                    self.logger.info("Created arc_spec HDU.")

                if (self.arc_spec_hdulist is not None) and (
                    self.arc_spec is not None
                ):
                    hdu_output += self.arc_spec_hdulist
                    self.hdu_content["arc_spec"] = True

            if "arc_lines" in output_split:
                if not self.hdu_content["arc_lines"]:
                    self.logger.info("Creating arc_lines HDU now.")
                    self.create_arc_lines_fits()
                    self.logger.info("Created HDU.")

                if self.arc_lines_hdulist is not None:
                    hdu_output += self.arc_lines_hdulist
                    self.hdu_content["arc_lines"] = True

            if "wavecal_coefficients" in output_split:
                if not self.hdu_content["wavecal_coefficients"]:
                    self.logger.info("Creat wavecal_coefficients HDU now.")
                    self.create_wavecal_fits()
                    self.logger.info("Creat wavecal_coefficients HDU.")

                if (self.wavecal_hdulist is not None) and (
                    self.fit_coeff is not None
                ):
                    hdu_output += self.wavecal_hdulist
                    self.hdu_content["wavecal_coefficients"] = True

            if "wavelength" in output_split:
                if not self.hdu_content["wavelength"]:
                    self.logger.info("Creating wavelength HDU now.")
                    self.create_wavelength_fits()
                    self.logger.info("Created wavelength HDU.")

                if (self.wavelength_hdulist is not None) and (
                    self.wave is not None
                ):
                    hdu_output += self.wavelength_hdulist
                    self.hdu_content["wavelength"] = True

            if "wavelength_resampled" in output_split:
                if not self.hdu_content["wavelength_resampled"]:
                    self.logger.info("Creating wavelength_resampled HDU now.")
                    self.create_wavelength_resampled_fits()
                    self.logger.info("Created wavelength_resampled HDU.")

                if (self.wavelength_resampled_hdulist is not None) and (
                    self.wave_resampled is not None
                ):
                    hdu_output += self.wavelength_resampled_hdulist
                    self.hdu_content["wavelength_resampled"] = True

            if "count_resampled" in output_split:
                if not self.hdu_content["count_resampled"]:
                    self.logger.info("Creating count_resampled HDU now.")
                    self.create_count_resampled_fits()
                    self.logger.info("Created count_resampled HDU.")

                if (self.count_resampled_hdulist is not None) and (
                    self.count_resampled is not None
                ):
                    hdu_output += self.count_resampled_hdulist
                    self.hdu_content["count_resampled"] = True

            if "sensitivity" in output_split:
                if not self.hdu_content["sensitivity"]:
                    self.logger.info("Creating sensitivity HDU now.")
                    self.create_sensitivity_fits()
                    self.logger.info("Created sensitivity HDU.")

                if (self.sensitivity_hdulist is not None) and (
                    self.sensitivity is not None
                ):
                    hdu_output += self.sensitivity_hdulist
                    self.hdu_content["sensitivity"] = True

            if "flux" in output_split:
                if not self.hdu_content["flux"]:
                    self.logger.info("Creating flux HDU now.")
                    self.create_flux_fits()
                    self.logger.info("Created flux HDU.")

                if (self.flux_hdulist is not None) and (self.flux is not None):
                    hdu_output += self.flux_hdulist
                    self.hdu_content["flux"] = True

            if "atm_ext" in output_split:
                if not self.hdu_content["atm_ext"]:
                    self.logger.info("Creating atm_ext HDU now.")
                    self.create_atm_ext_fits()
                    self.logger.info("Created atm_ext HDU.")

                if (self.atm_ext_hdulist is not None) and (
                    self.atm_ext is not None
                ):
                    hdu_output += self.atm_ext_hdulist
                    self.hdu_content["atm_ext"] = True

            if "flux_atm_ext_corrected" in output_split:
                if not self.hdu_content["flux_atm_ext_corrected"]:
                    self.logger.info(
                        "Creating flux_atm_ext_corrected HDU now."
                    )
                    self.create_flux_atm_ext_corrected_fits()
                    self.logger.info("Created flux_atm_ext_corrected HDU.")

                if (self.flux_atm_ext_corrected_hdulist is not None) and (
                    self.flux_atm_ext_corrected is not None
                ):
                    hdu_output += self.flux_atm_ext_corrected_hdulist
                    self.hdu_content["flux_atm_ext_corrected"] = True

            if "telluric_profile" in output_split:
                if not self.hdu_content["telluric_profile"]:
                    self.logger.info("Creating telluric_profile HDU now.")
                    self.create_telluric_profile_fits()
                    self.logger.info("Created telluric_profile HDU.")

                if (self.telluric_profile_hdulist is not None) and (
                    self.telluric_profile is not None
                ):
                    hdu_output += self.telluric_profile_hdulist
                    self.hdu_content["telluric_profile"] = True

            if "flux_telluric_corrected" in output_split:
                if not self.hdu_content["flux_telluric_corrected"]:
                    self.logger.info(
                        "Creating flux_telluric_corrected HDU now."
                    )
                    self.create_flux_telluric_corrected_fits()
                    self.logger.info("Created flux_telluric_corrected HDU.")

                if (self.flux_telluric_corrected_hdulist is not None) and (
                    self.flux_telluric_corrected is not None
                ):
                    hdu_output += self.flux_telluric_corrected_hdulist
                    self.hdu_content["flux_telluric_corrected"] = True

            if "flux_atm_ext_telluric_corrected" in output_split:
                if not self.hdu_content["flux_atm_ext_telluric_corrected"]:
                    self.logger.info(
                        "Creating flux_atm_ext_telluric_corrected HDU now."
                    )
                    self.create_flux_atm_ext_telluric_corrected_fits()
                    self.logger.info(
                        "Created flux_atm_ext_telluric_corrected HDU."
                    )

                if (
                    self.flux_atm_ext_telluric_corrected_hdulist is not None
                ) and (self.flux_atm_ext_telluric_corrected is not None):
                    hdu_output += self.flux_atm_ext_telluric_corrected_hdulist
                    self.hdu_content["flux_atm_ext_telluric_corrected"] = True

            if "sensitivity_resampled" in output_split:
                if not self.hdu_content["sensitivity_resampled"]:
                    self.logger.info("Creating sensitivity_resampled HDU now.")
                    self.create_sensitivity_resampled_fits()
                    self.logger.info("Created sensitivity_resampled HDU.")

                if (self.sensitivity_resampled_hdulist is not None) and (
                    self.sensitivity_resampled is not None
                ):
                    hdu_output += self.sensitivity_resampled_hdulist
                    self.hdu_content["sensitivity_resampled"] = True

            if "flux_resampled" in output_split:
                if not self.hdu_content["flux_resampled"]:
                    self.logger.info("Creating flux_resampled HDU now.")
                    self.create_flux_resampled_fits()
                    self.logger.info("Created flux_resampled HDU.")

                if (self.flux_resampled_hdulist is not None) and (
                    self.flux_resampled is not None
                ):
                    hdu_output += self.flux_resampled_hdulist
                    self.hdu_content["flux_resampled"] = True

            if "atm_ext_resampled" in output_split:
                if not self.hdu_content["atm_ext_resampled"]:
                    self.logger.info("Creating atm_ext_resampled HDU now.")
                    self.create_atm_ext_resampled_fits()
                    self.logger.info("Created atm_ext_resampled HDU.")

                if (self.atm_ext_resampled_hdulist is not None) and (
                    self.atm_ext_resampled is not None
                ):
                    hdu_output += self.atm_ext_resampled_hdulist
                    self.hdu_content["atm_ext_resampled"] = True

            if "flux_resampled_atm_ext_corrected" in output_split:
                if not self.hdu_content["flux_resampled_atm_ext_corrected"]:
                    self.logger.info(
                        "Creating flux_resampled_atm_ext_corrected HDU now."
                    )
                    self.create_flux_resampled_atm_ext_corrected_fits()
                    self.logger.info(
                        "Created flux_resampled_atm_ext_corrected HDU."
                    )

                if (
                    self.flux_resampled_atm_ext_corrected_hdulist is not None
                ) and (self.flux_resampled_atm_ext_corrected is not None):
                    hdu_output += self.flux_resampled_atm_ext_corrected_hdulist
                    self.hdu_content["flux_resampled_atm_ext_corrected"] = True

            if "telluric_profile_resampled" in output_split:
                if not self.hdu_content["telluric_profile_resampled"]:
                    self.logger.info(
                        "Creating telluric_profile_resampled HDU now."
                    )
                    self.create_telluric_profile_resampled_fits()
                    self.logger.info("Created telluric_profile_resampled HDU.")

                if (self.telluric_profile_resampled_hdulist is not None) and (
                    self.telluric_profile_resampled is not None
                ):
                    hdu_output += self.telluric_profile_resampled_hdulist
                    self.hdu_content["telluric_profile_resampled"] = True

            if "flux_resampled_telluric_corrected" in output_split:
                if not self.hdu_content["flux_resampled_telluric_corrected"]:
                    self.logger.info(
                        "Creating flux_resampled_telluric_corrected HDU now."
                    )
                    self.create_flux_resampled_telluric_corrected_fits()
                    self.logger.info(
                        "Created flux_resampled_telluric_corrected HDU."
                    )

                if (
                    self.flux_resampled_telluric_corrected_hdulist is not None
                ) and (self.flux_resampled_telluric_corrected is not None):
                    hdu_output += (
                        self.flux_resampled_telluric_corrected_hdulist
                    )
                    self.hdu_content[
                        "flux_resampled_telluric_corrected"
                    ] = True

            if "flux_resampled_atm_ext_telluric_corrected" in output_split:
                if not self.hdu_content[
                    "flux_resampled_atm_ext_telluric_corrected"
                ]:
                    self.logger.info(
                        "Creating flux_resampled_atm_ext_telluric_corrected HDU"
                        " now."
                    )
                    self.create_flux_resampled_atm_ext_telluric_corrected_fits()
                    self.logger.info(
                        "Created flux_resampled_atm_ext_telluric_corrected HDU."
                    )

                if (
                    self.flux_resampled_atm_ext_telluric_corrected_hdulist
                    is not None
                ) and (
                    self.flux_resampled_atm_ext_telluric_corrected is not None
                ):
                    hdu_output += (
                        self.flux_resampled_atm_ext_telluric_corrected_hdulist
                    )
                    self.hdu_content[
                        "flux_resampled_atm_ext_telluric_corrected"
                    ] = True

            # If the primary HDU is not chosen to be empty
            if empty_primary_hdu:
                hdu_output.update_extend()
                self.empty_primary_hdu = True
            else:
                # Convert the first HDU to PrimaryHDU
                hdu_output[0] = fits.PrimaryHDU(
                    hdu_output[0].data, hdu_output[0].header
                )
                hdu_output.update_extend()
                self.empty_primary_hdu = False

            self.hdu_output = hdu_output

        if return_hdu_list:
            return self.hdu_output

    def save_fits(
        self,
        output: str,
        filename: str,
        overwrite: bool = False,
        recreate: bool = True,
        empty_primary_hdu: bool = True,
        create_folder: bool = False,
    ):
        """
        Save the reduced data to disk, with a choice of any combination of the
        data, see below the 'output' parameters for details.

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
                arc_spec: 1 HDU
                    1D arc spectrum
                arc_lines: 2 HDUs
                    Arc line position (pixel), and arc line effective
                    position (pixel)
                wavecal: 1 HDU
                    Polynomial coefficients for wavelength calibration
                wavelength: 1 HDU
                    Wavelength of each pixel
                wavelength_resampled: 1 HDU
                    Wavelength of each resampled position
                count_resampled: 3 HDUs
                    Resampled Count, uncertainty, and sky (wavelength)
                sensitivity: 1 HDU
                    Sensitivity (pixel)
                flux: 3 HDUs
                    Flux, uncertainty, and sky (pixel)
                atm_ext: 1 HDU
                    Atmospheric extinction correction factor
                flux_atm_ext_corrected: 3 HDUs
                    Atmospheric extinction corrected flux, uncertainty, and
                    sky (pixel)
                telluric_profile: 1 HDU
                    Telluric absorption profile
                flux_telluric_corrected: 3 HDUs
                    Telluric corrected flux, uncertainty, and
                    sky (pixel)
                flux_atm_ext_telluric_corrected: 3 HDUs
                    Atmospheric extinction and telluric corrected flux,
                    uncertainty, and sky (pixel)
                sensitivity_resampled: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 4 HDUs
                    Flux, uncertainty, and sky (wavelength)
                atm_ext_resampled: 1 HDU
                    Atmospheric extinction correction factor
                flux_resampled_atm_ext_corrected: 3 HDUs
                    Atmospheric extinction corrected flux, uncertainty, and
                    sky (wavelength)
                telluric_profile_resampled: 1 HDU
                    Telluric absorption profile
                flux_resampled_telluic_corrected: 3 HDUs
                    Telluric corrected flux, uncertainty, and
                    sky (wavelength)
                flux_resampled_atm_ext_telluric_corrected: 3 HDUs
                    Atmospheric extinction and telluric  corrected flux,
                    uncertainty, and sky (wavelength)

        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        overwrite: boolean
            Default is False.
        recreate: boolean (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: boolean (Default: True)
            Set to True to leave the Primary HDU blank (Default: True)
        create_folder: boolean (Default: False)
            Create folder if not exist. Use with caution.

        """

        self.create_fits(
            output, recreate=recreate, empty_primary_hdu=empty_primary_hdu
        )

        # create the director if not exist
        if create_folder:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

        # Save file to disk
        if os.path.splitext(filename)[-1].lower() in [".fits", ".fit", ".fts"]:
            self.hdu_output.writeto(
                filename, overwrite=overwrite, output_verify="fix+ignore"
            )

        else:
            self.hdu_output.writeto(
                filename + ".fits",
                overwrite=overwrite,
                output_verify="fix+ignore",
            )

    def save_csv(
        self,
        output: str,
        filename: str,
        overwrite: bool = False,
        recreate: bool = True,
        create_folder: bool = False,
    ):
        """
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
                count: 3 HDUs
                    Count, uncertainty, and sky (pixel)
                weight_map: 1 HDU
                    Weight (pixel)
                arc_spec: 1 HDU
                    1D arc spectrum
                arc_lines: 2 HDUs
                    Arc line position (pixel), and arc line effective
                    position (pixel)
                wavecal: 1 HDU
                    Polynomial coefficients for wavelength calibration
                wavelength: 1 HDU
                    Wavelength of each pixel
                wavelength_resampled: 1 HDU
                    Wavelength of each resampled position
                count_resampled: 3 HDUs
                    Resampled Count, uncertainty, and sky (wavelength)
                sensitivity: 1 HDU
                    Sensitivity (pixel)
                flux: 3 HDUs
                    Flux, uncertainty, and sky (pixel)
                atm_ext: 1 HDU
                    Atmospheric extinction correction factor
                flux_atm_ext_corrected: 3 HDUs
                    Atmospheric extinction corrected flux, uncertainty, and
                    sky (pixel)
                telluric_profile: 1 HDU
                    Telluric absorption profile
                flux_telluric_corrected: 3 HDUs
                    Telluric corrected flux, uncertainty, and
                    sky (pixel)
                flux_atm_ext_telluric_corrected: 3 HDUs
                    Atmospheric extinction and telluric corrected flux,
                    uncertainty, and sky (pixel)
                sensitivity_resampled: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 4 HDUs
                    Flux, uncertainty, and sky (wavelength)
                atm_ext_resampled: 1 HDU
                    Atmospheric extinction correction factor
                flux_resampled_atm_ext_corrected: 3 HDUs
                    Atmospheric extinction corrected flux, uncertainty, and
                    sky (wavelength)
                telluric_profile_resampled: 1 HDU
                    Telluric absorption profile
                flux_resampled_telluic_corrected: 3 HDUs
                    Telluric corrected flux, uncertainty, and
                    sky (wavelength)
                flux_resampled_atm_ext_telluric_corrected: 3 HDUs
                    Atmospheric extinction and telluric  corrected flux,
                    uncertainty, and sky (wavelength)

        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        overwrite: boolean
            Default is False.
        recreate: boolean (Default: False)
            Set to True to overwrite the FITS data and header.
        create_folder: boolean (Default: False)
            Create folder if not exist. Use with caution.

        """

        self.create_fits(output, recreate=recreate, empty_primary_hdu=False)

        # create the director if not exist
        if create_folder:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

        output_split = output.split("+")

        start = 0

        for output_type in self.hdu_order.keys():
            if (output_type in output_split) & self.hdu_content[output_type]:
                logging.info(f"Exporting {output_type} to CSV.")
                end = start + self.n_hdu[output_type]

                logging.info(f"The HDU index is from {start} to {end - 1}.")

                if output_type != "arc_lines":
                    output_data = np.column_stack(
                        [hdu.data for hdu in self.hdu_output[start:end]]
                    )

                    if overwrite or (
                        not os.path.exists(
                            filename + "_" + output_type + ".csv"
                        )
                    ):
                        np.savetxt(
                            filename + "_" + output_type + ".csv",
                            output_data,
                            delimiter=",",
                            header=self.header[output_type],
                        )

                    else:
                        self.logger.warning(
                            filename
                            + f"_{output_type}.csv cannot be saved to disk. "
                            + "Please check the path and/or set overwrite to "
                            + "True."
                        )

                else:
                    output_data_arc_peaks = self.hdu_output[start].data
                    output_data_arc_peaks_refined = self.hdu_output[
                        start + 1
                    ].data

                    if overwrite or (
                        not os.path.exists(filename + "_arc_peaks.csv")
                    ):
                        np.savetxt(
                            f"{filename}_arc_peaks.csv",
                            output_data_arc_peaks,
                            delimiter=",",
                            header=self.header[output_type],
                        )

                    else:
                        self.logger.warning(
                            f"{filename}_arc_peaks.csv cannot be saved to "
                            + "disk. Please check the path and/or set "
                            + "overwrite to True."
                        )

                    if overwrite or (
                        not os.path.exists(filename + "_arc_peaks_refined.csv")
                    ):
                        np.savetxt(
                            filename + "_arc_peaks_refined.csv",
                            output_data_arc_peaks_refined,
                            delimiter=",",
                            header=self.header[output_type],
                        )

                    else:
                        self.logger.warning(
                            f"{filename}_arc_peaks_refined.csv cannot be "
                            + "saved to disk. Please check the path and/or "
                            + "set overwrite to True."
                        )

                start = end
                logging.info(f"Exported {output_type} to CSV successfully.")
