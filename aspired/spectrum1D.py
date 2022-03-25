# -*- coding: utf-8 -*-
import datetime
import logging
import os

import numpy as np
from astropy.io import fits
from scipy import interpolate as itp

__all__ = ["Spectrum1D"]


class Spectrum1D:
    """
    Base class of a 1D spectral object to hold the information of each
    extracted spectrum and the raw headers if was provided during the
    data reduction. The FITS or CSV file output are all done here.

    """

    def __init__(
        self,
        spec_id=None,
        verbose=True,
        logger_name="Spectrum1D",
        log_level="INFO",
        log_file_folder="default",
        log_file_name=None,
    ):
        """
        Initialise the object with a logger.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object. Note that this
            ID is unique in each reduction only.
        verbose: boolean (Default: True)
            Set to False to suppress all verbose warnings, except for
            critical failure.
        logger_name: str (Default: Spectrum1D)
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
        if (log_level == "CRITICAL") or (not verbose):
            self.logger.setLevel(logging.CRITICAL)
        elif log_level == "ERROR":
            self.logger.setLevel(logging.ERROR)
        elif log_level == "WARNING":
            self.logger.setLevel(logging.WARNING)
        elif log_level == "INFO":
            self.logger.setLevel(logging.INFO)
        elif log_level == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
        else:
            raise ValueError("Unknonw logging level.")

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] "
            "%(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
        )

        if log_file_name is None:
            # Only print log to screen
            self.handler = logging.StreamHandler()
        else:
            if log_file_name == "default":
                log_file_name = "{}_{}.log".format(
                    logger_name,
                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                )
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
        elif type(spec_id) == int:
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
        self.pixel_list = None
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

        # Continuum
        self.count_continuum = None
        self.count_resampled_continuum = None
        self.flux_continuum = None
        self.flux_resampled_continuum = None

        # Sensitivity curve smoothing properties
        self.smooth = None
        self.slength = None
        self.sorder = None

        # Tellurics
        self.telluric_func = None
        self.telluric_profile = None
        self.telluric_profile_resampled = None
        self.telluric_factor = 0.0
        self.telluric_factor_resampled = 0.0

        # Sensitivity curve properties
        self.sensitivity = None
        self.sensitivity_resampled = None
        self.sensitivity_func = None
        self.wave_literature = None
        self.flux_literature = None

        # Atmospheric extinction properties
        self.standard_flux_extinction_fraction = None
        self.standard_flux_resampled_extinction_fraction = None
        self.science_flux_extinction_fraction = None
        self.science_flux_resampled_extinction_fraction = None

        # HDU lists for output
        self.trace_hdulist = None
        self.count_hdulist = None
        self.arc_spec_hdulist = None
        self.wavecal_hdulist = None
        self.count_resampled_hdulist = None
        self.flux_hdulist = None
        self.flux_resampled_hdulist = None
        self.wavelength_hdulist = None

        # FITS output properties
        self.hdu_output = None
        self.empty_primary_hdu = True

        self.header = {
            "flux_resampled": "Resampled Flux, Resampled Flux Uncertainty, Resampled Sky Flux, "
            "Resampled Sensitivity Curve",
            "count_resampled": "Resampled Count, Resampled Count Uncertainty, "
            "Resampled Sky Count",
            "arc_spec": "1D Arc Spectrum, Arc Line Position, Arc Line Effective Position",
            "wavecal": "Polynomial coefficients for wavelength calibration",
            "wavelength": "The pixel-to-wavelength mapping",
            "flux": "Flux, Flux Uncertainty, Sky Flux",
            "sensitivity": "Sensitivity Curve",
            "weight_map": "Weight map of the extration (variance)",
            "count": "Count, Count Uncertainty, Sky Count",
            "trace": "Pixel positions of the trace in the spatial direction, "
            "Width of the trace",
        }

        self.n_hdu = {
            "trace": 2,
            "count": 3,
            "weight_map": 1,
            "arc_spec": 3,
            "wavecal": 1,
            "wavelength": 1,
            "count_resampled": 3,
            "sensitivity": 1,
            "flux": 3,
            "sensitivity_resampled": 1,
            "flux_resampled": 3,
        }

        self.hdu_order = {
            "trace": 0,
            "count": 1,
            "weight_map": 2,
            "arc_spec": 3,
            "wavecal": 4,
            "wavelength": 5,
            "count_resampled": 6,
            "sensitivity": 7,
            "flux": 8,
            "sensitivity_resampled": 9,
            "flux_resampled": 10,
        }

        self.hdu_content = {
            "trace": False,
            "count": False,
            "weight_map": False,
            "arc_spec": False,
            "wavecal": False,
            "wavelength": False,
            "count_resampled": False,
            "sensitivity": False,
            "flux": False,
            "sensitivity_resampled": False,
            "flux_resampled": False,
        }

    def merge(self, spectrum1D, overwrite=False):
        """
        This function copies all the info from the supplied spectrum1D to
        this one, including the spec_id.

        Parameters
        ----------
        spectrum1D: Spectrum1D object
            The source Spectrum1D to be deep copied over.
        overwrite: boolean (Default: False)
            Set to True to overwrite all the data in this Spectrum1D.

        """

        for attr, value in self.__dict__.items():

            if attr == "spec_id":

                if getattr(spectrum1D, attr) != 0:

                    setattr(self, attr, value)

            if getattr(self, attr) is None or []:

                setattr(self, attr, getattr(spectrum1D, attr))

            if overwrite:

                setattr(self, attr, getattr(spectrum1D, attr))

            else:

                # if not overwrite, do nothing
                pass

    def add_spectrum_header(self, header):
        """
        Add a header for the spectrum. Typically put the header of the
        raw 2D spectral image of the science target(s) here. Some
        automated operations rely on reading from the header.

        Parameters
        ----------
        header: astropy.io.fits.Header object

        """

        if header is not None:
            if type(header) == fits.Header:
                self.spectrum_header = header
                self.logger.info("spectrum_header is stored.")
            elif type(header[0]) == fits.Header:
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

    def add_standard_header(self, header):
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
            if type(header) == fits.Header:
                self.standard_header = header
                self.logger.info("standard_header is stored.")
            elif type(header[0]) == fits.Header:
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

    def add_arc_header(self, header):
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
            if type(header) == fits.Header:
                self.arc_header = header
                self.logger.info("arc_header is stored.")
            elif type(header[0]) == fits.Header:
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

    def add_trace(self, trace, trace_sigma, pixel_list=None):
        """
        Add the trace of the spectrum.

        Parameters
        ----------
        trace: list or numpy.ndarray (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: list or numpy.ndarray (N)
            Standard deviation of the Gaussian profile of a trace
        pixel_list: list or numpy array (Default: None)
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

        if pixel_list is None:

            pixel_list = list(np.arange(len(trace)).astype("int"))

        else:

            assert isinstance(
                pixel_list, (list, np.ndarray)
            ), "pixel_list has to be a list or a numpy array"
            assert len(pixel_list) == len(trace), "trace and pixel_list have "
            "to be the same size."

        pixel_mapping_itp = itp.interp1d(
            np.arange(len(trace)),
            pixel_list,
            kind="cubic",
            fill_value="extrapolate",
        )

        # Only add if all assertions are passed.
        self.trace = trace
        self.trace_sigma = trace_sigma
        self.len_trace = len(trace)
        self.add_pixel_list(pixel_list)
        self.add_pixel_mapping_itp(pixel_mapping_itp)

    def remove_trace(self):
        """
        Remove the trace, trace_sigma and len_trace, also remove the pixel_list
        and the interpolation function that maps the pixel values to the
        effective pixel values.

        """

        self.trace = None
        self.trace_sigma = None
        self.len_trace = None
        self.remove_pixel_list()
        self.remove_pixel_mapping_itp()

    def add_aperture(
        self, widthdn, widthup, sepdn, sepup, skywidthdn, skywidthup
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

    def add_count(self, count, count_err=None, count_sky=None):
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

        if self.pixel_list is None:

            pixel_list = list(np.arange(len(count)).astype("int"))
            self.add_pixel_list(pixel_list)

        else:

            assert len(self.pixel_list) == len(count), (
                "count and pixel_list have " + "to be the same size."
            )

    def remove_count(self):
        """
        Remove the count, count_err and count_sky.

        """

        self.count = None
        self.count_err = None
        self.count_sky = None

    def add_variances(self, var):
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

    def add_profile(self, profile):
        """
        Add the extraction profile.

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

    def add_arc_spec(self, arc_spec):
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

    def add_pixel_list(self, pixel_list):
        """
        Add the pixel list, which contain the effective pixel values that
        has accounted for chip gaps and non-integer pixel value.

        Parameters
        ----------
        pixel_list: 1-d array
            The effective position of the pixel.

        """

        assert isinstance(
            pixel_list, (list, np.ndarray)
        ), "pixel_list has to be a list or a numpy array"
        self.pixel_list = pixel_list

    def remove_pixel_list(self):
        """
        Remove the pixel list.

        """

        self.pixel_list = None

    def add_pixel_mapping_itp(self, pixel_mapping_itp):
        """
        Add the interpolated callable function that convert raw pixel
        values into effective pixel values.

        Parameters
        ----------
        pixel_mapping_itp: callable function
            The mapping function for raw-effective pixel position.

        """

        assert (
            type(pixel_mapping_itp) == itp.interpolate.interp1d
        ), "pixel_mapping_itp has to be a "
        "scipy.interpolate.interpolate.interp1d object."
        self.pixel_mapping_itp = pixel_mapping_itp

    def remove_pixel_mapping_itp(self):
        """
        Remove the interpolation function that maps the pixel values to the
        effective pixel value.

        """

        self.pixel_mapping_itp = None

    def add_peaks(self, peaks):
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

    def add_peaks_refined(self, peaks_refined):
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

    def add_peaks_wave(self, peaks_wave):
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

    def add_calibrator(self, calibrator):
        """
        Add a RASCAL Calibrator object.

        Parameters
        ----------
        calibrator: rascal.Calibrator()
            A RASCAL Calibrator object.

        """

        self.calibrator = calibrator

    def remove_calibrator(self):
        """
        Remove the Calibrator.

        """

        self.calibrator = None

    def add_atlas_wavelength_range(
        self, min_atlas_wavelength, max_atlas_wavelength
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

    def add_min_atlas_intensity(self, min_atlas_intensity):
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

    def add_min_atlas_distance(self, min_atlas_distance):
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

    def add_gain(self, gain):
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

    def add_readnoise(self, readnoise):
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

    def add_exptime(self, exptime):
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

    def add_airmass(self, airmass):
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

    def add_seeing(self, seeing):
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

    def add_weather_condition(self, pressure, temperature, relative_humidity):
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

    def add_fit_type(self, fit_type):
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

    def add_fit_coeff(self, fit_coeff):
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
        self, num_pix, pixel_list, plotting_library, log_level
    ):
        """
        Add the properties of the RASCAL Calibrator.

        Parameters
        ----------
        num_pix: int
            The number of pixels in the dispersion direction
        pixel_list: list or numpy array
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(num_pix), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        plotting_library : string
            Choose between matplotlib and plotly.
        log_level : string
            Choose from {CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET}.

        """

        self.num_pix = num_pix
        self.plotting_library = plotting_library
        self.log_level = log_level
        self.pixel_list = pixel_list

    def remove_calibrator_properties(self):
        """
        Remove the properties of the RASCAL Calibrator as recorded in ASPIRED.
        This does NOT remove the calibrator properites INSIDE the calibrator.

        """

        self.num_pix = None
        self.pixel_list = None
        self.plotting_library = None
        self.log_level = None

    def add_hough_properties(
        self,
        num_slopes,
        xbins,
        ybins,
        min_wavelength,
        max_wavelength,
        range_tolerance,
        linearity_tolerance,
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
        sample_size,
        top_n_candidate,
        linear,
        filter_close,
        ransac_tolerance,
        candidate_weighted,
        hough_weight,
        minimum_matches,
        minimum_peak_utilisation,
        minimum_fit_error,
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
        fit_coeff,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
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
        fit_coeff,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
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
        fit_coeff,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
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

    def add_wavelength(self, wave):
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

    def add_wavelength_resampled(self, wave_resampled):
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
        self, count_resampled, count_err_resampled, count_sky_resampled
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

    def add_standard_star(self, library, target):
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

    def add_smoothing(self, smooth, slength, sorder):
        """
        Add the SG smoothing parameters for computing the sensitivity curve.

        Parameters
        ----------
        smooth: bool
            Indicate if smoothing was applied.
        slength: int
            The size of the smoothing window.
        sorder: int
            The order of polynomial used for SG smoothing.

        """

        self.smooth = smooth
        self.slength = slength
        self.sorder = sorder

    def remove_smoothing(self):
        """
        Remove the smoothing parameters for computing the sensitivity curve.

        """

        self.smooth = None
        self.slength = None
        self.sorder = None

    def add_sensitivity_func(self, sensitivity_func):
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

    def add_sensitivity(self, sensitivity):
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

    def add_sensitivity_resampled(self, sensitivity_resampled):
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

    def add_literature_standard(self, wave_literature, flux_literature):
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

    def add_count_continuum(self, count_continuum):
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

    def add_count_resampled_continuum(self, count_resampled_continuum):
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

    def add_flux_continuum(self, flux_continuum):
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

    def add_flux_resampled_continuum(self, flux_resampled_continuum):
        """
        Add the continuum flux_resampled value (should be the same size as
        flux_resampled).

        Parameters
        ----------
        flux_resampled_continuum: list or 1-d array
            The flux of the continuum at the resampled wavelength.

        """

        self.flux_resampled_continuum = flux_resampled_continuum

    def remove_flux_resampled_continuum(self):
        """
        Remove the continuum flux_resampled values.

        """

        self.flux_resampled_continuum = None

    def add_telluric_func(self, telluric_func):
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

    def add_telluric_profile(self, telluric_profile):
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

    def add_telluric_profile_resampled(self, telluric_profile_resampled):
        """
        Add the Telluric profile - relative intensity at each resampled
        wavelength.

        Parameters
        ----------
        telluric_profile: list or numpy.ndarray
            The relative intensity of the telluric absorption strength [0, 1]

        """

        self.telluric_profile_resampled = telluric_profile_resampled

    def remove_telluric_profile_resampled(self):
        """
        Remove the Telluric profile at the resampled wavelengths.

        """

        self.telluric_profile_resampled = None

    def add_telluric_factor(self, telluric_factor):
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

    def add_telluric_factor_resampled(self, telluric_factor_resampled):
        """
        Add the Telluric factor found using the resampled flux.

        Parameters
        ----------
        telluric_factor_resampled: float
            The multiplier to the telluric profile that minimises the sum
            of the deviation of corrected spectrum from the resampled
            continuum spectrum.

        """

        self.telluric_factor_resampled = telluric_factor_resampled

    def remove_telluric_factor_resampled(self):
        """
        Remove the Telluric factor for the resampled spectrum.

        """

        self.telluric_factor_resampled = None

    def add_flux(self, flux, flux_err, flux_sky):
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

    def add_flux_resampled(
        self, flux_resampled, flux_err_resampled, flux_sky_resampled
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

    def _modify_imagehdu_data(self, hdulist, idx, method, *args):
        """
        Wrapper function to modify the data of an ImageHDU object.

        """

        method_to_call = getattr(hdulist[idx].data, method)
        method_to_call(*args)

    def _modify_imagehdu_header(self, hdulist, idx, method, *args):
        """
        Wrapper function to modify the header of an ImageHDU object.

        e.g.
        method = 'set'
        args = 'BUNIT', 'Angstroms'

        method_to_call(*args) becomes hdu[idx].header.set('BUNIT', 'Angstroms')

        """

        method_to_call = getattr(hdulist[idx].header, method)
        method_to_call(*args)

    def modify_trace_header(self, idx, method, *args):
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

    def modify_count_header(self, idx, method, *args):
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

    def modify_weight_map_header(self, method, *args):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+----------------------+
        | HDU | Data                 |
        +-----+----------------------+
        |  0  | Line spread function |
        +-----+----------------------+

        """

        self._modify_imagehdu_header(self.weight_map_hdulist, 0, method, *args)

    def modify_count_resampled_header(self, idx, method, *args):
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

    def modify_arc_spec_header(self, idx, method, *args):
        """
        for method 'set', it takes
        keyword, value=None, comment=None, before=None, after=None

        +-----+-------------------+
        | HDU | Data              |
        +-----+-------------------+
        |  0  | Arc spectrum      |
        |  1  | Peaks (pixel)     |
        |  2  | Peaks (sub-pixel) |
        +-----+-------------------+

        """

        self._modify_imagehdu_header(self.arc_spec_hdulist, idx, method, *args)

    def modify_wavecal_header(self, method, *args):
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

    def modify_wavelength_header(self, method, *args):
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

    def modify_sensitivity_header(self, method, *args):
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

    def modify_flux_header(self, idx, method, *args):
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

    def modify_sensitivity_resampled_header(self, method, *args):
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

    def modify_flux_resampled_header(self, idx, method, *args):
        """
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

        """

        self._modify_imagehdu_header(
            self.flux_resampled_hdulist, idx, method, *args
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

            # Add the trace
            self.modify_trace_header(0, "set", "EXTNAME", "Trace")
            self.modify_trace_header(0, "set", "LABEL", "Trace")
            self.modify_trace_header(0, "set", "CRPIX1", 1)
            self.modify_trace_header(0, "set", "CDELT1", 1)
            self.modify_trace_header(0, "set", "CRVAL1", self.pixel_list[0])
            self.modify_trace_header(0, "set", "CTYPE1", "Pixel (Dispersion)")
            self.modify_trace_header(0, "set", "CUNIT1", "Pixel")
            self.modify_trace_header(0, "set", "BUNIT", "Pixel (Spatial)")

            # Add the trace_sigma
            self.modify_trace_header(1, "set", "EXTNAME", "Trace width/sigma")
            self.modify_trace_header(1, "set", "LABEL", "Trace width/sigma")
            self.modify_trace_header(1, "set", "CRPIX1", 1)
            self.modify_trace_header(1, "set", "CDELT1", 1)
            self.modify_trace_header(1, "set", "CRVAL1", self.pixel_list[0])
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

            # Add the count
            self.modify_count_header(0, "set", "WIDTHDN", self.widthdn)
            self.modify_count_header(0, "set", "WIDTHUP", self.widthup)
            self.modify_count_header(0, "set", "SEPDN", self.sepdn)
            self.modify_count_header(0, "set", "SEPUP", self.sepup)
            self.modify_count_header(0, "set", "SKYDN", self.skywidthdn)
            self.modify_count_header(0, "set", "SKYUP", self.skywidthup)
            self.modify_count_header(0, "set", "XTYPE", self.extraction_type)
            self.modify_count_header(0, "set", "EXTNAME", "Electron Count")
            self.modify_count_header(0, "set", "LABEL", "Electron Count")
            self.modify_count_header(0, "set", "CRPIX1", 1)
            self.modify_count_header(0, "set", "CDELT1", 1)
            self.modify_count_header(0, "set", "CRVAL1", self.pixel_list[0])
            self.modify_count_header(0, "set", "CTYPE1", " Pixel (Dispersion)")
            self.modify_count_header(0, "set", "CUNIT1", "Pixel")
            self.modify_count_header(0, "set", "BUNIT", "electron")

            # Add the uncertainty count
            self.modify_count_header(
                1, "set", "EXTNAME", "Electron Count (Uncertainty)"
            )
            self.modify_count_header(
                1, "set", "LABEL", "Electron Count (Uncertainty)"
            )
            self.modify_count_header(1, "set", "CRPIX1", 1)
            self.modify_count_header(1, "set", "CDELT1", 1)
            self.modify_count_header(1, "set", "CRVAL1", self.pixel_list[0])
            self.modify_count_header(1, "set", "CTYPE1", "Pixel (Dispersion)")
            self.modify_count_header(1, "set", "CUNIT1", "Pixel")
            self.modify_count_header(1, "set", "BUNIT", "electron")

            # Add the sky count
            self.modify_count_header(
                2, "set", "EXTNAME", "Electron Count (Sky)"
            )
            self.modify_count_header(2, "set", "LABEL", "Electron Count (Sky)")
            self.modify_count_header(2, "set", "CRPIX1", 1)
            self.modify_count_header(2, "set", "CDELT1", 1)
            self.modify_count_header(2, "set", "CRVAL1", self.pixel_list[0])
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

            # Add the extraction weights
            self.modify_weight_map_header(
                "set", "EXTNAME", "Optimal Extraction Profile"
            )
            self.modify_weight_map_header(
                "set", "LABEL", "Optimal Extraction Profile"
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

            # Add the resampled count
            self.modify_count_resampled_header(
                0, "set", "EXTNAME", "Resampled Electron Count"
            )
            self.modify_count_resampled_header(
                0, "set", "LABEL", "Resampled Electron Count"
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

            # Add the resampled uncertainty count
            self.modify_count_resampled_header(
                1, "set", "EXTNAME", "Resampled Electron Count (Uncertainty)"
            )
            self.modify_count_resampled_header(
                1, "set", "LABEL", "Resampled Electron Count (Uncertainty)"
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
                2, "set", "EXTNAME", "Resampled Electron Count (Sky)"
            )
            self.modify_count_resampled_header(
                2, "set", "LABEL", "Resampled Electron Count (Sky)"
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
                peaks_ImageHDU = fits.ImageHDU(
                    self.peaks, header=self.arc_header
                )
                peaks_refined_ImageHDU = fits.ImageHDU(
                    self.peaks_refined, header=self.arc_header
                )
            else:
                arc_spec_ImageHDU = fits.ImageHDU(self.arc_spec)
                peaks_ImageHDU = fits.ImageHDU(self.peaks)
                peaks_refined_ImageHDU = fits.ImageHDU(self.peaks_refined)

            # Create an empty HDU list and populate with ImageHDUs
            self.arc_spec_hdulist = fits.HDUList()
            self.arc_spec_hdulist += [arc_spec_ImageHDU]
            self.arc_spec_hdulist += [peaks_ImageHDU]
            self.arc_spec_hdulist += [peaks_refined_ImageHDU]

            # Add the arc spectrum
            self.modify_arc_spec_header(0, "set", "EXTNAME", "Electron Count")
            self.modify_arc_spec_header(0, "set", "LABEL", "Electron Count")
            self.modify_arc_spec_header(0, "set", "CRPIX1", 1)
            self.modify_arc_spec_header(0, "set", "CDELT1", 1)
            self.modify_arc_spec_header(0, "set", "CRVAL1", self.pixel_list[0])
            self.modify_arc_spec_header(
                0, "set", "CTYPE1", "Pixel (Dispersion)"
            )
            self.modify_arc_spec_header(0, "set", "CUNIT1", "Pixel")
            self.modify_arc_spec_header(0, "set", "BUNIT", "electron")

            # Add the peaks in native pixel value
            self.modify_arc_spec_header(
                1, "set", "EXTNAME", "Peaks (Detector Pixel)"
            )
            self.modify_arc_spec_header(
                1, "set", "LABEL", "Peaks (Detector Pixel)"
            )
            self.modify_arc_spec_header(1, "set", "BUNIT", "Pixel")

            # Add the peaks in effective pixel value
            self.modify_arc_spec_header(
                2, "set", "EXTNAME", "Peaks (Effective Pixel)"
            )
            self.modify_arc_spec_header(
                2, "set", "LABEL", "Peaks (Effective Pixel)"
            )
            self.modify_arc_spec_header(2, "set", "BUNIT", "Pixel")

        except Exception as e:

            self.logger.error(str(e))

            # Set it to None if the above failed
            self.logger.error("arc_spec ImageHDU cannot be created.")
            self.arc_spec_hdulist = None

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
            self.modify_wavecal_header("set", "EXTNAME", "Peak Electron Count")
            self.modify_wavecal_header("set", "LABEL", "Peak Electron Count")
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

            # Add the calibrated wavelength
            self.modify_wavelength_header(
                "set", "EXTNAME", "Pixel-Wavelength Mapping"
            )
            self.modify_wavelength_header(
                "set", "LABEL", "Pixel-Wavelength Mapping"
            )
            self.modify_wavelength_header("set", "CRPIX1", 1)
            self.modify_wavelength_header("set", "CDELT1", 1)
            self.modify_wavelength_header("set", "CRVAL1", self.pixel_list[0])
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

            self.modify_sensitivity_header("set", "EXTNAME", "Sensitivity")
            self.modify_sensitivity_header("set", "LABEL", "Sensitivity")
            self.modify_sensitivity_header("set", "CRPIX1", 1.00e00)
            self.modify_sensitivity_header("set", "CDELT1", 1)
            self.modify_sensitivity_header("set", "CRVAL1", self.pixel_list[0])
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

            if self.standard_header is not None:
                if header is None:
                    header = self.standard_header
                else:
                    header += self.standard_header

            if self.arc_header is not None:
                if header is None:
                    header = self.arc_header
                else:
                    header += self.arc_header

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

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_header(0, "set", "EXTNAME", "Flux")
            self.modify_flux_header(0, "set", "LABEL", "Flux")
            self.modify_flux_header(0, "set", "CRPIX1", 1.00e00)
            self.modify_flux_header(0, "set", "CDELT1", 1)
            self.modify_flux_header(0, "set", "CRVAL1", self.pixel_list[0])
            self.modify_flux_header(0, "set", "CTYPE1", "Pixel")
            self.modify_flux_header(0, "set", "CUNIT1", "Pixel")
            self.modify_flux_header(
                0, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_header(1, "set", "EXTNAME", "Flux (Uncertainty)")
            self.modify_flux_header(1, "set", "LABEL", "Flux (Uncertainty)")
            self.modify_flux_header(1, "set", "CRPIX1", 1.00e00)
            self.modify_flux_header(1, "set", "CDELT1", 1)
            self.modify_flux_header(1, "set", "CRVAL1", self.pixel_list[0])
            self.modify_flux_header(1, "set", "CTYPE1", "Pixel")
            self.modify_flux_header(1, "set", "CUNIT1", "Pixel")
            self.modify_flux_header(
                1, "set", "BUNIT", "erg/(s*cm**2*Angstrom)"
            )

            self.modify_flux_header(2, "set", "EXTNAME", "Flux (Sky)")
            self.modify_flux_header(2, "set", "LABEL", "Flux (Sky)")
            self.modify_flux_header(2, "set", "CRPIX1", 1.00e00)
            self.modify_flux_header(2, "set", "CDELT1", 1)
            self.modify_flux_header(2, "set", "CRVAL1", self.pixel_list[0])
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

            self.modify_sensitivity_resampled_header(
                "set", "EXTNAME", "Sensitivity"
            )
            self.modify_sensitivity_resampled_header(
                "set", "LABEL", "Sensitivity"
            )
            self.modify_sensitivity_resampled_header("set", "CRPIX1", 1.00e00)
            self.modify_sensitivity_resampled_header("set", "CDELT1", 1)
            self.modify_sensitivity_resampled_header(
                "set", "CRVAL1", self.pixel_list[0]
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

            if self.standard_header is not None:
                if header is None:
                    header = self.standard_header
                else:
                    header += self.standard_header

            if self.arc_header is not None:
                if header is None:
                    header = self.arc_header
                else:
                    header += self.arc_header

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

            # Note that wave_start is the centre of the starting bin
            self.modify_flux_resampled_header(0, "set", "EXTNAME", "Flux")
            self.modify_flux_resampled_header(0, "set", "LABEL", "Flux")
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
                1, "set", "EXTNAME", "Flux (Uncertainty)"
            )
            self.modify_flux_resampled_header(
                1, "set", "LABEL", "Flux (Uncertainty)"
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
                2, "set", "EXTNAME", "Flux (Sky)"
            )
            self.modify_flux_resampled_header(2, "set", "LABEL", "Flux (Sky)")
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

    def remove_flux_resampled_fits(self):
        """
        Remove the flux_resampled FITS HDUList.

        """

        self.flux_resampled_hdulist = None

    def remove_wavelength_fits(self):
        """
        Remove the wavelength FITS HDUList.

        """

        self.wavelength_hdulist = None

    def remove_weight_map_fits(self):
        """
        Remove the weight_map FITS HDUList.

        """

        self.weight_map = None

    def remove_flux_fits(self):
        """
        Remove the flux FITS HDUList.

        """

        self.flux = None

    def remove_wavelength_resampled_fits(self):
        """
        Remove the resampled wavelength FITS HDUList.

        """

        self.wavelength_resampled_hdulist = None

    def remove_sensitivity_resampled_fits(self):
        """
        Remove the sensitivity_resampled FITS HDUList.

        """

        self.sensitivity_resampled = None

    def create_fits(
        self,
        output,
        recreate=False,
        empty_primary_hdu=True,
        return_hdu_list=False,
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
                arc_spec: 3 HDUs
                    1D arc spectrum, arc line position (pixel), and arc
                    line effective position (pixel)
                wavecal: 1 HDU
                    Polynomial coefficients for wavelength calibration
                wavelength: 1 HDU
                    Wavelength of each pixel
                count_resampled: 3 HDUs
                    Resampled Count, uncertainty, and sky (wavelength)
                sensitivity: 1 HDU
                    Sensitivity (pixel)
                flux: 4 HDUs
                    Flux, uncertainty, and sky (pixel)
                sensitivity_resampled: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 4 HDUs
                    Flux, uncertainty, and sky (wavelength)

        recreate: boolean (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: boolean (Default: True)
            Set to True to leave the Primary HDU blank.
        return_hdu_list: boolean (Default: False)
            Set to True to return the HDU List.

        """

        output_split = output.split("+")

        # If the requested list of HDUs is already good to go
        if set([k for k, v in self.hdu_content.items() if v]) == set(
            output_split
        ):

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
            if recreate:
                for k, v in self.hdu_content.items():
                    self.hdu_content[k] = False

            # Empty list for appending HDU lists
            hdu_output = fits.HDUList()

            # If leaving the primary HDU empty
            if empty_primary_hdu:

                hdu_output.append(fits.PrimaryHDU())

            # Join the list(s)
            if "trace" in output_split:

                if not self.hdu_content["trace"]:
                    self.create_trace_fits()

                hdu_output += self.trace_hdulist
                self.hdu_content["trace"] = True

            if "count" in output_split:

                if not self.hdu_content["count"]:
                    self.create_count_fits()

                hdu_output += self.count_hdulist
                self.hdu_content["count"] = True

            if "weight_map" in output_split:

                if not self.hdu_content["weight_map"]:
                    self.create_weight_map_fits()

                hdu_output += self.weight_map_hdulist
                self.hdu_content["weight_map"] = True

            if "arc_spec" in output_split:

                if not self.hdu_content["arc_spec"]:
                    self.create_arc_spec_fits()

                hdu_output += self.arc_spec_hdulist
                self.hdu_content["arc_spec"] = True

            if "wavecal" in output_split:

                if not self.hdu_content["wavecal"]:
                    self.create_wavecal_fits()

                hdu_output += self.wavecal_hdulist
                self.hdu_content["wavecal"] = True

            if "wavelength" in output_split:

                if not self.hdu_content["wavelength"]:
                    self.create_wavelength_fits()

                hdu_output += self.wavelength_hdulist
                self.hdu_content["wavelength"] = True

            if "count_resampled" in output_split:

                if not self.hdu_content["count_resampled"]:
                    self.create_count_resampled_fits()

                hdu_output += self.count_resampled_hdulist
                self.hdu_content["count_resampled"] = True

            if "sensitivity" in output_split:

                if not self.hdu_content["sensitivity"]:
                    self.create_sensitivity_fits()

                hdu_output += self.sensitivity_hdulist
                self.hdu_content["sensitivity"] = True

            if "flux" in output_split:

                if not self.hdu_content["flux"]:
                    self.create_flux_fits()

                hdu_output += self.flux_hdulist
                self.hdu_content["flux"] = True

            if "sensitivity_resampled" in output_split:

                if not self.hdu_content["sensitivity_resampled"]:
                    self.create_sensitivity_resampled_fits()

                hdu_output += self.sensitivity_resampled_hdulist
                self.hdu_content["sensitivity_resampled"] = True

            if "flux_resampled" in output_split:

                if not self.hdu_content["flux_resampled"]:
                    self.create_flux_resampled_fits()

                hdu_output += self.flux_resampled_hdulist
                self.hdu_content["flux_resampled"] = True

            # If the primary HDU is not chosen to be empty
            if not empty_primary_hdu:

                # Convert the first HDU to PrimaryHDU
                hdu_output[0] = fits.PrimaryHDU(
                    hdu_output[0].data, hdu_output[0].header
                )
                hdu_output.update_extend()
                self.empty_primary_hdu = False

            else:

                hdu_output.update_extend()
                self.empty_primary_hdu = True

            self.hdu_output = hdu_output

        if return_hdu_list:

            return self.hdu_output

    def save_fits(
        self,
        output,
        filename,
        overwrite=False,
        recreate=False,
        empty_primary_hdu=True,
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
                arc_spec: 3 HDUs
                    1D arc spectrum, arc line pixels, and arc line effective
                    pixels
                wavecal: 1 HDU
                    Polynomial coefficients for wavelength calibration
                wavelength: 1 HDU
                    Wavelength of each pixel
                count_resampled: 3 HDUs
                    Resampled Count, uncertainty, and sky (wavelength)
                sensitivity: 1 HDU
                    Sensitivity (pixel)
                flux: 4 HDUs
                    Flux, uncertainty, and sky (pixel)
                sensitivity: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 4 HDUs
                    Flux, uncertainty, and sky (wavelength)

        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        overwrite: boolean
            Default is False.
        recreate: boolean (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: boolean (Default: True)
            Set to True to leave the Primary HDU blank (Default: True)

        """

        self.create_fits(
            output, recreate=recreate, empty_primary_hdu=empty_primary_hdu
        )

        # Save file to disk
        self.hdu_output.writeto(filename + ".fits", overwrite=overwrite)

    def save_csv(self, output, filename, overwrite, recreate):
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
                count: 4 HDUs
                    Count, uncertainty, sky, and optimal flag (pixel)
                weight_map: 1 HDU
                    Weight (pixel)
                arc_spec: 3 HDUs
                    1D arc spectrum, arc line pixels, and arc line effective
                    pixels
                wavecal: 1 HDU
                    Polynomial coefficients for wavelength calibration
                wavelength: 1 HDU
                    Wavelength of each pixel
                count_resampled: 3 HDUs
                    Resampled Count, uncertainty, and sky (wavelength)
                sensitivity: 1 HDU
                    Sensitivity (pixel)
                flux: 4 HDUs
                    Flux, uncertainty, and sky (pixel)
                sensitivity: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 3 HDUs
                    Flux, uncertainty, and sky (wavelength)

        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        overwrite: boolean
            Default is False.
        recreate: boolean (Default: False)
            Set to True to overwrite the FITS data and header.

        """

        self.create_fits(output, recreate=recreate, empty_primary_hdu=False)

        output_split = output.split("+")

        start = 0

        for output_type in self.hdu_order.keys():

            if output_type in output_split:

                end = start + self.n_hdu[output_type]

                if output_type != "arc_spec":

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
                            + "_"
                            + output_type
                            + ".csv cannot be saved to disk. Please check "
                            + "the path and/or set overwrite to True."
                        )

                else:

                    output_data_arc_spec = self.hdu_output[start].data
                    output_data_arc_peaks = self.hdu_output[start + 1].data
                    output_data_arc_peaks_refined = self.hdu_output[
                        start + 2
                    ].data

                    if overwrite or (
                        not os.path.exists(filename + "_arc_spec.csv")
                    ):

                        np.savetxt(
                            filename + "_arc_spec.csv",
                            output_data_arc_spec,
                            delimiter=",",
                            header=self.header[output_type],
                        )

                    else:

                        self.logger.warning(
                            filename
                            + "_arc_spec.csv cannot be saved to disk. Please "
                            + "check the path and/or set overwrite to True."
                        )

                    if overwrite or (
                        not os.path.exists(filename + "_arc_peaks.csv")
                    ):

                        np.savetxt(
                            filename + "_arc_peaks.csv",
                            output_data_arc_peaks,
                            delimiter=",",
                            header=self.header[output_type],
                        )

                    else:

                        self.logger.warning(
                            filename
                            + "_arc_peaks.csv cannot be saved to disk. Please "
                            + "check the path and/or set overwrite to True."
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
                            filename
                            + "_arc_peaks_refined.csv cannot be saved to "
                            + "disk. Please check the path and/or set "
                            + "overwrite to True."
                        )

                start = end
