# -*- coding: utf-8 -*-
from aspired.util import get_continuum
import copy
import datetime
import logging
import os
import pkg_resources

from astropy.stats import sigma_clip
import numpy as np
from plotly import graph_objects as go
from plotly import io as pio
from spectres import spectres
from scipy import optimize
from scipy.interpolate import interp1d

from .wavelength_calibration import WavelengthCalibration
from .flux_calibration import FluxCalibration
from .spectrum1D import Spectrum1D

__all__ = ["OneDSpec"]


class OneDSpec:
    def __init__(
        self,
        verbose=True,
        logger_name="OneDSpec",
        log_level="INFO",
        log_file_folder="default",
        log_file_name=None,
    ):
        """
        This class applies the wavelength calibrations and compute & apply the
        flux calibration to the extracted 1D spectra. The standard TwoDSpec
        object is not required for data reduction, but the flux calibrated
        standard observation will not be available for diagnostic.

        Parameters
        ----------
        verbose: bool (Default: True)
            Set to False to suppress all verbose warnings, except for
            critical failure.
        logger_name: str (Default: 'OneDSpec')
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
        log_file_folder: None or str (Default: 'default')
            Folder in which the file is save, set to default to save to the
            current path.
        log_file_name: None or str (Default: None)
            File name of the log, set to None to print to screen only.

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

        self.verbose = verbose
        self.logger_name = logger_name
        self.log_level = log_level
        self.log_file_folder = log_file_folder
        self.log_file_name = log_file_name

        # Initialise empty calibration objects
        self.science_wavecal = {}
        self.standard_wavecal = WavelengthCalibration(
            verbose=self.verbose,
            logger_name=self.logger_name,
            log_level=self.log_level,
            log_file_folder=self.log_file_folder,
            log_file_name=self.log_file_name,
        )
        self.fluxcal = FluxCalibration(
            verbose=self.verbose,
            logger_name=self.logger_name,
            log_level=self.log_level,
            log_file_folder=self.log_file_folder,
            log_file_name=self.log_file_name,
        )

        # Create empty dictionary
        self.science_spectrum_list = {}
        self.standard_spectrum_list = {
            0: Spectrum1D(
                spec_id=0,
                verbose=self.verbose,
                logger_name=self.logger_name,
                log_level=self.log_level,
                log_file_folder=self.log_file_folder,
                log_file_name=self.log_file_name,
            )
        }

        self.add_science_spectrum1D(0)

        # Link them up
        self.science_wavecal[0].from_spectrum1D(self.science_spectrum_list[0])
        self.standard_wavecal.from_spectrum1D(self.standard_spectrum_list[0])
        self.fluxcal.from_spectrum1D(self.standard_spectrum_list[0])

        # Tracking data availability
        self.science_data_available = False
        self.standard_data_available = False

        self.science_arc_available = False
        self.standard_arc_available = False

        self.science_trace_available = False
        self.standard_trace_available = False

        self.science_arc_spec_available = False
        self.standard_arc_spec_available = False

        self.science_arc_lines_available = False
        self.standard_arc_lines_available = False

        self.science_atlas_available = False
        self.standard_atlas_available = False

        self.science_hough_pairs_available = False
        self.standard_hough_pairs_available = False

        self.science_wavecal_polynomial_available = False
        self.standard_wavecal_polynomial_available = False

        self.science_wavelength_calibrated = False
        self.standard_wavelength_calibrated = False

        self.science_wavelength_resampled = False
        self.standard_wavelength_resampled = False

        self.atmospheric_extinction_correction_available = False
        self.atmospheric_extinction_corrected = False

        self.sensitivity_curve_available = False

        self.science_flux_calibrated = False
        self.standard_flux_calibrated = False

        self.science_flux_resampled = False
        self.standard_flux_calibrated = False

    def add_science_spectrum1D(self, spec_id):
        """
        Add a new Spectrum1D with the ID spec_id. This overwrite the existing
        Spectrum1D object if it already exists.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object

        """

        # Create the wavelength calibration object for the given spec_id
        self.science_wavecal.update(
            {
                spec_id: WavelengthCalibration(
                    verbose=self.verbose,
                    logger_name=self.logger_name,
                    log_level=self.log_level,
                    log_file_folder=self.log_file_folder,
                    log_file_name=self.log_file_name,
                )
            }
        )

        # Create the Spectrum1D object for the given spec_id
        self.science_spectrum_list.update(
            {
                spec_id: Spectrum1D(
                    spec_id=spec_id,
                    verbose=self.verbose,
                    logger_name=self.logger_name,
                    log_level=self.log_level,
                    log_file_folder=self.log_file_folder,
                    log_file_name=self.log_file_name,
                )
            }
        )

        # Reference the wavecal to the Spectrum1D object just created
        self.science_wavecal[spec_id].from_spectrum1D(
            self.science_spectrum_list[spec_id]
        )

        self.logger.info(
            "spectrm1D object is added to spec_id: {}".format(spec_id)
        )

    def add_fluxcalibration(self, fluxcal):
        """
        Provide the pre-calibrated FluxCalibration object.

        Parameters
        ----------
        fluxcal: FluxCalibration object
            The true mag/flux values.

        """

        if type(fluxcal) == FluxCalibration:

            self.fluxcal = fluxcal
            self.logger.info("fluxcal object is added")

        else:

            err_msg = "Please provide a valid FluxCalibration object"
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

    def add_wavelengthcalibration(
        self, wavecal, spec_id=None, stype="science+standard"
    ):
        """
        Provide the pre-calibrated WavelengthCalibration object.

        Parameters
        ----------
        wavecal: list of WavelengthCalibration object
            The WavelengthPolyFit object for the science target, flux will
            not be calibrated if this is not provided.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if type(wavecal) == WavelengthCalibration:

            wavecal = [wavecal]

        elif type(wavecal) == list:

            pass

        else:

            err_msg = (
                "Please provide a WavelengthCalibration object or "
                + "a list of them."
            )
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            # Check the sizes of the wave and spec_id and convert wave
            # into a dictionary
            if len(wavecal) == len(spec_id):

                wavecal = {spec_id[i]: wavecal[i] for i in range(len(spec_id))}

            elif len(wavecal) == 1:

                wavecal = {spec_id[i]: wavecal[0] for i in range(len(spec_id))}

            else:

                error_msg = (
                    "wavecal must be the same length of shape " + "as spec_id."
                )
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

            for i in spec_id:

                if type(wavecal[i]) == WavelengthCalibration:

                    self.science_wavecal[i] = wavecal[i]
                    self.logger.info(
                        "Added WavelengthCalibration to the "
                        "science_spectrum_list for spec_id: {}.".format(i)
                    )

                else:

                    err_msg = (
                        "Please provide a valid "
                        + "WavelengthCalibration object."
                    )
                    self.logger.critical(err_msg)
                    raise TypeError(err_msg)

        if "standard" in stype_split:

            if type(wavecal[0]) == WavelengthCalibration:

                self.standard_wavecal = wavecal[0]
                self.logger.info(
                    "Added WavelengthCalibration to "
                    "the standard spectrum_list."
                )

            else:

                err_msg = (
                    "Please provide a valid " + "WavelengthCalibration object"
                )
                self.logger.critical(err_msg)
                raise TypeError(err_msg)

    def add_wavelength(self, wave, spec_id=None, stype="science+standard"):
        """
        Three combinations of wave and spec_id shapes are accepted.

        +-----------+-----------------+
        | Parameter |       Size      |
        +-----------+-----+-----+-----+
        | wave      |  1  |  1  |  N  |
        +-----------+-----+-----+-----+
        | spec_id   |  1  |  N  |  N  |
        +-----------+-----+-----+-----+

        Parameters
        ----------
        wave : numeric value, list or numpy 1D array (N)
            The wavelength of each pixels of the spectrum.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if type(wave) == np.ndarray:

            wave = [wave]

        elif type(wave) == list:

            pass

        else:

            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_data_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(wave) == len(spec_id):

                    wave = {spec_id[i]: wave[i] for i in range(len(spec_id))}

                elif len(wave) == 1:

                    wave = {spec_id[i]: wave[0] for i in range(len(spec_id))}

                else:

                    error_msg = (
                        "wave must be the same length of shape "
                        + "as spec_id."
                    )
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

                for i in spec_id:

                    if len(wave[i]) == len(
                        self.science_spectrum_list[i].count
                    ):

                        self.science_spectrum_list[i].add_wavelength(
                            wave=wave[i]
                        )
                        self.logger.info(
                            "Added wavelength list to the "
                            "science_spectrum_list for spec_id: {}.".format(i)
                        )

                    else:

                        err_msg = (
                            "The wavelength provided has a different "
                            + "size to that of the extracted science spectrum."
                        )
                        self.logger.critical(err_msg)
                        raise RuntimeError(err_msg)

                self.science_wavelength_calibrated = True

            else:

                err_msg = (
                    "science data is not available, wavelength "
                    + "cannot be added."
                )
                self.logger.critical(err_msg)
                raise RuntimeError(err_msg)

        if "standard" in stype_split:

            if self.standard_data_available:

                if len(wave[0]) == len(self.standard_spectrum_list[0].count):

                    self.standard_spectrum_list[0].add_wavelength(wave=wave[0])

                else:

                    err_msg = (
                        "The wavelength provided is of a different "
                        + "size to that of the extracted standard spectrum."
                    )
                    self.logger.critical(err_msg)
                    raise RuntimeError(err_msg)

                self.standard_wavelength_calibrated = True

            else:

                err_msg = (
                    "standard data is not available, wavelength "
                    + "cannot be added."
                )
                self.logger.critical(err_msg)
                raise RuntimeError(err_msg)

    def add_wavelength_resampled(
        self, wave_resampled, spec_id=None, stype="science+standard"
    ):
        """
        Three combinations of wave and spec_id shapes are accepted.

        +-----------+-----------------+
        | Parameter |       Size      |
        +-----------+-----+-----+-----+
        | wave      |  1  |  1  |  N  |
        +-----------+-----+-----+-----+
        | spec_id   |  1  |  N  |  N  |
        +-----------+-----+-----+-----+

        Parameters
        ----------
        wave_resampled:
            The wavelength of the resampled spectrum.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if type(wave_resampled) == np.ndarray:

            wave_resampled = [wave_resampled]

        elif type(wave_resampled) == list:

            pass

        else:

            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_data_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(wave_resampled) == len(spec_id):

                    wave_resampled = {
                        spec_id[i]: wave_resampled[i]
                        for i in range(len(spec_id))
                    }

                elif len(wave_resampled) == 1:

                    wave_resampled = {
                        spec_id[i]: wave_resampled[0]
                        for i in range(len(spec_id))
                    }

                else:

                    error_msg = (
                        "wave must be the same length of shape "
                        + "as spec_id."
                    )
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

                for i in spec_id:

                    if len(wave_resampled[i]) == len(
                        self.science_spectrum_list[i].count
                    ):

                        self.science_spectrum_list[i].add_wavelength_resampled(
                            wave_resampled=wave_resampled[i]
                        )
                        self.logger.info(
                            "Added wavelength_resampled list to "
                            "the science_spectrum_list for spec_id: "
                            "{}.".format(i)
                        )

                    else:

                        err_msg = (
                            "The wavelength provided has a different "
                            + "size to that of the extracted science spectrum."
                        )
                        self.logger.critical(err_msg)
                        raise RuntimeError(err_msg)

                self.science_wavelength_resampled = True

            else:

                err_msg = (
                    "science data is not available, "
                    + "wavelength_resampled cannot be added."
                )
                self.logger.critical(err_msg)
                raise RuntimeError(err_msg)

        if "standard" in stype_split:

            if self.standard_data_available:

                if len(wave_resampled[0]) == len(
                    self.standard_spectrum_list[0].count
                ):

                    self.standard_spectrum_list[0].add_wavelength_resampled(
                        wave_resampled=wave_resampled[0]
                    )
                    self.logger.info(
                        "Added wavelength list to the "
                        "standard_spectrum_list."
                    )

                else:

                    err_msg = (
                        "The wavelength provided is of a different "
                        + "size to that of the extracted standard spectrum."
                    )
                    self.logger.critical(err_msg)
                    raise RuntimeError(err_msg)

                self.standard_wavelength_resampled_calibrated = True

            else:

                err_msg = (
                    "standard data is not available, "
                    + "wavelength_resampled cannot be added."
                )
                self.logger.critical(err_msg)
                raise RuntimeError(err_msg)

    def add_spec(
        self,
        count,
        spec_id=None,
        count_err=None,
        count_sky=None,
        stype="science+standard",
    ):
        """
        Parameters
        ----------
        count: 1-d array
            The summed count at each column about the trace.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        count_err: 1-d array (Default: None)
            the uncertainties of the count values
        count_sky: 1-d array (Default: None)
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if type(count) == np.ndarray:

            count = [count]

        elif type(count) == list:

            pass

        else:

            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        if count_err is not None:

            if type(count_err) == np.ndarray:

                count_err = [count_err]

            elif type(count_err) == list:

                pass

            else:

                err_msg = "Please provide a numpy array or a list of them."
                self.logger.critical(err_msg)
                raise TypeError(err_msg)

        else:

            count_err = [None]

        if count_sky is not None:

            if type(count_sky) == np.ndarray:

                count_sky = [count_sky]

            elif type(count_sky) == list:

                pass

            else:

                err_msg = "Please provide a numpy array or a list of them."
                self.logger.critical(err_msg)
                raise TypeError(err_msg)

        else:

            count_sky = [None]

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    for i in spec_id:

                        if i not in list(self.science_spectrum_list.keys()):

                            self.add_science_spectrum1D(i)

                            self.logger.warning(
                                "The given spec_id, {}, does not exist. A new "
                                "spectrum1D is created. Please check you are "
                                "providing the correct spec_id.".format(
                                    spec_id
                                )
                            )

                        else:

                            pass

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            # Check the sizes of the count and spec_id and convert count
            # into a dictionary
            if len(count) == len(spec_id):

                count = {spec_id[i]: count[i] for i in range(len(spec_id))}

            elif len(count) == 1:

                count = {spec_id[i]: count[0] for i in range(len(spec_id))}

            else:

                error_msg = (
                    "count must be the same length of shape "
                    + "as spec_id, or of size 1."
                )
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            # Check the sizes of the count_sky and spec_id and convert
            # count_sky into a dictionary
            if count_sky is [None]:

                count_sky = {spec_id[i]: None for i in range(len(spec_id))}

            elif len(count_sky) == len(spec_id):

                count_sky = {
                    spec_id[i]: count_sky[i] for i in range(len(spec_id))
                }

            elif len(count_sky) == 1:

                count_sky = {
                    spec_id[i]: count_sky[0] for i in range(len(spec_id))
                }

            else:

                error_msg = (
                    "count_sky must be the same length of shape "
                    + "as spec_id, or of size 1."
                )
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            # Check the sizes of the count_err and spec_id and convert
            # count_err into a dictionary
            if count_err is [None]:

                count_err = {spec_id[i]: None for i in range(len(spec_id))}

            elif len(count_err) == len(spec_id):

                count_err = {
                    spec_id[i]: count_err[i] for i in range(len(spec_id))
                }

            elif len(count_err) == 1:

                count_err = {
                    spec_id[i]: count_err[0] for i in range(len(spec_id))
                }

            else:

                error_msg = (
                    "count_err must be the same length of shape "
                    + "as spec_id, or of size 1."
                )
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            for i in spec_id:

                self.science_spectrum_list[i].add_count(
                    count=count[i],
                    count_err=count_err[i],
                    count_sky=count_sky[i],
                )
                self.logger.info(
                    "Added count, count_err, and count_sky to"
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

            self.science_data_available = True

        if "standard" in stype_split:

            self.standard_spectrum_list[0].add_count(
                count=count[0], count_err=count_err[0], count_sky=count_sky[0]
            )
            self.logger.info(
                "Added count, count_err, and count_sky to "
                "standard_spectrum_list."
            )

            self.standard_data_available = True

    def add_arc_spec(self, arc_spec, spec_id=None, stype="science+standard"):
        """
        Parameters
        ----------
        arc_spec: 1-d array
            The count of the summed 1D arc spec
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """
        if isinstance(arc_spec, np.ndarray):

            arc_spec = [arc_spec]

        elif isinstance(arc_spec, list):

            pass

        else:

            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    for i in spec_id:

                        if i not in list(self.science_spectrum_list.keys()):

                            self.add_science_spectrum1D(i)

                            self.logger.warning(
                                "The given spec_id, {}, does not "
                                "exist. A new spectrum1D is created. "
                                "Please check you are providing the "
                                "correct spec_id.".format(spec_id)
                            )

                        else:

                            pass

                else:

                    pass

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            # Check the sizes of the wave and spec_id and convert wave
            # into a dictionary
            if len(arc_spec) == len(spec_id):

                arc_spec = {
                    spec_id[i]: arc_spec[i] for i in range(len(spec_id))
                }

            elif len(arc_spec) == 1:

                arc_spec = {
                    spec_id[i]: arc_spec[0] for i in range(len(spec_id))
                }

            else:

                error_msg = (
                    "arc_spec must be the same length or shape as spec_id. "
                    + "arc_spec has shape {} and ".format(np.shape(arc_spec))
                    + "spec_id has shape {}.".format(np.shape(spec_id))
                )
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

            for i in spec_id:

                self.science_spectrum_list[i].add_arc_spec(
                    arc_spec=arc_spec[i]
                )
                self.logger.info(
                    "Added arc_spec to"
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

            self.science_arc_spec_available = True

        if "standard" in stype_split:

            self.standard_spectrum_list[0].add_arc_spec(arc_spec=arc_spec[0])
            self.logger.info("Added arc_spec to" "standard_spectrum_list.")

            self.standard_arc_spec_available = True

    def add_arc_lines(self, peaks, spec_id=None, stype="science+standard"):
        """
        Parameters
        ----------
        peaks: list of list or list of arrays
            The pixel locations of the arc lines. Multiple traces of the arc
            can be provided as list of list or list of arrays.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if type(peaks) == np.ndarray:

            peaks = [peaks]

        elif type(peaks) == list:

            pass

        else:

            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    for i in spec_id:

                        if i not in list(self.science_spectrum_list.keys()):

                            self.add_science_spectrum1D(i)

                            self.logger.warning(
                                "The given spec_id, {}, does not "
                                "exist. A new spectrum1D is created. "
                                "Please check you are providing the "
                                "correct spec_id.".format(spec_id)
                            )

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            # Check the sizes of the wave and spec_id and convert wave
            # into a dictionary
            if len(peaks) == len(spec_id):

                peaks = {spec_id[i]: peaks[i] for i in range(len(spec_id))}

            elif len(peaks) == 1:

                peaks = {i: peaks[0] for i in spec_id}

            else:

                error_msg = (
                    "peaks must be the same length of shape " + "as spec_id."
                )
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            for i in spec_id:

                self.science_spectrum_list[i].add_peaks(peaks=peaks[i])
                self.logger.info(
                    "Added peaks to"
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

            self.science_arc_lines_available = True

        if "standard" in stype_split:

            self.standard_spectrum_list[0].add_peaks(peaks=peaks[0])
            self.logger.info("Added peaks to standard_spectrum_list.")

            self.standard_arc_lines_available = True

    def add_trace(
        self,
        trace,
        trace_sigma,
        spec_id=None,
        pixel_list=None,
        stype="science+standard",
    ):
        """
        Parameters
        ----------
        trace: list or numpy.ndarray (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: list or numpy.ndarray (N)
            Standard deviation of the Gaussian profile of a trace
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        pixel_list: list or numpy.narray (Default: None)
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(len(spec.trace[0])), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if type(trace) == np.ndarray:

            trace = [trace]

        elif type(trace) == list:

            pass

        else:

            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        if type(trace_sigma) == np.ndarray:

            trace_sigma = [trace_sigma]

        elif type(trace_sigma) == list:

            pass

        else:

            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    for i in spec_id:

                        if i not in list(self.science_spectrum_list.keys()):

                            self.add_science_spectrum1D(i)

                            self.logger.warning(
                                "The given spec_id, {}, does not "
                                "exist. A new spectrum1D is created. "
                                "Please check you are providing the "
                                "correct spec_id.".format(spec_id)
                            )

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            # Check the sizes of the wave and spec_id and convert wave
            # into a dictionary
            if len(trace) == len(spec_id):

                trace = {spec_id[i]: trace[i] for i in range(len(spec_id))}

            elif len(trace) == 1:

                trace = {spec_id[i]: trace[0] for i in range(len(spec_id))}

            else:

                error_msg = (
                    "trace must be the same length of shape " + "as spec_id."
                )
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            # Check the sizes of the wave and spec_id and convert wave
            # into a dictionary
            if len(trace_sigma) == len(spec_id):

                trace_sigma = {
                    spec_id[i]: trace_sigma[i] for i in range(len(spec_id))
                }

            elif len(trace_sigma) == 1:

                trace_sigma = {
                    spec_id[i]: trace_sigma[0] for i in range(len(spec_id))
                }

            else:

                error_msg = (
                    "wave must be the same length of shape " + "as spec_id."
                )
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

            for i in spec_id:

                self.science_spectrum_list[i].add_trace(
                    trace=trace[i],
                    trace_sigma=trace_sigma[i],
                    pixel_list=pixel_list,
                )
                self.logger.info(
                    "Added trace, trace_sigma, and pixel_list to"
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

            self.science_trace_available = True

        if "standard" in stype_split:

            self.standard_spectrum_list[0].add_trace(
                trace=trace[0],
                trace_sigma=trace_sigma[0],
                pixel_list=pixel_list,
            )
            self.logger.info(
                "Added trace, trace_sigma, and pixel_list to"
                "standard_spectrum_list"
            )

            self.standard_trace_available = True

    def add_fit_coeff(
        self,
        fit_coeff,
        fit_type="poly",
        spec_id=None,
        stype="science+standard",
    ):
        """
        Parameters
        ----------
        fit_coeff: list or numpy array, or a list of them
            Polynomial fit coefficients.
        fit_type: str or list of str
            Strings starting with 'poly', 'leg' or 'cheb' for polynomial,
            legendre and chebyshev fits. Case insensitive.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if type(fit_coeff) == np.ndarray:

            fit_coeff = [fit_coeff]

        elif all(isinstance(i, list) for i in fit_coeff):

            pass

        elif isinstance(fit_coeff, list):

            if isinstance(fit_coeff[0], (list, np.ndarray)):

                pass

            elif isinstance(fit_coeff[0], (int, float)):

                fit_coeff = [fit_coeff]

            else:

                pass

        elif all(isinstance(i, np.ndarray) for i in fit_coeff):

            pass

        else:

            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        if type(fit_type) == str:

            fit_type = [fit_type]

        elif all(isinstance(i, (str, list, np.ndarray)) for i in fit_type):

            for i, ft in enumerate(fit_type):

                if isinstance(ft, (list, np.ndarray)):

                    fit_type[i] = ft[0]

        else:

            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_data_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(fit_coeff) == len(spec_id):

                    fit_coeff = {
                        spec_id[i]: fit_coeff[i] for i in range(len(spec_id))
                    }

                elif len(fit_coeff) == 1:

                    fit_coeff = {
                        spec_id[i]: fit_coeff[0] for i in range(len(spec_id))
                    }

                else:

                    error_msg = (
                        "fit_coeff must be the same length of "
                        + "shape as spec_id."
                    )
                    self.logger.critical(error_msg)
                    raise RuntimeError(error_msg)

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(fit_type) == len(spec_id):

                    fit_type = {
                        spec_id[i]: fit_type[i] for i in range(len(spec_id))
                    }

                elif len(fit_type) == 1:

                    fit_type = {
                        spec_id[i]: fit_type[0] for i in range(len(spec_id))
                    }

                else:

                    error_msg = (
                        "wave must be the same length of shape "
                        + "as spec_id."
                    )
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

                for i in spec_id:

                    self.science_spectrum_list[i].add_fit_coeff(
                        fit_coeff=fit_coeff[i]
                    )
                    self.science_spectrum_list[i].add_fit_type(
                        fit_type=fit_type[i]
                    )
                    self.logger.info(
                        "Added fit_coeff and fit_type to"
                        "science_spectrum_list for spec_id: {}.".format(i)
                    )

            self.science_wavecal_polynomial_available = True

        if "standard" in stype_split:

            if self.standard_data_available:

                self.standard_spectrum_list[0].add_fit_coeff(
                    fit_coeff=fit_coeff[0]
                )
                self.standard_spectrum_list[0].add_fit_type(
                    fit_type=fit_type[0]
                )
                self.logger.info(
                    "Added fit_coeff and fit_type to" "standard_spectrum_list."
                )

            self.standard_wavecal_polynomial_available = True

    def from_twodspec(self, twodspec, spec_id=None, stype="science+standard"):
        """
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
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(twodspec.spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(twodspec.spectrum_list.keys())

            # reference the spectrum1D to the WavelengthCalibration
            for i in spec_id:

                self.add_science_spectrum1D(i)
                self.science_wavecal[i] = WavelengthCalibration(
                    verbose=self.verbose,
                    logger_name=self.logger_name,
                    log_level=self.log_level,
                    log_file_folder=self.log_file_folder,
                    log_file_name=self.log_file_name,
                )

                # By reference
                self.science_wavecal[i].from_spectrum1D(
                    twodspec.spectrum_list[i]
                )
                self.science_spectrum_list[i] = self.science_wavecal[
                    i
                ].spectrum1D

                self.logger.info(
                    "Referenced Spectrum1D of the"
                    "science_spectrum_list for spec_id: {}.".format(i)
                    + "to the corresponding science_wavecal."
                )

            self.science_data_available = True
            self.science_arc_available = True
            self.science_arc_spec_available = True

        if "standard" in stype_split:

            # By reference
            self.standard_wavecal = WavelengthCalibration(
                verbose=self.verbose,
                logger_name=self.logger_name,
                log_level=self.log_level,
                log_file_folder=self.log_file_folder,
                log_file_name=self.log_file_name,
            )
            self.standard_wavecal.from_spectrum1D(twodspec.spectrum_list[0])
            self.fluxcal.from_spectrum1D(twodspec.spectrum_list[0])
            self.standard_spectrum_list[0] = self.standard_wavecal.spectrum1D

            self.logger.info(
                "Referenced Spectrum1D of the"
                "standard_spectrum_list to the standard_wavecal."
            )

            self.standard_data_available = True
            self.standard_arc_available = True
            self.standard_arc_spec_available = True

    def find_arc_lines(
        self,
        spec_id=None,
        prominence=5.0,
        top_n_peaks=None,
        distance=5.0,
        refine=False,
        refine_window_width=5,
        display=False,
        width=1280,
        height=720,
        return_jsonstring=False,
        renderer="default",
        save_fig=False,
        fig_type="iframe+png",
        filename=None,
        open_iframe=False,
        stype="science+standard",
    ):
        """
        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        background: int or None (Default: None)
            User-supplied estimated background level
        percentile: float (Default: 2.)
            The percentile of the flux to be used as the estimate of the
            background sky level to the first order. Only used if background
            is None. [Count]
        prominence: float (Default: 5.)
            The minimum prominence to be considered as a peak (normalised)
        distance: float (Default: 5.)
            Minimum separation between peaks
        refine: bool (Default: True)
            Set to true to fit a gaussian to get the peak at sub-pixel
            precision
        refine_window_width: int or float (Default: 5)
            The number of pixels (on each side of the existing peaks) to be
            fitted with gaussian profiles over.
        display: bool (Default: False)
            Set to True to display disgnostic plot.
        renderer: str (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool (Default: False)
            set to True to return  JSON-string that can be rendered by Plotly
            in any support language.
        renderer: str (Default: 'default')
            plotly renderer options.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        Returns
        -------
        JSON strings if return_jsonstring is set to True

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_arc_spec_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    print(list(self.science_spectrum_list.keys()))
                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].find_arc_lines(
                        prominence=prominence,
                        top_n_peaks=top_n_peaks,
                        distance=distance,
                        refine=refine,
                        refine_window_width=refine_window_width,
                        display=display,
                        renderer=renderer,
                        width=width,
                        height=height,
                        return_jsonstring=return_jsonstring,
                        save_fig=save_fig,
                        fig_type=fig_type,
                        filename=filename,
                        open_iframe=open_iframe,
                    )

                    n_peaks = len(self.science_spectrum_list[i].peaks)
                    self.logger.info(
                        "{} arc lines are found in ".format(n_peaks)
                        + "science_spectrum_list for spec_id: {}.".format(i)
                    )

                self.science_arc_lines_available = True

            else:

                self.logger.warning("Science arc spectrum/a are not imported.")

        if "standard" in stype_split:

            if self.standard_arc_spec_available:

                self.standard_wavecal.find_arc_lines(
                    prominence=prominence,
                    top_n_peaks=top_n_peaks,
                    distance=distance,
                    refine=refine,
                    refine_window_width=refine_window_width,
                    display=display,
                    renderer=renderer,
                    width=width,
                    height=height,
                    return_jsonstring=return_jsonstring,
                    save_fig=save_fig,
                    fig_type=fig_type,
                    filename=filename,
                    open_iframe=open_iframe,
                )

                n_peaks = len(self.standard_spectrum_list[0].peaks)
                self.logger.info(
                    "{} arc lines are found in ".format(n_peaks)
                    + "standard_spectrum_list."
                )

                self.standard_arc_lines_available = True

            else:

                self.logger.warning(
                    "Standard arc spectrum/a are not imported."
                )

    def initialise_calibrator(
        self, spec_id=None, peaks=None, arc_spec=None, stype="science+standard"
    ):
        """
        If the peaks were found with find_arc_lines(), peaks and spectrum can
        be None.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        peaks: list, numpy.ndarray or None (Default: None)
            The pixel values of the peaks (start from zero)
        spectrum: list, numpy.ndarray or None (Default: None)
            The spectral intensity as a function of pixel.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    for i in spec_id:

                        if i not in list(self.science_spectrum_list.keys()):

                            self.add_science_spectrum1D(i)

                            self.logger.warning(
                                "The given spec_id, {}, does not "
                                "exist. A new spectrum1D is created. "
                                "Please check you are providing the "
                                "correct spec_id.".format(spec_id)
                            )

                        else:

                            pass

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_wavecal[i].from_spectrum1D(
                    self.science_spectrum_list[i]
                )
                self.science_wavecal[i].initialise_calibrator(
                    peaks=peaks, arc_spec=arc_spec
                )
                self.science_wavecal[i].set_calibrator_properties()
                self.science_wavecal[i].set_hough_properties()
                self.science_wavecal[i].set_ransac_properties()

                self.logger.info(
                    "Calibrator is initialised for "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_wavecal.from_spectrum1D(
                self.standard_spectrum_list[0]
            )
            self.standard_wavecal.initialise_calibrator(
                peaks=peaks, arc_spec=arc_spec
            )
            self.standard_wavecal.set_calibrator_properties()
            self.standard_wavecal.set_hough_properties()
            self.standard_wavecal.set_ransac_properties()

            self.logger.info(
                "Calibrator is initialised for the " "standard_spectrum_list."
            )

    def set_calibrator_properties(
        self,
        spec_id=None,
        num_pix=None,
        pixel_list=None,
        plotting_library="plotly",
        logger_name="Calibrator",
        log_level="info",
        stype="science+standard",
    ):
        """
        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        num_pix: int (Default: None)
            The number of pixels in the dispersion direction
        pixel_list: list or numpy array (Default: None)
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(num_pix), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        plotting_library : str (Default: 'plotly')
            Choose between matplotlib and plotly.
        logger_name: str (Default: 'Calibrator')
            This will set the name of the logger, if the name is used already,
            it will reference to the existing logger. This will be the
            first part of the default log file name unless log_file_name is
            provided.
        log_level : str (Default: 'info')
            Choose {critical, error, warning, info, debug, notset}.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_wavecal[i].set_calibrator_properties(
                    num_pix=num_pix,
                    pixel_list=pixel_list,
                    plotting_library=plotting_library,
                    logger_name=logger_name,
                    log_level=log_level,
                )
                self.logger.info(
                    "Calibrator properties are set for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_wavecal.set_calibrator_properties(
                num_pix=num_pix,
                pixel_list=pixel_list,
                plotting_library=plotting_library,
                logger_name=logger_name,
                log_level=log_level,
            )
            self.logger.info(
                "Calibrator properties are set for the "
                "standard_spectrum_list."
            )

    def set_hough_properties(
        self,
        spec_id=None,
        num_slopes=5000,
        xbins=200,
        ybins=200,
        min_wavelength=3000.0,
        max_wavelength=10000.0,
        range_tolerance=500,
        linearity_tolerance=100,
        stype="science+standard",
    ):
        """
        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        num_slopes: int (Default: 5000)
            Number of slopes to consider during Hough transform
        xbins: int (Default: 200)
            Number of bins for Hough accumulation
        ybins: int (Default: 200)
            Number of bins for Hough accumulation
        min_wavelength: float (Default: 3000.)
            Minimum wavelength of the spectrum.
        max_wavelength: float (Default: 10000.)
            Maximum wavelength of the spectrum.
        range_tolerance: float (Default: 500)
            Estimation of the error on the provided spectral range
            e.g. 3000-5000 with tolerance 500 will search for
            solutions that may satisfy 2500-5500
        linearity_tolerance: float (Default: 100)
            A toleranceold (Ansgtroms) which defines some padding around the
            range tolerance to allow for non-linearity. This should be the
            maximum expected excursion from linearity.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_wavecal[i].set_hough_properties(
                    num_slopes=num_slopes,
                    xbins=xbins,
                    ybins=ybins,
                    min_wavelength=min_wavelength,
                    max_wavelength=max_wavelength,
                    range_tolerance=range_tolerance,
                    linearity_tolerance=linearity_tolerance,
                )
                self.logger.info(
                    "Hough properties are set for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_wavecal.set_hough_properties(
                num_slopes=num_slopes,
                xbins=xbins,
                ybins=ybins,
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                range_tolerance=range_tolerance,
                linearity_tolerance=linearity_tolerance,
            )
            self.logger.info(
                "Hough properties are set for the " "standard_spectrum_list."
            )

    def set_ransac_properties(
        self,
        spec_id=None,
        sample_size=5,
        top_n_candidate=5,
        linear=True,
        filter_close=False,
        ransac_tolerance=5,
        candidate_weighted=True,
        hough_weight=1.0,
        minimum_matches=3,
        minimum_peak_utilisation=0.0,
        minimum_fit_error=1e-4,
        stype="science+standard",
    ):
        """
        Configure the Calibrator. This may require some manual twiddling before
        the calibrator can work efficiently. However, in theory, a large
        max_tries in fit() should provide a good solution in the expense of
        performance (minutes instead of seconds).

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        sample_size: int (Default: 5)
            Number of pixel-wavelength hough pairs to be used for each arc line
            being picked.
        top_n_candidate: int (Default: 5)
            Top ranked lines to be fitted.
        linear: bool (Default: True)
            True to use the hough transformed gradient, otherwise, use the
            known polynomial.
        filter_close: bool (Default: False)
            Remove the pairs that are out of bounds in the hough space.
        ransac_tolerance: float (Default: 5)
            The distance criteria  (Angstroms) to be considered an inlier to a
            fit. This should be close to the size of the expected residuals on
            the final fit (e.g. 1A is typical)
        candidate_weighted: bool (Default: True)
            Set to True to down-weight pairs that are far from the fit.
        hough_weight: float or None (Default: 1.0)
            Set to use the hough space to weigh the fit. The theoretical
            optimal weighting is unclear. The larger the value, the heavily it
            relies on the overdensity in the hough space for a good fit.
        minimum_matches: int (Default: 3)
            Minimum number of fitted peaks to accept as a solution. This has
            to be smaller than or equal to the sample size.
        minimum_peak_utilisation: float (Default: 0.)
            The minimum percentage of peaks used in order to accept as a
            valid solution.
        minimum_fit_error: float (Default 1e-4)
            Set to remove overfitted/unrealistic fits.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_wavecal[i].set_ransac_properties(
                    sample_size=sample_size,
                    top_n_candidate=top_n_candidate,
                    linear=linear,
                    filter_close=filter_close,
                    ransac_tolerance=ransac_tolerance,
                    candidate_weighted=candidate_weighted,
                    hough_weight=hough_weight,
                    minimum_matches=minimum_matches,
                    minimum_peak_utilisation=minimum_peak_utilisation,
                    minimum_fit_error=minimum_fit_error,
                )
                self.logger.info(
                    "Ransac properties are set for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_wavecal.set_ransac_properties(
                sample_size=sample_size,
                top_n_candidate=top_n_candidate,
                linear=linear,
                filter_close=filter_close,
                ransac_tolerance=ransac_tolerance,
                candidate_weighted=candidate_weighted,
                hough_weight=hough_weight,
                minimum_matches=minimum_matches,
                minimum_peak_utilisation=minimum_peak_utilisation,
                minimum_fit_error=minimum_fit_error,
            )
            self.logger.info(
                "Ransac properties are set for the " "standard_spectrum_list."
            )

    def set_known_pairs(
        self, pix=None, wave=None, spec_id=None, stype="science+standard"
    ):
        """
        Parameters
        ----------
        pix : numeric value, list or numpy 1D array (N) (Default: None)
            Any pixel value, can be outside the detector chip and
            serve purely as anchor points.
        wave : numeric value, list or numpy 1D array (N) (Default: None)
            The matching wavelength for each of the pix.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_wavecal[i].set_known_pairs(pix=pix, wave=wave)
                self.logger.info(
                    "Known pixel-wavelength pairs are added to "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_wavecal.set_known_pairs(pix=pix, wave=wave)
            self.logger.info(
                "Known pixel-wavelength pairs are added to "
                "standard_spectrum_list."
            )

    def add_user_atlas(
        self,
        elements,
        wavelengths,
        spec_id=None,
        intensities=None,
        candidate_tolerance=10.0,
        constrain_poly=False,
        vacuum=False,
        pressure=101325.0,
        temperature=273.15,
        relative_humidity=0.0,
        stype="science+standard",
    ):
        """
        Append the user supplied arc lines to the calibrator.

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
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        intensities : list or None (Default: None)
            Relative line intensities
        candidate_tolerance: float (Default: 10.)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: bool (Default: False)
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: bool (Default: False)
            Set to true to convert the input wavelength to air-wavelengths
            based on the given pressure, temperature and humidity.
        pressure: float (Default: 101325.)
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float (Default: 273.15)
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float (Default: 0.)
            In percentage.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if pressure is None:
            pressure = 101325.0
            self.logger.warning(
                "Pressure is not provided, set to 1 unit of "
                "standard atmosphere."
            )

        if temperature is None:
            temperature = 273.15
            self.logger.warning(
                "Temperature is not provided, set to 0 degrees " "Celsius."
            )

        if relative_humidity is None:
            relative_humidity = 0.0
            self.logger.warning(
                "Relative humidity is not provided, set to 0%."
            )

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_wavecal[i].add_user_atlas(
                    elements=elements,
                    wavelengths=wavelengths,
                    intensities=intensities,
                    candidate_tolerance=candidate_tolerance,
                    constrain_poly=constrain_poly,
                    vacuum=vacuum,
                    pressure=pressure,
                    temperature=temperature,
                    relative_humidity=relative_humidity,
                )
                self.logger.info(
                    "Added user supplied atlas to "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

            self.science_atlas_available = True

        if "standard" in stype_split:

            self.standard_wavecal.add_user_atlas(
                elements=elements,
                wavelengths=wavelengths,
                intensities=intensities,
                candidate_tolerance=candidate_tolerance,
                constrain_poly=constrain_poly,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
            )
            self.logger.info(
                "Added user supplied atlas to " "standard_spectrum_list."
            )

            self.standard_atlas_available = True

    def add_atlas(
        self,
        elements,
        spec_id=None,
        min_atlas_wavelength=3000.0,
        max_atlas_wavelength=10000.0,
        min_intensity=10.0,
        min_distance=10.0,
        candidate_tolerance=10.0,
        constrain_poly=False,
        vacuum=False,
        pressure=101325.0,
        temperature=273.15,
        relative_humidity=0,
        stype="science+standard",
    ):
        """
        Parameters
        ----------
        elements: str or list of strings
            Chemical symbol, case insensitive
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        min_atlas_wavelength: float (Default: 3000.)
            Minimum wavelength of the arc lines.
        max_atlas_wavelength: float (Default: 10000.)
            Maximum wavelength of the arc lines.
        min_intensity: float (Default: 10.)
            Minimum intensity of the arc lines. Refer to NIST for the
            intensity.
        min_distance: float (Default: 10.)
            Minimum separation between neighbouring arc lines.
        candidate_tolerance: float (Default: 10.)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: bool (Default: False)
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: bool (Default: False)
            Set to true to convert the input wavelength to air-wavelengths
            based on the given pressure, temperature and humidity.
        pressure: float (Default: 101325.)
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float (Default: 273.15)
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float (Default: 0)
            In percentage.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

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
                    relative_humidity=relative_humidity,
                )
                self.logger.info(
                    "Added atlas to "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

            self.science_atlas_available = True

        if "standard" in stype_split:

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
                relative_humidity=relative_humidity,
            )
            self.logger.info(
                "Added atlas to "
                "standard_spectrum_list for spec_id: {}.".format(i)
            )

            self.standard_atlas_available = True

    def remove_atlas_lines_range(
        self,
        wavelength,
        tolerance=10.0,
        spec_id=None,
        stype="science+standard",
    ):
        """
        Remove arc lines within a certain wavelength range.

        Parameters
        ----------
        wavelength: float
            Wavelength to remove (Angstrom)
        tolerance: float (Default: 10.)
            Tolerance around this wavelength where atlas lines will be removed
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_atlas_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].remove_atlas_lines_range(
                        wavelength, tolerance
                    )
                    self.logger.info(
                        "Remove atlas in the range of "
                        "{} +/- {}".format(wavelength, tolerance)
                        + "science_spectrum_list for spec_id: {}.".format(i)
                    )

            else:

                self.logger.warning("Science atlas is not available.")

        if "standard" in stype_split:

            if self.standard_atlas_available:

                self.standard_wavecal.remove_atlas_lines_range(
                    wavelength, tolerance
                )
                self.logger.info(
                    "Remove atlas in the range of "
                    "{} +/- {}".format(wavelength, tolerance)
                    + "standard_spectrum_list."
                )

            else:

                self.logger.warning("Standard atlas is not available.")

    def clear_atlas(self, spec_id=None, stype="science+standard"):
        """
        Remove all the atlas lines from the calibrator.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_atlas_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].clear_atlas()
                    self.logger.info(
                        "Atlas is removed from "
                        "science_spectrum_list for spec_id: {}.".format(i)
                    )

                self.science_atlas_available = False

            else:

                self.logger.warning("Science atlas is not available.")

        if "standard" in stype_split:

            if self.standard_atlas_available:

                self.standard_wavecal.clear_atlas()
                self.logger.info(
                    "Atlas is removed from standard_spectrum_list."
                )

                self.standard_atlas_available = False

            else:

                self.logger.warning("Standard atlas is not available.")

    def list_atlas(self, spec_id=None, stype="science+standard"):
        """
        Remove all the atlas lines from the calibrator.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_atlas_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].list_atlas()
                    self.logger.info(
                        "Listing the atlas of "
                        "science_spectrum_list for spec_id: {}.".format(i)
                    )

            else:

                self.logger.warning("Science atlas is not available.")

        if "standard" in stype_split:

            if self.standard_atlas_available:

                self.standard_wavecal.list_atlas()
                self.logger.info(
                    "Listing the atlas of " "standard_spectrum_list."
                )

            else:

                self.logger.warning("Standard atlas is not available.")

    def do_hough_transform(
        self, spec_id=None, brute_force=False, stype="science+standard"
    ):
        """
        ** brute_force is EXPERIMENTAL as of 1 Oct 2021 **
        The brute force method is supposed to provide all the possible
        solution, hence given a sufficiently large max_tries, the solution
        should always be the best possible outcome. However, it does not
        seem to work in a small fraction of our tests. Use with caution,
        and it is not the recommended way for now.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        brute_force: bool (Default: False)
            Set to true to compute the gradient and intercept between
            every two data points
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_wavecal[i].do_hough_transform(
                    brute_force=brute_force
                )
                self.logger.info(
                    "Hough Transform is performed on "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

            self.science_hough_pairs_available = True

        if "standard" in stype_split:

            self.standard_wavecal.do_hough_transform(brute_force=brute_force)
            self.logger.info(
                "Hough Transform is performed on " "standard_spectrum_list."
            )

            self.standard_hough_pairs_available = True

    def plot_search_space(
        self,
        spec_id=None,
        fit_coeff=None,
        top_n_candidate=3,
        weighted=True,
        save_fig=False,
        fig_type="iframe+png",
        filename=None,
        return_jsonstring=False,
        renderer="default",
        display=False,
        stype="science+standard",
    ):
        """
        A wrapper function to plot the search space in the Hough space.

        If fit fit_coefficients are provided, the model solution will be
        overplotted.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_hough_pairs_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].plot_search_space(
                        fit_coeff=fit_coeff,
                        top_n_candidate=top_n_candidate,
                        weighted=weighted,
                        save_fig=save_fig,
                        fig_type=fig_type,
                        filename=filename,
                        return_jsonstring=return_jsonstring,
                        renderer=renderer,
                        display=display,
                    )
                    self.logger.info(
                        "Search area of the Hough space is plotted for the "
                        "science_spectrum_list for spec_id: {}.".format(i)
                    )

            else:

                self.logger.warning("Science atlas is not available.")

        if "standard" in stype_split:

            if self.standard_hough_pairs_available:

                self.standard_wavecal.plot_search_space(
                    fit_coeff=fit_coeff,
                    top_n_candidate=top_n_candidate,
                    weighted=weighted,
                    save_fig=save_fig,
                    fig_type=fig_type,
                    filename=filename,
                    return_jsonstring=return_jsonstring,
                    renderer=renderer,
                    display=display,
                )
                self.logger.info(
                    "Search area of the Hough space is plotted for the "
                    "standard_spectrum_list."
                )

            else:

                self.logger.warning("Standard atlas is not available.")

    def fit(
        self,
        spec_id=None,
        max_tries=5000,
        fit_deg=4,
        fit_coeff=None,
        fit_tolerance=10.0,
        fit_type="poly",
        candidate_tolerance=2.0,
        brute_force=False,
        progress=True,
        return_solution=False,
        display=False,
        renderer="default",
        save_fig=False,
        fig_type="iframe+png",
        filename=None,
        stype="science+standard",
    ):
        """
        A wrapper function to perform wavelength calibration with RASCAL. As of
        14 January 2020, it supports He, Ne, Ar, Cu, Kr, Cd, Xe, Hg and Th from
        `NIST <https://physics.nist.gov/PhysRefData/ASD/lines_form.html>`_.

        Parameters
        ----------
        max_tries: int
            Number of trials of polynomial fitting.
        fit_deg: int (Default: 4)
            The degree of the polynomial to be fitted.
        fit_coeff: list (Default: None)
            *NOT CURRENTLY USED, as of 17 Jan 2021*
            Set the baseline of the least square fit. If no fits outform this
            set of polynomial coefficients, this will be used as the best fit.
        fit_tolerance: float (Default: 10)
            Sets a tolerance on whether a fit found by RANSAC is considered
            acceptable.
        fit_type: string (Default: 'poly')
            One of 'poly', 'legendre' or 'chebyshev'.
        candidate_tolerance: float (default: 2.0)
            toleranceold  (Angstroms) for considering a point to be an inlier
        brute_force: bool (Default: False)
            Set to True to try all possible combination in the given parameter
            space.
        progress: bool (Default: True)
            Set to show the progress using tdqm (if imported).
        return_jsonstring: (default: False)
            Set to True to save the plotly figure as json string.
        display: bool (Default: False)
            Set to show diagnostic plot.
        renderer: str (Default: 'default')
            plotly renderer options.
        save_fig: string (Default: False)
            Set to save figure.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        solution = {}

        if "science" in stype_split:

            if self.science_hough_pairs_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                solution_science = []

                for i in spec_id:

                    solution_science.append(
                        self.science_wavecal[i].fit(
                            max_tries=max_tries,
                            fit_deg=fit_deg,
                            fit_coeff=fit_coeff,
                            fit_tolerance=fit_tolerance,
                            fit_type=fit_type,
                            candidate_tolerance=candidate_tolerance,
                            brute_force=brute_force,
                            progress=progress,
                            display=display,
                            renderer=renderer,
                            save_fig=save_fig,
                            fig_type=fig_type,
                            filename=filename,
                            return_solution=return_solution,
                        )
                    )

                    self.logger.info(
                        "Wavelength solution is fitted for the "
                        "science_spectrum_list for spec_id: {}.".format(i)
                    )

                self.science_wavecal_polynomial_available = True
                solution["science"] = solution_science

            else:

                self.logger.warning("Science hough pairs are not available.")

        if "standard" in stype_split:

            if self.standard_hough_pairs_available:

                solution["standard"] = self.standard_wavecal.fit(
                    max_tries=max_tries,
                    fit_deg=fit_deg,
                    fit_coeff=fit_coeff,
                    fit_tolerance=fit_tolerance,
                    fit_type=fit_type,
                    candidate_tolerance=candidate_tolerance,
                    brute_force=brute_force,
                    progress=progress,
                    display=display,
                    renderer=renderer,
                    save_fig=save_fig,
                    fig_type=fig_type,
                    filename=filename,
                    return_solution=return_solution,
                )
                self.logger.info(
                    "Wavelength solution is fitted for the "
                    "standard_spectrum_list."
                )

                self.standard_wavecal_polynomial_available = True

            else:

                self.logger.warning("Standard spectrum/a are not imported.")

        if return_solution:

            return solution

    def robust_refit(
        self,
        spec_id=None,
        fit_coeff=None,
        n_delta=None,
        refine=False,
        tolerance=10.0,
        method="Nelder-Mead",
        convergence=1e-6,
        robust_refit=True,
        fit_deg=None,
        return_solution=False,
        display=False,
        renderer="default",
        save_fig=False,
        filename=None,
        stype="science+standard",
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **
        Refine the fitted solution with a minimisation method as provided by
        scipy.optimize.minimize().

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        fit_coeff: list or None (Default: None)
            List of polynomial fit coefficients.
        n_delta: int (Default: None)
            The number of the highest polynomial order to be adjusted
        refine: bool (Default: True)
            Set to True to refine solution.
        tolerance : float (Default: 10.)
            Absolute difference between fit and model in the unit of nm.
        method: str (Default: 'Nelder-Mead')
            scipy.optimize.minimize method.
        convergence: float (Default: 1e-6)
            scipy.optimize.minimize tol.
        robust_refit: bool (Default: True)
            Set to True to fit all the detected peaks with the given polynomial
            solution.
        fit_deg: int (Default: length of the input coefficients - 1)
            Order of polynomial fit with all the detected peaks.
        return_solution: bool (Default: True)
            Set to True to return the best fit polynomial coefficients.
        display: bool (Default: False)
            Set to show diagnostic plot.
        renderer: str (Default: 'default')
            plotly renderer options.
        save_fig: bool (Default: False)
            Set to save figure.
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        solution = {}

        if "science" in stype_split:

            if self.science_wavecal_polynomial_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)
                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                solution_science = []

                for i in spec_id:

                    if fit_coeff is None:

                        fit_coeff = self.science_wavecal[
                            i
                        ].spectrum1D.calibrator.fit_coeff

                    solution_science.append(
                        self.science_wavecal[i].robust_refit(
                            fit_coeff=fit_coeff,
                            n_delta=n_delta,
                            refine=refine,
                            tolerance=tolerance,
                            method=method,
                            convergence=convergence,
                            robust_refit=robust_refit,
                            fit_deg=fit_deg,
                            display=display,
                            renderer=renderer,
                            save_fig=save_fig,
                            filename=filename,
                            return_solution=return_solution,
                        )
                    )
                    self.logger.info(
                        "Wavelength solution is refined for the "
                        "science_spectrum_list for spec_id: {}.".format(i)
                    )

                solution["science"] = solution_science

            else:

                self.logger.warning("Science spectrum/a are not imported.")

        if "standard" in stype_split:

            if self.standard_wavecal_polynomial_available:

                if fit_coeff is None:

                    fit_coeff = self.standard_wavecal[
                        0
                    ].spectrum1D.calibrator.fit_coeff

                solution["standard"] = self.standard_wavecal.robust_refit(
                    fit_coeff=fit_coeff,
                    n_delta=n_delta,
                    refine=refine,
                    tolerance=tolerance,
                    method=method,
                    convergence=convergence,
                    robust_refit=robust_refit,
                    fit_deg=fit_deg,
                    display=display,
                    renderer=renderer,
                    save_fig=save_fig,
                    filename=filename,
                    return_solution=return_solution,
                )
                self.logger.info(
                    "Wavelength solution is refined for the "
                    "standard_spectrum_list."
                )

            else:

                self.logger.warning("Standard spectrum/a are not imported.")

        if return_solution:

            return solution

    def get_pix_wave_pairs(self, spec_id=None, stype="science+standard"):
        """
        Return the list of matched_peaks and matched_atlas with their
        position in the array.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        Return
        ------
        pw_pairs: dictionary
            Dictionary of 'science' and/or 'standard' where the values are
            lists of tuples each containing the array position, peak (pixel)
            and atlas (wavelength) in the order of the given spec_id.

        """

        stype_split = stype.split("+")

        pw_pairs = {}

        if "science" in stype_split:

            if self.science_wavecal_polynomial_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)
                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                pw_pairs_science = []

                for i in spec_id:

                    pw_pairs_science.append(
                        self.science_wavecal[i].get_pix_wave_pairs()
                    )

                pw_pairs["science"] = pw_pairs_science

        if "standard" in stype_split:

            if self.standard_wavecal_polynomial_available:

                pw_pairs_standard = self.standard_wavecal[
                    0
                ].get_pix_wave_pairs()

                pw_pairs["standard"] = pw_pairs_standard

        return pw_pairs

    def add_pix_wave_pair(
        self, pix, wave, spec_id=None, stype="science+standard"
    ):
        """
        Adding extra pixel-wavelength pair to the Calibrator for refitting.
        This DOES NOT work before the Calibrator having fit for a solution
        yet: use set_known_pairs() for that purpose.

        Parameters
        ----------
        pix: float
            pixel position
        wave: float
            wavelength
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_wavecal_polynomial_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)
                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].add_pix_wave_pair(pix, wave)

        if "standard" in stype_split:

            if self.standard_wavecal_polynomial_available:

                self.standard_wavecal.add_pix_wave_pair(pix, wave)

    def remove_pix_wave_pair(
        self, arg, spec_id=None, stype="science+standard"
    ):
        """
        Remove fitted pixel-wavelength pair from the Calibrator for refitting.
        The positions can be found from get_pix_wave_pairs(). One at a time.

        Parameters
        ----------
        arg: int
            The position of the pairs in the arrays.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_wavecal_polynomial_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)
                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].remove_pix_wave_pair(arg)

        if "standard" in stype_split:

            if self.standard_wavecal_polynomial_available:

                self.standard_wavecal.remove_pix_wave_pair(arg)

    def manual_refit(
        self,
        matched_peaks=None,
        matched_atlas=None,
        degree=None,
        x0=None,
        return_solution=False,
        spec_id=None,
        stype="science+standard",
    ):
        """
        Perform a refinement of the matched peaks and atlas lines.

        This function takes lists of matched peaks and atlases, along with
        user-specified lists of lines to add/remove from the lists.

        Any given peaks or atlas lines to remove are selected within a
        user-specified tolerance, by default 1 pixel and 5 atlas Angstrom.

        The final set of matching peaks/lines is then matched using a
        robust polyfit of the desired degree. Optionally, an initial
        fit x0 can be provided to condition the optimiser.

        The parameters are identical in the format in the fit() and
        match_peaks() functions, however, with manual changes to the lists of
        peaks and atlas, peak_utilisation and atlas_utilisation are
        meaningless so this function does not return in the same format.

        Parameters
        ----------
        matched_peaks: list (Default: None)
            List of matched peaks
        matched_atlas: list (Default: None)
            List of matched atlas lines
        degree: int (Default: None)
            Polynomial fit degree (Only used if x0 is None)
        x0: list (Default: None)
            Initial fit coefficients
        return_solution: bool (Default: False)
            Set to True to return the best fit polynomial coefficients.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """
        stype_split = stype.split("+")

        solution = {}

        if "science" in stype_split:

            if self.science_wavecal_polynomial_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)
                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                solution_science = []

                for i in spec_id:

                    solution_science.append(
                        self.science_wavecal[i].manual_refit(
                            matched_peaks=matched_peaks,
                            matched_atlas=matched_atlas,
                            degree=degree,
                            x0=x0,
                            return_solution=return_solution,
                        )
                    )

                solution["science"] = solution_science

        if "standard" in stype_split:

            if self.standard_wavecal_polynomial_available:

                solution["standard"] = self.standard_wavecal.manual_refit(
                    matched_peaks=matched_peaks,
                    matched_atlas=matched_atlas,
                    degree=degree,
                    x0=x0,
                    return_solution=return_solution,
                )

        if return_solution:

            return solution

    def get_calibrator(self, spec_id=None, stype="science+standard"):

        stype_split = stype.split("+")

        calibrators = {}

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)
            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            calibrator_science = []

            for i in spec_id:

                calibrator_science.append(
                    getattr(self.science_wavecal[i].spectrum1D, "calibrator")
                )

            calibrators["science"] = calibrator_science

        if "standard" in stype_split:

            calibrators["standard"] = getattr(
                self.standard_wavecal[0].spectrum1D, "calibrator"
            )

        return calibrators

    def apply_wavelength_calibration(
        self,
        spec_id=None,
        wave_start=None,
        wave_end=None,
        wave_bin=None,
        stype="science+standard",
    ):
        """
        Apply the wavelength calibration. Because the polynomial fit can run
        away at the two ends, the default minimum and maximum are limited to
        1,000 and 12,000 A, respectively. This can be overridden by providing
        user's choice of wave_start and wave_end.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        wave_start: float or None (Default: None)
            Provide the minimum wavelength for resampling.
        wave_end: float or None (Default: None)
            Provide the maximum wavelength for resampling
        wave_bin: float or None (Default: None)
            Provide the resampling bin size
        stype: str or None (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if self.science_wavecal_polynomial_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, contraints are applied to all
                    #  calibrators
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    spec = self.science_spectrum_list[i]

                    # Adjust for pixel shift due to chip gaps
                    wave = (
                        self.science_wavecal[i]
                        .polyval[spec.fit_type](
                            np.array(spec.pixel_list), spec.fit_coeff
                        )
                        .reshape(-1)
                    )

                    # compute the new equally-spaced wavelength array
                    if wave_bin is None:

                        wave_bin = np.nanmedian(np.ediff1d(wave))

                    if wave_start is None:

                        wave_start = wave[0]

                    if wave_end is None:

                        wave_end = wave[-1]

                    wave_resampled = np.arange(wave_start, wave_end, wave_bin)

                    count_resampled = spectres(
                        np.array(wave_resampled).reshape(-1),
                        np.array(wave).reshape(-1),
                        np.array(spec.count).reshape(-1),
                        verbose=True,
                    )

                    if spec.count_err is not None:

                        count_err_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.count_err).reshape(-1),
                            verbose=True,
                        )

                    if spec.count_sky is not None:

                        count_sky_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.count_sky).reshape(-1),
                            verbose=True,
                        )

                    spec.add_wavelength(wave)
                    spec.add_wavelength_resampled(wave_resampled)
                    spec.add_count_resampled(
                        count_resampled,
                        count_err_resampled,
                        count_sky_resampled,
                    )

                self.logger.info(
                    "Wavelength calibration is applied for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )
                self.science_wavelength_calibrated = True
                self.science_wavelength_resampled = True

            else:

                self.logger.warning("Science spectrum/a are not imported.")

        if "standard" in stype_split:

            if self.standard_wavecal_polynomial_available:

                spec = self.standard_spectrum_list[0]

                # Adjust for pixel shift due to chip gaps
                wave = self.standard_wavecal.polyval[spec.fit_type](
                    np.array(spec.pixel_list), spec.fit_coeff
                ).reshape(-1)

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
                    verbose=True,
                )

                if spec.count_err is not None:

                    count_err_resampled = spectres(
                        np.array(wave_resampled).reshape(-1),
                        np.array(wave).reshape(-1),
                        np.array(spec.count_err).reshape(-1),
                        verbose=True,
                    )

                if spec.count_sky is not None:

                    count_sky_resampled = spectres(
                        np.array(wave_resampled).reshape(-1),
                        np.array(wave).reshape(-1),
                        np.array(spec.count_sky).reshape(-1),
                        verbose=True,
                    )

                spec.add_wavelength(wave)
                spec.add_wavelength_resampled(wave_resampled)
                spec.add_count_resampled(
                    count_resampled, count_err_resampled, count_sky_resampled
                )

                self.logger.info(
                    "Wavelength calibration is applied for the "
                    "standard_spectrum_list."
                )
                self.standard_wavelength_calibrated = True
                self.standard_wavelength_resampled = True

            else:

                self.logger.warning("Standard spectrum is not imported.")

            self.standard_wavelength_calibrated = True

    def lookup_standard_libraries(self, target, cutoff=0.4):
        """
        Parameters
        ----------
        target: str
            Name of the standard star
        cutoff: float (Default: 0.4)
            The similarity tolerance [0=completely different, 1=identical]

        """

        self.fluxcal.lookup_standard_libraries(target=target, cutoff=cutoff)

    def load_standard(self, target, library=None, ftype="flux", cutoff=0.4):
        """
        Read the standard flux/magnitude file. And return the wavelength and
        flux/mag. The units of the data are always in

        | wavelength: A
        | flux:       ergs / cm / cm / s / A
        | mag:        mag (AB)

        Parameters
        ----------
        target: string
            Name of the standard star
        library: string (Default: None)
            Name of the library of standard star
        ftype: string (Default: 'flux')
            'flux' or 'mag'
        cutoff: float (Default: 0.4)
            The toleranceold for the word similarity in the range of [0, 1].

        """

        self.fluxcal.load_standard(
            target=target, library=library, ftype=ftype, cutoff=cutoff
        )
        self.logger.info("Loaded standard: {} from {}".format(target, library))

    def inspect_standard(
        self,
        display=True,
        renderer="default",
        width=1280,
        height=720,
        return_jsonstring=False,
        save_fig=False,
        fig_type="iframe+png",
        filename=None,
        open_iframe=False,
    ):
        """
        Parameters
        ----------
        display: bool (Default: True)
            Set to True to display disgnostic plot.
        renderer: str (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool (Default: False)
            set to True to return json str that can be rendered by Plotly
            in any support language.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        """

        self.fluxcal.inspect_standard(
            renderer=renderer,
            return_jsonstring=return_jsonstring,
            display=display,
            height=height,
            width=width,
            save_fig=save_fig,
            fig_type=fig_type,
            filename=filename,
            open_iframe=open_iframe,
        )
        self.logger.info("Inspect standard.")

        if return_jsonstring:

            return return_jsonstring

    def get_sensitivity(
        self,
        k=3,
        method="interpolate",
        mask_range=[[6850, 6960], [7580, 7700]],
        mask_fit_order=1,
        mask_fit_size=5,
        smooth=False,
        slength=5,
        sorder=3,
        return_function=False,
        sens_deg=7,
        **kwargs
    ):
        """
        Parameters
        ----------
        k: integer [1,2,3,4,5 only]
            The order of the spline.
        method: str (Default: 'interpolate')
            This should be either 'interpolate' of 'polynomial'. Note that the
            polynomial is computed from the interpolated function. The
            default is interpolate because it is much more stable at the
            wavelength limits of a spectrum in an automated system.
        mask_range: None or list of list
            (Default: 6850-6960, 7575-7700, 8925-9050)
            Masking out regions not suitable for fitting the sensitivity curve.
            None for no mask. List of list has the pattern
            [[min1, max1], [min2, max2],...]
        mask_fit_order: int (Default: 1)
            Order of polynomial to be fitted over the masked regions
        mask_fit_size: int (Default: 5)
            Number of "pixels" to be fitted on each side of the masked regions.
        smooth: bool (Default: False)
            set to smooth the input spectrum with scipy.signal.savgol_filter
        slength: int (Default:5)
            SG-filter window size
        sorder: int (Default: 3)
            SG-filter polynomial order
        return_function: bool (Default: False)
            Set to True to return the callable function of the sensitivity
            curve.
        sens_deg: int (Default: 7)
            The degree of polynomial of the sensitivity curve, only used if
            the method is 'polynomial'.
        **kwargs:
            keyword arguments for passing to the LOWESS function, see
            `statsmodels.nonparametric.smoothers_lowess.lowess()`

        """

        if self.standard_wavelength_calibrated:

            self.fluxcal.get_sensitivity(
                k=k,
                method=method,
                mask_range=mask_range,
                mask_fit_order=mask_fit_order,
                mask_fit_size=mask_fit_size,
                smooth=smooth,
                slength=slength,
                sorder=sorder,
                return_function=return_function,
                sens_deg=sens_deg,
                **kwargs
            )
            self.logger.info("Sensitivity curve computed.")
            self.sensitivity_curve_available = True

        else:

            error_msg = (
                "Standard star is not wavelength calibrated, "
                + "sensitivity curve cannot be computed."
            )
            self.logger.critical(error_msg)
            raise RuntimeError(error_msg)

    def save_sensitivity_func(self, filename="sensitivity_func.npy"):
        """
        Not-implemented wrapper.

        Parameters
        ----------
        filename: str
            Filename for the output interpolated sensivity curve.

        """

        self.fluxcal.save_sensitivity_func(filename=filename)
        self.logger.info("Sensitivity curve saved at {}.".format(filename))

    def add_sensitivity_func(self, sensitivity_func):
        """
        Provide a callable function of the detector sensitivity response.

        Parameters
        ----------
        sensitivity_func: str
            Interpolated sensivity curve object.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        self.fluxcal.add_sensitivity_func(sensitivity_func=sensitivity_func)
        self.logger.info("User supplied sensitivity curve added.")
        self.sensitivity_curve_available = True

    def inspect_sensitivity(
        self,
        display=True,
        renderer="default",
        width=1280,
        height=720,
        return_jsonstring=False,
        save_fig=False,
        fig_type="iframe+png",
        filename=None,
        open_iframe=False,
    ):
        """
        Parameters
        ----------
        display: bool (Default: True)
            Set to True to display disgnostic plot.
        renderer: str (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool (Default: False)
            set to True to return json str that can be rendered by Plotly
            in any support language.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.

        """

        if self.sensitivity_curve_available:

            self.fluxcal.inspect_sensitivity(
                renderer=renderer,
                width=width,
                height=height,
                return_jsonstring=return_jsonstring,
                display=display,
                save_fig=save_fig,
                fig_type=fig_type,
                filename=filename,
                open_iframe=open_iframe,
            )
            self.logger.info("Inspect sensitivity function.")

        else:

            self.logger.warning(
                "Sensitivity function not available, it "
                "cannot be inspected."
            )

    def apply_flux_calibration(
        self,
        spec_id=None,
        inspect=False,
        wave_min=3500.0,
        wave_max=8500.0,
        display=False,
        renderer="default",
        width=1280,
        height=720,
        return_jsonstring=False,
        save_fig=False,
        fig_type="iframe+png",
        filename=None,
        open_iframe=False,
        stype="science+standard",
    ):
        """
        Apply the computed sensitivity curve. And resample the spectra to
        match the highest resolution (the smallest wavelength bin) part of the
        spectrum.

        Note: This function directly modify the *target_spectrum1D*.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        inspect: bool (Default: False)
            Set to True to create/display/save figure
        wave_min: float (Default: 3500)
            Minimum wavelength to display
        wave_max: float (Default: 8500)
            Maximum wavelength to display
        display: bool (Default: False)
            Set to True to display disgnostic plot.
        renderer: string (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool (Default: False)
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if self.sensitivity_curve_available:

            if "science" in stype_split:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())
                    ):

                        error_msg = "The given spec_id does not exist."
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, contraints are applied to all
                    #  calibrators
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.fluxcal.apply_flux_calibration(
                        target_spectrum1D=self.science_spectrum_list[i],
                        inspect=inspect,
                        wave_min=wave_min,
                        wave_max=wave_max,
                        display=display,
                        renderer=renderer,
                        width=width,
                        height=height,
                        return_jsonstring=return_jsonstring,
                        save_fig=save_fig,
                        fig_type=fig_type,
                        filename=filename,
                        open_iframe=open_iframe,
                    )
                    self.science_spectrum_list[i].add_standard_header(
                        self.standard_spectrum_list[0].spectrum_header
                    )
                    self.logger.info(
                        "Flux calibration is applied for the "
                        "science_spectrum_list for spec_id: {}.".format(i)
                    )

                self.science_flux_calibrated = True

            if "standard" in stype_split:

                self.fluxcal.apply_flux_calibration(
                    target_spectrum1D=self.standard_spectrum_list[0],
                    inspect=inspect,
                    wave_min=wave_min,
                    wave_max=wave_max,
                    display=display,
                    renderer=renderer,
                    width=width,
                    height=height,
                    return_jsonstring=return_jsonstring,
                    save_fig=save_fig,
                    fig_type=fig_type,
                    filename=filename,
                    open_iframe=open_iframe,
                )
                self.standard_spectrum_list[0].add_standard_header(
                    self.standard_spectrum_list[0].spectrum_header
                )
                self.logger.info(
                    "Flux calibration is applied for the "
                    "standard_spectrum_list."
                )

                self.standard_flux_calibrated = True

        else:

            self.logger.warning(
                "Sensitivity function is not available, "
                "flux calibration is not possible."
            )

    def _min_std(self, factor, flux, telluric_profile, continuum, sigma=4.5):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        Minimisation function to get the best mutiplier for the strength
        of the Telluric profile.

        Parameters
        ----------
        factor: float
            The multiplier for the strength of the Telluric profile.
        flux: 1-d array (N)
            Flux of the target.
        telluric_profile: 1-d array (N)
            Telluric Profile normalised to 1 being the most strongly absorbed,
            0 being outside the Telluric regions.
        continuum: 1-d array (N)
            Continuum flux array.
        sigma: float (default: 4.5)
            Level of sigma clipping.

        """

        mask = telluric_profile != 0
        telluric_absorption = factor * telluric_profile
        diff = flux + telluric_absorption - continuum
        nansum = np.nansum(diff[mask] ** 2.0) * 1e20

        return nansum

    def add_telluric_function(
        self, telluric, spec_id=None, stype="science+standard"
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        Provide a callable function that gives the Telluric profile.

        Parameters
        ----------
        telluric: callable function
            A function that gives the absorption profile as a function
            of wavelength.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                science_spec = self.science_spectrum_list[i]

                if callable(telluric):

                    science_spec.add_telluric_func(telluric)

                elif isinstance(telluric, (np.ndarray, list)):

                    science_spec.add_telluric_func(
                        interp1d(telluric[0], telluric[1])
                    )

                else:

                    self.logger.warning(
                        "telluric provided has to be a callable function, "
                        "a list or a np.ndarray. "
                        "{} is given".format(type(telluric))
                    )

                if science_spec.wave is not None:

                    science_spec.add_telluric_profile(
                        science_spec.telluric_func(science_spec.wave)
                    )

                else:

                    self.logger.warning(
                        "wave is not available. Telluric correction cannot"
                        "be performed."
                    )

                if science_spec.wave_resampled is not None:

                    science_spec.add_telluric_profile_resampled(
                        science_spec.telluric_func(science_spec.wave_resampled)
                    )

                else:

                    self.logger.warning(
                        "wave_resampled is not available. Telluric correction "
                        "cannot be performed."
                    )

        if "standard" in stype_split:

            # Add to the standard spectrum
            standard_spec = self.standard_spectrum_list[0]

            if callable(telluric):

                standard_spec.add_telluric_func(telluric)

            elif isinstance(telluric, (np.ndarray, list)):

                standard_spec.add_telluric_func(
                    interp1d(telluric[0], telluric[1])
                )

            else:

                self.logger.warning(
                    "telluric provided has to be a callable function, "
                    "a list or a np.ndarray. "
                    "{} is given".format(type(telluric))
                )

            if standard_spec.wave is not None:

                standard_spec.add_telluric_profile(
                    standard_spec.telluric_func(standard_spec.wave)
                )

            else:

                self.logger.warning(
                    "wave is not available. Telluric correction cannot"
                    "be performed."
                )

            if standard_spec.wave_resampled is not None:

                standard_spec.add_telluric_profile_resampled(
                    standard_spec.telluric_func(standard_spec.wave_resampled)
                )

            else:

                self.logger.warning(
                    "wave_resampled is not available. Telluric correction "
                    "cannot be performed."
                )

    def get_continuum(self, spec_id=None, **kwargs):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        Get the continnum from the wave, count and flux.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        **kwargs: dictionary
            The keyword arguments to be passed to the lowess function
            for generating the continuum.

        """

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if spec_id is not None:

            if not set(spec_id).issubset(
                list(self.science_spectrum_list.keys())
            ):

                error_msg = "The given spec_id does not exist."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:

            spec_id = list(self.science_spectrum_list.keys())

        # Get the telluric profile
        for i in spec_id:

            science_spec = self.science_spectrum_list[i]

            wave = science_spec.wave
            count = science_spec.count
            flux = science_spec.flux

            science_spec.add_count_continuum(
                get_continuum(wave, count, **kwargs)
            )

            if flux is not None:

                science_spec.add_flux_continuum(
                    get_continuum(wave, flux, **kwargs)
                )

            else:

                self.logger.warning(
                    "flux is None, only count_continuum is found."
                )

    def get_resampled_continuum(self, spec_id=None, **kwargs):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        Get the continnum from the resampled wave, count and flux.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        **kwargs: dictionary
            The keyword arguments to be passed to the lowess function
            for generating the continuum.

        """

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if spec_id is not None:

            if not set(spec_id).issubset(
                list(self.science_spectrum_list.keys())
            ):

                error_msg = "The given spec_id does not exist."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:

            spec_id = list(self.science_spectrum_list.keys())

        # Get the telluric profile
        for i in spec_id:

            science_spec = self.science_spectrum_list[i]

            wave_resampled = science_spec.wave_resampled
            count_resampled = science_spec.count_resampled
            flux_resampled = science_spec.flux_resampled

            science_spec.add_flux_resampled_continuum(
                get_continuum(wave_resampled, flux_resampled, **kwargs)
            )
            science_spec.add_count_resampled_continuum(
                get_continuum(wave_resampled, count_resampled, **kwargs)
            )

    def get_telluric_profile(
        self,
        spec_id=None,
        mask_range=[[6850, 6960], [7580, 7700]],
        return_function=False,
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        Getting the Telluric absorption profile from the continuum of the
        standard star spectrum.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        mask_range: list of list
            list of lists with 2 values indicating the range marked by each
            of the Telluric regions.
        return_function: bool (Default: False)
            Set to True to explicitly return the interpolated function of
            the Telluric profile.

        """

        (
            telluric_func,
            telluric_profile,
            telluric_factor,
        ) = self.fluxcal.get_telluric_profile(
            wave=self.standard_spectrum_list[0].wave,
            flux=self.standard_spectrum_list[0].flux,
            continuum=self.standard_spectrum_list[0].flux_continuum,
            mask_range=mask_range,
            return_function=True,
        )

        self.logger.info(
            "Copying the telluric absorption profile to "
            "the science spectrum1D(s)."
        )

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if spec_id is not None:

            if not set(spec_id).issubset(
                list(self.science_spectrum_list.keys())
            ):

                error_msg = "The given spec_id does not exist."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:

            spec_id = list(self.science_spectrum_list.keys())

        # Add the telluric profile from fluxcal to science onedspec
        for i in spec_id:

            self.science_spectrum_list[i].add_telluric_func(telluric_func)
            self.science_spectrum_list[i].add_telluric_profile(
                telluric_profile
            )

        # Add the telluric profile from fluxcal to standard onedspec
        self.standard_spectrum_list[0].add_telluric_func(telluric_func)
        self.standard_spectrum_list[0].add_telluric_profile(telluric_profile)
        self.standard_spectrum_list[0].add_telluric_factor(telluric_factor)

        if return_function:

            return telluric_func

    def get_telluric_correction(
        self, spec_id=None, factor=1.0, auto_apply=False, **kwargs
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        Get the telluric absorption profile from the standard star based on
        the masked regions given in generating the sensitivity curve. Note
        that the profile has a "positive" flux so that in the step of applying
        a correction, a POSITIVE constant is found to multiply with the
        normalised telluric profile before ADDING to the spectrum for
        telluric absorption correction (counter-intuitive to the term
        telluric absorption subtraction).

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        factor: float (Default: 1.0)
            The extra fudge factor multiplied to the telluric profile to
            manally adjust the strength.
        auto_apply: bool (Default: False)
            Set to True to accept the computed telluric absorption correction
            automatically, which is currently an irresversible process through
            the public API.

        """

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if spec_id is not None:

            if not set(spec_id).issubset(
                list(self.science_spectrum_list.keys())
            ):

                error_msg = "The given spec_id does not exist."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:

            spec_id = list(self.science_spectrum_list.keys())

        # Get the telluric profile
        for i in spec_id:

            science_spec = self.science_spectrum_list[i]

            # If there isn't a telluric profile, try to get it from the
            # standard star
            if science_spec.telluric_func is None:

                if self.standard_spectrum_list[0].telluric_func is None:

                    err_msg = (
                        "Telluric profile is not available, please "
                        + "compute from the standard star, or manually "
                        + "supply one."
                    )
                    self.logger.error(err_msg)

                else:

                    science_spec.add_telluric_func(
                        self.standard_spectrum_list[0].telluric_func
                    )

            wave = science_spec.wave
            flux = science_spec.flux

            if (science_spec.flux_continuum is None) or (
                len(kwargs.keys()) > 0
            ):

                self.get_continuum(i, **kwargs)

            flux_continuum = science_spec.flux_continuum

            if science_spec.telluric_profile is None:

                science_spec.add_telluric_profile(
                    science_spec.telluric_func(wave)
                )

            telluric_factor = optimize.minimize(
                self._min_std,
                np.nanmedian(np.abs(flux)),
                args=(flux, science_spec.telluric_profile, flux_continuum),
                tol=1e-20,
                method="Nelder-Mead",
                options={"maxiter": 10000},
            ).x

            science_spec.add_telluric_factor(telluric_factor)
            self.logger.info("telluric_factor is {}.".format(telluric_factor))

            if self.science_wavelength_resampled:

                wave_resampled = science_spec.wave_resampled
                flux_resampled = science_spec.flux_resampled

                if science_spec.flux_resampled_continuum is None:

                    self.get_resampled_continuum(i, **kwargs)

                flux_resampled_continuum = (
                    science_spec.flux_resampled_continuum
                )

                if (science_spec.telluric_profile_resampled is None) or (
                    len(kwargs.keys()) > 0
                ):

                    science_spec.add_telluric_profile_resampled(
                        science_spec.telluric_func(wave_resampled)
                    )

                telluric_factor_resampled = optimize.minimize(
                    self._min_std,
                    np.nanmedian(np.abs(flux_resampled)),
                    args=(
                        flux_resampled,
                        science_spec.telluric_profile_resampled,
                        flux_resampled_continuum,
                    ),
                    tol=1e-20,
                    method="Nelder-Mead",
                    options={"maxiter": 10000},
                ).x

                self.logger.info(
                    "telluric_factor_resampled is {}.".format(
                        telluric_factor_resampled
                    )
                )

                science_spec.add_telluric_factor_resampled(
                    telluric_factor_resampled
                )

        if self.standard_wavelength_resampled:

            standard_spec = self.standard_spectrum_list[i]
            wave_standard_resampled = standard_spec.wave_resampled

            if (standard_spec.telluric_profile_resampled is None) or (
                len(kwargs.keys()) > 0
            ):

                standard_spec.add_telluric_profile_resampled(
                    standard_spec.telluric_func(wave_standard_resampled)
                )

                telluric_factor_resampled = optimize.minimize(
                    self._min_std,
                    np.nanmedian(np.abs(flux_resampled)),
                    args=(
                        standard_spec.flux_resampled,
                        standard_spec.telluric_profile_resampled,
                        standard_spec.flux_resampled_continuum,
                    ),
                    tol=1e-20,
                    method="Nelder-Mead",
                    options={"maxiter": 10000},
                ).x

                self.logger.info(
                    "telluric_factor_resampled is {}.".format(
                        telluric_factor_resampled
                    )
                )

                standard_spec.add_telluric_factor_resampled(
                    telluric_factor_resampled
                )

        if auto_apply:

            self.apply_telluric_correction(
                factor=factor, spec_id=spec_id, stype="science+standard"
            )

    def inspect_telluric_profile(
        self,
        display=True,
        renderer="default",
        width=1280,
        height=720,
        return_jsonstring=False,
        save_fig=False,
        fig_type="iframe+png",
        filename=None,
        open_iframe=False,
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        Display the Telluric profile.

        Parameters
        ----------
        display: bool (Default: True)
            Set to True to display disgnostic plot.
        renderer: string (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool (Default: False)
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        """

        self.fluxcal.inspect_telluric_profile(
            display=display,
            renderer=renderer,
            width=width,
            height=height,
            return_jsonstring=return_jsonstring,
            save_fig=save_fig,
            fig_type=fig_type,
            filename=filename,
            open_iframe=open_iframe,
        )
        self.logger.info("Inspecting the telluric absorption profile.")

    def inspect_telluric_correction(
        self,
        factor=1.0,
        spec_id=None,
        display=True,
        renderer="default",
        width=1280,
        height=720,
        return_jsonstring=False,
        save_fig=False,
        fig_type="iframe+png",
        filename=None,
        open_iframe=False,
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        Inspect the Telluric absorption correction on top of pre-corrected
        spectra.

        Parameters
        ----------
        factor: float (Default: 1.0)
            The extra fudge factor multiplied to the telluric profile to
            manally adjust the strength.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        display: bool (Default: True)
            Set to True to display disgnostic plot.
        renderer: string (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool (Default: False)
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        """

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if spec_id is not None:

            if not set(spec_id).issubset(
                list(self.science_spectrum_list.keys())
            ):

                error_msg = "The given spec_id does not exist."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:

            # if spec_id is None, contraints are applied to all
            #  calibrators
            spec_id = list(self.science_spectrum_list.keys())

        to_return = []

        # Get the telluric profile
        for i in spec_id:

            spec = self.science_spectrum_list[i]

            wave = spec.wave

            fluxcount = spec.flux
            fluxcount_name = "Flux"
            fluxcount_continuum = spec.flux_continuum
            telluric_factor = spec.telluric_factor
            telluric_profile = spec.telluric_profile

            flux_low = (
                np.nanpercentile(np.array(fluxcount).reshape(-1), 5) / 1.5
            )
            flux_high = (
                np.nanpercentile(np.array(fluxcount).reshape(-1), 95) * 1.5
            )
            flux_mask = (np.array(fluxcount).reshape(-1) > flux_low) & (
                np.array(fluxcount).reshape(-1) < flux_high
            )

            if np.sum(flux_mask) > 0:

                flux_min = np.nanmin(
                    np.array(fluxcount).reshape(-1)[flux_mask]
                )
                flux_max = np.nanmax(
                    np.array(fluxcount).reshape(-1)[flux_mask]
                )

            else:

                flux_min = np.nanmin(np.array(fluxcount).reshape(-1))
                flux_max = np.nanmax(np.array(fluxcount).reshape(-1))

            fig_sci = go.Figure(
                layout=dict(
                    autosize=False,
                    height=height,
                    width=width,
                    updatemenus=list(
                        [
                            dict(
                                active=0,
                                buttons=list(
                                    [
                                        dict(
                                            label="Log Scale",
                                            method="update",
                                            args=[
                                                {"visible": [True, True]},
                                                {
                                                    "title": "Log scale",
                                                    "yaxis": {"type": "log"},
                                                },
                                            ],
                                        ),
                                        dict(
                                            label="Linear Scale",
                                            method="update",
                                            args=[
                                                {"visible": [True, False]},
                                                {
                                                    "title": "Linear scale",
                                                    "yaxis": {
                                                        "type": "linear"
                                                    },
                                                },
                                            ],
                                        ),
                                    ]
                                ),
                            )
                        ]
                    ),
                    title="Log scale",
                )
            )

            # show the image on the top
            fig_sci.add_trace(
                go.Scatter(
                    x=wave,
                    y=fluxcount,
                    line=dict(color="royalblue"),
                    name=fluxcount_name,
                )
            )

            fig_sci.add_trace(
                go.Scatter(
                    x=wave,
                    y=fluxcount_continuum,
                    line=dict(color="firebrick"),
                    name="Continuum Flux",
                )
            )

            fig_sci.add_trace(
                go.Scatter(
                    x=wave,
                    y=(
                        fluxcount + telluric_profile * telluric_factor * factor
                    ),
                    line=dict(color="orange"),
                    name="Telluric Corrected Spectrum",
                )
            )

            fig_sci.add_trace(
                go.Scatter(
                    x=wave,
                    y=(telluric_profile * telluric_factor * factor),
                    line=dict(color="grey"),
                    name="Telluric Profile",
                )
            )

            fig_sci.update_layout(
                hovermode="closest",
                showlegend=True,
                xaxis=dict(title="Wavelength / A"),
                yaxis=dict(
                    title="Flux", range=[flux_min, flux_max], type="linear"
                ),
                legend=go.layout.Legend(
                    x=0,
                    y=1,
                    traceorder="normal",
                    font=dict(family="sans-serif", size=12, color="black"),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )

            if filename is None:

                filename = "telluric_inspection"

            if save_fig:

                fig_type_split = fig_type.split("+")

                for t in fig_type_split:

                    save_path = filename + "_" + str(i) + "." + t

                    if t == "iframe":

                        pio.write_html(
                            fig_sci, save_path, auto_open=open_iframe
                        )

                    elif t in ["jpg", "png", "svg", "pdf"]:

                        pio.write_image(fig_sci, save_path)

                    self.logger.info(
                        "Figure is saved to {} for the ".format(save_path)
                        + "science_spectrum_list for spec_id: {}.".format(i)
                    )

            if display:

                if renderer == "default":

                    fig_sci.show()

                else:

                    fig_sci.show(renderer)

            if return_jsonstring:

                to_return.append(fig_sci.to_json())

        if return_jsonstring:

            return to_return

    def apply_telluric_correction(
        self, factor=1.0, spec_id=None, stype="science+standard"
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        Apply the telluric correction with the extra multiplier 'factor'.
        The 'factor' provided in the profile() is NOT
        propagated to this function, it has to be explicitly provided
        to this function.

        The telluric absorption profile is normalised to 1 at the most
        absorpted wavelegnth, the factor manually provided can be
        negative in case of over/under-subtraction.

        Parameters
        ----------
        factor: float (Default: None)
            The extra fudge factor multiplied to the telluric profile to
            manally adjust the strength.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, contraints are applied to all
                #  calibrators
                spec_id = list(self.science_spectrum_list.keys())

            # Get the telluric profile
            for i in spec_id:

                science_spec = self.science_spectrum_list[i]

                if science_spec.telluric_profile_resampled is None:

                    self.logger.warning(
                        "A resampled telluric profile is not available, "
                        "please construct a profile with "
                        "get_telluric_profile()."
                    )

                else:

                    science_spec.flux_resampled = (
                        science_spec.flux_resampled
                        + science_spec.telluric_profile_resampled
                        * science_spec.telluric_factor_resampled
                        * factor
                    )

                if science_spec.telluric_profile is None:

                    self.logger.warning(
                        "A resampled telluric profile is not available, "
                        "please construct a profile with "
                        "get_telluric_profile()."
                    )

                else:

                    science_spec.flux = (
                        science_spec.flux
                        + science_spec.telluric_profile
                        * science_spec.telluric_factor
                        * factor
                    )

        if "standard" in stype_split:

            standard_spec = self.standard_spectrum_list[0]

            if standard_spec.telluric_profile_resampled is None:

                self.logger.warning(
                    "A resampled telluric profile is not available, "
                    "please construct a profile with "
                    "get_telluric_profile()."
                )

            else:

                standard_spec.flux_resampled = (
                    standard_spec.flux_resampled
                    + standard_spec.telluric_profile_resampled
                    * standard_spec.telluric_factor_resampled
                    * factor
                )

            if standard_spec.telluric_profile is None:

                self.logger.warning(
                    "A resampled telluric profile is not available, "
                    "please construct a profile with "
                    "get_telluric_profile()."
                )

            else:

                standard_spec.flux = (
                    standard_spec.flux
                    + standard_spec.telluric_profile
                    * standard_spec.telluric_factor
                    * factor
                )

    def set_atmospheric_extinction(
        self,
        location="orm",
        extinction_func=None,
        kind="cubic",
        fill_value="extrapolate",
        **kwargs
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        The ORM atmospheric extinction correction table is taken from
        http://www.ing.iac.es/astronomy/observing/manuals/ps/tech_notes/tn031.pdf

        The MK atmospheric extinction correction table is taken from
        Buton et al. (2013A&A...549A...8B)

        The CP atmospheric extinction correction table is taken from
        Patat et al. (2011A&A...527A..91P)

        The LS atmospheric extinction correction table is taken from
        THE ESO USERS MANUAL 1993
        https://www.eso.org/public/archives/techdocs/pdf/report_0003.pdf

        Parameters
        ----------
        location: str (Default: orm)
            Location of the observatory, currently contains:
            (1) orm - Roque de los Muchachos Observatory (2420 m)
            (2) mk - Mauna Kea (4205 m)
            (3) cp - Cerro Paranal (2635 m)
            (4) ls - La Silla (2400 m) [up to 9000A only]
            Only used if extinction_func is None.
        extinction_func: callable function (Default: None)
            Input wavelength in Angstrom, output magnitude of extinction per
            airmass. It will override the 'location'.

        """

        if (extinction_func is not None) and (callable(extinction_func)):

            self.extinction_func = extinction_func
            self.logger.info(
                "Manual extinction correction function is loaded."
            )

        else:

            filename = pkg_resources.resource_filename(
                "aspired", "extinction/{}_atm_extinct.txt".format(location)
            )
            extinction_table = np.loadtxt(filename, delimiter=",")
            self.extinction_func = interp1d(
                extinction_table[:, 0],
                extinction_table[:, 1],
                kind=kind,
                fill_value=fill_value,
                **kwargs
            )
            self.logger.info(
                "{} extinction correction function is loaded.".format(location)
            )

        self.atmospheric_extinction_correction_available = True

    def apply_atmospheric_extinction_correction(
        self, science_airmass=None, standard_airmass=None, spec_id=None
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        This is the first step in allowing atmospheric extinction correction
        of the spectra. Currently it only works if both the science and
        standard spectra are present and both airmass values are provided.
        Towards completion, this function should allow atmospheric
        extinction correction on any meaningful combination of
        (1) science and/or standard spectrum/a, and (2) airmass of either
        or both science and standard observations.

        Parameters
        ----------
        science_airmass: float, str or None (Default: None)
            - If None, it will look for the airmass in the header, if the
              keyword AIRMASS is not found, correction will not be performed.
            - A string input will be used as the header keyword of the airmass,
              if the keyword or header is not found, correction will not be
              performed.
            - A floatpoint value will override the other two and directly be
              use as the airmass
        standard_airmass: float, str or None (Default: None)
            The same as science_airmass.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object

        """

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if spec_id is not None:

            if not set(spec_id).issubset(
                list(self.science_spectrum_list.keys())
            ):

                error_msg = "The given spec_id does not exist."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:

            # if spec_id is None, contraints are applied to all
            #  calibrators
            spec_id = list(self.science_spectrum_list.keys())

        if not self.atmospheric_extinction_correction_available:

            self.logger.warning(
                "Atmospheric extinction correction is not configured, "
                "The default ORM extinction curve is used."
            )

            self.set_atmospheric_extinction()

        standard_spec = self.standard_spectrum_list[0]

        if standard_airmass is not None:

            if isinstance(standard_airmass, (int, float)):

                standard_am = standard_airmass
                self.logger.info(
                    "Airmass is set to be {}.".format(standard_am)
                )

            if isinstance(standard_airmass, str):

                try:

                    standard_am = standard_spec.spectrum_header[
                        standard_airmass
                    ]

                except Exception as e:

                    self.logger.warning(str(e))

                    standard_am = 1.0
                    self.logger.warning(
                        "Keyword for airmass: {} cannot be found "
                        "in header.".format(standard_airmass)
                    )
                    self.logger.warning("Airmass is set to be 1.0")

        else:

            try:

                standard_am = standard_spec.spectrum_header["AIRMASS"]

            except Exception as e:

                self.logger.warning(str(e))

                standard_am = 1.0
                self.logger.warning(
                    "Keyword for airmass: AIRMASS cannot be found "
                    "in header."
                )
                self.logger.warning("Airmass is set to be 1.0")

        if spec_id is not None:

            if not set(spec_id).issubset(
                list(self.science_spectrum_list.keys())
            ):

                error_msg = "The given spec_id does not exist."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:

            # if spec_id is None, contraints are applied to all
            #  calibrators
            spec_id = list(self.science_spectrum_list.keys())

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        for i in spec_id:

            science_spec = self.science_spectrum_list[i]

            if science_airmass is not None:

                if isinstance(science_airmass, (int, float)):

                    science_am = science_airmass

                if isinstance(science_airmass, str):

                    try:

                        science_am = science_spec.spectrum_header[
                            science_airmass
                        ]

                    except Exception as e:

                        self.logger.warning(str(e))
                        science_am = 1.0

            else:

                if science_airmass is None:

                    try:

                        science_am = science_spec.spectrum_header["AIRMASS"]

                    except Exception as e:

                        self.logger.warning(str(e))
                        science_am = 1.0

            if science_am is None:

                science_am = 1.0

            self.logger.info("Standard airmass is {}.".format(standard_am))
            self.logger.info("Science airmass is {}.".format(science_am))

            # Get the atmospheric extinction correction factor
            science_flux_extinction_factor = 10.0 ** (
                -(self.extinction_func(science_spec.wave) * science_am) / 2.5
            )
            standard_flux_extinction_factor = 10.0 ** (
                -(self.extinction_func(science_spec.wave) * standard_am) / 2.5
            )
            science_spec.flux /= (
                science_flux_extinction_factor
                / standard_flux_extinction_factor
            )

            science_flux_resampled_extinction_factor = 10.0 ** (
                -(
                    self.extinction_func(science_spec.wave_resampled)
                    * science_am
                )
                / 2.5
            )
            standard_flux_resampled_extinction_factor = 10.0 ** (
                -(
                    self.extinction_func(science_spec.wave_resampled)
                    * standard_am
                )
                / 2.5
            )
            science_spec.flux_resampled /= (
                science_flux_resampled_extinction_factor
                / standard_flux_resampled_extinction_factor
            )

        self.atmospheric_extinction_corrected = True
        self.logger.info("Atmospheric extinction is corrected.")

    def inspect_reduced_spectrum(
        self,
        spec_id=None,
        stype="science+standard",
        wave_min=3500.0,
        wave_max=8500.0,
        display=True,
        renderer="default",
        width=1280,
        height=720,
        save_fig=False,
        fig_type="iframe+png",
        filename=None,
        open_iframe=False,
        return_jsonstring=False,
    ):
        """
        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        wave_min: float (Default: 3500.)
            Minimum wavelength to display
        wave_max: float (Default: 8500.)
            Maximum wavelength to display
        display: bool (Default: True)
            Set to True to display disgnostic plot.
        renderer: str (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.
        return_jsonstring: bool (Default: False)
            set to True to return JSON-string that can be rendered by Plotly
            in any support language.

        """

        stype_split = stype.split("+")
        to_return = []

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                spec = self.science_spectrum_list[i]
                telluric = None

                if self.science_wavelength_resampled:

                    wave = spec.wave_resampled

                    if self.science_flux_calibrated:
                        fluxcount = spec.flux_resampled
                        fluxcount_sky = spec.flux_sky_resampled
                        fluxcount_err = spec.flux_err_resampled
                        fluxcount_name = "Flux"
                        fluxcount_sky_name = "Sky Flux"
                        fluxcount_err_name = "Flux Uncertainty"
                        telluric = spec.telluric_profile_resampled
                        telluric_factor = spec.telluric_factor_resampled
                        fluxcount_continuum = spec.flux_resampled_continuum
                    else:
                        fluxcount = spec.count_resampled
                        fluxcount_sky = spec.count_sky_resampled
                        fluxcount_err = spec.count_err_resampled
                        fluxcount_name = "Count / (e- / s)"
                        fluxcount_sky_name = "Sky Count / (e- / s)"
                        fluxcount_err_name = "Count Uncertainty / (e- / s)"
                        fluxcount_continuum = spec.count_resampled_continuum

                elif self.science_wavelength_calibrated:

                    wave = spec.wave

                    if self.science_flux_calibrated:
                        fluxcount = spec.flux
                        fluxcount_sky = spec.flux_sky
                        fluxcount_err = spec.flux_err
                        fluxcount_name = "Flux"
                        fluxcount_sky_name = "Sky Flux"
                        fluxcount_err_name = "Flux Uncertainty"
                        telluric = spec.telluric_profile
                        fluxcount_continuum = spec.flux_continuum
                    else:
                        fluxcount = spec.count
                        fluxcount_sky = spec.count_sky
                        fluxcount_err = spec.count_err
                        fluxcount_name = "Count / (e- / s)"
                        fluxcount_sky_name = "Sky Count / (e- / s)"
                        fluxcount_err_name = "Count Uncertainty / (e- / s)"
                        fluxcount_continuum = spec.count_continuum

                else:

                    self.logger.warning(
                        "Spectrum is not wavelength "
                        "calibrated, it cannot be plotted."
                    )
                    continue

                wave_mask = (np.array(wave).reshape(-1) > wave_min) & (
                    np.array(wave).reshape(-1) < wave_max
                )

                flux_low = (
                    np.nanpercentile(
                        np.array(fluxcount).reshape(-1)[wave_mask], 10
                    )
                    / 1.5
                )
                flux_high = (
                    np.nanpercentile(
                        np.array(fluxcount).reshape(-1)[wave_mask], 90
                    )
                    * 1.5
                )
                flux_mask = (np.array(fluxcount).reshape(-1) > flux_low) & (
                    np.array(fluxcount).reshape(-1) < flux_high
                )

                if np.sum(flux_mask) > 0:

                    flux_min = np.nanmin(
                        np.array(fluxcount).reshape(-1)[flux_mask]
                    )
                    flux_max = np.nanmax(
                        np.array(fluxcount).reshape(-1)[flux_mask]
                    )

                else:

                    flux_min = np.nanmin(np.array(fluxcount).reshape(-1))
                    flux_max = np.nanmax(np.array(fluxcount).reshape(-1))

                fig_sci = go.Figure(
                    layout=dict(
                        autosize=False,
                        height=height,
                        width=width,
                        updatemenus=list(
                            [
                                dict(
                                    active=0,
                                    buttons=list(
                                        [
                                            dict(
                                                label="Log Scale",
                                                method="update",
                                                args=[
                                                    {"visible": [True, True]},
                                                    {
                                                        "title": "Log scale",
                                                        "yaxis": {
                                                            "type": "log"
                                                        },
                                                    },
                                                ],
                                            ),
                                            dict(
                                                label="Linear Scale",
                                                method="update",
                                                args=[
                                                    {"visible": [True, False]},
                                                    {
                                                        "title": "Linear scale",
                                                        "yaxis": {
                                                            "type": "linear"
                                                        },
                                                    },
                                                ],
                                            ),
                                        ]
                                    ),
                                )
                            ]
                        ),
                        title="Science Spectrum",
                    )
                )

                # show the image on the top
                fig_sci.add_trace(
                    go.Scatter(
                        x=wave,
                        y=fluxcount,
                        line=dict(color="royalblue"),
                        name=fluxcount_name,
                    )
                )

                if fluxcount_err is not None:

                    fig_sci.add_trace(
                        go.Scatter(
                            x=wave,
                            y=fluxcount_err,
                            line=dict(color="firebrick"),
                            name=fluxcount_err_name,
                        )
                    )

                if fluxcount_sky is not None:

                    fig_sci.add_trace(
                        go.Scatter(
                            x=wave,
                            y=fluxcount_sky,
                            line=dict(color="orange"),
                            name=fluxcount_sky_name,
                        )
                    )

                if telluric is not None:

                    fig_sci.add_trace(
                        go.Scatter(
                            x=wave,
                            y=telluric * telluric_factor,
                            line=dict(color="grey"),
                            name="Telluric Correction",
                        )
                    )

                if fluxcount_continuum is not None:

                    fig_sci.add_trace(
                        go.Scatter(
                            x=wave,
                            y=fluxcount_continuum,
                            line=dict(color="black"),
                            name="Continuum",
                        )
                    )

                fig_sci.update_layout(
                    hovermode="closest",
                    showlegend=True,
                    xaxis=dict(
                        title="Wavelength / A", range=[wave_min, wave_max]
                    ),
                    yaxis=dict(
                        title="Flux", range=[flux_min, flux_max], type="linear"
                    ),
                    legend=go.layout.Legend(
                        x=0,
                        y=1,
                        traceorder="normal",
                        font=dict(family="sans-serif", size=12, color="black"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )

                if filename is None:

                    filename = "spectrum"

                if save_fig:

                    fig_type_split = fig_type.split("+")

                    for t in fig_type_split:

                        save_path = filename + "_" + str(i) + "." + t

                        if t == "iframe":

                            pio.write_html(
                                fig_sci, save_path, auto_open=open_iframe
                            )

                        elif t in ["jpg", "png", "svg", "pdf"]:

                            pio.write_image(fig_sci, save_path)

                        self.logger.info(
                            "Figure is saved to {} for the ".format(save_path)
                            + "science_spectrum_list for spec_id: {}.".format(
                                i
                            )
                        )

                if display:

                    if renderer == "default":

                        fig_sci.show()

                    else:

                        fig_sci.show(renderer)

                if return_jsonstring:

                    to_return.append(fig_sci.to_json())

        if "standard" in stype_split:

            spec = self.standard_spectrum_list[0]

            if self.standard_wavelength_calibrated:

                standard_wave = spec.wave

                if self.standard_flux_calibrated:
                    standard_fluxcount = spec.flux
                    standard_fluxcount_sky = spec.flux_sky
                    standard_fluxcount_err = spec.flux_err
                    standard_fluxcount_name = "Flux"
                    standard_fluxcount_sky_name = "Sky Flux"
                    standard_fluxcount_err_name = "Flux Uncertainty"
                    standard_fluxcount_continuum = spec.flux_continuum
                else:
                    standard_fluxcount = spec.count
                    standard_fluxcount_sky = spec.count_sky
                    standard_fluxcount_err = spec.count_err
                    standard_fluxcount_name = "Count / (e- / s)"
                    standard_fluxcount_sky_name = "Sky Count / (e- / s)"
                    standard_fluxcount_err_name = (
                        "Count Uncertainty / (e- / s)"
                    )
                    standard_fluxcount_continuum = spec.count_continuum

            if self.standard_wavelength_resampled:

                standard_wave = spec.wave_resampled

                if self.standard_flux_calibrated:
                    standard_fluxcount = spec.flux_resampled
                    standard_fluxcount_sky = spec.flux_sky_resampled
                    standard_fluxcount_err = spec.flux_err_resampled
                    standard_fluxcount_name = "Flux"
                    standard_fluxcount_sky_name = "Sky Flux"
                    standard_fluxcount_err_name = "Flux Uncertainty"
                    standard_fluxcount_continuum = (
                        spec.flux_resampled_continuum
                    )
                else:
                    standard_fluxcount = spec.count_resampled
                    standard_fluxcount_sky = spec.count_sky_resampled
                    standard_fluxcount_err = spec.count_err_resampled
                    standard_fluxcount_name = "Count / (e- / s)"
                    standard_fluxcount_sky_name = "Sky Count / (e- / s)"
                    standard_fluxcount_err_name = (
                        "Count Uncertainty / (e- / s)"
                    )
                    standard_fluxcount_continuum = (
                        spec.count_resampled_continuum
                    )

            standard_wave_mask = (
                np.array(standard_wave).reshape(-1) > wave_min
            ) & (np.array(standard_wave).reshape(-1) < wave_max)
            standard_flux_mask = (
                np.array(standard_fluxcount).reshape(-1)
                > np.nanpercentile(
                    np.array(standard_fluxcount).reshape(-1)[
                        standard_wave_mask
                    ],
                    10,
                )
                / 1.5
            ) & (
                np.array(standard_fluxcount).reshape(-1)
                < np.nanpercentile(
                    np.array(standard_fluxcount).reshape(-1)[
                        standard_wave_mask
                    ],
                    90,
                )
                * 1.5
            )

            if np.nansum(standard_flux_mask) > 0:

                standard_flux_min = np.nanmin(
                    np.array(standard_fluxcount).reshape(-1)[
                        standard_flux_mask
                    ]
                )
                standard_flux_max = np.nanmax(
                    np.array(standard_fluxcount).reshape(-1)[
                        standard_flux_mask
                    ]
                )

            else:

                standard_flux_min = np.nanmin(
                    np.array(standard_fluxcount).reshape(-1)
                )
                standard_flux_max = np.nanmax(
                    np.array(standard_fluxcount).reshape(-1)
                )

            fig_standard = go.Figure(
                layout=dict(
                    updatemenus=list(
                        [
                            dict(
                                active=0,
                                buttons=list(
                                    [
                                        dict(
                                            label="Log Scale",
                                            method="update",
                                            args=[
                                                {"visible": [True, True]},
                                                {
                                                    "title": "Log scale",
                                                    "yaxis": {"type": "log"},
                                                },
                                            ],
                                        ),
                                        dict(
                                            label="Linear Scale",
                                            method="update",
                                            args=[
                                                {"visible": [True, False]},
                                                {
                                                    "title": "Linear scale",
                                                    "yaxis": {
                                                        "type": "linear"
                                                    },
                                                },
                                            ],
                                        ),
                                    ]
                                ),
                            )
                        ]
                    ),
                    autosize=False,
                    height=height,
                    width=width,
                    title="Standard Spectrum",
                )
            )

            # show the image on the top
            fig_standard.add_trace(
                go.Scatter(
                    x=standard_wave,
                    y=standard_fluxcount,
                    line=dict(color="royalblue"),
                    name=standard_fluxcount_name,
                )
            )

            if standard_fluxcount_err is not None:

                fig_standard.add_trace(
                    go.Scatter(
                        x=standard_wave,
                        y=standard_fluxcount_err,
                        line=dict(color="firebrick"),
                        name=standard_fluxcount_err_name,
                    )
                )

            if standard_fluxcount_sky is not None:

                fig_standard.add_trace(
                    go.Scatter(
                        x=standard_wave,
                        y=standard_fluxcount_sky,
                        line=dict(color="orange"),
                        name=standard_fluxcount_sky_name,
                    )
                )

            if self.fluxcal.standard_fluxmag_true is not None:

                fig_standard.add_trace(
                    go.Scatter(
                        x=self.fluxcal.standard_wave_true,
                        y=self.fluxcal.standard_fluxmag_true,
                        line=dict(color="black"),
                        name="Standard",
                    )
                )

            if standard_fluxcount_continuum is not None:

                fig_standard.add_trace(
                    go.Scatter(
                        x=standard_wave,
                        y=standard_fluxcount_continuum,
                        line=dict(color="grey"),
                        name="Continuum",
                    )
                )

            fig_standard.update_layout(
                hovermode="closest",
                showlegend=True,
                xaxis=dict(title="Wavelength / A", range=[wave_min, wave_max]),
                yaxis=dict(
                    title="Flux",
                    range=[standard_flux_min, standard_flux_max],
                    type="linear",
                ),
                legend=go.layout.Legend(
                    x=0,
                    y=1,
                    traceorder="normal",
                    font=dict(family="sans-serif", size=12, color="black"),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )

            if filename is None:

                filename = "spectrum_standard"

            if save_fig:

                fig_type_split = fig_type.split("+")

                for t in fig_type_split:

                    save_path = filename + "." + t

                    if t == "iframe":

                        pio.write_html(
                            fig_standard, save_path, auto_open=open_iframe
                        )

                    elif t in ["jpg", "png", "svg", "pdf"]:

                        pio.write_image(fig_standard, save_path)

                    self.logger.info(
                        "Figure is saved to {} for the ".format(save_path)
                        + "standard_spectrum_list."
                    )

            if display:

                if renderer == "default":

                    fig_standard.show(height=height, width=width)

                else:

                    fig_standard.show(renderer, height=height, width=width)

            if return_jsonstring:

                to_return.append(fig_standard.to_json())

        if return_jsonstring:

            return to_return

        if ("science" not in stype_split) and ("standard" not in stype_split):

            error_msg = (
                "Unknown stype, please choose from (1) science; "
                + "and/or (2) standard. use + as delimiter."
            )
            self.logger.critical(error_msg)
            raise TypeError(error_msg)

    def create_fits(
        self,
        output="arc_spec+wavecal+wavelength+flux+flux_resampled",
        spec_id=None,
        stype="science+standard",
        recreate=True,
        empty_primary_hdu=True,
    ):
        """
        Create a HDU list, with a choice of any combination of the
        data, see below the 'output' parameters for details.

        Parameters
        ----------
        output: String
            (Default: 'arc_spec+wavecal+wavelength+flux+flux_resampled')
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 4 HDUs
                    Count, uncertainty, sky, optimal flag, and weight (pixel)
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
                    Flux, uncertainty, sky, and sensitivity (pixel)
                sensitivity_resampled: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 4 HDUs
                    Flux, uncertainty, sky, and sensitivity (wavelength)

        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        recreate: bool (Default: True)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank

        """

        # Split the string into strings
        stype_split = stype.split("+")
        output_split = output.split("+")

        for i in output_split:

            if i not in [
                "trace",
                "count",
                "weight_map",
                "arc_spec",
                "wavecal",
                "wavelength",
                "count_resampled",
                "sensitivity",
                "flux",
                "sensitivity_resampled",
                "flux_resampled",
            ]:

                error_msg = "{} is not a valid output.".format(i)
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        if ("science" not in stype_split) and ("standard" not in stype_split):

            error_msg = (
                "Unknown stype, please choose from (1) science; "
                + "and/or (2) standard. use + as delimiter."
            )
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, contraints are applied to all
                #  calibrators
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].create_fits(
                    output=output,
                    recreate=recreate,
                    empty_primary_hdu=empty_primary_hdu,
                )
                self.logger.info(
                    "FITS is created for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].create_fits(
                output=output,
                recreate=recreate,
                empty_primary_hdu=empty_primary_hdu,
            )
            self.logger.info(
                "FITS is created for the "
                "standard_spectrum_list for spec_id: {}.".format(i)
            )

    def modify_trace_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the trace header.

        Parameters
        ----------
        idx: int
            The HDU number of the trace FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_trace_header(
                    idx, method, *args
                )
                self.logger.info(
                    "trace header is moldified for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_trace_header(
                idx, method, *args
            )
            self.logger.info(
                "trace header is moldified for the " "standard_spectrum_list."
            )

    def modify_count_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the count header.

        Parameters
        ----------
        idx: int
            The HDU number of the trace FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_count_header(
                    idx, method, *args
                )
                self.logger.info(
                    "count header is moldified for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_count_header(
                idx, method, *args
            )
            self.logger.info(
                "count header is moldified for the " "standard_spectrum_list."
            )

    def modify_weight_map_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the weight map header.

        Parameters
        ----------
        idx: int
            The HDU number of the weight map FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_weight_map_header(
                    idx, method, *args
                )
                self.logger.info(
                    "weight_map header is moldified for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_weight_map_header(
                idx, method, *args
            )
            self.logger.info(
                "weight_map header is moldified for the "
                "standard_spectrum_list."
            )

    def modify_count_resampled_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the count resampled header.

        Parameters
        ----------
        idx: int
            The HDU number of the count resampled FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_count_resampled_header(
                    idx, method, *args
                )
                self.logger.info(
                    "count_resampled header is moldified for "
                    "the science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_count_resampled_header(
                idx, method, *args
            )
            self.logger.info(
                "count_resampled header is moldified for the "
                "standard_spectrum_list."
            )

    def modify_arc_spec_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the arc spectrum header.

        Parameters
        ----------
        idx: int
            The HDU number of the arc spectrum FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_arc_spec_header(
                    idx, method, *args
                )
                self.logger.info(
                    "arc_spec header is moldified for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_arc_spec_header(
                idx, method, *args
            )
            self.logger.info(
                "arc_spec header is moldified for the "
                "standard_spectrum_list."
            )

    def modify_wavecal_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the wavecal header.

        Parameters
        ----------
        idx: int
            The HDU number of the wavecal FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_wavecal_header(
                    idx, method, *args
                )
                self.logger.info(
                    "wavecal header is moldified for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_wavecal_header(
                idx, method, *args
            )
            self.logger.info(
                "wavecal header is moldified for the "
                "standard_spectrum_list."
            )

    def modify_wavelength_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the wavelength header.

        Parameters
        ----------
        idx: int
            The HDU number of the wavelength FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_wavelength_header(
                    idx, method, *args
                )
                self.logger.info(
                    "wavelength header is moldified for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_wavelength_header(
                idx, method, *args
            )
            self.logger.info(
                "wavelength header is moldified for the "
                "standard_spectrum_list."
            )

    def modify_sensitivity_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the sensitivity header.

        Parameters
        ----------
        idx: int
            The HDU number of the sensitivity FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_sensitivity_header(
                    idx, method, *args
                )
                self.logger.info(
                    "sensitivity header is moldified for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_sensitivity_header(
                idx, method, *args
            )
            self.logger.info(
                "sensitivity header is moldified for the "
                "standard_spectrum_list."
            )

    def modify_flux_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the flux header.

        Parameters
        ----------
        idx: int
            The HDU number of the flux FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_flux_header(
                    idx, method, *args
                )
                self.logger.info(
                    "flux header is moldified for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_flux_header(
                idx, method, *args
            )
            self.logger.info(
                "flux header is moldified for the " "standard_spectrum_list."
            )

    def modify_sensitivity_resampled_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the sensitivity resampled header.

        Parameters
        ----------
        idx: int
            The HDU number of the sensitivity resampled FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[
                    i
                ].modify_sensitivity_resampled_header(idx, method, *args)
                self.logger.info(
                    "sensitivity_resampled header is moldified "
                    "for the science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_sensitivity_resampled_header(
                idx, method, *args
            )
            self.logger.info(
                "sensitivity_resampled header is moldified for "
                "the standard_spectrum_list."
            )

    def modify_flux_resampled_header(
        self, idx, method, *args, spec_id=None, stype="science+standard"
    ):
        """
        Wrapper function to modify the flux resampled header.

        Parameters
        ----------
        idx: int
            The HDU number of the flux resampled FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_flux_resampled_header(
                    idx, method, *args
                )
                self.logger.info(
                    "flux_resampled header is moldified for the "
                    "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].modify_flux_resampled_header(
                idx, method, *args
            )
            self.logger.info(
                "flux_resampled header is moldified for the "
                "standard_spectrum_list."
            )

    def save_fits(
        self,
        spec_id=None,
        output="arc_spec+wavecal+wavelength+flux+flux_resampled",
        filename="reduced",
        stype="science+standard",
        recreate=False,
        empty_primary_hdu=True,
        overwrite=False,
    ):
        """
        Save the reduced data to disk, with a choice of any combination of the
        data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        output: String
            (Default: 'arc_spec+wavecal+wavelength+flux+flux_resampled')
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 4 HDUs
                    Count, uncertainty, sky, optimal flag, and weight (pixel)
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
                flux: 3 HDUs
                    Flux, uncertainty, and sky (pixel)
                sensitivity_resampled: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 3 HDUs
                    Flux, uncertainty, and sky (wavelength)

        filename: String (Default: 'reduced')
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank
        overwrite: bool (Default: False)
            Default is False.

        """

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        stype_split = stype.split("+")
        output_split = output.split("+")

        for i in output_split:

            if i not in [
                "trace",
                "count",
                "weight_map",
                "arc_spec",
                "wavecal",
                "wavelength",
                "count_resampled",
                "sensitivity",
                "flux",
                "sensitivity_resampled",
                "flux_resampled",
            ]:

                error_msg = "{} is not a valid output.".format(i)
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        if ("science" not in stype_split) and ("standard" not in stype_split):

            error_msg = (
                "Unknown stype, please choose from (1) science; "
                + "and/or (2) standard. use + as delimiter."
            )
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    error_msg = "The given spec_id does not exist."
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, contraints are applied to all
                #  calibrators
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                filename_i = filename + "_science_" + str(i)

                self.science_spectrum_list[i].save_fits(
                    output=output,
                    filename=filename_i,
                    overwrite=overwrite,
                    recreate=recreate,
                    empty_primary_hdu=empty_primary_hdu,
                )
                self.logger.info(
                    "FITS file is saved to {} for the ".format(filename_i)
                    + "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].save_fits(
                output=output,
                filename=filename + "_standard",
                overwrite=overwrite,
                recreate=recreate,
                empty_primary_hdu=empty_primary_hdu,
            )
            self.logger.info(
                "FITS file is saved to {}_standard ".format(filename)
                + "for the standard_spectrum_list."
            )

    def save_csv(
        self,
        spec_id=None,
        output="arc_spec+wavecal+wavelength+flux+flux_resampled",
        filename="reduced",
        stype="science+standard",
        recreate=False,
        overwrite=False,
    ):
        """
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        output: String
            (Default: 'arc_spec+wavecal+wavelength+flux+flux_resampled')
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 4 HDUs
                    Count, uncertainty, sky, optimal flag, and weight (pixel)
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
                flux: 3 HDUs
                    Flux, uncertainty, sky, and sensitivity (pixel)
                sensitivity_resampled: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 3 HDUs
                    Flux, uncertainty, sky, and sensitivity (wavelength)

        filename: String (Default: 'reduced')
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.
        overwrite: bool (Default: False)
            Default is False.

        """

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        stype_split = stype.split("+")
        output_split = output.split("+")

        for i in output_split:

            if i not in [
                "trace",
                "count",
                "weight_map",
                "arc_spec",
                "wavecal",
                "wavelength",
                "count_resampled",
                "sensitivity",
                "flux",
                "sensitivity",
                "flux_resampled",
            ]:

                error_msg = "{} is not a valid output.".format(i)
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        if ("science" not in stype_split) and ("standard" not in stype_split):

            error_msg = (
                "Unknown stype, please choose from (1) science; "
                + "and/or (2) standard. use + as delimiter."
            )
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        if "science" in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):

                    raise ValueError("The given spec_id does not exist.")

            else:

                # if spec_id is None, contraints are applied to all
                #  calibrators
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                filename_i = filename + "_science_" + str(i)

                self.science_spectrum_list[i].save_csv(
                    output=output,
                    filename=filename_i,
                    recreate=recreate,
                    overwrite=overwrite,
                )
                self.logger.info(
                    "CSV file is saved to {} for the ".format(filename_i)
                    + "science_spectrum_list for spec_id: {}.".format(i)
                )

        if "standard" in stype_split:

            self.standard_spectrum_list[0].save_csv(
                output=output,
                filename=filename + "_standard",
                recreate=recreate,
                overwrite=overwrite,
            )
            self.logger.info(
                "FITS file is saved to {}_standard ".format(filename)
                + "for the standard_spectrum_list."
            )
