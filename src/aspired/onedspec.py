#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""For One Dimensional operations"""

import copy
import datetime
import logging
import os
from typing import Callable, Union

import numpy as np
import pkg_resources
from astropy.io import fits
from astropy.modeling.polynomial import Chebyshev1D
from plotly import graph_objects as go
from plotly import io as pio
from scipy import optimize
from scipy.interpolate import interp1d
from spectresc import spectres

from .flux_calibration import FluxCalibration
from .spectrum_oneD import SpectrumOneD
from .twodspec import TwoDSpec
from .util import get_continuum
from .wavelength_calibration import WavelengthCalibration

__all__ = ["OneDSpec"]


class OneDSpec:
    def __init__(
        self,
        verbose: bool = True,
        logger_name: str = "OneDSpec",
        log_level: str = "INFO",
        log_file_folder: str = "default",
        log_file_name: str = None,
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
            "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
        )

        if log_file_name is None:
            # Only print log to screen
            self.handler = logging.StreamHandler()
        else:
            if log_file_name == "default":
                t_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                log_file_name = f"{logger_name}_{t_str}.log"
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
            0: SpectrumOneD(
                spec_id=0,
                verbose=self.verbose,
                logger_name=self.logger_name,
                log_level=self.log_level,
                log_file_folder=self.log_file_folder,
                log_file_name=self.log_file_name,
            )
        }

        self.add_science_spectrum_oned(0)

        # Link them up
        self.science_wavecal[0].from_spectrum_oned(
            self.science_spectrum_list[0]
        )
        self.standard_wavecal.from_spectrum_oned(
            self.standard_spectrum_list[0]
        )
        self.fluxcal.from_spectrum_oned(self.standard_spectrum_list[0])

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

        self.science_wavelength_calibrator_available = False
        self.standard_wavelength_calibrator_available = False

        self.science_atlas_available = False
        self.standard_atlas_available = False

        self.science_hough_pairs_available = False
        self.standard_hough_pairs_available = False

        # this means the wavelength is fitted, but not necessarily applied
        self.science_wavecal_coefficients_available = False
        self.standard_wavecal_coefficients_available = False

        self.science_wavelength_calibrated = False
        self.standard_wavelength_calibrated = False

        self.science_wavelength_resampled = False
        self.standard_wavelength_resampled = False

        # This concerns the extinction itself
        self.atmospheric_extinction_correction_available = False

        self.science_telluric_profile_available = False
        self.standard_telluric_profile_available = False
        self.science_telluric_function_available = False
        self.standard_telluric_function_available = False
        self.science_telluric_strength_available = False
        self.standard_telluric_strength_available = False

        # This concerns the spectrum being corrected or not
        self.atmospheric_extinction_corrected = False
        self.science_telluric_corrected = False
        self.standard_telluric_corrected = False
        self.extinction_fraction = 1.0
        self.extinction_func = None

        self.sensitivity_curve_available = False

        self.science_flux_calibrated = False
        self.standard_flux_calibrated = False

        self.science_flux_resampled = False
        self.standard_flux_resampled = False

        self.science_wavelength_resampled_calibrated = False
        self.standard_wavelength_resampled_calibrated = False

    def add_science_spectrum_oned(self, spec_id: int):
        """
        Add a new SpectrumOneD with the ID spec_id. This overwrite the existing
        SpectrumOneD object if it already exists.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object

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

        # Create the SpectrumOneD object for the given spec_id
        self.science_spectrum_list.update(
            {
                spec_id: SpectrumOneD(
                    spec_id=spec_id,
                    verbose=self.verbose,
                    logger_name=self.logger_name,
                    log_level=self.log_level,
                    log_file_folder=self.log_file_folder,
                    log_file_name=self.log_file_name,
                )
            }
        )

        # Reference the wavecal to the SpectrumOneD object just created
        self.science_wavecal[spec_id].from_spectrum_oned(
            self.science_spectrum_list[spec_id]
        )

        self.logger.info(f"spectrm1D object is added to spec_id: {spec_id}")

    def _check_spec_id(
        self, spec_id: Union[int, list, np.ndarray], add_missing=False
    ):
        """
        Check if the spec_id exists and return in the right format.

        Parameters
        ----------
        spec_id: int, list or np.ndarray
            The ID corresponding to the spectrum_oned object.
        add_missing: bool (Default: False)
            Add the spec_id if not exist.

        """

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if spec_id is not None:
            if add_missing:
                for i in spec_id:
                    if i not in list(self.science_spectrum_list.keys()):
                        self.add_science_spectrum_oned(i)

                        self.logger.warning(
                            f"The given spec_id, {spec_id}, does not "
                            "exist. A new spectrum_oned is created. "
                            "Please check you are providing the "
                            "correct spec_id."
                        )

            else:
                if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())
                ):
                    error_msg = (
                        f"The given spec_id: {spec_id} does not exist in the "
                        "spectrum_list "
                        f"{list(self.science_spectrum_list.keys())}."
                    )
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

        else:
            # if spec_id is None, calibrators are initialised to all
            spec_id = list(self.science_spectrum_list.keys())

        return spec_id

    def add_fluxcalibration(self, fluxcal: FluxCalibration):
        """
        Provide the pre-calibrated FluxCalibration object.

        Parameters
        ----------
        fluxcal: FluxCalibration object
            The true mag/flux values.

        """

        if isinstance(fluxcal, FluxCalibration):
            self.fluxcal = fluxcal
            self.logger.info("fluxcal object is added")

        else:
            err_msg = "Please provide a valid FluxCalibration object"
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

    def add_wavelengthcalibration(
        self,
        wavecal: Union[WavelengthCalibration, list],
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Provide the pre-calibrated WavelengthCalibration object.

        Parameters
        ----------
        wavecal: list of WavelengthCalibration object
            The WavelengthPolyFit object for the science target, flux will
            not be calibrated if this is not provided.
        spec_id: int or None (Default: 0)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if isinstance(wavecal, WavelengthCalibration):
            wavecal = [wavecal]

        elif isinstance(wavecal, list):
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
            spec_id = self._check_spec_id(spec_id)

            # Check the sizes of the wave and spec_id and convert wave
            # into a dictionary
            if len(wavecal) == len(spec_id):
                wavecal = {spec_id[i]: wavecal[i] for i in range(len(spec_id))}

            elif len(wavecal) == 1:
                wavecal = {spec_id[i]: wavecal[0] for i in range(len(spec_id))}

            else:
                error_msg = (
                    "wavecal must be the same length of shape as spec_id."
                )
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

            for i in spec_id:
                if isinstance(wavecal[i], WavelengthCalibration):
                    self.science_wavecal[i] = wavecal[i]
                    self.logger.info(
                        "Added WavelengthCalibration to the "
                        f"science_spectrum_list for spec_id: {i}."
                    )

                else:
                    err_msg = (
                        "Please provide a valid WavelengthCalibration object."
                    )
                    self.logger.critical(err_msg)
                    raise TypeError(err_msg)

        if "standard" in stype_split:
            if isinstance(wavecal[0], WavelengthCalibration):
                self.standard_wavecal = wavecal[0]
                self.logger.info(
                    "Added WavelengthCalibration to the standard spectrum_list."
                )

            else:
                err_msg = "Please provide a valid WavelengthCalibration object"
                self.logger.critical(err_msg)
                raise TypeError(err_msg)

    def add_wavelength(
        self,
        wave: Union[np.ndarray, list],
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
        wave : numeric value, list or numpy 1D array (N)
            The wavelength of each pixels of the spectrum.
        spec_id: int (Default: 0)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if isinstance(wave, np.ndarray):
            wave = [wave]

        elif isinstance(wave, list):
            pass

        else:
            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(wave) == len(spec_id):
                    wave = {spec_id[i]: wave[i] for i in range(len(spec_id))}

                elif len(wave) == 1:
                    wave = {spec_id[i]: wave[0] for i in range(len(spec_id))}

                else:
                    error_msg = (
                        "wave must be the same length of shape as spec_id."
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
                            f"science_spectrum_list for spec_id: {i}."
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
                    "Science data is not available, wavelength "
                    + "cannot be added."
                )
                self.logger.warning(err_msg)

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
                    "Standard data is not available, wavelength "
                    + "cannot be added."
                )
                self.logger.warning(err_msg)

    def add_wavelength_resampled(
        self,
        wave_resampled: Union[np.ndarray, list],
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
        spec_id: int (Default: 0)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if isinstance(wave_resampled, np.ndarray):
            wave_resampled = [wave_resampled]

        elif isinstance(wave_resampled, list):
            pass

        else:
            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

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
                            "Added wavelength_resampled list to the science_"
                            f"spectrum_list for spec_id: {i}."
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
                self.logger.warning(err_msg)

        if "standard" in stype_split:
            if self.standard_data_available:
                if len(wave_resampled[0]) == len(
                    self.standard_spectrum_list[0].count
                ):
                    self.standard_spectrum_list[0].add_wavelength_resampled(
                        wave_resampled=wave_resampled[0]
                    )
                    self.logger.info(
                        "Added wavelength list to the standard_spectrum_list."
                    )

                else:
                    err_msg = (
                        "The wavelength provided is of a different "
                        + "size to that of the extracted standard spectrum."
                    )
                    self.logger.critical(err_msg)

                self.standard_wavelength_resampled_calibrated = True

            else:
                err_msg = (
                    "standard data is not available, "
                    + "wavelength_resampled cannot be added."
                )
                self.logger.warning(err_msg)

    def add_spec(
        self,
        count: Union[np.ndarray, list],
        count_err: Union[np.ndarray, list] = None,
        count_sky: Union[np.ndarray, list] = None,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
        count: 1-d array
            The summed count at each column about the trace.
        count_err: 1-d array (Default: None)
            the uncertainties of the count values
        count_sky: 1-d array (Default: None)
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if isinstance(count, np.ndarray):
            count = [count]

        elif isinstance(count, list):
            pass

        else:
            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        if count_err is not None:
            if isinstance(count_err, np.ndarray):
                count_err = [count_err]

            elif isinstance(count_err, list):
                pass

            else:
                err_msg = "Please provide a numpy array or a list of them."
                self.logger.critical(err_msg)
                raise TypeError(err_msg)

        else:
            count_err = [None]

        if count_sky is not None:
            if isinstance(count_sky, np.ndarray):
                count_sky = [count_sky]

            elif isinstance(count_sky, list):
                pass

            else:
                err_msg = "Please provide a numpy array or a list of them."
                self.logger.critical(err_msg)
                raise TypeError(err_msg)

        else:
            count_sky = [None]

        stype_split = stype.split("+")

        if "science" in stype_split:
            spec_id = self._check_spec_id(spec_id, add_missing=True)

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
                    f"science_spectrum_list for spec_id: {i}."
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

    def add_arc_spec(
        self,
        arc_spec: Union[np.ndarray, list],
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
        arc_spec: 1-d array
            The count of the summed 1D arc spec
        spec_id: int (Default: 0)
            The ID corresponding to the spectrum_oned object
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
            spec_id = self._check_spec_id(spec_id, add_missing=True)

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
                    "arc_spec has shape {np.shape(arc_spec)} and spec_id "
                    "has shape {np.shape(spec_id)}."
                )
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

            for i in spec_id:
                self.science_spectrum_list[i].add_arc_spec(
                    arc_spec=arc_spec[i]
                )
                self.logger.info(
                    f"Added arc_spec toscience_spectrum_list for spec_id: {i}."
                )

            self.science_arc_spec_available = True

        if "standard" in stype_split:
            self.standard_spectrum_list[0].add_arc_spec(arc_spec=arc_spec[0])
            self.logger.info("Added arc_spec to standard_spectrum_list.")

            self.standard_arc_spec_available = True

    def add_arc_lines(
        self,
        peaks: Union[np.ndarray, list],
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
        peaks: list of list or list of arrays
            The pixel locations of the arc lines. Multiple traces of the arc
            can be provided as list of list or list of arrays.
        spec_id: int (Default: 0)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if isinstance(peaks, np.ndarray):
            peaks = [peaks]

        elif isinstance(peaks, list):
            pass

        else:
            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:
            spec_id = self._check_spec_id(spec_id, add_missing=True)

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
                    f"Added peaks toscience_spectrum_list for spec_id: {i}."
                )

            self.science_arc_lines_available = True

        if "standard" in stype_split:
            self.standard_spectrum_list[0].add_peaks(peaks=peaks[0])
            self.logger.info("Added peaks to standard_spectrum_list.")

            self.standard_arc_lines_available = True

    def add_trace(
        self,
        trace: Union[np.ndarray, list],
        trace_sigma: Union[np.ndarray, list],
        effective_pixel: Union[np.ndarray, list] = None,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
        trace: list or numpy.ndarray (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: list or numpy.ndarray (N)
            Standard deviation of the Gaussian profile of a trace
        effective_pixel: list or numpy.narray (Default: None)
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(len(spec.trace[0])), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        spec_id: int or None (Default: 0)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if isinstance(trace, np.ndarray):
            trace = [trace]

        elif isinstance(trace, list):
            pass

        else:
            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        if isinstance(trace_sigma, np.ndarray):
            trace_sigma = [trace_sigma]

        elif isinstance(trace_sigma, list):
            pass

        else:
            err_msg = "Please provide a numpy array or a list of them."
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:
            spec_id = self._check_spec_id(spec_id, add_missing=True)

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
                error_msg = "wave must be the same length of shape as spec_id."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

            for i in spec_id:
                self.science_spectrum_list[i].add_trace(
                    trace=trace[i],
                    trace_sigma=trace_sigma[i],
                    effective_pixel=effective_pixel,
                )
                self.logger.info(
                    "Added trace, trace_sigma, and effective_pixel to"
                    f"science_spectrum_list for spec_id: {i}."
                )

            self.science_trace_available = True

        if "standard" in stype_split:
            self.standard_spectrum_list[0].add_trace(
                trace=trace[0],
                trace_sigma=trace_sigma[0],
                effective_pixel=effective_pixel,
            )
            self.logger.info(
                "Added trace, trace_sigma, and effective_pixel to"
                "standard_spectrum_list"
            )

            self.standard_trace_available = True

    def add_fit_coeff(
        self,
        fit_coeff: Union[np.ndarray, list],
        fit_type: str = "poly",
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
        fit_coeff: list or numpy array, or a list of them
            Polynomial fit coefficients.
        fit_type: str or list of str
            Strings starting with 'poly', 'leg' or 'cheb' for polynomial,
            legendre and chebyshev fits. Case insensitive.
        spec_id: int or None (Default: 0)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if isinstance(fit_coeff, np.ndarray):
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

        if isinstance(fit_type, str):
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
            spec_id = self._check_spec_id(spec_id, add_missing=True)

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
                    "wave must be the same length of shape " + "as spec_id."
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
                    f"science_spectrum_list for spec_id: {i}."
                )

            self.science_wavecal_coefficients_available = True

        if "standard" in stype_split:
            self.standard_spectrum_list[0].add_fit_coeff(
                fit_coeff=fit_coeff[0]
            )
            self.standard_spectrum_list[0].add_fit_type(fit_type=fit_type[0])
            self.logger.info(
                "Added fit_coeff and fit_type tostandard_spectrum_list."
            )

            self.standard_wavecal_coefficients_available = True

    def from_twodspec(
        self,
        twodspec: TwoDSpec,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        To add a TwoDSpec object or numpy array to provide the traces, line
        spread function of the traces, optionally the pixel values
        correcponding to the traces. The arc_spec will be imported if
        available.

        Parameters
        ----------
        twodspec: TwoDSpec object
            TwoDSpec containing `the trace` and `trace_sigma`.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            # This cannot use the _check_spec_id because it is looking up the
            # number of spec_id in the TWODSPEC, not ONEDSPEC here
            # spec_id = self._check_spec_id(spec_id)

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

            # reference the spectrum_oned to the WavelengthCalibration
            for i in spec_id:
                self.add_science_spectrum_oned(i)
                self.science_wavecal[i] = WavelengthCalibration(
                    verbose=self.verbose,
                    logger_name=self.logger_name,
                    log_level=self.log_level,
                    log_file_folder=self.log_file_folder,
                    log_file_name=self.log_file_name,
                )

                # By reference
                self.science_wavecal[i].from_spectrum_oned(
                    twodspec.spectrum_list[i]
                )
                self.science_spectrum_list[i] = self.science_wavecal[
                    i
                ].spectrum_oned

                self.logger.info(
                    "Referenced SpectrumOneD of the"
                    f"science_spectrum_list for spec_id: {i}."
                    "to the corresponding science_wavecal."
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
            self.standard_wavecal.from_spectrum_oned(twodspec.spectrum_list[0])
            self.fluxcal.from_spectrum_oned(twodspec.spectrum_list[0])
            self.standard_spectrum_list[
                0
            ] = self.standard_wavecal.spectrum_oned

            self.logger.info(
                "Referenced SpectrumOneD of the"
                "standard_spectrum_list to the standard_wavecal."
            )

            self.standard_data_available = True
            self.standard_arc_available = True
            self.standard_arc_spec_available = True

    def from_fits(
        self,
        fits_file: Union[str, fits.hdu.hdulist.HDUList],
        spec_id: int = 0,
        stype: str = "science",
    ):
        """
        To add a FITS files/object saved from a TwoDSpec object to provide
        the trace, line spread function of the trace, optionally the pixel
        values correcponding to the trace. The arc_spec will be imported if
        available. Note that TwoDSpec exports each trace as a separate file.

        Parameters
        ----------
        fits: FITS filepath/object
            A FITS HDUList containining the `trace`, `trace_sigma`,
            and optionally the `weight_map` and `arc_spec`.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object, If not given,
            it will assign the smallest positive integer that is not taken.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        # If a filepath to a TwoDSpec output FITS is provided
        # Note that HDU0 is an empty PrimaryHDU
        if isinstance(fits_file, str):
            fits_file = fits.open(fits_file)

        if not isinstance(fits_file, fits.hdu.hdulist.HDUList):
            self.logger.critical(
                "A FITS file containing an HDU list is required, "
                f"{type(fits_file)} is given"
            )

        if "science" in stype_split:
            self.add_science_spectrum_oned(spec_id)
            # fits_file[0] is the empty PrimaryHDU
            try:
                self.science_spectrum_list[spec_id].add_trace(
                    fits_file["trace"].data, fits_file["trace_sigma"].data
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.science_spectrum_list[spec_id].add_count(
                    fits_file["count"].data,
                    fits_file["count_err"].data,
                    fits_file["count_sky"].data,
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.science_spectrum_list[spec_id].add_variances(
                    fits_file["weight_map"].data
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.science_spectrum_list[spec_id].add_arc_spec(
                    fits_file["arc_spec"].data
                )
            except KeyError as err:
                self.logger.warning(err)

            try:
                self.science_spectrum_list[spec_id].add_gain(
                    fits_file["count"].header["GAIN"]
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.science_spectrum_list[spec_id].add_readnoise(
                    fits_file["count"].header["RNOISE"]
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.science_spectrum_list[spec_id].add_exptime(
                    fits_file["count"].header["XPOSURE"]
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.science_spectrum_list[spec_id].add_seeing(
                    fits_file["count"].header["SEEING"]
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.science_spectrum_list[spec_id].add_airmass(
                    fits_file["count"].header["AIRMASS"]
                )
            except KeyError as err:
                self.logger.warning(err)

            # reference the spectrum_oned to the WavelengthCalibration
            self.science_wavecal[spec_id] = WavelengthCalibration(
                verbose=self.verbose,
                logger_name=self.logger_name,
                log_level=self.log_level,
                log_file_folder=self.log_file_folder,
                log_file_name=self.log_file_name,
            )

            # By reference
            self.science_wavecal[spec_id].from_spectrum_oned(
                self.science_spectrum_list[spec_id]
            )
            self.science_spectrum_list[spec_id] = self.science_wavecal[
                spec_id
            ].spectrum_oned

            self.logger.info(
                "Referenced SpectrumOneD of the science_spectrum_list "
                f"for spec_id: {spec_id} to the corresponding "
                "science_wavecal."
            )

            self.science_data_available = True
            self.science_arc_available = True
            self.science_arc_spec_available = True

        if "standard" in stype_split:
            try:
                self.standard_spectrum_list[0].add_trace(
                    fits_file["trace"].data, fits_file["trace_sigma"].data
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.standard_spectrum_list[0].add_count(
                    fits_file["count"].data,
                    fits_file["count_err"].data,
                    fits_file["count_sky"].data,
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.standard_spectrum_list[0].add_variances(
                    fits_file["weight_map"].data
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.standard_spectrum_list[0].add_arc_spec(
                    fits_file["arc_spec"].data
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.standard_spectrum_list[0].add_gain(
                    fits_file["count"].header["GAIN"]
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.standard_spectrum_list[0].add_readnoise(
                    fits_file["count"].header["RNOISE"]
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.standard_spectrum_list[0].add_exptime(
                    fits_file["count"].header["XPOSURE"]
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.standard_spectrum_list[0].add_seeing(
                    fits_file["count"].header["SEEING"]
                )
            except KeyError as err:
                self.logger.warning(err)
            try:
                self.standard_spectrum_list[0].add_airmass(
                    fits_file["count"].header["AIRMASS"]
                )
            except KeyError as err:
                self.logger.warning(err)

            # By reference
            self.standard_wavecal = WavelengthCalibration(
                verbose=self.verbose,
                logger_name=self.logger_name,
                log_level=self.log_level,
                log_file_folder=self.log_file_folder,
                log_file_name=self.log_file_name,
            )
            self.standard_wavecal.from_spectrum_oned(
                self.standard_spectrum_list[0]
            )
            self.fluxcal.from_spectrum_oned(self.standard_spectrum_list[0])
            self.standard_spectrum_list[
                0
            ] = self.standard_wavecal.spectrum_oned

            self.logger.info(
                "Referenced SpectrumOneD of the"
                "standard_spectrum_list to the standard_wavecal."
            )

            self.standard_data_available = True
            self.standard_arc_available = True
            self.standard_arc_spec_available = True

    def add_variance(
        self,
        variance: Union[list, np.ndarray],
        stype: str,
        spec_id: int = None,
    ):
        """
        Add variance manually.

        Parameters
        ----------
        variance: 1-d array
            The variance.
        stype: str
            'science' or 'standard' to indicate type.
        spec_id: int or None (Default: 0)
            The ID corresponding to the spectrum_oned object.

        """

        if stype == "science":
            spec_id = self._check_spec_id(spec_id)
            self.science_spectrum_list[spec_id].add_variances(
                variance,
            )

        elif stype == "standard":
            self.standard_spectrum_list[spec_id].add_variances(
                variance,
            )

        else:
            self.logger.error(f"Unknown stype: {stype}.")

    def add_gain(self, gain: float, stype: str, spec_id: int = None):
        """
        Add arc_spec manually.

        Parameters
        ----------
        gain: float
            The gain.
        stype: str
            'science' or 'standard' to indicate type.
        spec_id: int or None (default: 0)
            The ID corresponding to the spectrum_oned object.

        """

        if stype == "science":
            spec_id = self._check_spec_id(spec_id)
            self.science_spectrum_list[spec_id].add_gain(
                gain,
            )

        elif stype == "standard":
            self.standard_spectrum_list[spec_id].add_gain(
                gain,
            )

        else:
            self.logger.error(f"Unknown stype: {stype}.")

    def add_readnoise(self, readnoise: float, stype: str, spec_id: int = None):
        """
        Add arc_spec manually.

        Parameters
        ----------
        readnoise: float
            The readnoise.
        stype: str
            'science' or 'standard' to indicate type.
        spec_id: int or None (default: 0)
            The ID corresponding to the spectrum_oned object.

        """

        if stype == "science":
            spec_id = self._check_spec_id(spec_id)
            self.science_spectrum_list[spec_id].add_readnoise(
                readnoise,
            )

        elif stype == "standard":
            self.standard_spectrum_list[spec_id].add_readnoise(
                readnoise,
            )

        else:
            self.logger.error(f"Unknown stype: {stype}.")

    def add_exptime(self, exptime: float, stype: str, spec_id: int = None):
        """
        Add exptime manually.

        Parameters
        ----------
        exptime: float
            The exptime.
        stype: str
            'science' or 'standard' to indicate type.
        spec_id: int or None (default: 0)
            The ID corresponding to the spectrum_oned object.

        """

        if stype == "science":
            spec_id = self._check_spec_id(spec_id)
            self.science_spectrum_list[spec_id].add_exptime(
                exptime,
            )

        elif stype == "standard":
            self.standard_spectrum_list[spec_id].add_exptime(
                exptime,
            )

        else:
            self.logger.error(f"Unknown stype: {stype}.")

    def add_seeing(self, seeing: float, stype: str, spec_id: int = None):
        """
        Add seeing manually.

        Parameters
        ----------
        seeing: float
            The seeing.
        stype: str
            'science' or 'standard' to indicate type.
        spec_id: int or None (default: 0)
            The ID corresponding to the spectrum_oned object.

        """

        if stype == "science":
            spec_id = self._check_spec_id(spec_id)
            self.science_spectrum_list[spec_id].add_seeing(
                seeing,
            )

        elif stype == "standard":
            self.standard_spectrum_list[spec_id].add_seeing(
                seeing,
            )

        else:
            self.logger.error(f"Unknown stype: {stype}.")

    def add_airmass(self, airmass: float, stype: str, spec_id: int = None):
        """
        Add airmass manually.

        Parameters
        ----------
        airmass: float
            The airmass.
        stype: str
            'science' or 'standard' to indicate type.
        spec_id: int or None (default: 0)
            The ID corresponding to the spectrum_oned object.

        """

        if stype == "science":
            spec_id = self._check_spec_id(spec_id)
            self.science_spectrum_list[spec_id].add_airmass(
                airmass,
            )

        elif stype == "standard":
            self.standard_spectrum_list[spec_id].add_airmass(
                airmass,
            )

        else:
            self.logger.error(f"Unknown stype: {stype}.")

    def find_arc_lines(
        self,
        prominence: float = 5.0,
        top_n_peaks: int = None,
        distance: float = 5.0,
        refine: bool = False,
        refine_window_width: int = 5,
        display: bool = False,
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        renderer: str = "default",
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
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
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        Returns
        -------
        JSON strings if return_jsonstring is set to True

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_arc_spec_available:
                spec_id = self._check_spec_id(spec_id)

                if filename is None:
                    filename = "arc_lines"

                for i in spec_id:
                    if len(spec_id) == 1:
                        filename_i = filename

                    else:
                        filename_i = filename + "_" + str(i)

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
                        filename=filename_i,
                        open_iframe=open_iframe,
                    )

                    n_peaks = len(self.science_spectrum_list[i].peaks)
                    self.logger.info(
                        f"{n_peaks} arc lines are found in "
                        f"science_spectrum_list for spec_id: {i}."
                    )

                self.science_arc_lines_available = True

            else:
                self.logger.warning(
                    "Science arc spectrum/a are not imported."
                    "Unable to find arc lines."
                )

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
                    f"{n_peaks} arc lines are found in standard_spectrum_list."
                )

                self.standard_arc_lines_available = True

            else:
                self.logger.warning(
                    "Standard arc spectrum/a are not imported."
                    "Unable to find arc lines."
                )

    def inspect_arc_lines(
        self,
        display: bool = True,
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        renderer: str = "default",
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
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
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        Returns
        -------
        JSON strings if return_jsonstring is set to True

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_arc_lines_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    if len(spec_id) == 1:
                        filename_i = filename

                    else:
                        filename_i = filename + "_" + str(i)

                    self.science_wavecal[i].inspect_arc_lines(
                        display=display,
                        width=width,
                        height=height,
                        return_jsonstring=return_jsonstring,
                        renderer=renderer,
                        save_fig=save_fig,
                        fig_type=fig_type,
                        filename=filename_i,
                        open_iframe=open_iframe,
                    )

            else:
                self.logger.warning(
                    "Science arc spectrum/a are not imported."
                    "Nothing to inspect."
                )

        if "standard" in stype_split:
            if self.standard_arc_lines_available:
                self.standard_wavecal.inspect_arc_lines(
                    display=display,
                    width=width,
                    height=height,
                    return_jsonstring=return_jsonstring,
                    renderer=renderer,
                    save_fig=save_fig,
                    fig_type=fig_type,
                    filename=filename,
                    open_iframe=open_iframe,
                )

            else:
                self.logger.warning(
                    "Standard arc spectrum/a are not imported. Nothing to"
                    " inspect."
                )

    def initialise_calibrator(
        self,
        peaks: Union[list, np.ndarray] = None,
        arc_spec: Union[list, np.ndarray] = None,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        If the peaks were found with find_arc_lines(), peaks and spectrum can
        be None.

        Parameters
        ----------
        peaks: list, numpy.ndarray or None (Default: None)
            The pixel values of the peaks (start from zero)
        spectrum: list, numpy.ndarray or None (Default: None)
            The spectral intensity as a function of pixel.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            spec_id = self._check_spec_id(spec_id)

            for i in spec_id:
                self.science_wavecal[i].from_spectrum_oned(
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
                    f"science_spectrum_list for spec_id: {i}."
                )

            self.science_wavelength_calibrator_available = True

        if "standard" in stype_split:
            self.standard_wavecal.from_spectrum_oned(
                self.standard_spectrum_list[0]
            )
            self.standard_wavecal.initialise_calibrator(
                peaks=peaks, arc_spec=arc_spec
            )
            self.standard_wavecal.set_calibrator_properties()
            self.standard_wavecal.set_hough_properties()
            self.standard_wavecal.set_ransac_properties()

            self.logger.info(
                "Calibrator is initialised for the standard_spectrum_list."
            )
            self.standard_wavelength_calibrator_available = True

    # placeholder for rascal v0.4
    def set_calibrator_logger(
        self,
        logger_name: str = "Calibrator",
        log_level: str = "info",
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
        logger_name: str (Default: 'Calibrator')
            This will set the name of the logger, if the name is used already,
            it will reference to the existing logger. This will be the
            first part of the default log file name unless log_file_name is
            provided.
        log_level : str (Default: 'info')
            Choose {critical, error, warning, info, debug, notset}.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavelength_calibrator_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_wavecal[i].set_logger(
                        logger_name=logger_name,
                        log_level=log_level,
                    )

            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavelength_calibrator_available:
                self.standard_wavecal.set_logger(
                    logger_name=logger_name,
                    log_level=log_level,
                )

            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

    def set_calibrator_properties(
        self,
        num_pix: int = None,
        effective_pixel: Union[list, np.ndarray] = None,
        plotting_library: str = "plotly",
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
        num_pix: int (Default: None)
            The number of pixels in the dispersion direction
        effective_pixel: list or numpy array (Default: None)
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(num_pix), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        plotting_library : str (Default: 'plotly')
            Choose between matplotlib and plotly.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavelength_calibrator_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_wavecal[i].set_calibrator_properties(
                        num_pix=num_pix,
                        effective_pixel=effective_pixel,
                        plotting_library=plotting_library,
                    )
                    self.logger.info(
                        "Calibrator properties are set for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )

            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavelength_calibrator_available:
                self.standard_wavecal.set_calibrator_properties(
                    num_pix=num_pix,
                    effective_pixel=effective_pixel,
                    plotting_library=plotting_library,
                )
                self.logger.info(
                    "Calibrator properties are set for the"
                    " standard_spectrum_list."
                )

            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

    def set_hough_properties(
        self,
        num_slopes: int = 5000,
        xbins: int = 200,
        ybins: int = 200,
        min_wavelength: float = 3000.0,
        max_wavelength: float = 10000.0,
        range_tolerance: float = 500.0,
        linearity_tolerance: float = 100.0,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
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
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavelength_calibrator_available:
                spec_id = self._check_spec_id(spec_id)

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
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavelength_calibrator_available:
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
                    "Hough properties are set for the standard_spectrum_list."
                )
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

    def set_ransac_properties(
        self,
        sample_size: int = 5,
        top_n_candidate: int = 5,
        linear: bool = True,
        filter_close: bool = False,
        ransac_tolerance: float = 5.0,
        candidate_weighted: bool = True,
        hough_weight: float = 1.0,
        minimum_matches: int = 3,
        minimum_peak_utilisation: float = 0.0,
        minimum_fit_error: float = 1e-4,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Configure the Calibrator. This may require some manual twiddling before
        the calibrator can work efficiently. However, in theory, a large
        max_tries in fit() should provide a good solution in the expense of
        performance (minutes instead of seconds).

        Parameters
        ----------
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
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavelength_calibrator_available:
                spec_id = self._check_spec_id(spec_id)

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
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )
        if "standard" in stype_split:
            if self.standard_wavelength_calibrator_available:
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
                    "Ransac properties are set for the standard_spectrum_list."
                )

            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

    def set_known_pairs(
        self,
        pix: Union[list, np.ndarray] = None,
        wave: Union[list, np.ndarray] = None,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavelength_calibrator_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_wavecal[i].set_known_pairs(pix=pix, wave=wave)
                    self.logger.info(
                        "Known pixel-wavelength pairs are added to "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavelength_calibrator_available:
                self.standard_wavecal.set_known_pairs(pix=pix, wave=wave)
                self.logger.info(
                    "Known pixel-wavelength pairs are added to "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

    def add_user_atlas(
        self,
        elements: Union[list, np.ndarray],
        wavelengths: Union[list, np.ndarray],
        intensities: Union[list, np.ndarray] = None,
        candidate_tolerance: float = 10.0,
        constrain_poly: bool = False,
        vacuum: bool = False,
        pressure: float = 101325.0,
        temperature: float = 273.15,
        relative_humidity: float = 0.0,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
        spec_id: int (Default: 0)
            The ID corresponding to the spectrum_oned object
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
                "Temperature is not provided, set to 0 degrees Celsius."
            )

        if relative_humidity is None:
            relative_humidity = 0.0
            self.logger.warning(
                "Relative humidity is not provided, set to 0%."
            )

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavelength_calibrator_available:
                spec_id = self._check_spec_id(spec_id)

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
                        f"science_spectrum_list for spec_id: {i}."
                    )

                self.science_atlas_available = True

            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavelength_calibrator_available:
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
                    "Added user supplied atlas to standard_spectrum_list."
                )

                self.standard_atlas_available = True
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

    def add_atlas(
        self,
        elements: Union[list, np.ndarray],
        min_atlas_wavelength: float = 3000.0,
        max_atlas_wavelength: float = 10000.0,
        min_intensity: float = 10.0,
        min_distance=10.0,
        candidate_tolerance=10.0,
        constrain_poly=False,
        vacuum=False,
        pressure=101325.0,
        temperature=273.15,
        relative_humidity=0,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
        elements: str or list of strings
            Chemical symbol, case insensitive
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
        spec_id: int (Default: 0)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavelength_calibrator_available:
                spec_id = self._check_spec_id(spec_id)

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
                        "Added atlas to science_spectrum_list for spec_id:"
                        f" {i}."
                    )

                self.science_atlas_available = True
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavelength_calibrator_available:
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
                self.logger.info("Added atlas to standard_spectrum_list")

                self.standard_atlas_available = True
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

    def remove_atlas_lines_range(
        self,
        wavelength: Union[list, np.ndarray],
        tolerance: float = 10.0,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_atlas_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_wavecal[i].remove_atlas_lines_range(
                        wavelength, tolerance
                    )
                    self.logger.info(
                        f"Remove atlas in the range of {wavelength} +/- "
                        f"{tolerance} science_spectrum_list for spec_id: {i}."
                    )

            else:
                self.logger.warning("Science atlas is not available.")

        if "standard" in stype_split:
            if self.standard_atlas_available:
                self.standard_wavecal.remove_atlas_lines_range(
                    wavelength, tolerance
                )
                self.logger.info(
                    f"Remove atlas in the range of {wavelength} +/- "
                    f"{tolerance} standard_spectrum_list."
                )

            else:
                self.logger.warning("Standard atlas is not available.")

    def clear_atlas(
        self,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Remove all the atlas lines from the calibrator.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_atlas_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_wavecal[i].clear_atlas()
                    self.logger.info(
                        "Atlas is removed from "
                        f"science_spectrum_list for spec_id: {i}."
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

    def list_atlas(
        self,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Remove all the atlas lines from the calibrator.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_atlas_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_wavecal[i].list_atlas()
                    self.logger.info(
                        "Listing the atlas of "
                        f"science_spectrum_list for spec_id: {i}."
                    )

            else:
                self.logger.warning("Science atlas is not available.")

        if "standard" in stype_split:
            if self.standard_atlas_available:
                self.standard_wavecal.list_atlas()
                self.logger.info(
                    "Listing the atlas of standard_spectrum_list."
                )

            else:
                self.logger.warning("Standard atlas is not available.")

    def do_hough_transform(
        self,
        brute_force: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
        brute_force: bool (Default: False)
            Set to true to compute the gradient and intercept between
            every two data points
        spec_id: int (Default: 0)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavelength_calibrator_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_wavecal[i].do_hough_transform(
                        brute_force=brute_force
                    )
                    self.logger.info(
                        "Hough Transform is performed on "
                        f"science_spectrum_list for spec_id: {i}."
                    )

                self.science_hough_pairs_available = True

            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavelength_calibrator_available:
                self.standard_wavecal.do_hough_transform(
                    brute_force=brute_force
                )
                self.logger.info(
                    "Hough Transform is performed on standard_spectrum_list."
                )

                self.standard_hough_pairs_available = True
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

    def plot_search_space(
        self,
        fit_coeff: Union[list, np.ndarray] = None,
        top_n_candidate: int = 3,
        weighted: bool = True,
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        return_jsonstring: bool = False,
        renderer: str = "default",
        display: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        A wrapper function to plot the search space in the Hough space.

        If fit fit_coefficients are provided, the model solution will be
        overplotted.

        Parameters
        ----------
        fit_coeff: list (default: None)
            List of best polynomial fit_coefficients
        top_n_candidate: int (default: 3)
            Top ranked lines to be fitted.
        weighted: (default: True)
            Draw sample based on the distance from the matched known wavelength
            of the atlas.
        save_fig: boolean (default: False)
            Save an image if set to True. matplotlib uses the pyplot.save_fig()
            while the plotly uses the pio.write_html() or pio.write_image().
            The support format types should be provided in fig_type.
        fig_type: string (default: 'png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: (default: None)
            The destination to save the image.
        return_jsonstring: (default: False)
            Set to True to save the plotly figure as json string. Ignored if
            matplotlib is used.
        renderer: (default: 'default')
            Set the rendered for the plotly display. Ignored if matplotlib is
            used.
        display: boolean (Default: False)
            Set to True to display disgnostic plot.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_hough_pairs_available:
                spec_id = self._check_spec_id(spec_id)

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
                        f"science_spectrum_list for spec_id: {i}."
                    )

            else:
                self.logger.warning("Science hough pairs are not available.")

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
                self.logger.warning("Standard hour pairs are not available.")

    def fit(
        self,
        max_tries: int = 5000,
        fit_deg: int = 4,
        fit_coeff: Union[list, np.ndarray] = None,
        fit_tolerance: float = 10.0,
        fit_type: str = "poly",
        candidate_tolerance: float = 2.0,
        brute_force: bool = False,
        progress: bool = True,
        return_solution: bool = False,
        display: bool = False,
        renderer: str = "default",
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        solution = {}

        if "science" in stype_split:
            if self.science_hough_pairs_available:
                spec_id = self._check_spec_id(spec_id)

                solution_science = []

                for i in spec_id:
                    self.logger.info(
                        "Attempting to fit wavelength solution for "
                        f"science_spectrum_list for spec_id: {i}."
                    )

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
                        f"science_spectrum_list for spec_id: {i}."
                    )

                self.science_wavecal_coefficients_available = True
                solution["science"] = solution_science

            else:
                self.logger.warning("Science hough pairs are not available.")

        if "standard" in stype_split:
            if self.standard_hough_pairs_available:
                self.logger.info(
                    "Attempting to fit wavelength solution for "
                    "standard_spectrum_list[0]."
                )
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

                self.standard_wavecal_coefficients_available = True

            else:
                self.logger.warning("Standard hough pairs are not imported.")

        if return_solution:
            return solution

    def robust_refit(
        self,
        fit_coeff: Union[list, np.ndarray] = None,
        n_delta: int = None,
        refine: bool = False,
        tolerance: float = 10.0,
        method: str = "Nelder-Mead",
        convergence: float = 1e-6,
        robust_refit: bool = True,
        fit_deg: int = None,
        return_solution: bool = False,
        display: bool = False,
        renderer: str = "default",
        save_fig: bool = False,
        filename: str = None,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **
        Refine the fitted solution with a minimisation method as provided by
        scipy.optimize.minimize().

        Parameters
        ----------
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
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        solution = {}

        if "science" in stype_split:
            if self.science_wavecal_coefficients_available:
                spec_id = self._check_spec_id(spec_id)

                solution_science = []

                for i in spec_id:
                    if fit_coeff is None:
                        fit_coeff = self.science_wavecal[
                            i
                        ].spectrum_oned.calibrator.fit_coeff

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
                        f"science_spectrum_list for spec_id: {i}."
                    )

                solution["science"] = solution_science

            else:
                self.logger.warning(
                    "Wavelength solution is not fitted for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavecal_coefficients_available:
                if fit_coeff is None:
                    fit_coeff = self.standard_wavecal[
                        0
                    ].spectrum_oned.calibrator.fit_coeff

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
                self.logger.warning(
                    "Wavelength solution is not fitted for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

        if return_solution:
            return solution

    def get_pix_wave_pairs(
        self,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Return the list of matched_peaks and matched_atlas with their
        position in the array.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
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
            if self.science_wavecal_coefficients_available:
                spec_id = self._check_spec_id(spec_id)
                pw_pairs_science = []

                for i in spec_id:
                    pw_pairs_science.append(
                        self.science_wavecal[i].get_pix_wave_pairs()
                    )

                pw_pairs["science"] = pw_pairs_science
            else:
                self.logger.warning(
                    "Science pix-wave pairs are not available."
                )

        if "standard" in stype_split:
            if self.standard_wavecal_coefficients_available:
                pw_pairs_standard = self.standard_wavecal.get_pix_wave_pairs()

                pw_pairs["standard"] = pw_pairs_standard
            else:
                self.logger.warning(
                    "Standard pix-wave pairs are not available."
                )
        return pw_pairs

    def add_pix_wave_pair(
        self,
        pix: float,
        wave: float,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
        spec_id: int or None (Default: 0)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavecal_coefficients_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_wavecal[i].add_pix_wave_pair(pix, wave)
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavecal_coefficients_available:
                self.standard_wavecal.add_pix_wave_pair(pix, wave)
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

    def remove_pix_wave_pair(
        self,
        arg: int,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Remove fitted pixel-wavelength pair from the Calibrator for refitting.
        The positions can be found from get_pix_wave_pairs(). One at a time.

        Parameters
        ----------
        arg: int
            The position of the pairs in the arrays.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavecal_coefficients_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_wavecal[i].remove_pix_wave_pair(arg)
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavecal_coefficients_available:
                self.standard_wavecal.remove_pix_wave_pair(arg)
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

    def manual_refit(
        self,
        matched_peaks: Union[list, np.ndarray] = None,
        matched_atlas: Union[list, np.ndarray] = None,
        degree: int = None,
        x0: float = None,
        return_solution: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """
        stype_split = stype.split("+")

        solution = {}

        if "science" in stype_split:
            if self.science_wavecal_coefficients_available:
                spec_id = self._check_spec_id(spec_id)

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
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavecal_coefficients_available:
                solution["standard"] = self.standard_wavecal.manual_refit(
                    matched_peaks=matched_peaks,
                    matched_atlas=matched_atlas,
                    degree=degree,
                    x0=x0,
                    return_solution=return_solution,
                )
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

        if return_solution:
            return solution

    def get_calibrator(
        self,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        stype_split = stype.split("+")

        calibrators = {}

        if "science" in stype_split:
            if self.science_wavelength_calibrator_available:
                spec_id = self._check_spec_id(spec_id)
                calibrator_science = []

                for i in spec_id:
                    calibrator_science.append(
                        getattr(
                            self.science_wavecal[i].spectrum_oned, "calibrator"
                        )
                    )

                calibrators["science"] = calibrator_science
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the science"
                    " spectrum/a, please initialise one before proceeding."
                )

        if "standard" in stype_split:
            if self.standard_wavelength_calibrator_available:
                calibrators["standard"] = getattr(
                    self.standard_wavecal.spectrum_oned, "calibrator"
                )
            else:
                self.logger.warning(
                    "Wavelength calibrator is not available for the standard"
                    " spectrum/a, please initialise one before proceeding."
                )

        return calibrators

    def apply_wavelength_calibration(
        self,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Apply the wavelength calibration.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str or None (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavecal_coefficients_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    spec = self.science_spectrum_list[i]

                    # Adjust for pixel shift due to chip gaps
                    wave = (
                        self.science_wavecal[i]
                        .polyval[spec.fit_type](
                            np.array(spec.effective_pixel), spec.fit_coeff
                        )
                        .reshape(-1)
                    )

                    spec.add_wavelength(wave)

                    self.logger.info(
                        "Wavelength calibration is applied for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )

                self.science_wavelength_calibrated = True

            else:
                self.logger.warning(
                    "Science wavelength calibration cofficients are not"
                    " available."
                )

        if "standard" in stype_split:
            if self.standard_wavecal_coefficients_available:
                spec = self.standard_spectrum_list[0]

                # Adjust for pixel shift due to chip gaps
                wave = self.standard_wavecal.polyval[spec.fit_type](
                    np.array(spec.effective_pixel), spec.fit_coeff
                ).reshape(-1)

                spec.add_wavelength(wave)

                self.logger.info(
                    "Wavelength calibration is applied for the "
                    "standard_spectrum_list."
                )
                self.standard_wavelength_calibrated = True

            else:
                self.logger.warning(
                    "Standard wavelength calibration cofficients are not"
                    " available."
                )

    def lookup_standard_libraries(self, target: str, cutoff: float = 0.4):
        """
        Parameters
        ----------
        target: str
            Name of the standard star
        cutoff: float (Default: 0.4)
            The similarity tolerance [0=completely different, 1=identical]

        """

        self.fluxcal.lookup_standard_libraries(target=target, cutoff=cutoff)

    def load_standard(
        self,
        target: str,
        library: str = None,
        ftype: str = "flux",
        cutoff: float = 0.4,
    ):
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
        self.logger.info(
            f"Loaded standard: {self.fluxcal.target} from"
            f" {self.fluxcal.library}"
        )

    def inspect_standard(
        self,
        display: bool = True,
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        renderer: str = "default",
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
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
        k: int = 3,
        method: str = "interpolate",
        mask_range: list = [[6850, 6960], [7580, 7700]],
        mask_fit_order: int = 1,
        mask_fit_size: int = 5,
        smooth: bool = True,
        return_function: bool = False,
        sens_deg: int = 7,
        use_continuum: bool = False,
        **kwargs,
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
        smooth: bool (Default: True)
            set to smooth the input spectrum with a lowess function with
            statsmodels
        return_function: bool (Default: False)
            Set to True to return the callable function of the sensitivity
            curve.
        sens_deg: int (Default: 7)
            The degree of polynomial of the sensitivity curve, only used if
            the method is 'polynomial'.
        use_continuum: bool (Default: False)
            Set to True to use continuum for finding the sensitivity function.
            If used, the smoothing filter will be applied on the continuum.
        **kwargs:
            keyword arguments for passing to the LOWESS function for getting
            the continuum, see
            `statsmodels.nonparametric.smoothers_lowess.lowess()`

        """

        if getattr(self.standard_spectrum_list[0], "count_continuum") is None:
            self.logger.warning(
                "Continuum is not available, we are runing "
                "get_count_continuum with the default params which "
                "is certainly not optimal. Please check your "
                "results carefully."
            )
            self.get_count_continuum()

        if self.standard_wavelength_calibrated:
            self.fluxcal.get_sensitivity(
                k=k,
                method=method,
                mask_range=mask_range,
                mask_fit_order=mask_fit_order,
                mask_fit_size=mask_fit_size,
                smooth=smooth,
                return_function=return_function,
                sens_deg=sens_deg,
                use_continuum=use_continuum,
                **kwargs,
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

    def save_sensitivity_func(self, filename: str = "sensitivity_func.npy"):
        """
        Not-implemented wrapper.

        Parameters
        ----------
        filename: str
            Filename for the output interpolated sensivity curve.

        """

        self.fluxcal.save_sensitivity_func(filename=filename)
        self.logger.info("Sensitivity curve saved at {filename}.")

    def add_sensitivity_func(self, sensitivity_func: Callable):
        """
        Provide a callable function of the detector sensitivity response.

        Parameters
        ----------
        sensitivity_func: Callable
            Interpolated sensivity curve object.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        self.fluxcal.add_sensitivity_func(sensitivity_func=sensitivity_func)
        self.logger.info("User supplied sensitivity curve added.")
        self.sensitivity_curve_available = True

    def inspect_sensitivity(
        self,
        display: bool = True,
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        renderer: str = "default",
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
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
                "Sensitivity function not available, it cannot be inspected."
            )

    def apply_flux_calibration(
        self,
        inspect: bool = True,
        wave_min: float = 3500.0,
        wave_max: float = 8500.0,
        display: bool = False,
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        renderer: str = "default",
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Apply the computed sensitivity curve. And resample the spectra to
        match the highest resolution (the smallest wavelength bin) part of the
        spectrum.

        Note: This function directly modify the *target_spectrum_oned*.

        Parameters
        ----------
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
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if self.sensitivity_curve_available:
            if "science" in stype_split:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.fluxcal.apply_flux_calibration(
                        target_spectrum_oned=self.science_spectrum_list[i],
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
                        f"science_spectrum_list for spec_id: {i}."
                    )

                self.science_flux_calibrated = True

            if "standard" in stype_split:
                self.fluxcal.apply_flux_calibration(
                    target_spectrum_oned=self.standard_spectrum_list[0],
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

    def _min_std(
        self,
        factor: float,
        flux: Union[list, np.ndarray],
        telluric_profile: Union[list, np.ndarray],
        continuum: Union[list, np.ndarray],
    ):
        """
        ** EXPERIMENTAL, as of 1 October 2021 **

        Minimisation function to get the best mutiplier for the strength
        of the Telluric profile.

        Parameters
        ----------
        factor: float
            The multiplier for the strength of the Telluric profile.
        flux: list or 1-d array (N)
            Flux of the target.
        telluric_profile: list or 1-d array (N)
            Telluric Profile normalised to 1 being the most strongly absorbed,
            0 being outside the Telluric regions.
        continuum: list or 1-d array (N)
            Continuum flux array.

        """

        mask = np.asarray(telluric_profile) != 0
        telluric_absorption = factor * np.asarray(telluric_profile)
        diff = np.asarray(flux) + telluric_absorption - np.asarray(continuum)
        nansum = np.nansum(diff[mask] ** 2.0) * 1e20

        return nansum

    def add_telluric_function(
        self,
        telluric: Callable,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Provide a callable function that gives the Telluric profile.

        Parameters
        ----------
        telluric: callable function
            A function that gives the absorption profile as a function
            of wavelength.
        spec_id: int or None (Default: 0)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    science_spec = self.science_spectrum_list[i]

                    if callable(telluric):
                        science_spec.add_telluric_func(telluric)
                        self.science_telluric_function_available = True

                    elif isinstance(telluric, (np.ndarray, list)):
                        science_spec.add_telluric_func(
                            interp1d(telluric[0], telluric[1])
                        )
                        self.science_telluric_function_available = True

                    else:
                        self.logger.warning(
                            "telluric provided has to be a callable function, "
                            "a list or a np.ndarray. "
                            f"{type(telluric)} is given"
                        )

                    if science_spec.wave is not None:
                        science_spec.add_telluric_profile(
                            science_spec.telluric_func(science_spec.wave)
                        )
                        self.science_telluric_profile_available = True

                    else:
                        self.logger.warning(
                            "wave is not available. Telluric correction cannot"
                            "be performed."
                        )
            else:
                err_msg = (
                    "science data is not available, "
                    + "wavelength_resampled cannot be added."
                )
                self.logger.warning(err_msg)

        if "standard" in stype_split:
            if self.standard_data_available:
                # Add to the standard spectrum
                standard_spec = self.standard_spectrum_list[0]

                if callable(telluric):
                    standard_spec.add_telluric_func(telluric)
                    self.standard_telluric_function_available = True

                elif isinstance(telluric, (np.ndarray, list)):
                    standard_spec.add_telluric_func(
                        interp1d(telluric[0], telluric[1])
                    )
                    self.standard_telluric_function_available = True

                else:
                    self.logger.warning(
                        "telluric provided has to be a callable function, "
                        "a list or a np.ndarray. "
                        "{type(telluric)} is given"
                    )

                if standard_spec.wave is not None:
                    standard_spec.add_telluric_profile(
                        standard_spec.telluric_func(standard_spec.wave)
                    )
                    self.standard_telluric_profile_available = True

                else:
                    self.logger.warning(
                        "wave is not available. Telluric correction cannot"
                        "be performed."
                    )
            else:
                err_msg = (
                    "standard data is not available, "
                    + "wavelength_resampled cannot be added."
                )
                self.logger.warning(err_msg)

    def get_count_continuum(
        self,
        spec_id: Union[int, list, np.ndarray] = None,
        method: str = "lowess",
        stype: str = "science+standard",
        **kwargs: dict,
    ):
        """
        ** fit_generic_continuum is EXPERIMENTAL, as of 23 May 2023 **

        Get the continnum from the wave, count and flux.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        method: str
            "lowess" or "fit". The former uses the lowess function from
            statsmodels. The latter fits with specutil's fit_generic_continuum.
        **kwargs: dictionary
            The keyword arguments for the lowess function or the
            fit_generic_continuum function.

        """

        stype_split = stype.split("+")

        if kwargs is None:
            sci_kwargs = {}
            std_kwargs = {}

        else:
            sci_kwargs = kwargs.copy()
            std_kwargs = kwargs.copy()

        # Note that the order of polynomial is higher in the science than in standard
        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)
                # Get the continuum here
                for i in spec_id:
                    science_spec = self.science_spectrum_list[i]

                    wave = science_spec.wave
                    count = science_spec.count

                    if method == "fit":
                        if "model" not in sci_kwargs:
                            sci_kwargs["model"] = Chebyshev1D(15)

                        if "median_window" not in sci_kwargs:
                            sci_kwargs["median_window"] = 11

                    else:
                        if "frac" not in sci_kwargs:
                            sci_kwargs["frac"] = 0.1

                    science_spec.add_count_continuum(
                        get_continuum(wave, count, method=method, **sci_kwargs)
                    )
            else:
                err_msg = "science data is not available."
                self.logger.warning(err_msg)

        if "standard" in stype_split:
            if self.standard_data_available:
                # Add to the standard spectrum
                standard_spec = self.standard_spectrum_list[0]

                wave = standard_spec.wave
                count = standard_spec.count

                if method == "fit":
                    if "model" not in std_kwargs:
                        std_kwargs["model"] = Chebyshev1D(6)

                    if "median_window" not in std_kwargs:
                        std_kwargs["median_window"] = 11

                else:
                    if "frac" not in std_kwargs:
                        std_kwargs["frac"] = 0.1

                standard_spec.add_count_continuum(
                    get_continuum(wave, count, method=method, **std_kwargs)
                )
            else:
                err_msg = "standard data is not available."
                self.logger.warning(err_msg)

    def get_flux_continuum(
        self,
        spec_id: Union[int, list, np.ndarray] = None,
        method: str = "lowess",
        stype: str = "science+standard",
        **kwargs: dict,
    ):
        """
        ** fit_generic_continuum is EXPERIMENTAL, as of 23 May 2023 **

        Get the continnum from the wave, count and flux.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        method: str
            "lowess" or "fit". The former uses the lowess function from
            statsmodels. The latter fits with specutil's fit_generic_continuum.
        **kwargs: dictionary
            The keyword arguments for the lowess function or the
            fit_generic_continuum function.

        """

        stype_split = stype.split("+")

        if kwargs is None:
            sci_kwargs = {}
            std_kwargs = {}

        else:
            sci_kwargs = kwargs.copy()
            std_kwargs = kwargs.copy()

        # Note that the order of polynomial is higher in the science than in standard
        if "science" in stype_split:
            if self.science_flux_calibrated:
                spec_id = self._check_spec_id(spec_id)
                # Get the continuum here
                for i in spec_id:
                    science_spec = self.science_spectrum_list[i]

                    wave = science_spec.wave
                    flux = science_spec.flux

                    if method == "fit":
                        if "model" not in sci_kwargs:
                            sci_kwargs["model"] = Chebyshev1D(15)

                        if "median_window" not in sci_kwargs:
                            sci_kwargs["median_window"] = 11

                    else:
                        if "frac" not in sci_kwargs:
                            sci_kwargs["frac"] = 0.1

                    science_spec.add_flux_continuum(
                        get_continuum(wave, flux, method=method, **sci_kwargs)
                    )

            else:
                err_msg = "Science flux is not calibrated."
                self.logger.warning(err_msg)

        if "standard" in stype_split:
            if self.standard_flux_calibrated:
                # Add to the standard spectrum
                standard_spec = self.standard_spectrum_list[0]

                wave = standard_spec.wave
                flux = standard_spec.flux

                if method == "fit":
                    if "model" not in std_kwargs:
                        std_kwargs["model"] = Chebyshev1D(6)

                    if "median_window" not in std_kwargs:
                        std_kwargs["median_window"] = 11

                else:
                    if "frac" not in std_kwargs:
                        std_kwargs["frac"] = 0.1

                standard_spec.add_flux_continuum(
                    get_continuum(wave, flux, method=method, **std_kwargs)
                )
            else:
                err_msg = "Standard flux is not calibrated."
                self.logger.warning(err_msg)

    def get_telluric_profile(
        self,
        spec_id: Union[int, list, np.ndarray] = None,
        mask_range: list = [[6850, 6960], [7580, 7700]],
        use_continuum: bool = False,
        return_function: bool = False,
    ):
        """
        Getting the Telluric absorption profile from the continuum of the
        standard star spectrum.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        mask_range: list of list
            list of lists with 2 values indicating the range marked by each
            of the Telluric regions.
        return_function: bool (Default: False)
            Set to True to explicitly return the interpolated function of
            the Telluric profile.

        """

        if self.standard_spectrum_list[0].flux_continuum is None:
            self.logger.error(
                "Flux continuum is not available, please provide a set of"
                " continuum or run get_flux_continuum."
            )

        if use_continuum:
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

        else:
            (
                telluric_func,
                telluric_profile,
                telluric_factor,
            ) = self.fluxcal.get_telluric_profile(
                wave=self.standard_spectrum_list[0].wave,
                flux=self.standard_spectrum_list[0].flux,
                continuum=spectres(
                    self.standard_spectrum_list[0].wave,
                    self.standard_spectrum_list[0].wave_literature,
                    self.standard_spectrum_list[0].flux_literature,
                ),
                mask_range=mask_range,
                return_function=True,
            )

        self.logger.info(
            "Copying the telluric absorption profile to "
            "the science spectrum_oned(s)."
        )

        spec_id = self._check_spec_id(spec_id)

        # Add the telluric profile from fluxcal to science onedspec
        for i in spec_id:
            self.science_spectrum_list[i].add_telluric_func(telluric_func)
            self.science_spectrum_list[i].add_telluric_profile(
                telluric_profile
            )
            self.science_spectrum_list[i].add_telluric_factor(telluric_factor)

        # Add the telluric profile from fluxcal to standard onedspec
        self.standard_spectrum_list[0].add_telluric_func(telluric_func)
        self.standard_spectrum_list[0].add_telluric_profile(telluric_profile)
        self.standard_spectrum_list[0].add_telluric_factor(telluric_factor)

        self.science_telluric_profile_available = True
        self.standard_telluric_profile_available = True
        self.science_telluric_function_available = True
        self.standard_telluric_function_available = True

        if return_function:
            return telluric_func

    def get_telluric_correction(
        self,
        factor: float = 1.0,
        auto_apply: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        **kwargs: dict,
    ):
        self.logger.warning(
            DeprecationWarning(
                "get_telluric_correction() will be removed in version >=0.6."
                "Please use get_telluric_strength()."
            )
        )
        if self.standard_telluric_profile_available:
            self.get_telluric_strength(
                factor=factor, auto_apply=auto_apply, spec_id=spec_id, **kwargs
            )
        elif self.science_telluric_profile_available:
            self.standard_spectrum_list[0].add_telluric_func(
                self.science_spectrum_list[0].telluric_func
            )
            self.standard_spectrum_list[0].add_telluric_profile(
                self.science_spectrum_list[0].telluric_profile
            )
            self.standard_spectrum_list[0].add_telluric_factor(
                self.science_spectrum_list[0].telluric_factor
            )
            self.standard_telluric_profile_available = True
            self.get_telluric_strength(
                factor=factor, auto_apply=auto_apply, spec_id=spec_id, **kwargs
            )
            err_msg = (
                "Telluric profile is missing in the standard, the profile "
                + "from the science is copied over."
            )
            self.logger.warning(err_msg)

        else:
            err_msg = (
                "Telluric profile is missing, try getting one with"
                + "get_telluric_profile()."
            )
            self.logger.warning(err_msg)

    def get_telluric_strength(
        self,
        factor: float = 1.0,
        auto_apply: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        *args: str,
    ):
        """
        Get the telluric absorption profile from the standard star based on
        the masked regions given in generating the sensitivity curve. Note
        that the profile has a "positive" flux so that in the step of applying
        a correction, a POSITIVE constant is found to multiply with the
        normalised telluric profile before ADDING to the spectrum for
        telluric absorption correction (counter-intuitive to the term
        telluric absorption subtraction).

        Parameters
        ----------
        factor: float (Default: 1.0)
            The extra fudge factor multiplied to the telluric profile to
            manally adjust the strength.
        auto_apply: bool (Default: False)
            Set to True to accept the computed telluric absorption correction
            automatically, which is currently an irresversible process through
            the public API.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object

        """

        if (not self.science_telluric_profile_available) & (
            not self.standard_telluric_profile_available
        ):
            error_msg = (
                "Telluric profile is not available. Please provide "
                "one or get one with get_telluric_profile(). Fine tuning can "
                "be done using also get_continuum() on the standard spectrum."
            )
            raise ValueError(error_msg)

        spec_id = self._check_spec_id(spec_id)
        # Get the telluric profile
        for i in spec_id:
            if self.science_data_available:
                if not self.science_telluric_function_available:
                    self.standard_spectrum_list[0].add_telluric_func(
                        self.science_spectrum_list[0].telluric_func
                    )
                    self.standard_spectrum_list[0].add_telluric_profile(
                        self.science_spectrum_list[0].telluric_profile
                    )
                    self.standard_spectrum_list[0].add_telluric_factor(
                        self.science_spectrum_list[0].telluric_factor
                    )
                    self.science_telluric_function_available = True

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
                        self.science_telluric_function_available = True

                wave = science_spec.wave
                flux = science_spec.flux

                if (science_spec.flux_continuum is None) or (len(args) > 0):
                    self.get_flux_continuum(i, *args)

                flux_continuum = science_spec.flux_continuum

                if science_spec.telluric_profile is None:
                    science_spec.add_telluric_profile(
                        science_spec.telluric_func(wave)
                    )
                    self.science_telluric_profile_available = True
                telluric_factor = optimize.minimize(
                    self._min_std,
                    np.nanmedian(np.abs(flux)),
                    args=(flux, science_spec.telluric_profile, flux_continuum),
                    tol=1e-20,
                    method="Nelder-Mead",
                    options={"maxiter": 10000},
                ).x

                science_spec.add_telluric_factor(telluric_factor)
                self.logger.info(f"telluric_factor is {telluric_factor}.")

                self.science_telluric_strength_available = True

            else:
                err_msg = "science data is not available."
                self.logger.warning(err_msg)

        if self.standard_wavelength_calibrated:
            if (
                self.standard_data_available
                & self.standard_telluric_function_available
            ):
                standard_spec = self.standard_spectrum_list[0]
                wave_standard = standard_spec.wave

                if (standard_spec.telluric_profile is None) or (len(args) > 0):
                    standard_spec.add_telluric_profile(
                        standard_spec.telluric_func(wave_standard)
                    )

                    telluric_factor = optimize.minimize(
                        self._min_std,
                        np.nanmedian(np.abs(standard_spec.flux)),
                        args=(
                            standard_spec.flux,
                            standard_spec.telluric_profile,
                            standard_spec.flux_continuum,
                        ),
                        tol=1e-20,
                        method="Nelder-Mead",
                        options={"maxiter": 10000},
                    ).x

                    self.logger.info(f"telluric_factor is {telluric_factor}.")

                    standard_spec.add_telluric_factor(telluric_factor)

                    self.standard_telluric_strength_available = True

            else:
                err_msg = "standard data is not available."
                self.logger.warning(err_msg)

        if auto_apply:
            self.apply_telluric_correction(
                factor=factor, spec_id=spec_id, stype="science+standard"
            )

    def inspect_telluric_profile(
        self,
        display: bool = True,
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        renderer: str = "default",
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
    ):
        """
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

        if not self.standard_telluric_profile_available:
            self.standard_spectrum_list[0].add_telluric_func(
                self.science_spectrum_list[0].telluric_func
            )
            self.standard_spectrum_list[0].add_telluric_profile(
                self.science_spectrum_list[0].telluric_profile
            )
            self.standard_spectrum_list[0].add_telluric_factor(
                self.science_spectrum_list[0].telluric_factor
            )
            self.standard_telluric_profile_available = True

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
        factor: float = 1.0,
        display: bool = True,
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        renderer: str = "default",
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
    ):
        """
        Inspect the Telluric absorption correction on top of the spectra. This
        does NOT apply the correction to the spectrum. This is for inspection
        and manually modifying an extrac multiplier (fnudge factor) to the
        absorption strength.

        Parameters
        ----------
        factor: float (Default: 1.0)
            The extra fudge factor multiplied to the telluric profile to
            manally adjust the strength.
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
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        """

        if (not self.science_telluric_profile_available) & (
            not self.standard_telluric_profile_available
        ):
            error_msg = (
                "Telluric profile is not available. Please provide "
                "one or get one with get_telluric_profile(). Fine tuning can "
                "be done using also get_continuum() on the standard spectrum."
            )
            raise ValueError(error_msg)

        if not self.science_telluric_strength_available:
            error_msg = (
                "Telluric strength is not available. executing "
                "get_telluric_strength()."
            )
            self.get_telluric_strength()

        spec_id = self._check_spec_id(spec_id)

        to_return = []

        # Get the telluric profile
        for i in spec_id:
            spec = self.science_spectrum_list[i]

            wave = spec.wave

            fluxcount = spec.flux
            fluxcount_name = "Flux"
            fluxcount_continuum = spec.flux_continuum
            telluric_factor = spec.telluric_factor
            telluric_func = spec.telluric_func
            spec.add_telluric_nudge_factor(factor)

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
                        fluxcount
                        + telluric_func(wave) * telluric_factor * factor
                    ),
                    line=dict(color="orange"),
                    name="Telluric Corrected Spectrum",
                )
            )

            fig_sci.add_trace(
                go.Scatter(
                    x=wave,
                    y=(telluric_func(wave) * telluric_factor * factor),
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
                    if len(spec_id) == 1:
                        save_path = filename + "." + t

                    else:
                        save_path = filename + "_" + str(i) + "." + t

                    if t == "iframe":
                        pio.write_html(
                            fig_sci, save_path, auto_open=open_iframe
                        )

                    elif t in ["jpg", "png", "svg", "pdf"]:
                        pio.write_image(fig_sci, save_path)

                    self.logger.info(
                        f"Figure is saved to {save_path} for the "
                        f"science_spectrum_list for spec_id: {i}."
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
        self,
        factor: float = None,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Apply the telluric correction with the extra multiplier 'factor'.
        The 'factor' provided in the profile() is propagated to this
        function, it has to be explicitly provided to this function.

        The telluric absorption profile is normalised to 1 at the most
        absorpted wavelegnth, the factor manually provided can be
        negative in case of over/under-subtraction.

        Parameters
        ----------
        factor: float (Default: None)
            The extra fudge factor multiplied to the telluric profile to
            manally adjust the strength.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        if (not self.science_telluric_function_available) and (
            not self.standard_telluric_function_available
        ):
            error_msg = (
                "Telluric function is not available. Please provide "
                "one or get one with get_telluric_profile(). Fine tuning can "
                "be done using also get_continuum() on the standard spectrum."
            )
            raise ValueError(error_msg)

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                if not self.science_telluric_strength_available:
                    error_msg = (
                        "Telluric strength is not available. executing "
                        "get_telluric_strength()."
                    )
                    self.get_telluric_strength()

                spec_id = self._check_spec_id(spec_id)

                # Get the telluric function
                for i in spec_id:
                    science_spec = self.science_spectrum_list[i]

                    if science_spec.telluric_func is None:
                        self.logger.warning(
                            "A resampled telluric function is not available, "
                            "please construct a function with "
                            "get_telluric_profile()."
                        )

                    else:
                        case_a = (
                            self.science_telluric_corrected
                            & self.atmospheric_extinction_corrected
                        )
                        case_b = (
                            not self.science_telluric_corrected
                        ) & self.atmospheric_extinction_corrected
                        # case_c = self.science_telluric_corrected & (
                        #    not self.atmospheric_extinction_corrected
                        # )
                        # case_d = (not self.science_telluric_corrected) & (
                        #    not self.atmospheric_extinction_corrected
                        # )

                        if factor is None:
                            factor = science_spec.telluric_nudge_factor

                        else:
                            science_spec.add_telluric_nudge_factor(factor)

                        # in all cases
                        flux_telluric_corrected = (
                            science_spec.flux
                            + science_spec.telluric_func(science_spec.wave)
                            * science_spec.telluric_factor
                            * factor
                        )
                        science_spec.add_flux_telluric_corrected(
                            flux_telluric_corrected,
                            science_spec.flux_err,
                            science_spec.flux_sky,
                        )

                        if case_a or case_b:
                            flux_atm_ext_telluric_corrected = (
                                science_spec.flux_atm_ext_corrected
                                + science_spec.telluric_func(science_spec.wave)
                                * science_spec.telluric_factor
                                * factor
                            )
                            science_spec.add_flux_atm_ext_telluric_corrected(
                                flux_atm_ext_telluric_corrected,
                                science_spec.flux_err_atm_ext_corrected,
                                science_spec.flux_sky_atm_ext_corrected,
                            )

                # Flag it as corrected
                self.science_telluric_corrected = True
                self.logger.info(
                    "Telluric absorption in the science spectrum is corrected."
                )
            else:
                err_msg = "science data is not available."
                self.logger.warning(err_msg)

        if "standard" in stype_split:
            if self.standard_data_available:
                standard_spec = self.standard_spectrum_list[0]

                if standard_spec.telluric_func is None:
                    self.logger.warning(
                        "A resampled telluric function is not available, "
                        "please construct a function with "
                        "get_telluric_profile()."
                    )

                else:
                    if factor is None:
                        factor = standard_spec.telluric_nudge_factor

                    else:
                        standard_spec.add_telluric_nudge_factor(factor)

                    case_a = (
                        self.standard_telluric_corrected
                        & self.atmospheric_extinction_corrected
                    )
                    case_b = (
                        not self.standard_telluric_corrected
                    ) & self.atmospheric_extinction_corrected
                    # case_c = self.standard_telluric_corrected & (
                    #    not self.atmospheric_extinction_corrected
                    # )
                    # case_d = (not self.standard_telluric_corrected) & (
                    #    not self.atmospheric_extinction_corrected
                    # )

                    # in all cases
                    flux_telluric_corrected = (
                        standard_spec.flux
                        + standard_spec.telluric_func(standard_spec.wave)
                        * standard_spec.telluric_factor
                        * factor
                    )
                    standard_spec.add_flux_telluric_corrected(
                        flux_telluric_corrected,
                        standard_spec.flux_err,
                        standard_spec.flux_sky,
                    )

                    if case_a or case_b:
                        # standard doesn't require atmospheic extinction correction
                        flux_atm_ext_telluric_corrected = (
                            standard_spec.flux
                            + standard_spec.telluric_func(standard_spec.wave)
                            * standard_spec.telluric_factor
                            * factor
                        )
                        standard_spec.add_flux_atm_ext_telluric_corrected(
                            flux_atm_ext_telluric_corrected,
                            standard_spec.flux_err_atm_ext_corrected,
                            standard_spec.flux_sky_atm_ext_corrected,
                        )

                    # Flag it as corrected
                    self.standard_telluric_corrected = True
                    self.logger.info(
                        "Telluric absorption in the standard spectrum is"
                        " corrected."
                    )

            else:
                err_msg = "Standard data is not available."
                self.logger.warning(err_msg)

    def set_atmospheric_extinction(
        self,
        location: str = "orm",
        extinction_func: Callable = None,
        kind: str = "cubic",
        fill_value: str = "extrapolate",
        **kwargs: dict,
    ):
        """
        The ORM atmospheric extinction correction table is taken from
        http://www.ing.iac.es/astronomy/observing/manuals/ps/tech_notes/tn031.pdf

        The MK atmospheric extinction correction table is taken from
        Buton et al. (2013A&A...549A...8B)

        The CP atmospheric extinction correction table is taken from
        Patat et al. (2011A&A...527A..91P)

        The LS atmospheric extinction correction table is taken from
        THE ESO USERS MANUAL 1993
        https://www.eso.org/public/archives/techdocs/pdf/report_0003.pdf

        The KP (Kitt Peak) atmospheric extinction correction table is taken
        from iraf

        The CT (Cerro Tololo) atmospheric extinctioncorrection table is taken
        from iraf

        Parameters
        ----------
        location: str (Default: orm)
            Location of the observatory, currently contains:
            (1) orm - Roque de los Muchachos Observatory (2420 m)
            (2) mk - Mauna Kea (4205 m)
            (3) cp - Cerro Paranal (2635 m)
            (4) ls - La Silla (2400 m) [up to 9000A only]
            (5) kp - Kitt Peak (2096 m)
            (6) ct - Cerro Tololo (2207 m)
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
                "aspired",
                f"extinction/{location.lower()}_atm_extinct.txt",
            )
            extinction_table = np.loadtxt(filename, delimiter=",")
            self.extinction_func = interp1d(
                extinction_table[:, 0],
                extinction_table[:, 1],
                kind=kind,
                fill_value=fill_value,
                **kwargs,
            )
            self.logger.info(
                f"{location.lower()} extinction correction function is loaded."
            )

        self.atmospheric_extinction_correction_available = True

    def apply_atmospheric_extinction_correction(
        self,
        science_airmass: float = None,
        standard_airmass: float = None,
        spec_id: Union[np.ndarray, list, int] = None,
    ):
        """
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
            The ID corresponding to the spectrum_oned object

        """

        spec_id = self._check_spec_id(spec_id)

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
                self.logger.info(f"Airmass is set to be {standard_am}.")

            if isinstance(standard_airmass, str):
                try:
                    standard_am = standard_spec.spectrum_header[
                        standard_airmass
                    ]

                except Exception as e:
                    self.logger.warning(str(e))

                    standard_am = 1.0
                    self.logger.warning(
                        f"Keyword for airmass: {standard_airmass} cannot be "
                        "found in header."
                    )
                    self.logger.warning("Airmass is set to be 1.0")

        else:
            try:
                standard_am = standard_spec.spectrum_header["AIRMASS"]

            except Exception as e:
                self.logger.warning(str(e))

                standard_am = 1.0
                self.logger.warning(
                    "Keyword for airmass: AIRMASS cannot be found in header."
                )
                self.logger.warning("Airmass is set to be 1.0")

        spec_id = self._check_spec_id(spec_id)

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

            self.logger.info(f"Standard airmass is {standard_am}.")
            self.logger.info(f"Science airmass is {science_am}.")

            _interpolated_ext = self.extinction_func(science_spec.wave)
            # Get the atmospheric extinction correction factor
            science_flux_extinction_factor = 10.0 ** (
                -(_interpolated_ext * science_am) / 2.5
            )
            # note that we are still using the science_spec.wave because we
            # want to "uncorrect" the atmospheric correction on the standard
            # star at the wavelength of of the science target
            standard_flux_extinction_factor = 10.0 ** (
                -(_interpolated_ext * standard_am) / 2.5
            )

            # ratio of the +ve flux adjustment due to the airmass of the
            # science observation, and the -ve flux adjustment due to the
            # airmass of the standard observation
            self.extinction_fraction = (
                science_flux_extinction_factor
                / standard_flux_extinction_factor
            )

            self.science_spectrum_list[i].add_atm_ext(self.extinction_fraction)

            case_a = (
                self.science_telluric_corrected
                & self.atmospheric_extinction_corrected
            )
            # case_b = (not self.science_telluric_corrected) &
            #  self.atmospheric_extinction_corrected
            case_c = self.science_telluric_corrected & (
                not self.atmospheric_extinction_corrected
            )
            # case_d = (not self.science_telluric_corrected) &
            # (not self.atmospheric_extinction_corrected)

            # Apply the correction
            science_flux_atm_ext_corrected = (
                copy.deepcopy(science_spec.flux) / self.extinction_fraction
            )
            science_flux_err_atm_ext_corrected = (
                copy.deepcopy(science_spec.flux_err) / self.extinction_fraction
            )
            science_flux_sky_atm_ext_corrected = (
                copy.deepcopy(science_spec.flux_sky) / self.extinction_fraction
            )

            # Add the corrected spectra to the spectrum_oned
            science_spec.add_flux_atm_ext_corrected(
                science_flux_atm_ext_corrected,
                science_flux_err_atm_ext_corrected,
                science_flux_sky_atm_ext_corrected,
            )

            # Add the corrected spectra to the spectrum_oned
            standard_spec.add_flux_atm_ext_corrected(
                standard_spec.flux,
                standard_spec.flux_err,
                standard_spec.flux_sky,
            )

            if case_a or case_c:
                # Apply the correction
                science_flux_atm_ext_telluric_corrected = (
                    copy.deepcopy(science_spec.flux_telluric_corrected)
                    / self.extinction_fraction
                )
                science_flux_err_atm_ext_telluric_corrected = (
                    copy.deepcopy(science_spec.flux_err_telluric_corrected)
                    / self.extinction_fraction
                )
                science_flux_sky_atm_ext_telluric_corrected = (
                    copy.deepcopy(science_spec.flux_sky_telluric_corrected)
                    / self.extinction_fraction
                )

                # Add the corrected spectra to the spectrum_oned
                science_spec.add_flux_atm_ext_telluric_corrected(
                    science_flux_atm_ext_telluric_corrected,
                    science_flux_err_atm_ext_telluric_corrected,
                    science_flux_sky_atm_ext_telluric_corrected,
                )

                # Add the corrected spectra to the spectrum_oned
                standard_spec.add_flux_atm_ext_telluric_corrected(
                    standard_spec.flux_telluric_corrected,
                    standard_spec.flux_err_telluric_corrected,
                    standard_spec.flux_sky_telluric_corrected,
                )

        # Flag it as corrected
        self.atmospheric_extinction_corrected = True
        self.logger.info("Atmospheric extinction is corrected.")

    def inspect_reduced_spectrum(
        self,
        wave_min: float = 3500.0,
        wave_max: float = 8500.0,
        atm_ext_corrected: bool = True,
        telluric_corrected: bool = True,
        display: bool = True,
        width: int = 1280,
        height: int = 720,
        renderer: str = "default",
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
        return_jsonstring: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Parameters
        ----------
        wave_min: float (Default: 3500.)
            Minimum wavelength to display
        wave_max: float (Default: 8500.)
            Maximum wavelength to display
        atm_ext_corrected: bool (Default: True)
            Set to True to use the atmospheric extinction corrected
            spectrum (if available).
        telluric_corrected: bool (Default: True)
            Set to True to use the telluric corrected spectrum (if available).
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
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")
        to_return = []

        if "science" in stype_split:
            spec_id = self._check_spec_id(spec_id)

            for i in spec_id:
                spec = self.science_spectrum_list[i]
                telluric = None

                if self.science_wavelength_calibrated:
                    wave = spec.wave

                    if self.science_flux_calibrated:
                        if (
                            atm_ext_corrected
                            & self.atmospheric_extinction_corrected
                        ):
                            if (
                                telluric_corrected
                                & self.science_telluric_corrected
                            ):
                                fluxcount = (
                                    spec.flux_atm_ext_telluric_corrected
                                )
                                fluxcount_sky = (
                                    spec.flux_sky_atm_ext_telluric_corrected
                                )
                                fluxcount_err = (
                                    spec.flux_err_atm_ext_telluric_corrected
                                )
                                fluxcount_name = "Flux"
                                fluxcount_sky_name = "Sky Flux"
                                fluxcount_err_name = "Flux Uncertainty"
                                telluric = spec.telluric_profile
                                telluric_factor = spec.telluric_factor
                                telluric_nudge_factor = (
                                    spec.telluric_nudge_factor
                                )
                                fluxcount_continuum = spec.flux_continuum

                            else:
                                fluxcount = spec.flux_atm_ext_corrected
                                fluxcount_sky = spec.flux_sky_atm_ext_corrected
                                fluxcount_err = spec.flux_err_atm_ext_corrected
                                fluxcount_name = "Flux"
                                fluxcount_sky_name = "Sky Flux"
                                fluxcount_err_name = "Flux Uncertainty"
                                telluric = spec.telluric_profile
                                telluric_factor = spec.telluric_factor
                                telluric_nudge_factor = (
                                    spec.telluric_nudge_factor
                                )
                                fluxcount_continuum = spec.flux_continuum

                        elif (
                            telluric_corrected
                            & self.science_telluric_corrected
                        ):
                            fluxcount = spec.flux_telluric_corrected
                            fluxcount_sky = spec.flux_sky_telluric_corrected
                            fluxcount_err = spec.flux_err_telluric_corrected
                            fluxcount_name = "Flux"
                            fluxcount_sky_name = "Sky Flux"
                            fluxcount_err_name = "Flux Uncertainty"
                            telluric = spec.telluric_profile
                            telluric_factor = spec.telluric_factor
                            telluric_nudge_factor = spec.telluric_nudge_factor
                            fluxcount_continuum = spec.flux_continuum

                        else:
                            fluxcount = spec.flux
                            fluxcount_sky = spec.flux_sky
                            fluxcount_err = spec.flux_err
                            fluxcount_name = "Flux"
                            fluxcount_sky_name = "Sky Flux"
                            fluxcount_err_name = "Flux Uncertainty"
                            telluric = spec.telluric_profile
                            telluric_factor = spec.telluric_factor
                            telluric_nudge_factor = spec.telluric_nudge_factor
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
                                                        "title": "Log",
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
                                                        "title": "Linear",
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
                            y=telluric
                            * telluric_factor
                            * telluric_nudge_factor,
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
                        if len(spec_id) == 1:
                            save_path = filename + "." + t

                        else:
                            save_path = filename + "_" + str(i) + "." + t

                        if t == "iframe":
                            pio.write_html(
                                fig_sci, save_path, auto_open=open_iframe
                            )

                        elif t in ["jpg", "png", "svg", "pdf"]:
                            pio.write_image(fig_sci, save_path)

                        self.logger.info(
                            f"Figure is saved to {save_path} for the "
                            f"science_spectrum_list for spec_id: {i}."
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
            standard_telluric = None

            if self.standard_wavelength_calibrated:
                standard_wave = spec.wave

                if self.standard_flux_calibrated:
                    if (
                        atm_ext_corrected
                        & self.atmospheric_extinction_corrected
                    ):
                        if (
                            telluric_corrected
                            & self.standard_telluric_corrected
                        ):
                            standard_fluxcount = (
                                spec.flux_atm_ext_telluric_corrected
                            )
                            standard_fluxcount_sky = (
                                spec.flux_sky_atm_ext_telluric_corrected
                            )
                            standard_fluxcount_err = (
                                spec.flux_err_atm_ext_telluric_corrected
                            )
                            standard_fluxcount_name = "Flux"
                            standard_fluxcount_sky_name = "Sky Flux"
                            standard_fluxcount_err_name = "Flux Uncertainty"
                            standard_telluric = spec.telluric_profile
                            standard_telluric_factor = spec.telluric_factor
                            standard_telluric_nudge_factor = (
                                spec.telluric_nudge_factor
                            )
                            standard_fluxcount_continuum = spec.flux_continuum

                        else:
                            standard_fluxcount = spec.flux_atm_ext_corrected
                            standard_fluxcount_sky = (
                                spec.flux_sky_atm_ext_corrected
                            )
                            standard_fluxcount_err = (
                                spec.flux_err_atm_ext_corrected
                            )
                            standard_fluxcount_name = "Flux"
                            standard_fluxcount_sky_name = "Sky Flux"
                            standard_fluxcount_err_name = "Flux Uncertainty"
                            standard_telluric = spec.telluric_profile
                            standard_telluric_factor = spec.telluric_factor
                            standard_telluric_nudge_factor = (
                                spec.telluric_nudge_factor
                            )
                            standard_fluxcount_continuum = spec.flux_continuum

                    elif telluric_corrected & self.standard_telluric_corrected:
                        standard_fluxcount = spec.flux_telluric_corrected
                        standard_fluxcount_sky = (
                            spec.flux_sky_telluric_corrected
                        )
                        standard_fluxcount_err = (
                            spec.flux_err_telluric_corrected
                        )
                        standard_fluxcount_name = "Flux"
                        standard_fluxcount_sky_name = "Sky Flux"
                        standard_fluxcount_err_name = "Flux Uncertainty"
                        standard_telluric = spec.telluric_profile
                        standard_telluric_factor = spec.telluric_factor
                        standard_telluric_nudge_factor = (
                            spec.telluric_nudge_factor
                        )
                        standard_fluxcount_continuum = spec.flux_continuum

                    else:
                        standard_fluxcount = spec.flux
                        standard_fluxcount_sky = spec.flux_sky
                        standard_fluxcount_err = spec.flux_err
                        standard_fluxcount_name = "Flux"
                        standard_fluxcount_sky_name = "Sky Flux"
                        standard_fluxcount_err_name = "Flux Uncertainty"
                        standard_telluric = spec.telluric_profile
                        standard_telluric_factor = spec.telluric_factor
                        standard_telluric_nudge_factor = (
                            spec.telluric_nudge_factor
                        )
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

            else:
                self.logger.warning(
                    "Spectrum is not wavelength "
                    "calibrated, it cannot be plotted."
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

            if standard_telluric is not None:
                fig_standard.add_trace(
                    go.Scatter(
                        x=standard_wave,
                        y=standard_telluric
                        * standard_telluric_factor
                        * standard_telluric_nudge_factor,
                        line=dict(color="grey"),
                        name="Telluric Correction",
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
                        f"Figure is saved to {save_path} for the "
                        "standard_spectrum_list."
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

    def resample(
        self,
        wave_start: float = None,
        wave_end: float = None,
        wave_bin: int = None,
        stype: str = "science+standard",
        spec_id: Union[int, list, np.ndarray] = None,
    ):
        """

        Parameters
        ----------
        wave_min: None (Default to the minimum fitted wavlength)
            Minimum wavelength to display
        wave_max: None (Default to the maximum fitted wavlength)
            Maximum wavelength to display
        wave_bin: None (Deafult to median of the wavelength bin size)
            Provide the resampling bin size
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str or None (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_wavelength_calibrated:
                spec_id = self._check_spec_id(spec_id)
                for i in spec_id:
                    spec = self.science_spectrum_list[i]

                    if spec.wave is not None:
                        # Adjust for pixel shift due to chip gaps
                        wave = spec.wave

                        # compute the new equally-spaced wavelength array
                        if wave_bin is None:
                            wave_bin = np.nanmedian(np.ediff1d(wave))

                        if wave_start is None:
                            wave_start = wave[0]

                        if wave_end is None:
                            wave_end = wave[-1]

                        wave_resampled = np.arange(
                            wave_start, wave_end, wave_bin
                        )
                        spec.add_wavelength_resampled(wave_resampled)
                        self.science_wavelength_resampled = True

                        if spec.count is not None:
                            count_resampled = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(spec.count).reshape(-1),
                                verbose=True,
                            )

                            count_err_resampled = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(spec.count_err).reshape(-1),
                                verbose=True,
                            )

                            count_sky_resampled = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(spec.count_sky).reshape(-1),
                                verbose=True,
                            )

                            spec.add_count_resampled(
                                count_resampled,
                                count_err_resampled,
                                count_sky_resampled,
                            )

                            self.logger.info(
                                f"count is resampled for spec_id: {i}."
                            )

                        if spec.sensitivity is not None:
                            sensitivity_resampled = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(spec.sensitivity).reshape(-1),
                                verbose=True,
                            )

                            spec.add_sensitivity_resampled(
                                sensitivity_resampled,
                            )

                        if spec.flux is not None:
                            flux_resampled = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(spec.flux).reshape(-1),
                                verbose=True,
                            )

                            flux_err_resampled = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(spec.flux_err).reshape(-1),
                                verbose=True,
                            )

                            flux_sky_resampled = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(spec.flux_sky).reshape(-1),
                                verbose=True,
                            )

                            spec.add_flux_resampled(
                                flux_resampled,
                                flux_err_resampled,
                                flux_sky_resampled,
                            )

                            self.logger.info(
                                f"flux is resampled for spec_id: {i}."
                            )

                        if spec.flux_atm_ext_corrected is not None:
                            flux_resampled_atm_ext_corrected = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(spec.flux_atm_ext_corrected).reshape(
                                    -1
                                ),
                                verbose=True,
                            )

                            flux_err_resampled_atm_ext_corrected = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(
                                    spec.flux_err_atm_ext_corrected
                                ).reshape(-1),
                                verbose=True,
                            )

                            flux_sky_resampled_atm_ext_corrected = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(
                                    spec.flux_sky_atm_ext_corrected
                                ).reshape(-1),
                                verbose=True,
                            )

                            spec.add_flux_resampled_atm_ext_corrected(
                                flux_resampled_atm_ext_corrected,
                                flux_err_resampled_atm_ext_corrected,
                                flux_sky_resampled_atm_ext_corrected,
                            )
                            self.logger.info(
                                "flux_atm_ext_corrected is resampled for "
                                f"spec_id: {i}."
                            )

                        if spec.flux_telluric_corrected is not None:
                            flux_resampled_telluric_corrected = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(spec.flux_telluric_corrected).reshape(
                                    -1
                                ),
                                verbose=True,
                            )

                            flux_err_resampled_telluric_corrected = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(
                                    spec.flux_err_telluric_corrected
                                ).reshape(-1),
                                verbose=True,
                            )

                            flux_sky_resampled_telluric_corrected = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(
                                    spec.flux_sky_telluric_corrected
                                ).reshape(-1),
                                verbose=True,
                            )

                            spec.add_flux_resampled_telluric_corrected(
                                flux_resampled_telluric_corrected,
                                flux_err_resampled_telluric_corrected,
                                flux_sky_resampled_telluric_corrected,
                            )
                            self.logger.info(
                                "flux_telluric_corrected is resampled for "
                                f"spec_id: {i}."
                            )

                        if spec.flux_atm_ext_telluric_corrected is not None:
                            flux_resampled_atm_ext_telluric_corrected = (
                                spectres(
                                    np.array(wave_resampled).reshape(-1),
                                    np.array(wave).reshape(-1),
                                    np.array(
                                        spec.flux_atm_ext_telluric_corrected
                                    ).reshape(-1),
                                    verbose=True,
                                )
                            )

                            flux_err_resampled_atm_ext_telluric_corrected = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(
                                    spec.flux_err_atm_ext_telluric_corrected
                                ).reshape(-1),
                                verbose=True,
                            )

                            flux_sky_resampled_atm_ext_telluric_corrected = spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(
                                    spec.flux_sky_atm_ext_telluric_corrected
                                ).reshape(-1),
                                verbose=True,
                            )

                            spec.add_flux_resampled_atm_ext_telluric_corrected(
                                flux_resampled_atm_ext_telluric_corrected,
                                flux_err_resampled_atm_ext_telluric_corrected,
                                flux_sky_resampled_atm_ext_telluric_corrected,
                            )
                            self.logger.info(
                                "flux_resampled_atm_ext_telluric_corrected is "
                                f"resampled for spec_id: {i}."
                            )

            else:
                err_msg = (
                    "Science wavelength is not calibrated, cannot resample."
                )
                self.logger.warning(err_msg)

        if "standard" in stype_split:
            if self.standard_wavelength_calibrated:
                spec = self.standard_spectrum_list[0]

                if spec.wave is not None:
                    # Adjust for pixel shift due to chip gaps
                    wave = spec.wave

                    # compute the new equally-spaced wavelength array
                    if wave_bin is None:
                        wave_bin = np.nanmedian(np.ediff1d(wave))

                    if wave_start is None:
                        wave_start = wave[0]

                    if wave_end is None:
                        wave_end = wave[-1]

                    wave_resampled = np.arange(wave_start, wave_end, wave_bin)
                    spec.add_wavelength_resampled(wave_resampled)
                    self.standard_wavelength_resampled = True

                    if spec.count is not None:
                        count_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.count).reshape(-1),
                            verbose=True,
                        )

                        count_err_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.count_err).reshape(-1),
                            verbose=True,
                        )

                        count_sky_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.count_sky).reshape(-1),
                            verbose=True,
                        )

                        spec.add_count_resampled(
                            count_resampled,
                            count_err_resampled,
                            count_sky_resampled,
                        )

                    self.logger.info(
                        "Wavelength calibration is applied for the "
                        "standard_spectrum_list."
                    )

                    if spec.sensitivity is not None:
                        sensitivity_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.sensitivity).reshape(-1),
                            verbose=True,
                        )

                        spec.add_sensitivity_resampled(
                            sensitivity_resampled,
                        )

                    if spec.flux is not None:
                        flux_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.flux).reshape(-1),
                            verbose=True,
                        )

                        flux_err_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.flux_err).reshape(-1),
                            verbose=True,
                        )

                        flux_sky_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.flux_sky).reshape(-1),
                            verbose=True,
                        )

                        spec.add_flux_resampled(
                            flux_resampled,
                            flux_err_resampled,
                            flux_sky_resampled,
                        )

                    if spec.flux_atm_ext_corrected is not None:
                        flux_resampled_atm_ext_corrected = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.flux_atm_ext_corrected).reshape(-1),
                            verbose=True,
                        )

                        flux_err_resampled_atm_ext_corrected = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.flux_err_atm_ext_corrected).reshape(
                                -1
                            ),
                            verbose=True,
                        )

                        flux_sky_resampled_atm_ext_corrected = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.flux_sky_atm_ext_corrected).reshape(
                                -1
                            ),
                            verbose=True,
                        )

                        spec.add_flux_resampled_atm_ext_corrected(
                            flux_resampled_atm_ext_corrected,
                            flux_err_resampled_atm_ext_corrected,
                            flux_sky_resampled_atm_ext_corrected,
                        )

                    if spec.flux_telluric_corrected is not None:
                        flux_resampled_telluric_corrected = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.flux_telluric_corrected).reshape(-1),
                            verbose=True,
                        )

                        flux_err_resampled_telluric_corrected = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.flux_err_telluric_corrected).reshape(
                                -1
                            ),
                            verbose=True,
                        )

                        flux_sky_resampled_telluric_corrected = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.flux_sky_telluric_corrected).reshape(
                                -1
                            ),
                            verbose=True,
                        )

                        spec.add_flux_resampled_telluric_corrected(
                            flux_resampled_telluric_corrected,
                            flux_err_resampled_telluric_corrected,
                            flux_sky_resampled_telluric_corrected,
                        )

                    if spec.flux_atm_ext_telluric_corrected is not None:
                        flux_resampled_atm_ext_telluric_corrected = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(
                                spec.flux_atm_ext_telluric_corrected
                            ).reshape(-1),
                            verbose=True,
                        )

                        flux_err_resampled_atm_ext_telluric_corrected = (
                            spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(
                                    spec.flux_err_atm_ext_telluric_corrected
                                ).reshape(-1),
                                verbose=True,
                            )
                        )

                        flux_sky_resampled_atm_ext_telluric_corrected = (
                            spectres(
                                np.array(wave_resampled).reshape(-1),
                                np.array(wave).reshape(-1),
                                np.array(
                                    spec.flux_sky_atm_ext_telluric_corrected
                                ).reshape(-1),
                                verbose=True,
                            )
                        )

                        spec.add_flux_resampled_atm_ext_telluric_corrected(
                            flux_resampled_atm_ext_telluric_corrected,
                            flux_err_resampled_atm_ext_telluric_corrected,
                            flux_sky_resampled_atm_ext_telluric_corrected,
                        )

            else:
                err_msg = (
                    "Standard wavelength is not calibrated, cannot resample."
                )
                self.logger.warning(err_msg)

    def create_fits(
        self,
        output: str = "*",
        recreate: bool = True,
        empty_primary_hdu: bool = True,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Create a HDU list, with a choice of any combination of the
        data, see below the 'output' parameters for details.

        Parameters
        ----------
        output: String
            (Default: '*')
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
                    arc line position (pixel), and arc line effective
                    position (pixel)
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


        recreate: bool (Default: True)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # If output is *, chamge it to everything
        if output == "*":
            output = (
                "trace+count+weight_map+arc_spec+wavecal_coefficients+"
                + "wavelength+wavelength_resampled+count_resampled+"
                + "sensitivity+flux+atm_ext+flux_atm_ext_corrected+"
                + "flux_telluric_corrected+telluric_profile+"
                + "flux_atm_ext_telluric_corrected+sensitivity_resampled+"
                + "flux_resampled+atm_ext_resampled+"
                + "flux_resampled_atm_ext_corrected+"
                + "telluric_profile_resampled+"
                + "flux_resampled_telluric_corrected+"
                + "flux_resampled_atm_ext_telluric_corrected"
            )

        # Split the string into strings
        stype_split = stype.split("+")
        output_split = output.split("+")

        for i in output_split:
            if i not in [
                "trace",
                "count",
                "weight_map",
                "arc_spec",
                "arc_lines",
                "wavecal_coefficients",
                "wavelength",
                "wavelength_resampled",
                "count_resampled",
                "sensitivity",
                "flux",
                "atm_ext",
                "flux_atm_ext_corrected",
                "telluric_profile",
                "flux_telluric_corrected",
                "flux_atm_ext_telluric_corrected",
                "sensitivity_resampled",
                "flux_resampled",
                "atm_ext_resampled",
                "flux_resampled_atm_ext_corrected",
                "telluric_profile_resampled",
                "flux_resampled_telluric_corrected",
                "flux_resampled_atm_ext_telluric_corrected",
            ]:
                error_msg = f"{i} is not a valid output."
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
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    spec = self.science_spectrum_list[i]
                    for j in output_split:
                        if ("resampled" in j) and (not spec.hdu_content[j]):
                            self.resample(stype="science")

                    spec.create_fits(
                        output=output,
                        recreate=recreate,
                        empty_primary_hdu=empty_primary_hdu,
                    )

                    self.logger.info(
                        "FITS is created for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                spec = self.standard_spectrum_list[0]

                for j in output_split:
                    if ("resampled" in j) & (not spec.hdu_content[j]):
                        self.resample(stype="standard")

                spec.create_fits(
                    output=output,
                    recreate=recreate,
                    empty_primary_hdu=empty_primary_hdu,
                )
                self.logger.info(
                    "FITS is created for the "
                    f"standard_spectrum_list for spec_id: {i}."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_trace_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_trace_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "trace header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_trace_header(
                    idx, method, *args
                )
                self.logger.info(
                    "trace header is moldified for the standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_count_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_count_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "count header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_count_header(
                    idx, method, *args
                )
                self.logger.info(
                    "count header is moldified for the standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_weight_map_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_weight_map_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "weight_map header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_weight_map_header(
                    idx, method, *args
                )
                self.logger.info(
                    "weight_map header is moldified for the"
                    " standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_arc_spec_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_arc_spec_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "arc_spec header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_arc_spec_header(
                    idx, method, *args
                )
                self.logger.info(
                    "arc_spec header is moldified for the"
                    " standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_arc_lines_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the arc lines header.

        Parameters
        ----------
        idx: int
            The HDU number of the arc lines FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_arc_lines_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "arc_lines header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_arc_lines_header(
                    idx, method, *args
                )
                self.logger.info(
                    "arc_lines header is moldified for the"
                    " standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_wavecal_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_wavecal_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "wavecal header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_wavecal_header(
                    idx, method, *args
                )
                self.logger.info(
                    "wavecal header is moldified for the"
                    " standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_wavelength_header(
        self,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_wavelength_header(
                        method, *args
                    )
                    self.logger.info(
                        "wavelength header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_wavelength_header(
                    method, *args
                )
                self.logger.info(
                    "wavelength header is moldified for the"
                    " standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_wavelength_resampled_header(
        self,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the wavelength_resampled header.

        Parameters
        ----------
        idx: int
            The HDU number of the wavelength_resampled FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_wavelength_resampled_header(method, *args)
                    self.logger.info(
                        "wavelength header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[
                    0
                ].modify_wavelength_resampled_header(method, *args)
                self.logger.info(
                    "wavelength header is moldified for the"
                    " standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_count_resampled_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_count_resampled_header(idx, method, *args)
                    self.logger.info(
                        "count_resampled header is moldified for "
                        f"the science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_count_resampled_header(
                    idx, method, *args
                )
                self.logger.info(
                    "count_resampled header is moldified for the "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_sensitivity_header(
        self,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_sensitivity_header(
                        method, *args
                    )
                    self.logger.info(
                        "sensitivity header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_sensitivity_header(
                    method, *args
                )
                self.logger.info(
                    "sensitivity header is moldified for the "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_flux_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_flux_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "flux header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_flux_header(
                    idx, method, *args
                )
                self.logger.info(
                    "flux header is moldified for the standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_atm_ext_header(
        self,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the atmospheric extinction factor header.

        Parameters
        ----------
        idx: int
            The HDU number of the sensitivity FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_atm_ext_header(
                        method, *args
                    )
                    self.logger.info(
                        "atm_ext header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_atm_ext_header(
                    method, *args
                )
                self.logger.info(
                    "atm_ext header is moldified for the"
                    " standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_flux_atm_ext_corrected_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the flux_atm_ext_corrected header.

        Parameters
        ----------
        idx: int
            The HDU number of the flux FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_flux_atm_ext_corrected_header(idx, method, *args)
                    self.logger.info(
                        "flux_atm_ext_corrected header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[
                    0
                ].modify_flux_atm_ext_corrected_header(idx, method, *args)
                self.logger.info(
                    "flux_atm_ext_corrected header is moldified for the "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_telluric_profile_header(
        self,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the telluric_profile header.

        Parameters
        ----------
        idx: int
            The HDU number of the telluric_profile FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_telluric_profile_header(method, *args)
                    self.logger.info(
                        "telluric_profile header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_telluric_profile_header(
                    method, *args
                )
                self.logger.info(
                    "telluric_profile header is moldified for the "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_flux_telluric_corrected_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the flux_atm_ext_corrected header.

        Parameters
        ----------
        idx: int
            The HDU number of the flux FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_flux_telluric_corrected_header(idx, method, *args)
                    self.logger.info(
                        "flux_telluric_corrected header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[
                    0
                ].modify_flux_telluric_corrected_header(idx, method, *args)
                self.logger.info(
                    "flux_telluric_corrected header is moldified for the "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_flux_atm_ext_telluric_corrected_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the flux_atm_ext_telluric_corrected header.

        Parameters
        ----------
        idx: int
            The HDU number of the flux FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_flux_atm_ext_telluric_corrected_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "flux_atm_ext_telluric_corrected header is moldified"
                        f" for the science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[
                    0
                ].modify_flux_atm_ext_telluric_corrected_header(
                    idx, method, *args
                )
                self.logger.info(
                    "flux_atm_ext_telluric_corrected header is moldified for"
                    " the standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_sensitivity_resampled_header(
        self,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the sensitivity resampled header.

        Parameters
        ----------
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_sensitivity_resampled_header(method, *args)
                    self.logger.info(
                        "sensitivity_resampled header is moldified "
                        f"for the science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[
                    0
                ].modify_sensitivity_resampled_header(method, *args)
                self.logger.info(
                    "sensitivity_resampled header is moldified for "
                    "the standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_flux_resampled_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
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
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[i].modify_flux_resampled_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "flux_resampled header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_flux_resampled_header(
                    idx, method, *args
                )
                self.logger.info(
                    "flux_resampled header is moldified for the "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_atm_ext_resampled_header(
        self,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the resampled atmospheric extinction
        factor header.

        Parameters
        ----------
        idx: int
            The HDU number of the resampled atmospheric extinction FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_atm_ext_resampled_header(method, *args)
                    self.logger.info(
                        "atm_ext_resampled header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[0].modify_atm_ext_resampled_header(
                    method, *args
                )
                self.logger.info(
                    "atm_ext_resampled header is moldified for the "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_flux_resampled_atm_ext_corrected_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the atmospheric extinction corrected flux
        resampled header.

        Parameters
        ----------
        idx: int
            The HDU number of the flux resampled FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_flux_resampled_atm_ext_corrected_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "flux_resampled_atm_ext header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[
                    0
                ].modify_flux_resampled_atm_ext_corrected_header(
                    idx, method, *args
                )
                self.logger.info(
                    "flux_resampled_atm_ext header is moldified for the "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_telluric_profile_resampled_header(
        self,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the resampled telluric profile header.

        Parameters
        ----------
        idx: int
            The HDU number of the telluric_profile FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_telluric_profile_resampled_header(method, *args)
                    self.logger.info(
                        "telluric_profile_resampled header is moldified for"
                        f" the science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[
                    0
                ].modify_telluric_profile_resampled_header(method, *args)
                self.logger.info(
                    "telluric_profile_resampled header is moldified for the "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_flux_resampled_telluric_corrected_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the telluric absorption corrected flux
        resampled header.

        Parameters
        ----------
        idx: int
            The HDU number of the flux resampled FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_flux_resampled_telluric_corrected_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "flux_resampled_telluric header is moldified for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[
                    0
                ].modify_flux_resampled_telluric_corrected_header(
                    idx, method, *args
                )
                self.logger.info(
                    "flux_resampled_telluric header is moldified for the "
                    "standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def modify_flux_resampled_atm_ext_telluric_corrected_header(
        self,
        idx: int,
        method: str,
        *args: str,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Wrapper function to modify the telluric absorption corrected flux
        resampled header.

        Parameters
        ----------
        idx: int
            The HDU number of the flux resampled FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split("+")

        if "science" in stype_split:
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    self.science_spectrum_list[
                        i
                    ].modify_flux_resampled_atm_ext_telluric_corrected_header(
                        idx, method, *args
                    )
                    self.logger.info(
                        "flux_resampled_atm_ext_telluric header is moldified"
                        f" for the science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                self.standard_spectrum_list[
                    0
                ].modify_flux_resampled_atm_ext_telluric_corrected_header(
                    idx, method, *args
                )
                self.logger.info(
                    "flux_resampled_atm_ext_telluric header is moldified for"
                    " the standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def save_fits(
        self,
        output: str = "*",
        filename: str = "reduced",
        recreate: bool = False,
        empty_primary_hdu: bool = True,
        overwrite: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Save the reduced data to disk, with a choice of any combination of the
        data, see below the 'output' parameters for details.

        Parameters
        ----------
        output: String
            (Default: '*')
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

        filename: String (Default: 'reduced')
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank
        overwrite: bool (Default: False)
            Default is False.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # If output is *, chamge it to everything
        if output == "*":
            output = (
                "trace+count+weight_map+arc_spec+arc_lines+"
                + "wavecal_coefficients+wavelength+wavelength_resampled+"
                + "count_resampled+sensitivity+flux+atm_ext+"
                + "flux_atm_ext_corrected+flux_telluric_corrected+"
                + "telluric_profile+flux_atm_ext_telluric_corrected+"
                + "sensitivity_resampled+flux_resampled+atm_ext_resampled+"
                + "flux_resampled_atm_ext_corrected+"
                + "telluric_profile_resampled+"
                + "flux_resampled_telluric_corrected+"
                + "flux_resampled_atm_ext_telluric_corrected"
            )

        # Split the string into strings
        stype_split = stype.split("+")
        output_split = output.split("+")

        for i in output_split:
            if i not in [
                "trace",
                "count",
                "weight_map",
                "arc_spec",
                "arc_lines",
                "wavecal_coefficients",
                "wavelength",
                "wavelength_resampled",
                "count_resampled",
                "sensitivity",
                "flux",
                "atm_ext",
                "flux_atm_ext_corrected",
                "telluric_profile",
                "flux_telluric_corrected",
                "flux_atm_ext_telluric_corrected",
                "sensitivity_resampled",
                "flux_resampled",
                "atm_ext_resampled",
                "flux_resampled_atm_ext_corrected",
                "telluric_profile_resampled",
                "flux_resampled_telluric_corrected",
                "flux_resampled_atm_ext_telluric_corrected",
            ]:
                error_msg = f"{i} is not a valid output."
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
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)

                for i in spec_id:
                    if len(spec_id) == 1:
                        filename_i = filename + "_science"
                    else:
                        filename_i = filename + "_science_" + str(i)

                    spec = self.science_spectrum_list[i]
                    output_filtered = []

                    for j in output_split:
                        if spec.hdu_content[j]:
                            output_filtered.append(j)

                        elif not spec.hdu_content[j]:
                            self.create_fits(j)
                            output_filtered.append(j)

                        else:
                            pass

                    spec.save_fits(
                        output="+".join(output_filtered),
                        filename=filename_i,
                        overwrite=overwrite,
                        recreate=recreate,
                        empty_primary_hdu=empty_primary_hdu,
                    )
                    self.logger.info(
                        f"FITS file is saved to {filename_i} for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                spec = self.standard_spectrum_list[0]
                output_filtered = []

                for j in output_split:
                    if spec.hdu_content[j]:
                        output_filtered.append(j)

                    elif not spec.hdu_content[j]:
                        self.create_fits(j)
                        output_filtered.append(j)

                    else:
                        pass

                spec.save_fits(
                    output="+".join(output_filtered),
                    filename=filename + "_standard",
                    overwrite=overwrite,
                    recreate=recreate,
                    empty_primary_hdu=empty_primary_hdu,
                )
                self.logger.info(
                    f"FITS file is saved to {filename}_standard "
                    "for the standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")

    def save_csv(
        self,
        output: str = "*",
        filename: str = "reduced",
        recreate: bool = False,
        overwrite: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
        stype: str = "science+standard",
    ):
        """
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum_oned object
        output: String
            (Default: '*')
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

        # If output is *, chamge it to everything
        if output == "*":
            output = (
                "trace+count+weight_map+arc_spec+arc_lines+"
                + "wavecal_coefficients+wavelength+wavelength_resampled+"
                + "count_resampled+sensitivity+flux+atm_ext+"
                + "flux_atm_ext_corrected+telluric_profile+"
                + "flux_telluric_corrected+flux_atm_ext_telluric_corrected+"
                + "sensitivity_resampled+flux_resampled+atm_ext_resampled+"
                + "flux_resampled_atm_ext_corrected+"
                + "telluric_profile_resampled+"
                + "flux_resampled_telluric_corrected+"
                + "flux_resampled_atm_ext_telluric_corrected"
            )

        # Split the string into strings
        stype_split = stype.split("+")
        output_split = output.split("+")

        for i in output_split:
            if i not in [
                "trace",
                "count",
                "weight_map",
                "arc_spec",
                "arc_lines",
                "wavecal_coefficients",
                "wavelength",
                "wavelength_resampled",
                "count_resampled",
                "sensitivity",
                "flux",
                "atm_ext",
                "flux_atm_ext_corrected",
                "telluric_profile",
                "flux_telluric_corrected",
                "flux_atm_ext_telluric_corrected",
                "sensitivity_resampled",
                "flux_resampled",
                "atm_ext_resampled",
                "flux_resampled_atm_ext_corrected",
                "telluric_profile_resampled",
                "flux_resampled_telluric_corrected",
                "flux_resampled_atm_ext_telluric_corrected",
            ]:
                error_msg = f"{i} is not a valid output."
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
            if self.science_data_available:
                spec_id = self._check_spec_id(spec_id)
                for i in spec_id:
                    spec = self.science_spectrum_list[i]
                    output_filtered = []

                    for j in output_split:
                        if ("resampled" in j) and (not spec.hdu_content[j]):
                            self.resample(stype="science")

                        elif not spec.hdu_content[j]:
                            self.create_fits(j)

                        else:
                            pass

                        if spec.hdu_content[j]:
                            output_filtered.append(j)

                    if len(spec_id) == 1:
                        filename_i = filename + "_science"

                    else:
                        filename_i = filename + "_science_" + str(i)

                    spec.save_csv(
                        output="+".join(output_filtered),
                        filename=filename_i,
                        recreate=recreate,
                        overwrite=overwrite,
                    )
                    self.logger.info(
                        f"CSV file is saved to {filename_i} for the "
                        f"science_spectrum_list for spec_id: {i}."
                    )
            else:
                self.logger.warning("Science data is not available.")

        if "standard" in stype_split:
            if self.standard_data_available:
                spec = self.standard_spectrum_list[0]
                output_filtered = []

                for j in output_split:
                    if ("resampled" in j) and (not spec.hdu_content[j]):
                        self.resample(stype="standard")

                    elif not spec.hdu_content[j]:
                        self.create_fits(j)

                    else:
                        pass

                    if spec.hdu_content[j]:
                        output_filtered.append(j)

                spec.save_csv(
                    output="+".join(output_filtered),
                    filename=filename + "_standard",
                    recreate=recreate,
                    overwrite=overwrite,
                )
                self.logger.info(
                    f"FITS file is saved to {filename}_standard "
                    "for the standard_spectrum_list."
                )
            else:
                self.logger.warning("Standard data is not available.")
