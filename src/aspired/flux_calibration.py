#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""For Flux Calibration"""

import datetime
import difflib
import json
import logging
import os
from typing import Callable, Union

import numpy as np
import pkg_resources
from plotly import graph_objects as go
from plotly import io as pio
from scipy import interpolate as itp
from scipy import signal
from spectresc import spectres
from statsmodels.nonparametric.smoothers_lowess import lowess

from .spectrum_oneD import SpectrumOneD

base_dir = os.path.dirname(__file__)

__all__ = ["StandardLibrary", "FluxCalibration"]


class StandardLibrary:
    """
    This class handles flux calibration by comparing the extracted and
    wavelength-calibrated standard observation to the "ground truth"
    from

    https://github.com/iraf-community/iraf/tree/master/noao/lib/onedstandards
    https://www.eso.org/sci/observing/tools/standards/spectra.html

    See explanation notes at those links for details.

    Parameters
    ----------
    verbose: bool (Default: True)
        Set to False to suppress all verbose warnings, except for
        critical failure.
    logger_name: str (Default: StandardLibrary)
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
        File name of the log, set to None to logging.warning to screen
        only.

    """

    def __init__(
        self,
        verbose: bool = True,
        logger_name: str = "StandardLibrary",
        log_level: str = "INFO",
        log_file_folder: str = "default",
        log_file_name: str = None,
    ):
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
                d_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                log_file_name = f"{logger_name}_{d_str}.log"
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

        self.standard_fluxmag_true = False
        self.standard_wave_true = False

        self.library = None
        self.target = None
        self.designation = None
        self.ftype = "flux"
        self.cutoff = 0.4

        self.designation_to_lib_filename = None
        self.lib_to_filename = None
        self.lib_to_designation = None
        self.filename_to_lib = None

        self.spectrum_oned_imported = False

        self._load_standard_dictionary()

    def _load_standard_dictionary(self):
        """
        Load the dictionaries containing the names of all the standard stars.

        """

        self.designation_to_lib_filename = json.load(
            open(
                pkg_resources.resource_filename(
                    "aspired", "standards/designation_to_lib_filename.json"
                ),
                encoding="ascii",
            )
        )
        self.lib_to_filename = json.load(
            open(
                pkg_resources.resource_filename(
                    "aspired", "standards/lib_to_filename.json"
                ),
                encoding="ascii",
            )
        )
        self.lib_to_designation = json.load(
            open(
                pkg_resources.resource_filename(
                    "aspired", "standards/lib_to_designation.json"
                ),
                encoding="ascii",
            )
        )
        self.filename_to_lib = json.load(
            open(
                pkg_resources.resource_filename(
                    "aspired", "standards/filename_to_lib.json"
                ),
                encoding="ascii",
            )
        )

        self.designation_list = self.designation_to_lib_filename.keys()
        self.lib_list = self.lib_to_filename.keys()
        self.filename_list = self.filename_to_lib.keys()

    def _get_eso_standard(self):
        """
        Formatter for reading the ESO standards.

        """

        folder = self.library

        # first letter of the file name
        if self.ftype == "flux":
            filename = "f"

        else:
            filename = "m"

        # match the target designation to the file name
        filename += self.designation_to_lib_filename[self.designation][
            self.library
        ]

        # the extension
        filename += ".dat"

        filepath = os.path.join(base_dir, "standards", folder, filename)

        _f = np.loadtxt(filepath)

        self.standard_wave_true = _f[:, 0]
        self.standard_fluxmag_true = _f[:, 1]

        if self.ftype == "flux":
            if self.library != "esoxshooter":
                self.standard_fluxmag_true /= 10.0**16.0

    def _get_ing_standard(self):
        """
        Formatter for reading the ING standards.

        """

        folder = self.library

        # the first part of the file name
        filename = self.designation_to_lib_filename[self.designation][
            self.library
        ]
        extension = self.library[3:]

        # last letter (or nothing) of the file name
        if self.ftype == "flux":
            if (filename == "g24" or filename == "g157") and (
                extension == "fg"
            ):
                filename += "a"

            if (filename == "h102") and (extension == "sto"):
                filename += "a"

        else:
            filename += "a"

        # the extension
        filename += "." + extension

        filepath = os.path.join(base_dir, "standards", folder, filename)

        _f = open(filepath, encoding="ascii")
        wave = []
        fluxmag = []
        for line in _f.readlines():
            if line[0] in ["*", "S"]:
                if line.startswith("SET .Z.UNITS = "):
                    # remove all special characters and white spaces
                    unit = "".join(
                        e for e in line.split('"')[1].lower() if e.isalnum()
                    )

            else:
                _li = line.strip().strip(":").split()
                wave.append(_li[0])
                fluxmag.append(_li[1])

        _f.close()
        self.standard_wave_true = np.array(wave).astype("float")
        self.standard_fluxmag_true = np.array(fluxmag).astype("float")

        # See https://www.stsci.edu/~strolger/docs/UNITS.txt for the unit
        # conversion.
        if self.ftype == "flux":
            # Trap the ones without flux files
            if (
                extension == "mas"
                or filename == "g24a.fg"
                or filename == "g157a.fg"
                or filename == "h102a.sto"
            ):
                self.standard_fluxmag_true = (
                    10.0 ** (-(self.standard_fluxmag_true / 2.5))
                    * 3630.780548
                    / 3.33564095e4
                    / self.standard_wave_true**2
                )

            # convert milli-Jy into F_lambda
            if unit == "mjy":
                self.standard_fluxmag_true = (
                    self.standard_fluxmag_true
                    * 1e-3
                    * 2.99792458e-05
                    / self.standard_wave_true**2
                )

            # convert micro-Jy into F_lambda
            if unit == "microjanskys":
                self.standard_fluxmag_true = (
                    self.standard_fluxmag_true
                    * 1e-6
                    * 2.99792458e-05
                    / self.standard_wave_true**2
                )

    def _get_iraf_standard(self):
        """
        Formatter for reading the iraf standards.

        """

        folder = self.library

        # file name and extension
        filename = (
            self.designation_to_lib_filename[self.designation][self.library]
            + ".dat"
        )

        filepath = os.path.join(base_dir, "standards", folder, filename)

        _f = np.loadtxt(filepath, skiprows=1)

        self.standard_wave_true = _f[:, 0]
        self.standard_fluxmag_true = _f[:, 1]

        # iraf is always in AB magnitude
        if self.ftype == "flux":
            # Convert from AB mag to flux
            self.standard_fluxmag_true = (
                10.0 ** (-(self.standard_fluxmag_true / 2.5))
                * 3630.780548
                / 3.33564095e4
                / self.standard_wave_true**2
            )

    def lookup_standard_libraries(self, target: str, cutoff: float = 0.4):
        """
        Check if the requested standard and library exist. Return the three
        most similar words if the requested one does not exist. See

            https://docs.python.org/3.7/library/difflib.html

        This method tries to match the designation of the standard star
        that is available on SIMBAD or as named by the Isaac Newton Group of
        Telescopes. All comparisons are performed in lower case, all space
        and special symbols (expect ".", "+", "-", "_") are stripped.

        Parameters
        ----------
        target: str
            Name/designation of the standard star
        cutoff: float (Default: 0.4)
            The similarity toleranceold
            [0 (completely different) - 1 (identical)]

        """

        library_list = []
        filename_list = []

        if not isinstance(target, str):
            error_msg = (
                f"Target name has to be of type str, {type(target)} is"
                " provided."
            )
            self.logger.critical(error_msg)

        # If the provided designation exists
        if target.lower() in self.designation_list:
            _filename_list = self.designation_to_lib_filename[target.lower()]

            exact_match = True

        else:
            # If the requested target is not in any library, suggest the
            # closest match, Top 5 are returned.
            # difflib uses Gestalt pattern matching.
            designation_list = difflib.get_close_matches(
                target.lower(),
                list(self.designation_to_lib_filename.keys()),
                n=5,
                cutoff=cutoff,
            )

            for designation in designation_list:
                _filename_list = self.designation_to_lib_filename[designation]

            exact_match = False

        for libname, filename in _filename_list.items():
            filename_list.append(filename)
            library_list.append(libname)

        if len(filename_list) > 0:
            self.logger.warning(
                "Requested standard star cannot be found, a list of"
                " the closest matching names are returned:"
                f" {filename_list}."
            )
        else:
            error_msg = (
                (
                    "Please check the name of your standard star, nothing "
                    f"share a similarity above {cutoff}."
                ),
            )
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        # Return pair(s) of filename and library
        return [
            (f, l) for l, f in zip(library_list, filename_list)
        ], exact_match

    def lookup_closet_match_in_library(
        self, target: str, library: str, cutoff: float = 0.2
    ):
        """
        Check if the requested standard and library exist. Only if the
        similarity is better than (by default) 0.2 a target name will be returned. See

            https://docs.python.org/3.11/library/difflib.html

        Parameters
        ----------
        target: str
            Desination of the standard star
        library: str
            Name of the library
        cutoff: float (Default: 0.4)
            The toleranceold for the word similarity in the range of [0, 1].

        Return:
        -------
        The closest designation to the target in that (closet) library.

        """

        # Load the list of libraries
        library_name = difflib.get_close_matches(
            library, list(self.lib_to_designation.keys()), n=1, cutoff=cutoff
        )

        if library_name == []:
            return None, None

        else:
            library_name = library_name[0]

        # difflib uses Gestalt pattern matching.
        target_designation = difflib.get_close_matches(
            target.lower(),
            self.lib_to_designation[library_name],
            n=1,
            cutoff=cutoff,
        )
        if target_designation == []:
            target_designation = None

        else:
            target_designation = target_designation[0]

        return target_designation, library_name

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

        self.target = target.lower()
        self.ftype = ftype.lower()
        self.cutoff = cutoff

        # If there is a close match from the user-provided library, use
        # that first, it will only accept the library and target if the
        # cutoff is above 0.4
        self.library = library
        if self.library is not None:
            (
                _designation,
                _library,
            ) = self.lookup_closet_match_in_library(self.target, self.library)

            if _designation is not None:
                self.designation = _designation
                self.library = _library
                self.logger.info(
                    f"The requested standard star {self.target} is found in the"
                    f" given library {self.library}."
                )

        # If not, search again with the first one returned from lookup.
        else:
            libraries, success = self.lookup_standard_libraries(self.target)

            if success:
                self.target, self.library = libraries[0]

                if not np.in1d([library], libraries):
                    self.logger.warning(
                        "The requested standard star cannot be found in the"
                        " given library, or the library is not specified."
                        f" ASPIRED is using {self.library}."
                    )

            else:
                # When success is Flase, the libraries is a list of standards
                self.target, self.library = libraries[0]

                self.logger.warning(
                    f"The requested library does not exist, {self.library} "
                    "is used because it has the closest matching name."
                )

            self.designation, _ = self.lookup_closet_match_in_library(
                self.target, self.library
            )

        if self.library.startswith("iraf"):
            self._get_iraf_standard()

        if self.library.startswith("ing"):
            self._get_ing_standard()

        if self.library.startswith("eso"):
            self._get_eso_standard()

    def inspect_standard(
        self,
        display: bool = True,
        renderer: str = "default",
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
    ):
        """
        Display the standard star plot.

        Parameters
        ----------
        display: bool (Default: True)
            Set to True to display disgnostic plot.
        renderer: string (Default: default)
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

        fig = go.Figure(
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
                                                "yaxis": {"type": "linear"},
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
        fig.add_trace(
            go.Scatter(
                x=self.standard_wave_true,
                y=self.standard_fluxmag_true,
                line=dict(color="royalblue", width=4),
            )
        )

        fig.update_layout(
            title=self.library + ": " + self.target + " " + self.ftype,
            xaxis_title=r"$\text{Wavelength / A}$",
            yaxis_title=(
                r"$\text{Flux / ergs cm}^{-2} \text{s}^{-1}" + "\text{A}^{-1}$"
            ),
            hovermode="closest",
            showlegend=False,
        )

        if filename is None:
            filename = "standard"

        if save_fig:
            fig_type_split = fig_type.split("+")

            for _t in fig_type_split:
                if _t == "iframe":
                    pio.write_html(
                        fig, filename + "." + _t, auto_open=open_iframe
                    )

                elif _t in ["jpg", "png", "svg", "pdf"]:
                    pio.write_image(fig, filename + "." + _t)

        if display:
            if renderer == "default":
                fig.show()

            else:
                fig.show(renderer)

        if return_jsonstring:
            return fig.to_json()


class FluxCalibration(StandardLibrary):
    """
    For flux calibration using iraf, ING and ESO standards.

    """

    def __init__(
        self,
        verbose: bool = True,
        logger_name: str = "FluxCalibration",
        log_level: str = "INFO",
        log_file_folder: str = "default",
        log_file_name: str = None,
    ):
        """
        Initialise a FluxCalibration object.

        Parameters
        ----------
        verbose: bool (Default: True)
            Set to False to suppress all verbose warnings, except for
            critical failure.
        logger_name: str (Default: FluxCalibration)
            This will set the name of the logger, if the name is used already,
            it will reference to the existing logger. This will be the
            first part of the default log file name unless log_file_name is
            provided.
        log_level: str (Default: 'INFO')
            Four levels of logging are available, in decreasing order of
            information and increasing order of severity:
            CRITICAL, DEBUG, INFO, WARNING, ERROR
        log_file_folder: None or str (Default: "default")
            Folder in which the file is save, set to default to save to the
            current path.
        log_file_name: None or str (Default: None)
            File name of the log, set to None to logging.warning to screen
            only.

        """

        # Set-up logger
        self.logger = logging.getLogger(logger_name)
        if (log_level == "CRITICAL") or (not verbose):
            logging.basicConfig(level=logging.CRITICAL)
        if log_level == "ERROR":
            logging.basicConfig(level=logging.ERROR)
        if log_level == "WARNING":
            logging.basicConfig(level=logging.WARNING)
        if log_level == "INFO":
            logging.basicConfig(level=logging.INFO)
        if log_level == "DEBUG":
            logging.basicConfig(level=logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
        )

        if log_file_name is None:
            # Only logging.warning log to screen
            handler = logging.StreamHandler()
        else:
            if log_file_name == "default":
                d_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                log_file_name = f"{logger_name}_{d_str}.log"
            # Save log to file
            if log_file_folder == "default":
                log_file_folder = ""

            handler = logging.FileHandler(
                os.path.join(log_file_folder, log_file_name), "a+"
            )

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.verbose = verbose
        self.logger_name = logger_name
        self.log_level = log_level
        self.log_file_folder = log_file_folder
        self.log_file_name = log_file_name

        # Load the dictionary
        super().__init__(
            verbose=self.verbose,
            logger_name=self.logger_name,
            log_level=self.log_level,
            log_file_folder=self.log_file_folder,
            log_file_name=self.log_file_name,
        )
        self.verbose = verbose
        self.spectrum_oned = SpectrumOneD(
            spec_id=0,
            verbose=self.verbose,
            logger_name=self.logger_name,
            log_level=self.log_level,
            log_file_folder=self.log_file_folder,
            log_file_name=self.log_file_name,
        )
        self.target_spec_id = None
        self.standard_wave_true = None
        self.standard_fluxmag_true = None

        self.count_continuum = None
        self.flux_continuum = None

    def from_spectrum_oned(
        self,
        spectrum_oned: SpectrumOneD,
        merge: bool = False,
        overwrite: bool = False,
    ):
        """
        This function copies all the info from the spectrum_oned, because
        users may supply different level/combination of reduction, everything
        is copied from the spectrum_oned even though in most cases only a
        None will be passed.

        By default, this is passing object by reference by default, so it
        directly modifies the spectrum_oned supplied. By setting merger to
        True, it copies the data into the SpectrumOneD in the FluxCalibration
        object.

        Parameters
        ----------
        spectrum_oned: SpectrumOneD object
            The SpectrumOneD to be referenced or copied.
        merge: bool (Default: False)
            Set to True to copy everything over to the local SpectrumOneD,
            hence FluxCalibration will not be acting on the SpectrumOneD
            outside.

        """

        if merge:
            self.spectrum_oned.merge(spectrum_oned, overwrite=overwrite)

        else:
            self.spectrum_oned = spectrum_oned

        self.spectrum_oned_imported = True

    def remove_spectrum_oned(self):
        """
        Delete the spectrum_oned object.

        """

        self.spectrum_oned = SpectrumOneD(
            spec_id=0,
            verbose=self.verbose,
            logger_name=self.logger_name,
            log_level=self.log_level,
            log_file_folder=self.log_file_folder,
            log_file_name=self.log_file_name,
        )
        self.spectrum_oned_imported = False

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

        super().load_standard(
            target=target, library=library, ftype=ftype, cutoff=cutoff
        )
        # the best target and library found can be different from the input
        self.spectrum_oned.add_standard_star(
            library=self.library, target=self.target
        )

    def add_standard(
        self,
        wavelength: Union[np.ndarray, list],
        count: Union[np.ndarray, list],
        count_err: Union[np.ndarray, list] = None,
        count_sky: Union[np.ndarray, list] = None,
    ):
        """
        Add spectrum (wavelength, count, count_err & count_sky).

        Parameters
        ----------
        wavelength: 1-d array
            The wavelength at each pixel of the trace.
        count: 1-d array
            The summed count at each column about the trace.
        count_err: 1-d array (Default: None)
            the uncertainties of the count values
        count_sky: 1-d array (Default: None)
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract

        """

        self.spectrum_oned.add_wavelength(wavelength)
        self.spectrum_oned.add_count(count, count_err, count_sky)

    def get_telluric_profile(
        self,
        wave: Union[np.ndarray, list],
        flux: Union[np.ndarray, list],
        continuum: Union[np.ndarray, list],
        mask_range: Union[np.ndarray, list] = [[6850, 6960], [7580, 7700]],
        return_function: bool = False,
    ):
        """
        Getting the Telluric absorption profile from the continuum of the
        standard star spectrum.

        Parameters
        ----------
        wave: list or 1-d array (N)
            Wavelength.
        flux: list or 1-d array (N)
            Flux.
        continuum: list or 1-d array (N)
            Continuum Flux.
        mask_range: list of list
            list of lists with 2 values indicating the range marked by each
            of the Telluric regions.
        return_function: bool (Default: False)
            Set to True to explicitly return the interpolated function of
            the Telluric profile.

        """

        telluric_profile = np.zeros_like(wave)

        # Get the continuum of the spectrum
        # This should give the POSITIVE values over the telluric region
        residual = continuum - flux

        for m_range in mask_range:
            # Get the indices for the two sides of the masking region
            # at the native pixel scale
            left_of_mask = np.where(wave <= m_range[0])[0]
            right_of_mask = np.where(wave >= m_range[1])[0]

            if (len(left_of_mask) == 0) or (len(right_of_mask) == 0):
                continue

            left_telluric_start = int(max(left_of_mask))
            right_telluric_end = int(min(right_of_mask)) + 1

            telluric_profile[
                left_telluric_start:right_telluric_end
            ] = residual[left_telluric_start:right_telluric_end]

        # normalise the profile
        telluric_factor = np.ptp(telluric_profile)
        telluric_profile /= telluric_factor

        # If the spectrum doesn't cover any given telluric mask regions
        if np.isnan(telluric_profile).all():
            telluric_profile = np.zeros_like(telluric_profile)

        telluric_func = itp.interp1d(
            wave, telluric_profile, fill_value="extrapolate"
        )

        self.spectrum_oned.add_telluric_func(telluric_func)
        self.spectrum_oned.add_telluric_profile(telluric_profile)
        self.spectrum_oned.add_telluric_factor(telluric_factor)

        if return_function:
            return telluric_func, telluric_profile, telluric_factor

    def inspect_telluric_profile(
        self,
        display: bool = True,
        renderer: str = "default",
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
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
        fig = go.Figure(
            layout=dict(
                autosize=False, height=height, width=width, title="Log scale"
            )
        )
        # show the image on the top
        self.logger.debug(np.asarray(self.spectrum_oned.wave))
        self.logger.debug(
            self.spectrum_oned.telluric_func(
                np.asarray(self.spectrum_oned.wave)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.asarray(self.spectrum_oned.wave),
                y=self.spectrum_oned.telluric_func(
                    np.asarray(self.spectrum_oned.wave)
                ),
                line=dict(color="royalblue", width=4),
                name="Flux",
            )
        )

        fig.update_layout(
            hovermode="closest",
            title="Telluric Profile",
            showlegend=True,
            xaxis_title=r"$\text{Wavelength / A}$",
            yaxis_title="Arbitrary",
            yaxis=dict(title="Flux"),
            legend=go.layout.Legend(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="black"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        if filename is None:
            filename = "telluric_profile"

        if save_fig:
            fig_type_split = fig_type.split("+")

            for f_type in fig_type_split:
                if f_type == "iframe":
                    pio.write_html(
                        fig, filename + "." + f_type, auto_open=open_iframe
                    )

                elif f_type in ["jpg", "png", "svg", "pdf"]:
                    pio.write_image(fig, filename + "." + f_type)

        if display:
            if renderer == "default":
                fig.show()

            else:
                fig.show(renderer)

        if return_jsonstring:
            return fig.to_json()

    def get_sensitivity(
        self,
        k: int = 3,
        method: str = "interpolate",
        mask_range: Union[np.ndarray, list] = [
            [6850.0, 6960.0],
            [7580.0, 7700.0],
        ],
        mask_fit_order: int = 1,
        mask_fit_size: int = 3,
        smooth: bool = True,
        return_function: bool = True,
        sens_deg: int = 7,
        use_continuum: bool = False,
        **kwargs: str,
    ):
        """
        The sensitivity curve is computed by dividing the true values by the
        wavelength calibrated standard spectrum, which is resampled with the
        spectres.spectres(). The curve is then interpolated with a cubic spline
        by default and is stored as a scipy interp1d object.

        6850 - 6960, 7575 - 7700, and 8925 - 9050 A are masked by
        default.

        Parameters
        ----------
        k: integer [1,2,3,4,5 only]
            The order of the spline.
        method: str (Default: interpolate)
            This should be either 'interpolate' of 'polynomial'. Note that the
            polynomial is computed from the interpolated function. The
            default is interpolate because it is much more stable at the
            wavelength limits of a spectrum in an automated system.
        mask_range: None or list of list
            Masking out regions of Telluric absorption when fitting the
            sensitivity curve. None for no mask. List of list has the pattern
            [[min_pix_1, max_pix_1], [min_pix_2, max_pix_2],...]
        mask_fit_order: int (Default: 1)
            Order of polynomial to be fitted over the masked regions
        mask_fit_size: int (Default: 3)
            Number of "pixels" to be fitted on each side of the masked regions.
        smooth: bool (Default: False)
            set to smooth the input spectrum with a lowess function with
            statsmodels
        return_function: bool (Default: True)
            Set to True to explicity return the interpolated function of the
            sensitivity curve.
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

        Returns
        -------
        A callable function as a function of wavelength if return_function is
        set to True.

        """

        # resampling both the observed and the database standard spectra
        # in unit of flux per second. The higher resolution spectrum is
        # resampled to match the lower resolution one.
        if use_continuum:
            count = np.array(getattr(self.spectrum_oned, "count_continuum"))

        else:
            count = np.array(getattr(self.spectrum_oned, "count"))

        count_err = np.array(getattr(self.spectrum_oned, "count_err"))
        wave = np.array(getattr(self.spectrum_oned, "wave"))
        exptime = getattr(self.spectrum_oned, "exptime")
        if exptime is None:
            exptime = 1.0

        self.logger.info(
            (
                "The exposure time used for computing sensitivity curve "
                f"is {exptime} seconds."
            ),
        )

        # Mask regions before smoothing and avoiding telluric absorptions
        if mask_range is not None:
            for m in mask_range:
                # If the mask is partially outside the spectrum, ignore
                if (m[0] < min(wave)) or (m[1] > max(wave)):
                    continue

                # Get the indices for the two sides of the masking region
                left_end = int(max(np.where(wave <= m[0])[0])) + 1
                left_start = int(left_end - mask_fit_size)
                right_start = int(min(np.where(wave >= m[1])[0]))
                right_end = int(right_start + mask_fit_size) + 1

                # Get the wavelengths of the two sides
                wave_temp = np.concatenate(
                    (
                        wave[left_start:left_end],
                        wave[right_start:right_end],
                    )
                )

                # Get the count of the two sides
                count_temp = np.concatenate(
                    (
                        count[left_start:left_end],
                        count[right_start:right_end],
                    )
                )

                finite_mask = (
                    ~np.isnan(count_temp)
                    & (count_temp > 0.0)
                    & np.isfinite(count_temp)
                )

                # Fit the polynomial across the masked region
                coeff = np.polynomial.polynomial.polyfit(
                    wave_temp[finite_mask],
                    count_temp[finite_mask],
                    mask_fit_order,
                )

                # Replace the snsitivity values with the fitted values
                count[left_end:right_start] = np.polynomial.polynomial.polyval(
                    wave[left_end:right_start], coeff
                )

        # This applies a lowess filter from statsmodels that
        # uses a different lowess_frac default value that is more appropriate in
        # getting a first guess continuum which reject "outliers" much more
        # aggressively.
        if smooth:
            if kwargs is None:
                kwargs = {}

            if "frac" not in kwargs:
                kwargs["frac"] = 0.1

            if "return_sorted" not in kwargs:
                kwargs["return_sorted"] = False

            count = lowess(count, wave, **kwargs)
            # Set the smoothing parameters
            self.spectrum_oned.add_smoothing(smooth, **kwargs)

        # If the median resolution of the observed spectrum is higher than
        # the literature one
        if np.nanmedian(np.array(np.ediff1d(wave))) < np.nanmedian(
            np.array(np.ediff1d(self.standard_wave_true))
        ):
            standard_count, _ = spectres(
                np.array(self.standard_wave_true).reshape(-1),
                np.array(wave).reshape(-1),
                np.array(count).reshape(-1),
                np.array(count_err).reshape(-1),
                verbose=True,
            )
            standard_flux_true = self.standard_fluxmag_true
            standard_wave_true = self.standard_wave_true

        # If the median resolution of the observed spectrum is lower than
        # the literature one
        else:
            standard_count = count
            # standard_flux_err = count_err
            standard_flux_true = spectres(
                np.array(wave).reshape(-1),
                np.array(self.standard_wave_true).reshape(-1),
                np.array(self.standard_fluxmag_true).reshape(-1),
                verbose=True,
            )
            standard_wave_true = wave

        # Get the sensitivity curve and convert the unit to per second
        sensitivity = standard_flux_true / (standard_count / exptime)

        mask = np.isfinite(np.log10(sensitivity)) & ~np.isnan(
            np.log10(sensitivity)
        )
        sensitivity_masked = sensitivity.copy()[mask]
        standard_wave_masked = standard_wave_true[mask]
        standard_flux_masked = standard_flux_true[mask]

        if method == "interpolate":
            tck = itp.splrep(
                standard_wave_masked, np.log10(sensitivity_masked), k=k
            )

            def sensitivity_func(_x):
                return itp.splev(_x, tck)

        elif method == "polynomial":
            coeff = np.polynomial.polynomial.polyfit(
                standard_wave_masked,
                np.log10(sensitivity_masked),
                deg=sens_deg,
            )

            def sensitivity_func(x):
                return np.polynomial.polynomial.polyval(x, coeff)

        else:
            error_msg = "{method} is not implemented."
            self.logger.critical(error_msg)
            raise NotImplementedError(error_msg)

        self.spectrum_oned.add_sensitivity(sensitivity_masked)
        self.spectrum_oned.add_literature_standard(
            standard_wave_masked, standard_flux_masked
        )

        # Add to each SpectrumOneD object
        self.spectrum_oned.add_sensitivity_func(sensitivity_func)

        if return_function:
            return sensitivity_func

    def add_sensitivity_func(self, sensitivity_func: Callable):
        """
        parameters
        ----------
        sensitivity_func: callable function
            Interpolated sensivity curve object (in unit of per second).

        """

        # Add to both science and standard spectrum_list
        self.spectrum_oned.add_sensitivity_func(
            sensitivity_func=sensitivity_func
        )
        self.spectrum_oned.add_literature_standard(
            self.standard_wave_true, self.standard_fluxmag_true
        )

    def save_sensitivity_func(self):
        """
        Saving the sensitivity function to disk, to be implemented.

        """

        pass

    def inspect_sensitivity(
        self,
        display: bool = True,
        renderer: str = "default",
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
    ):
        """
        Display the computed sensitivity curve.

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

        wave_literature = getattr(self.spectrum_oned, "wave_literature")
        flux_literature = getattr(self.spectrum_oned, "flux_literature")
        sensitivity = getattr(self.spectrum_oned, "sensitivity")
        sensitivity_func = getattr(self.spectrum_oned, "sensitivity_func")

        library = getattr(self.spectrum_oned, "library")
        target = getattr(self.spectrum_oned, "target")

        fig = go.Figure(
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
                                                "yaxis": {"type": "linear"},
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
        fig.add_trace(
            go.Scatter(
                x=wave_literature,
                y=flux_literature,
                line=dict(color="royalblue", width=4),
                name="Count / s (Observed)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=wave_literature,
                y=sensitivity,
                yaxis="y2",
                line=dict(color="firebrick", width=4),
                name="Sensitivity Curve",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=wave_literature,
                y=10.0 ** sensitivity_func(wave_literature),
                yaxis="y2",
                line=dict(color="black", width=2),
                name="Best-fit Sensitivity Curve",
            )
        )

        fig.update_layout(
            title=library + ": " + target, yaxis_title="Count / s"
        )

        fig.update_layout(
            hovermode="closest",
            showlegend=True,
            xaxis_title=r"$\text{Wavelength / A}$",
            yaxis=dict(title="Count / s"),
            yaxis2=dict(
                title="Sensitivity Curve",
                type="log",
                anchor="x",
                overlaying="y",
                side="right",
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
            filename = "senscurve"

        if save_fig:
            fig_type_split = fig_type.split("+")

            for f_type in fig_type_split:
                if f_type == "iframe":
                    pio.write_html(
                        fig, filename + "." + f_type, auto_open=open_iframe
                    )

                elif f_type in ["jpg", "png", "svg", "pdf"]:
                    pio.write_image(fig, filename + "." + f_type)

        if display:
            if renderer == "default":
                fig.show()

            else:
                fig.show(renderer)

        if return_jsonstring:
            return fig.to_json()

    def apply_flux_calibration(
        self,
        target_spectrum_oned: SpectrumOneD,
        inspect: bool = False,
        wave_min: float = None,
        wave_max: float = None,
        display: bool = False,
        renderer: str = "default",
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
    ):
        """
        Apply the computed sensitivity curve. And resample the spectra to
        match the highest resolution (the smallest wavelength bin) part of the
        spectrum.

        Note: This function directly modify the *target_spectrum_oned*.

        Note 2: the wave_min and wave_max are for DISPLAY purpose only.

        Parameters
        ----------
        target_spectrum_oned: SpectrumOneD object
            The spectrum to be flux calibrated.
        inspect: bool (Default: False)
            Set to True to create/display/save figure
        wave_min: float (Default: None -> 3500)
            Minimum wavelength to display
        wave_max: float (Default: None -> 8500)
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

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        """

        self.target_spec_id = getattr(target_spectrum_oned, "spec_id")

        wave = getattr(target_spectrum_oned, "wave")
        count = getattr(target_spectrum_oned, "count")
        count_err = getattr(target_spectrum_oned, "count_err")
        count_sky = getattr(target_spectrum_oned, "count_sky")
        exptime = getattr(target_spectrum_oned, "exptime")
        if exptime is None:
            exptime = 1.0

        self.logger.info(
            f"The exposure time used for flux calibration is {exptime} seconds."
        )

        # apply the flux calibration
        sensitivity_func = getattr(self.spectrum_oned, "sensitivity_func")
        sensitivity = 10.0 ** sensitivity_func(wave) / exptime

        flux = sensitivity * count

        if count_err is not None:
            flux_err = sensitivity * count_err

        if count_sky is not None:
            flux_sky = sensitivity * count_sky

        target_spectrum_oned.add_flux(flux, flux_err, flux_sky)
        target_spectrum_oned.add_sensitivity(sensitivity)

        # Add the rest of the flux calibration parameters
        target_spectrum_oned.merge(self.spectrum_oned)

        if wave_min is None:
            wave_min = 3500.0

        if wave_max is None:
            wave_max = 8500.0

        if inspect:
            wave_mask = (np.array(wave).reshape(-1) > wave_min) & (
                np.array(wave).reshape(-1) < wave_max
            )

            flux_low = (
                np.nanpercentile(np.array(flux).reshape(-1)[wave_mask], 5)
                / 1.5
            )
            flux_high = (
                np.nanpercentile(np.array(flux).reshape(-1)[wave_mask], 95)
                * 1.5
            )
            flux_mask = (np.array(flux).reshape(-1) > flux_low) & (
                np.array(flux).reshape(-1) < flux_high
            )
            flux_min = np.log10(
                np.nanmin(np.array(flux).reshape(-1)[flux_mask])
            )
            flux_max = np.log10(
                np.nanmax(np.array(flux).reshape(-1)[flux_mask])
            )

            fig = go.Figure(
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
            fig.add_trace(
                go.Scatter(
                    x=wave,
                    y=flux,
                    line=dict(color="royalblue"),
                    name="Flux",
                )
            )

            if flux_err is not None:
                fig.add_trace(
                    go.Scatter(
                        x=wave,
                        y=flux_err,
                        line=dict(color="firebrick"),
                        name="Flux Uncertainty",
                    )
                )

            if flux_sky is not None:
                fig.add_trace(
                    go.Scatter(
                        x=wave,
                        y=flux_sky,
                        line=dict(color="orange"),
                        name="Sky Flux",
                    )
                )

            fig.update_layout(
                hovermode="closest",
                showlegend=True,
                xaxis=dict(title="Wavelength / A", range=[wave_min, wave_max]),
                yaxis=dict(
                    title="Flux", range=[flux_min, flux_max], type="log"
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
                filename = "spectrum_" + str(self.target_spec_id)

            else:
                filename = (
                    os.path.splitext(filename)[0]
                    + "_"
                    + str(self.target_spec_id)
                )

            if save_fig:
                fig_type_split = fig_type.split("+")

                for f_type in fig_type_split:
                    if f_type == "iframe":
                        pio.write_html(
                            fig, filename + "." + f_type, auto_open=open_iframe
                        )

                    elif f_type in ["jpg", "png", "svg", "pdf"]:
                        pio.write_image(fig, filename + "." + f_type)

            if display:
                if renderer == "default":
                    fig.show()

                else:
                    fig.show(renderer)

            if return_jsonstring:
                return fig.to_json()

    def create_fits(
        self,
        output: str = "count+wavelength+sensitivity+flux",
        empty_primary_hdu: bool = True,
        recreate: bool = False,
    ):
        """
        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            count: 4 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            wavelength: 1 HDU
                Wavelength of each pixel
            sensitivity: 1 HDU
                Sensitivity (pixel)
            flux: 4 HDUs
                Flux, uncertainty, and sky

        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank (Default: True)
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.

        """

        # If flux is calibrated
        self.spectrum_oned.create_fits(
            output=output,
            empty_primary_hdu=empty_primary_hdu,
            recreate=recreate,
        )

    def save_fits(
        self,
        output: str = "count+wavelength+sensitivity+flux",
        filename: str = "fluxcal",
        empty_primary_hdu: bool = True,
        overwrite: bool = False,
        recreate: bool = False,
    ):
        """
        Save the reduced data to disk, with a choice of any combination of
        the data that are already present in the SpectrumOneD, see below the
        'output' parameters for details.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+". Because a FluxCalibration
            only requires a subset of all the data, only a few data products
            are guaranteed to exist.

            count: 4 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            wavelength: 1 HDU
                Wavelength of each pixel
            sensitivity: 1 HDU
                Sensitivity (pixel)
            flux: 4 HDUs
                Flux, uncertainty, and sky

        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank (Default: True)
        overwrite: bool
            Default is False.
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.

        """

        # Fix the names and extensions
        if self.target_spec_id is not None:
            filename = (
                os.path.splitext(filename)[0] + "_" + str(self.target_spec_id)
            )
        else:
            filename = os.path.splitext(filename)[0]

        self.spectrum_oned.save_fits(
            output=output,
            filename=filename,
            overwrite=overwrite,
            recreate=recreate,
            empty_primary_hdu=empty_primary_hdu,
        )

    def save_csv(
        self,
        output: str = "count+wavelength+sensitivity+flux",
        filename: str = "fluxcal",
        overwrite: bool = False,
        recreate: bool = False,
    ):
        """
        Save the reduced data to disk, with a choice of any combination of
        the data that are already present in the SpectrumOneD, see below the
        'output' parameters for details.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+". Because a FluxCalibration
            only requires a subset of all the data, only a few data products
            are guaranteed to exist.

            count: 4 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            wavelength: 1 HDU
                Wavelength of each pixel
            sensitivity: 1 HDU
                Sensitivity (pixel)
            flux: 4 HDUs
                Flux, uncertainty, and sky

        filename: String (Default: None)
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        overwrite: bool (Default: False)
            Set to True to allow overwriting the FITS data at the file
            destination.
        recreate: bool (Default: False)
            Set to True to regenerate the FITS data and header.

        """

        # Fix the names and extensions
        if self.target_spec_id is not None:
            filename = (
                os.path.splitext(filename)[0] + "_" + str(self.target_spec_id)
            )
        else:
            filename = os.path.splitext(filename)[0]

        self.spectrum_oned.save_csv(
            output=output,
            filename=filename,
            overwrite=overwrite,
            recreate=recreate,
        )

    def get_spectrum_oned(self):
        """
        Return the spectrum_oned object.

        """

        return self.spectrum_oned
