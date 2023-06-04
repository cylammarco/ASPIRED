# -*- coding: utf-8 -*-
import datetime
import logging
import os
from typing import Union

import numpy as np
from plotly import graph_objects as go
from plotly import io as pio
from rascal.atlas import Atlas
from rascal.calibrator import Calibrator
from rascal.util import refine_peaks
from scipy import signal

from .spectrum_oneD import SpectrumOneD

__all__ = ["WavelengthCalibration"]


class WavelengthCalibration:
    def __init__(
        self,
        verbose: bool = True,
        logger_name: str = "WavelengthCalibration",
        log_level: str = "INFO",
        log_file_folder: str = "default",
        log_file_name: str = None,
    ):
        """
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
        verbose: bool (Default: True)
            Set to False to suppress all verbose warnings, except for
            critical failure.
        logger_name: str (Default: OneDSpec)
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

        self.spectrum_oned = SpectrumOneD(
            spec_id=0,
            verbose=self.verbose,
            logger_name=self.logger_name,
            log_level=self.log_level,
            log_file_folder=self.log_file_folder,
            log_file_name=self.log_file_name,
        )

        self.polyval = {
            "poly": np.polynomial.polynomial.polyval,
            "leg": np.polynomial.legendre.legval,
            "cheb": np.polynomial.chebyshev.chebval,
        }

    def from_spectrum_oned(
        self,
        spectrum_oned: SpectrumOneD,
        merge: bool = False,
        overwrite: bool = False,
    ):
        """
        This function copies all the info from the spectrum_oned, because users
        may supply different level/combination of reduction, everything is
        copied from the spectrum_oned even though in most cases only a None
        will be passed.

        By default, this is passing object by reference by default, so it
        directly modifies the spectrum_oned supplied. By setting merger to True,
        it copies the data into the SpectrumOneD in the FluxCalibration object.

        Parameters
        ----------
        spectrum_oned: SpectrumOneD object
            The SpectrumOneD to be referenced or copied.
        merge: bool (Default: False)
            Set to True to copy everything over to the local SpectrumOneD,
            hence FluxCalibration will not be acting on the SpectrumOneD
            outside.
        overwrite: bool (Default: False)
            Set to True to make a complete copy of the spectrum_oned to the
            target spectrum_oned, that includes all the Nones and other settings.
            Use with caution, as it removes the properties set before this
            function call.

        """

        # This DOES NOT modify the spectrum_oned outside of WavelengthCalibration
        if merge:
            self.spectrum_oned.merge(spectrum_oned, overwrite=overwrite)
        # This DOES modify the spectrum_oned outside of WavelengthCalibration
        else:
            self.spectrum_oned = spectrum_oned

    def add_arc_lines(self, peaks: Union[list, np.ndarray]):
        """
        Provide the pixel locations of the arc lines.

        Parameters
        ----------
        peaks: list
            The pixel locations of the arc lines. Multiple traces of the arc
            can be provided as list of list or list of arrays.

        """

        self.spectrum_oned.add_peaks_refined(peaks)

    def remove_arc_lines(self):
        """
        Remove all the refined arc lines.

        """

        self.spectrum_oned.remove_peaks_refined()

    def add_arc_spec(self, arc_spec: Union[list, np.ndarray]):
        """
        Provide the 1D spectrum of the arc image.

        Parameters
        ----------
        arc_spec: list
            The photoelectron count of the 1D arc spectrum.

        """

        self.spectrum_oned.add_arc_spec(arc_spec)

    def remove_arc_spec(self):
        """
        Remove the aspectrm of the arc

        """

        self.spectrum_oned.remove_arc_spec()

    def add_fit_type(self, fit_type: str):
        """
        Adding the polynomial type.

        Parameters
        ----------
        fit_type: str or list of str
            Strings starting with 'poly', 'leg' or 'cheb' for polynomial,
            legendre and chebyshev fits. Case insensitive.

        """

        self.spectrum_oned.add_fit_type(fit_type)

    def remove_fit_type(self):
        """
        To remove the polynomial fit type.

        """

        self.spectrum_oned.remove_fit_type()

    def add_fit_coeff(self, fit_coeff: Union[list, np.ndarray]):
        """
        Adding the polynomial coefficients.

        Parameters
        ----------
        fit_coeff: list or list of list
            Polynomial fit coefficients.

        """

        self.spectrum_oned.add_fit_coeff(fit_coeff)

    def remove_fit_coeff(self):
        """
        To remove the polynomial fit coefficients.

        """

        self.spectrum_oned.remove_fit_coeff()

    def inspect_arc_lines(
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

        Parameters
        ----------
        display: bool (Default: False)
            Set to True to display disgnostic plot.
        renderer: string (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs.
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs.
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
            Open the iframe in the default browser if set to True. Only used
            if an iframe is saved.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        """

        arc_spec = getattr(self.spectrum_oned, "arc_spec")

        arc_spec = arc_spec - np.nanmin(arc_spec)
        arc_spec = arc_spec / np.nanmax(arc_spec)

        peaks = getattr(self.spectrum_oned, "peaks")

        fig = go.Figure(
            layout=dict(autosize=False, height=height, width=width)
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(arc_spec)),
                y=arc_spec,
                mode="lines",
                line=dict(color="royalblue", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=peaks,
                y=np.array(arc_spec)[np.rint(peaks).astype("int")],
                mode="markers",
                line=dict(color="firebrick", width=1),
            )
        )

        fig.update_layout(
            xaxis=dict(
                zeroline=False,
                range=[0, len(arc_spec)],
                title="Spectral Direction / pixel",
            ),
            yaxis=dict(
                zeroline=False, range=[0, max(arc_spec)], title="e- / s"
            ),
            hovermode="closest",
            showlegend=False,
        )

        if filename is None:
            filename = "arc_lines"

        if save_fig:
            fig_type_split = fig_type.split("+")

            for t in fig_type_split:
                if t == "iframe":
                    pio.write_html(
                        fig, filename + "." + t, auto_open=open_iframe
                    )

                elif t in ["jpg", "png", "svg", "pdf"]:
                    pio.write_image(fig, filename + "." + t)

        if display:
            if renderer == "default":
                fig.show()

            else:
                fig.show(renderer)

        if return_jsonstring:
            return fig.to_json()

    def find_arc_lines(
        self,
        arc_spec: Union[list, np.ndarray] = None,
        prominence: float = 5.0,
        top_n_peaks: int = None,
        distance: float = 5.0,
        refine: bool = True,
        refine_window_width: int = 5,
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
        This function identifies the arc lines (peaks) with
        scipy.signal.find_peaks(), where only the distance and the prominence
        keywords are used. Distance is the minimum separation between peaks,
        the default value is roughly twice the nyquist sampling rate (i.e.
        pixel size is 2.3 times smaller than the object that is being resolved,
        hence, the sepration between two clearly resolved peaks are ~5 pixels
        apart). A crude estimate of the background can exclude random noise
        which look like small peaks.

        Parameters
        ----------
        arc_spec: list, array or None (Default: None)
            If not provided, it will look for the arc_spec in the spectrum_oned.
            Otherwise, the input arc_spec will be used.
        prominence: float (Default: 5.)
            The minimum prominence to be considered as a peak (% of max).
        top_n_peaks: int (Default: None)
            The N most prominent peaks. None means keeping all peaks.
        distance: float (Default: 5.)
            Minimum separation between peaks.
        refine: bool (Default: True)
            Set to true to fit a gaussian to get the peak at sub-pixel
            precision.
        refine_window_width: int (Default: 5)
            The number of pixels (on each side of the existing peaks) to be
            fitted with gaussian profiles over.
        display: bool (Default: False)
            Set to True to display disgnostic plot.
        renderer: string (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs.
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs.
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
            Open the iframe in the default browser if set to True. Only used
            if an iframe is saved.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        """

        if arc_spec is None:
            if getattr(self.spectrum_oned, "arc_spec") is None:
                error_msg = (
                    "arc_spec is not provided. Either provide when "
                    + "executing this function or provide a spectrum_oned that "
                    + "contains an arc_spec."
                )
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:
            if getattr(self.spectrum_oned, "arc_spec") is not None:
                self.logger.warning("arc_spec is replaced with the new one.")

            setattr(self.spectrum_oned, "arc_spec", arc_spec)

        arc_spec = getattr(self.spectrum_oned, "arc_spec")

        arc_spec = arc_spec - np.nanmin(arc_spec)
        arc_spec = arc_spec / np.nanmax(arc_spec)

        peaks, prop = signal.find_peaks(
            arc_spec, distance=distance, prominence=prominence / 100.0
        )
        prom = prop["prominences"]
        prom_sorted_arg = np.argsort(prom)[::-1]

        if isinstance(top_n_peaks, (int, float)):
            peaks = np.sort(peaks[prom_sorted_arg][: int(top_n_peaks)])

        self.spectrum_oned.add_peaks(peaks)

        # Fine tuning
        if refine:
            peaks = refine_peaks(
                arc_spec,
                getattr(self.spectrum_oned, "peaks"),
                window_width=int(refine_window_width),
            )
            self.spectrum_oned.add_peaks(peaks)

        # Adjust for chip gaps
        if getattr(self.spectrum_oned, "pixel_mapping_itp") is not None:
            self.spectrum_oned.add_peaks_refined(
                getattr(self.spectrum_oned, "pixel_mapping_itp")(peaks)
            )

        else:
            self.spectrum_oned.add_peaks_refined(peaks)

        if save_fig or display or return_jsonstring:
            to_return = self.inspect_arc_lines(
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

        if return_jsonstring:
            return to_return

    def initialise_calibrator(
        self,
        peaks: Union[list, np.ndarray] = None,
        arc_spec: Union[list, np.ndarray] = None,
    ):
        """
        Initialise a RASCAL calibrator.

        Parameters
        ----------
        peaks: list (Default: None)
            The pixel values of the peaks (start from zero).
        arc_spec: list
            The spectral intensity as a function of pixel.

        """

        if peaks is None:
            if getattr(self.spectrum_oned, "peaks_refined") is not None:
                peaks = getattr(self.spectrum_oned, "peaks_refined")

            elif getattr(self.spectrum_oned, "peaks") is not None:
                peaks = getattr(self.spectrum_oned, "peaks")

            else:
                error_msg = (
                    "arc_spec is not provided. Either provide when "
                    + "executing this function or provide a spectrum_oned that "
                    + "contains a peaks_refined."
                )
                self.logger.warning(error_msg)

        else:
            if getattr(self.spectrum_oned, "peaks_refined") is not None:
                self.logger.warning(
                    "peaks_refined is replaced with the new one."
                )

            self.spectrum_oned.add_peaks_refined(peaks)

        if arc_spec is None:
            if getattr(self.spectrum_oned, "arc_spec") is not None:
                arc_spec = getattr(self.spectrum_oned, "arc_spec")

            else:
                error_msg = (
                    "arc_spec is not provided. Either provide when "
                    + "executing this function or provide a spectrum_oned that "
                    + "contains an arc_spec."
                )
                self.logger.warning(error_msg)

        else:
            if getattr(self.spectrum_oned, "arc_spec") is not None:
                self.logger.warning("arc_spec is replaced with the new one.")

            self.spectrum_oned.add_arc_spec(arc_spec)

        self.spectrum_oned.add_calibrator(
            Calibrator(peaks=peaks, spectrum=arc_spec)
        )
        self.spectrum_oned.calibrator.atlas = Atlas()
        self.set_calibrator_properties()

    # placeholder for rascal v0.4
    def set_logger(
        self, logger_name: str = "Calibrator", log_level: str = "info"
    ):
        """
        Parameters
        ----------
        logger_name: str (Default: 'Calibrator')
            Name of the logger.
        log_level : str (Default: 'info')
            Choose from {CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET}.

        """

        self.spectrum_oned.calibrator.set_logger(
            logger_name=logger_name, log_level=log_level
        )

    def set_calibrator_properties(
        self,
        num_pix: int = None,
        effective_pixel: Union[list, np.ndarray] = None,
        plotting_library: str = "plotly",
        seed: int = None,
        logger_name: str = "Calibrator",
        log_level: str = "warning",
    ):
        """
        Set the properties of the calibrator.

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
        plotting_library : string (Default: 'plotly')
            Choose between matplotlib and plotly.
        seed: int
            Set an optional seed for random number generators. If used,
            this parameter must be set prior to calling RANSAC. Useful
            for deterministic debugging.
        logger_name: string (default: 'Calibrator')
            The name of the logger. It can use an existing logger if a
            matching name is provided.
        log_level: string (default: 'info')
            Choose {critical, error, warning, info, debug, notset}.

        """

        self.spectrum_oned.calibrator.set_calibrator_properties(
            num_pix=num_pix,
            pixel_list=effective_pixel,
            plotting_library=plotting_library,
            seed=seed,
            logger_name=logger_name,
            log_level=log_level,
        )

        self.spectrum_oned.add_calibrator_properties(
            self.spectrum_oned.calibrator.num_pix,
            self.spectrum_oned.calibrator.pixel_list,
            self.spectrum_oned.calibrator.plotting_library,
        )

    def set_hough_properties(
        self,
        num_slopes: int = 5000,
        xbins: int = 500,
        ybins: int = 500,
        min_wavelength: float = 3000.0,
        max_wavelength: float = 9000.0,
        range_tolerance: float = 500.0,
        linearity_tolerance: float = 100.0,
    ):
        """
        Set the properties of the hough transform.

        Parameters
        ----------
        num_slopes: int (Default: 1000)
            Number of slopes to consider during Hough transform
        xbins: int (Default: 50)
            Number of bins for Hough accumulation
        ybins: int (Default: 50)
            Number of bins for Hough accumulation
        min_wavelength: float (Default: 3000.0)
            Minimum wavelength of the spectrum.
        max_wavelength: float (Default: 9000.0)
            Maximum wavelength of the spectrum.
        range_tolerance: float (Default: 500.0)
            Estimation of the error on the provided spectral range
            e.g. 3000-5000 with tolerance 500 will search for
            solutions that may satisfy 2500-5500
        linearity_tolerance: float (Default: 100.0)
            A toleranceold (Ansgtroms) which defines some padding around the
            range tolerance to allow for non-linearity. This should be the
            maximum expected excursion from linearity.

        """

        self.spectrum_oned.calibrator.set_hough_properties(
            num_slopes=num_slopes,
            xbins=xbins,
            ybins=ybins,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
            range_tolerance=range_tolerance,
            linearity_tolerance=linearity_tolerance,
        )

        self.spectrum_oned.add_hough_properties(
            num_slopes,
            xbins,
            ybins,
            min_wavelength,
            max_wavelength,
            range_tolerance,
            linearity_tolerance,
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
    ):
        """
        Set the properties of the RANSAC process.

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
        ransac_tolerance: float (Default: 5.0)
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
        minimum_peak_utilisation: float (Default: 0.0)
            The minimum percentage of peaks used in order to accept as a
            valid solution.
        minimum_fit_error: float (Default 1e-4)
            Set to remove overfitted/unrealistic fits.

        """

        self.spectrum_oned.calibrator.set_ransac_properties(
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

        self.spectrum_oned.add_ransac_properties(
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

    def set_known_pairs(
        self,
        pix: Union[list, np.ndarray] = None,
        wave: Union[list, np.ndarray] = None,
    ):
        """
        Provide manual pixel-wavelength pair(s), they will be appended to the
        list of pixel-wavelength pairs after the random sample being drawn from
        the RANSAC step, i.e. they are ALWAYS PRESENT in the fitting step. Use
        with caution because it can skew or bias the fit significantly, make
        sure the pixel value is accurate to at least 1/10 of a pixel.

        This can be used, for example, for low intensity lines at the edge of
        the spectrum.

        Parameters
        ----------
        pix : numeric value, list or numpy 1D array (N) (Default: None)
            Any pixel value, can be outside the detector chip and
            serve purely as anchor points.
        wave : numeric value, list or numpy 1D array (N) (Default: None)
            The matching wavelength for each of the pix.

        """

        self.spectrum_oned.calibrator.set_known_pairs(pix=pix, wave=wave)

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
    ):
        """
        Append the user supplied arc lines to the calibrator.

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements: list
            Element (required). Preferably a standard (i.e. periodic table)
            name for convenience with built-in atlases
        wavelengths: list
            Wavelength to add (Angstrom)
        intensities: list
            Relative line intensities
        candidate_tolerance: float (Default: 10)
            Tolerance (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly : bool (Default: False)
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: bool (Default: False)
            Set to true to convert the input wavelength to air-wavelengths
            based on the given pressure, temperature and humidity.
        pressure: float (Default: 101325.)
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement from 1 atm (the default)
            per 1000 meter altitude.
        temperature: float (Default: 273.15)
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float (Default: 0.)
            In percentage.

        """

        self.spectrum_oned.calibrator.atlas.add_user_atlas(
            elements=elements,
            wavelengths=wavelengths,
            intensities=intensities,
            vacuum=vacuum,
            pressure=pressure,
            temperature=temperature,
            relative_humidity=relative_humidity,
        )
        self.spectrum_oned.calibrator.constrain_poly = constrain_poly
        self.spectrum_oned.calibrator.candidate_tolerance = candidate_tolerance

        self.spectrum_oned.add_weather_condition(
            pressure, temperature, relative_humidity
        )

        self.spectrum_oned.calibrator._generate_pairs()

    def add_atlas(
        self,
        elements: Union[list, np.ndarray],
        min_atlas_wavelength: float = 1000.0,
        max_atlas_wavelength: float = 30000.0,
        min_intensity: float = 10.0,
        min_distance: float = 10.0,
        candidate_tolerance: float = 10.0,
        constrain_poly: bool = False,
        vacuum: bool = False,
        pressure: float = 101325.0,
        temperature: float = 273.15,
        relative_humidity: float = 0.0,
    ):
        """
        Adds an atlas of arc lines to the calibrator, given an element.

        Arc lines are taken from a general list of NIST lines and can be
        filtered using the minimum relative intensity (note this may not be
        accurate due to instrumental effects such as detector response,
        dichroics, etc) and minimum line separation.

        Lines are filtered first by relative intensity, then by separation.
        This is to improve robustness in the case where there is a strong
        line very close to a weak line (which is within the separation limit).

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements: string or list of strings
            Chemical symbol, case insensitive
        min_atlas_wavelength: float (Default: None)
            Minimum wavelength of the arc lines.
        max_atlas_wavelength: float (Default: None)
            Maximum wavelength of the arc lines.
        min_intensity: float (Default: None)
            Minimum intensity of the arc lines. Refer to NIST for the
            intensity.
        min_distance: float (Default: None)
            Minimum separation between neighbouring arc lines.
        candidate_tolerance: float (Default: 10)
            Tolerance (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: bool (Default: Flase)
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: bool (Default: False)
            Set to True if the light path from the arc lamb to the detector
            plane is entirely in vacuum.
        pressure: float (Default: 101325.)
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float (Default: 273.15)
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float (Default: 0.)
            In percentage.

        """

        self.spectrum_oned.calibrator.atlas.add(
            elements=elements,
            min_atlas_wavelength=min_atlas_wavelength,
            max_atlas_wavelength=max_atlas_wavelength,
            min_intensity=min_intensity,
            min_distance=min_distance,
            vacuum=vacuum,
            pressure=pressure,
            temperature=temperature,
            relative_humidity=relative_humidity,
        )
        self.spectrum_oned.calibrator.constrain_poly = constrain_poly
        self.spectrum_oned.calibrator.candidate_tolerance = candidate_tolerance

        self.spectrum_oned.add_atlas_wavelength_range(
            min_atlas_wavelength, max_atlas_wavelength
        )

        self.spectrum_oned.add_min_atlas_intensity(min_intensity)

        self.spectrum_oned.add_min_atlas_distance(min_distance)

        self.spectrum_oned.add_weather_condition(
            pressure, temperature, relative_humidity
        )
        self.spectrum_oned.calibrator._generate_pairs()

    def remove_atlas_lines_range(
        self, wavelength: float, tolerance: float = 10.0
    ):
        """
        Remove arc lines within the given wavelength range (tolerance).

        Parameters
        ----------
        wavelength: float
            Wavelength to remove (Angstrom)
        tolerance: float
            Tolerance around this wavelength where atlas lines will be removed

        """

        self.spectrum_oned.calibrator.remove_atlas_lines_range(
            wavelength, tolerance=tolerance
        )
        self.spectrum_oned.calibrator._generate_pairs()

    def list_atlas(self):
        """
        List all the lines loaded to the Calibrator.

        """

        self.spectrum_oned.calibrator.list_atlas()

    def clear_atlas(self):
        """
        Remove all the lines loaded to the Calibrator.

        """

        self.spectrum_oned.calibrator.clear_atlas()
        self.spectrum_oned.calibrator._generate_pairs()

    def do_hough_transform(self, brute_force: bool = False):
        """
        ** brute_force is EXPERIMENTAL as of 1 Sept 2021 **
        The brute force method is supposed to provide all the possible
        solution, hence given a sufficiently large max_tries, the solution
        should always be the best possible outcome. However, it does not
        seem to work in a small fraction of our tests. Use with caution,
        and it is not the recommended way for now.

        Perform Hough transform on the pixel-wavelength pairs with the
        configuration set by the set_hough_properties().

        Parameters
        ----------
        brute_force: bool (Default: False)
            Set to true to compute the gradient and intercept between
            every two data points

        """

        self.spectrum_oned.calibrator.do_hough_transform(
            brute_force=brute_force
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
        save_fig: bool (default: False)
            Save an image if set to True. matplotlib uses the pyplot.save_fig()
            while the plotly uses the pio.write_html() or pio.write_image().
            The support format types should be provided in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: (default: None)
            The destination to save the image.
        return_jsonstring: (default: False)
            Set to True to save the plotly figure as json string.
        renderer: (default: 'default')
            Set the rendered for the plotly display. Ignored if matplotlib is
            used.
        display: bool (Default: False)
            Set to True to display disgnostic plot.

        Return
        ------
        json object if json is True.

        """

        self.spectrum_oned.calibrator.plot_search_space(
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

    def fit(
        self,
        max_tries: int = 5000,
        fit_deg: int = 4,
        fit_coeff: Union[list, np.ndarray] = None,
        fit_tolerance: float = 5.0,
        fit_type: str = "poly",
        candidate_tolerance: float = 2.0,
        brute_force: bool = False,
        progress: bool = True,
        return_jsonstring: bool = False,
        display: bool = False,
        renderer: str = "default",
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        return_solution: bool = True,
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
        fit_tolerance: float (Default: 5.0)
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
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.

        """

        (
            fit_coeff,
            matched_peaks,
            matched_atlas,
            rms,
            residual,
            peak_utilisation,
            atlas_utilisation,
        ) = self.spectrum_oned.calibrator.fit(
            max_tries=max_tries,
            fit_deg=fit_deg,
            fit_coeff=fit_coeff,
            fit_tolerance=fit_tolerance,
            fit_type=fit_type,
            candidate_tolerance=candidate_tolerance,
            brute_force=brute_force,
            progress=progress,
        )

        if fit_type == "poly":
            fit_type_rascal = "poly"

        if fit_type == "legendre":
            fit_type_rascal = "leg"

        if fit_type == "chebyshev":
            fit_type_rascal = "cheb"

        self.spectrum_oned.add_fit_type(fit_type_rascal)

        self.spectrum_oned.add_fit_output_rascal(
            fit_coeff,
            matched_peaks,
            matched_atlas,
            rms,
            residual,
            peak_utilisation,
            atlas_utilisation,
        )

        if display or return_jsonstring or save_fig:
            self.spectrum_oned.calibrator.plot_fit(
                fit_coeff=fit_coeff,
                plot_atlas=True,
                log_spectrum=False,
                tolerance=fit_tolerance,
                display=display,
                filename=filename,
                return_jsonstring=return_jsonstring,
                renderer=renderer,
                save_fig=save_fig,
                fig_type=fig_type,
            )

        if return_solution:
            return (
                fit_coeff,
                matched_peaks,
                matched_atlas,
                rms,
                residual,
                peak_utilisation,
                atlas_utilisation,
            )

    def robust_refit(
        self,
        fit_coeff: Union[list, np.ndarray],
        n_delta: int = None,
        refine: bool = False,
        tolerance: float = 10.0,
        method: str = "Nelder-Mead",
        convergence: float = 1e-6,
        robust_refit: bool = True,
        fit_deg: Union[list, np.ndarray] = None,
        display: bool = False,
        renderer: str = "default",
        filename: str = None,
        return_jsonstring: bool = False,
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        return_solution: bool = True,
    ):
        """
        ** refine option is EXPERIMENTAL, as of 17 Jan 2021 **
        A wrapper function to robustly refit the wavelength solution with
        RASCAL when there is already a set of good coefficienes.

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
        fit_coeff : list
            List of polynomial fit coefficients.
        n_delta : int (Default: None)
            The number of the highest polynomial order to be adjusted
        refine : bool (Default: False)
            Set to True to refine solution.
        tolerance : float (Default: 10.)
            Absolute difference between fit and model in the unit of nm.
        method : string (Default: 'Nelder-Mead')
            scipy.optimize.minimize method.
        convergence : float (Default: 1e-6)
            scipy.optimize.minimize tol.
        robust_refit : bool (Default: True)
            Set to True to fit all the detected peaks with the given polynomial
            solution.
        fit_deg : int (Default: length of the input coefficients)
            Order of polynomial fit with all the detected peaks.
        display: bool (Default: False)
            Set to show diagnostic plot.
        renderer: str (Default: 'default')
            plotly renderer options.
        save_fig: string (Default: False)
            Set to save figure.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        return_solution: bool (Default: True)
            Set to True to return the best fit polynomial coefficients.

        """

        if fit_deg is None:
            fit_deg = len(fit_coeff) - 1

        if n_delta is None:
            n_delta = len(fit_coeff) - 1

        (
            fit_coeff,
            matched_peaks,
            matched_atlas,
            rms,
            residual,
            peak_utilisation,
            atlas_utilisation,
        ) = self.spectrum_oned.calibrator.match_peaks(
            fit_coeff,
            n_delta=n_delta,
            refine=refine,
            tolerance=tolerance,
            method=method,
            convergence=convergence,
            robust_refit=robust_refit,
            fit_deg=fit_deg,
        )

        rms = np.sqrt(np.nanmean(residual**2.0))

        if display:
            self.spectrum_oned.calibrator.plot_fit(
                fit_coeff=fit_coeff,
                plot_atlas=True,
                log_spectrum=False,
                tolerance=1.0,
                return_jsonstring=return_jsonstring,
                display=display,
                renderer=renderer,
                save_fig=save_fig,
                fig_type=fig_type,
                filename=filename,
            )

        self.spectrum_oned.add_fit_output_refine(
            fit_coeff,
            matched_peaks,
            matched_atlas,
            rms,
            residual,
            peak_utilisation,
            atlas_utilisation,
        )

        if return_solution:
            return (
                fit_coeff,
                matched_peaks,
                matched_atlas,
                rms,
                residual,
                peak_utilisation,
                atlas_utilisation,
            )

    def get_pix_wave_pairs(self):
        """
        Return the list of matched_peaks and matched_atlas with their
        position in the array.

        Return
        ------
        pw_pairs: list
            List of tuples each containing the array position, peak (pixel)
            and atlas (wavelength).

        """

        pw_pairs = self.spectrum_oned.calibrator.get_pix_wave_pairs()

        return pw_pairs

    def add_pix_wave_pair(self, pix: float, wave: float):
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

        """

        self.spectrum_oned.calibrator.add_pix_wave_pair(pix, wave)

    def remove_pix_wave_pair(self, arg: int):
        """
        Remove fitted pixel-wavelength pair from the Calibrator for refitting.
        The positions can be found from get_pix_wave_pairs(). One at a time.

        Parameters
        ----------
        arg: int
            The position of the pairs in the arrays.

        """

        self.spectrum_oned.calibrator.remove_pix_wave_pair(arg)

    def manual_refit(
        self,
        matched_peaks: Union[list, np.ndarray] = None,
        matched_atlas: Union[list, np.ndarray] = None,
        degree: int = None,
        x0: Union[list, np.ndarray] = None,
        return_solution: bool = True,
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
        return_solution: bool (Default: True)
            Set to True to return the best fit polynomial coefficients.

        """

        (
            self.fit_coeff,
            self.matched_peaks,
            self.matched_atlas,
            self.rms,
            self.residual,
        ) = self.spectrum_oned.calibrator.manual_refit(
            matched_peaks, matched_atlas, degree, x0
        )

        if return_solution:
            return (
                self.fit_coeff,
                self.matched_peaks,
                self.matched_atlas,
                self.rms,
                self.residual,
            )

    def get_calibrator(self):
        """
        Get the calibrator object.

        """

        return getattr(self.spectrum_oned, "calibrator")

    def get_spectrum_oned(self):
        """
        Get the spectrum_oned object.

        """

        return self.spectrum_oned

    def save_fits(
        self,
        output: str = "wavecal",
        filename: str = "wavecal",
        overwrite: bool = False,
        recreate: bool = False,
        empty_primary_hdu: bool = True,
    ):
        """
        Save the reduced data to disk, with a choice of any combination of the
        data that are already present in the SpectrumOneD. Because a
        WavelengthCalibration only requires a subset of all the data, only
        'wavecal' is guaranteed to exist.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        overwrite: bool
            Default is False.
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank

        """

        self.spectrum_oned.save_fits(
            output=output,
            filename=filename,
            overwrite=overwrite,
            recreate=recreate,
            empty_primary_hdu=empty_primary_hdu,
        )

    def save_csv(
        self,
        output: str = "wavecal",
        filename: str = "wavecal",
        overwrite: bool = False,
        recreate: bool = False,
    ):
        """
        Save the reduced data to disk, with a choice of any combination of the
        data that are already present in the SpectrumOneD. Because a
        WavelengthCalibration only requires a subset of all the data, only
        'wavecal' is guaranteed to exist.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        overwrite: bool
            Default is False.
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.

        """

        self.spectrum_oned.save_csv(
            output=output,
            filename=filename,
            overwrite=overwrite,
            recreate=recreate,
        )
