#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""For Two Dimensional operations"""

import copy
import datetime
import logging
import os
from itertools import chain
from typing import Union

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
from plotly import graph_objects as go
from plotly import io as pio
from scipy import ndimage, signal
from spectresc import spectres
from statsmodels.nonparametric.smoothers_lowess import lowess

from .extraction import (
    optimal_extraction_horne86,
    optimal_extraction_marsh89,
    tophat_extraction,
)
from .image_reduction import ImageReducer, ImageReduction
from .line_spread_function import (
    build_line_spread_profile,
    get_line_spread_function,
)
from .spectrum_oneD import SpectrumOneD
from .util import bfixpix, create_bad_pixel_mask

__all__ = ["TwoDSpec"]


class TwoDSpec:
    """
    This is a class for processing a 2D spectral image.

    """

    def __init__(
        self,
        data: Union[
            np.ndarray,
            fits.hdu.hdulist.HDUList,
            fits.hdu.hdulist.PrimaryHDU,
            fits.hdu.hdulist.ImageHDU,
            ImageReducer,
            ImageReduction,
        ] = None,
        header: fits.Header = None,
        verbose: bool = True,
        logger_name: str = "TwoDSpec",
        log_level: str = "INFO",
        log_file_folder: str = "default",
        log_file_name: str = None,
        **kwargs,
    ):
        """
        The constructor takes the data and the header, and the the header
        infromation will be read automatically. See set_properties()
        for the detail information of the keyword arguments. The extraction
        always consider the x-direction as the dispersion direction, while
        the y-direction as the spatial direction.

        parameters
        ----------
        data: 2D numpy array (M x N) OR astropy.io.fits object
            2D spectral image in either format
        header: FITS header (deafult: None)
            THIS WILL OVERRIDE the header from the astropy.io.fits object
        verbose: bool (Default: True)
            Set to False to suppress all verbose warnings, except for
            critical failure.
        logger_name: str (Default: TwoDSpec)
            This will set the name of the logger, if the name is used already,
            it will reference to the existing logger. This will be the
            first part of the default log file name unless log_file_name is
            provided.
        log_level: str (Default: 'INFO')
            Four levels of logging are available, in decreasing order of
            information and increasing order of severity: (1) DEBUG, (2) INFO,
            (3) WARNING, (4) ERROR and (5) CRITICAL. WARNING means that
            there is is_optimal operations in some parts of that step. ERROR
            means that the requested operation cannot be performed, but the
            software can handle it by either using the default setting or
            skipping the operation. CRITICAL means that the requested
            operation cannot be resolved without human interaction, this is
            most usually coming from missing data.
        log_file_folder: None or str (Default: "default")
            Folder in which the file is save, set to default to save to the
            current path.
        log_file_name: None or str (Default: None)
            File name of the log, set to None to print to screen only.
        **kwargs: keyword arguments (Default: see set_properties())
            see set_properties().

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

        self.img = None
        self.img_residual = None
        self.img_rectified = None
        self.img_residual_rectified = None
        self.img_mean = None
        self.img_median = None
        self.img_1_percentile = None
        self.header = None
        self.arc = None
        self.arc_rectified = None
        self.arc_header = None
        self.arc_mean = None
        self.arc_median = None
        self.arc_1_percentile = None
        self.bad_mask = None

        self.saxis = 1
        self.waxis = 0

        # Cosmic ray removal properties
        self.cosmicray = False
        self.fsmode = None
        self.psfmodel = None
        self.cosmicray_sigma = None

        self.spatial_mask = (1,)
        self.spec_mask = (1,)
        self.flip = False
        self.spec_size = None
        self.spatial_size = None

        self.spatial_mask_applied = False
        self.spec_mask_applied = False
        self.transpose_applied = False
        self.flip_applied = False

        # Default values if not supplied
        self.airmass = 1.0
        self.readnoise = 0.0
        self.gain = 1.0
        self.seeing = 1.0
        self.exptime = 1.0

        self.airmass_is_default_value = True
        self.readnoise_is_default_value = True
        self.gain_is_default_value = True
        self.seeing_is_default_value = True
        self.exptime_is_default_value = True

        self.zmin = None
        self.zmax = None

        self.start_window_idx = None
        self.spec_idx = None
        self.spec_pix = None
        self.resample_factor = 1.0

        self.nspec_traced = 0

        # rectification parameters
        self.rec_coeff = None
        self.rec_n_down = None
        self.rec_n_up = None
        self.rec_upsample_factor = None
        self.rec_bin_size = None
        self.rec_n_bin = None
        self.rec_spline_order = None
        self.rec_order = None

        # profile
        self.line_spread_profile_upsampled = None
        self.line_spread_profile = None

        self.verbose = verbose
        self.logger_name = logger_name
        self.log_level = log_level
        self.log_file_folder = log_file_folder
        self.log_file_name = log_file_name

        # Default keywords to be searched in the order in the list
        self.readnoise_keyword = ["RDNOISE", "RNOISE", "RN"]
        self.gain_keyword = ["GAIN", "EGAIN"]
        self.seeing_keyword = [
            "SEEING",
            "L1SEEING",
            "ESTSEE",
            "DIMMSEE",
            "SEEDIMM",
            "DSEEING",
            "FWHM",
            "L1FWHM",
            "AGFWHM",
        ]
        # AEPOSURE is the average exposure time computed in ImageReducer
        # it is the effective exposure time suitable for computing
        # the sensitivity curve.
        self.exptime_keyword = [
            "AXPOSURE",
            "XPOSURE",
            "EXPOSURE",
            "EXPTIME",
            "EXPOSED",
            "TELAPSED",
            "ELAPSED",
        ]
        self.airmass_keyword = ["AIRMASS", "AMASS", "AIRM", "AIR"]

        self.add_data(data, header)
        self.spectrum_list = {}
        self.set_properties(**kwargs)

        if self.arc is not None:
            self.apply_mask_to_arc()

    def add_data(
        self,
        data: Union[
            np.ndarray,
            fits.hdu.hdulist.HDUList,
            fits.hdu.hdulist.PrimaryHDU,
            fits.hdu.hdulist.ImageHDU,
            CCDData,
            ImageReducer,
            ImageReduction,
        ],
        header: fits.Header = None,
    ):
        """
        Adding the 2D image data to be processed. The data can be a 2D numpy
        array, an AstroPy ImageHDU/Primary HDU object or an ImageReducer
        object.

        parameters
        ----------
        data: 2D numpy array (M x N) OR astropy.io.fits object
            2D spectral image in either format
        header: FITS header (deafult: None)
            This take priority over the header from the
            fits.hdu.hdulist.HDUList, fits.hdu.image.PrimaryHDU,
            or CCDData.

        """

        # If data provided is an numpy array
        if isinstance(data, np.ndarray):
            self.img = data
            self.logger.info("An numpy array is loaded as data.")
            self.set_header(header)
            self.bad_mask = create_bad_pixel_mask(self.img)[0]

        # If it is a fits.hdu.hdulist.HDUList object
        elif isinstance(data, fits.hdu.hdulist.HDUList):
            self.img = data[0].data
            if header is None:
                self.set_header(data[0].header)
            else:
                self.set_header(header)
            self.bad_mask = create_bad_pixel_mask(self.img)[0]
            self.logger.warning(
                "An HDU list is provided, only the first HDU will be read."
            )

        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(data, fits.hdu.image.PrimaryHDU) or isinstance(
            data, fits.hdu.image.ImageHDU
        ):
            self.img = data.data
            if header is None:
                self.set_header(data.header)
            else:
                self.set_header(header)
            self.bad_mask = create_bad_pixel_mask(self.img)[0]
            self.logger.info("A PrimaryHDU is loaded as data.")

        # If it is a CCDData
        elif isinstance(data, CCDData):
            self.img = data.data
            if header is None:
                self.set_header(data.header)
            else:
                self.set_header(header)
            self.bad_mask = create_bad_pixel_mask(self.img)[0]
            self.logger.info("A CCDData is loaded as data.")

        # If it is an ImageReducer object
        elif isinstance(data, (ImageReducer, ImageReduction)):
            # If the data is not reduced, reduce it here. Error handling is
            # done by the ImageReducer class
            if data.image_fits is None:
                data.create_image_fits()

            self.img = data.image_fits.data
            self.logger.info("An ImageReudction object is loaded as data.")

            if header is None:
                self.set_header(data.image_fits.header)
            else:
                self.set_header(header)
            if data.arc_main is not None:
                self.add_arc(data.arc_main, data.arc_header[0])

            else:
                self.logger.warning(
                    "Arc frame is not in the ImageReducer "
                    "object, please supplied manually if you wish to perform "
                    "wavelength calibration."
                )

            self.bad_mask = data.bad_mask

        # If a filepath is provided
        elif isinstance(data, str):
            # If HDU number is provided
            if data[-1] == "]":
                filepath, hdunum = data.split("[")
                hdunum = int(hdunum[:-1])

            # If not, assume the HDU idnex is 0
            else:
                filepath = data
                hdunum = 0

            # Load the file and dereference it afterwards
            fitsfile_tmp = fits.open(filepath)[hdunum]
            self.img = copy.deepcopy(fitsfile_tmp.data)
            self.set_header(copy.deepcopy(fitsfile_tmp.header))
            logging.info(
                "Loaded data from: %s, with hdunum: %s", filepath, hdunum
            )
            self.bad_mask = create_bad_pixel_mask(self.img)[0]

            fitsfile_tmp = None

        elif data is None:
            pass

        else:
            error_msg = (
                "Please provide a numpy array, an "
                + "astropy.io.fits.hdu.image.PrimaryHDU object "
                + "or an ImageReducer object."
            )
            self.logger.critical(error_msg)
            raise TypeError(error_msg)

        if self.img is not None:
            # We perform the tracing on a *pixel healed* temporary image
            if self.bad_mask is not None:
                if self.bad_mask.shape == self.img.shape:
                    self.img = bfixpix(self.img, self.bad_mask, retdat=True)

            self.img_residual = self.img.copy()
            self._get_image_size()
            self._get_image_zminmax()
            self.img_mean = np.nanmean(self.img)
            self.img_median = np.nanmedian(self.img)
            self.img_1_percentile = np.nanpercentile(self.img, 1.0)
            self.logger.info(f"mean value of the image is {self.img_mean}")
            self.logger.info(f"median value of the image is {self.img_median}")
            self.logger.info(
                f"0.1 percentile of the image is {self.img_1_percentile}"
            )

    def set_properties(
        self,
        saxis: int = None,
        variance: Union[int, float] = None,
        spatial_mask: np.ndarray = None,
        spec_mask: np.ndarray = None,
        flip: Union[bool, int] = None,
        cosmicray: Union[bool, int] = None,
        gain: Union[int, float] = -1,
        readnoise: Union[int, float] = -1,
        fsmode: str = None,
        psfmodel: str = None,
        seeing: Union[int, float] = -1,
        exptime: Union[int, float] = -1,
        airmass: Union[int, float] = -1,
        verbose: bool = None,
        **kwargs,
    ):
        """
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
        |  top   | bottom |   0   | False |
        +--------+--------+-------+-------+
        | bottom |  top   |   0   |  True |
        +--------+--------+-------+-------+

        Spectra are sorted by their brightness. If there are multiple spectra
        on the image, and the target is not the brightest source, use at least
        the number of spectra visible to eye and pick the one required later.
        The default automated outputs is the brightest one, which is the
        most common case for images from a long-slit spectrograph.

        Parameters
        ----------
        saxis: int (Default: 1)
            dispersion direction, 0 for vertical, 1 for horizontal.
        variance: 2D numpy array (M, N)
            The per-pixel-variance of the frame.
        spatial_mask: 1D numpy array (size: N. Default is (1,))
            Mask in the spatial direction, can be the indices of the pixels
            to be included (size <N) or a 1D numpy array of True/False (size N)
        spec_mask: 1D numpy array (Size: M. Default: (1,))
            Mask in the dispersion direction, can be the indices of the pixels
            to be included (size <M) or a 1D numpy array of True/False (size M)
        flip: bool (Deafult: False)
            If the frame has to be left-right flipped, set to True.
        cosmicray: bool (Default: True)
            Set to True to remove cosmic rays, this directly alter the reduced
            image data. We only explicitly include the 4 most important
            parameters in this function: `gain`, `readnoise`, `fsmode`, and
            `psfmodel`, the rest can be configured with kwargs.
        gain: float (Deafult: -1)
            Gain of the detector in unit of electron per photon, not important
            if noise estimation is not needed. Negative value means "pass",
            i.e. do nothing. None means grabbing from the header, though if it
            is not found, it is set to 1.0.
        readnoise: float (Deafult: -1)
            Readnoise of the detector in unit of electron, not important if
            noise estimation is not needed. Negative value means "pass",
            i.e. do nothing. None means grabbing from the header, though if it
            is not found, it is set to 0.0.
        fsmode: str (Default: None. Use 'convolve' if not set.)
            Method to build the fine structure image: `median`: Use the median
            filter in the standard LA Cosmic algorithm. `convolve`: Convolve
            the image with the psf kernel to calculate the fine structure
            image.
        psfmodel: str (Default: None. Use 'gaussy' if not set.)
            Model to use to generate the psf kernel if fsmode is `convolve`
            and psfk is None. The current choices are Gaussian and Moffat
            profiles. 'gauss' and 'moffat' produce circular PSF kernels. The
            `gaussx` and `gaussy` produce Gaussian kernels in the x and y
            directions respectively. `gaussxy` and `gaussyx` apply the
            Gaussian kernels in the x then the y direction, and first y then
            x direction, respectively.
        seeing: float (Deafult: -1)
            Seeing in unit of arcsec, use as the first guess of the line
            spread function of the spectra. Negative value means "pass",
            i.e. do nothing. None means grabbing from the header, though if it
            is not found, it is set to 1.0.
        exptime: float (Deafult: -1)
            Esposure time for the observation, not important if absolute flux
            calibration is not needed. Negative value means "pass",
            i.e. do nothing. None means grabbing from the header, though if it
            is not found, it is set to 1.0.
        airmass: float (Default: -1)
            The airmass where the observation carries out. Negative value
            means "pass", i.e. do nothing. None means grabbing from the
            header, though if it is not found, it is set to 0.0.
        verbose: bool
            Set to False to suppress all verbose warnings, except for
            critical failure.
        **kwargs:
            Extra keyword arguments for the astroscrappy.detect_cosmics:
            https://astroscrappy.readthedocs.io/en/latest/api/
            astroscrappy.detect_cosmics.html
            The default setting is::

                astroscrappy.detect_cosmics(indat, inmask=None, bkg=None,
                    var=None, sigclip=4.5, sigfrac=0.3, objlim=5.0, gain=1.0,
                    readnoise=6.5, satlevel=65536.0, niter=4, sepmed=True,
                    cleantype='meanmask', fsmode='median', psfmodel='gauss',
                    psffwhm=2.5, psfsize=7, psfk=None, psfbeta=4.765,
                    verbose=False)

        """

        if saxis is not None:
            self.saxis = saxis

            if self.saxis == 1:
                self.waxis = 0

            elif self.saxis == 0:
                self.waxis = 1

            else:
                self.saxis = 0
                self.logger.error(
                    "saxis can only be 0 or 1, %s is given. It is set to 0.",
                    saxis,
                )

        if spatial_mask is not None:
            self.spatial_mask = spatial_mask

        if spec_mask is not None:
            self.spec_mask = spec_mask

        if flip is not None:
            self.flip = flip

        self.set_readnoise(readnoise)
        self.set_gain(gain)
        self.set_seeing(seeing)
        self.set_exptime(exptime)
        self.set_airmass(airmass)

        if cosmicray is not None:
            self.cosmicray = cosmicray

        if fsmode is not None:
            self.fsmode = fsmode

        else:
            if self.fsmode is None:
                self.fsmode = "convolve"

        if psfmodel is not None:
            self.psfmodel = psfmodel

        else:
            if self.psfmodel is None:
                self.psfmodel = "gaussy"

        if kwargs is not None:
            self.cr_kwargs = kwargs

        # cosmic ray rejection
        if self.cosmicray:
            self.logger.info("Removing cosmic rays in mode: %s.", psfmodel)

            if self.fsmode == "convolve":
                if psfmodel == "gaussyx":
                    self.img = detect_cosmics(
                        self.img / self.gain,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode="convolve",
                        psfmodel="gaussy",
                        **kwargs,
                    )[1]

                    self.img = detect_cosmics(
                        self.img / self.gain,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode="convolve",
                        psfmodel="gaussx",
                        **kwargs,
                    )[1]

                elif psfmodel == "gaussxy":
                    self.img = detect_cosmics(
                        self.img / self.gain,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode="convolve",
                        psfmodel="gaussx",
                        **kwargs,
                    )[1]

                    self.img = detect_cosmics(
                        self.img / self.gain,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode="convolve",
                        psfmodel="gaussy",
                        **kwargs,
                    )[1]

                else:
                    self.img = detect_cosmics(
                        self.img / self.gain,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode="convolve",
                        psfmodel=self.psfmodel,
                        **kwargs,
                    )[1]

            else:
                self.img = detect_cosmics(
                    self.img / self.gain,
                    gain=self.gain,
                    readnoise=self.readnoise,
                    fsmode=self.fsmode,
                    psfmodel=self.psfmodel,
                    **kwargs,
                )[1]

        if verbose is not None:
            self.verbose = verbose

        if self.img is not None:
            # the valid y-range of the chip (i.e. spatial direction)
            if len(self.spatial_mask) > 1:
                if self.saxis == 1:
                    self.img = self.img[self.spatial_mask]

                    if self.img_residual is not None:
                        self.img_residual = self.img_residual[
                            self.spatial_mask
                        ]

                    if self.bad_mask is not None:
                        self.bad_mask = self.bad_mask[self.spatial_mask]

                else:
                    self.img = self.img[:, self.spatial_mask]

                    if self.img_residual is not None:
                        self.img_residual = self.img_residual[
                            :, self.spatial_mask
                        ]

                    if self.bad_mask is not None:
                        self.bad_mask = self.bad_mask[:, self.spatial_mask]

                self.spatial_mask_applied = True

            # the valid x-range of the chip (i.e. dispersion direction)
            if len(self.spec_mask) > 1:
                if self.saxis == 1:
                    self.img = self.img[:, self.spec_mask]

                    if self.img_residual is not None:
                        self.img_residual = self.img_residual[
                            :, self.spec_mask
                        ]

                    if self.bad_mask is not None:
                        self.bad_mask = self.bad_mask[:, self.spec_mask]

                else:
                    self.img = self.img[self.spec_mask]

                    if self.img_residual is not None:
                        self.img_residual = self.img_residual[self.spec_mask]

                    if self.bad_mask is not None:
                        self.bad_mask = self.bad_mask[self.spec_mask]

                self.spec_mask_applied = True

            if self.saxis == 0:
                self.img = np.transpose(self.img)

                if self.img_residual is not None:
                    self.img_residual = np.transpose(self.img_residual)

                if self.bad_mask is not None:
                    self.bad_mask = np.transpose(self.bad_mask)

                self.transpose_applied = True

            if self.flip:
                self.img = np.flip(self.img)

                if self.img_residual is not None:
                    self.img_residual = np.flip(self.img_residual)

                if self.bad_mask is not None:
                    self.bad_mask = np.flip(self.bad_mask)

                self.flip_applied = True

            self._get_image_size()
            self._get_image_zminmax()

            if (variance is not None) & (
                np.shape(variance) == np.shape(self.img)
            ):
                self.variance = variance

            elif isinstance(variance, (int, float)):
                self.variance = np.ones_like(self.img) * variance

            else:
                self.logger.info(
                    "Variance image is created from the modulus of the image "
                    "and the readnoise value."
                )
                self.variance = np.abs(self.img) + self.readnoise**2

        else:
            self.variance = None

    def _get_image_size(self):
        # get the length in the spectral and spatial directions
        self.spec_size = np.shape(self.img)[1]
        self.spatial_size = np.shape(self.img)[0]
        self.logger.info("spec_size is found to be %s.", self.spec_size)
        self.logger.info("spatial_size is found to be %s.", self.spatial_size)

    def _get_image_zminmax(self):
        # set the 2D histogram z-limits
        img_log = np.log10(self.img)
        img_log_finite = img_log[np.isfinite(img_log)]
        self.zmin = np.nanpercentile(img_log_finite, 5)
        self.zmax = np.nanpercentile(img_log_finite, 95)
        self.logger.info("zmin is set to %s.", self.zmin)
        self.logger.info("zmax is set to %s.", self.zmax)

    # Get the readnoise
    def set_readnoise(self, readnoise: Union[float, str] = None):
        """
        Set the readnoise of the image.

        Parameters
        ----------
        readnoise: str, float, int or None (Default: None)
            If a string is provided, it will be treated as a header keyword
            for the readnoise value. Float or int will be used as the
            readnoise value. If None is provided, the header will be searched
            with the set of default readnoise keywords.

        """

        if (readnoise is not None) and (self.readnoise is not None):
            if isinstance(readnoise, str):
                # use the supplied keyword
                self.readnoise = float(self.header[readnoise])
                self.logger.info(
                    "readnoise is found to be %s.", self.readnoise
                )
                self.readnoise_is_default_value = False

            elif isinstance(readnoise, (float, int)) & (~np.isnan(readnoise)):
                if readnoise < 0:
                    pass

                else:
                    # use the given readnoise value
                    self.readnoise = float(readnoise)
                    self.logger.info("readnoise is set to %s.", self.readnoise)
                    self.readnoise_is_default_value = False

            else:
                self.readnoise = 0.0
                self.logger.warning(
                    (
                        "readnoise has to be None, a numeric value or the FITS"
                        " header keyword, %s is  given. It is set to 0."
                    ),
                    readnoise,
                )
                self.readnoise_is_default_value = True

        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:
                readnoise_keyword_matched = np.in1d(
                    self.readnoise_keyword, self.header
                )

                if readnoise_keyword_matched.any():
                    self.readnoise = self.header[
                        self.readnoise_keyword[
                            np.where(readnoise_keyword_matched)[0][0]
                        ]
                    ]
                    self.logger.info(
                        "readnoise is found to be %s.", self.readnoise
                    )
                    self.readnoise_is_default_value = False

                else:
                    self.readnoise = 0.0
                    self.logger.warning(
                        "Readnoise value cannot be identified. It is set to 0."
                    )
                    self.readnoise_is_default_value = True

            else:
                self.readnoise = 0.0
                self.logger.warning(
                    "Header is not provided. Readnoise value "
                    "is not provided. It is set to 0."
                )
                self.readnoise_is_default_value = True

    # Get the gain
    def set_gain(self, gain: Union[float, str] = None):
        """
        Set the gain of the image.

        Parameters
        ----------
        gain: str, float, int or None (Default: None)
            If a string is provided, it will be treated as a header keyword
            for the gain value. Float or int will be used as the
            gain value. If None is provided, the header will be searched
            with the set of default gain keywords.

        """

        if (gain is not None) and (self.gain is not None):
            if isinstance(gain, str):
                # use the supplied keyword
                self.gain = float(self.header[gain])
                self.logger.info("gain is found to be %s.", self.gain)
                self.gain_is_default_value = False

            elif isinstance(gain, (float, int)) & (~np.isnan(gain)):
                if gain < 0:
                    pass

                else:
                    # use the given gain value
                    self.gain = float(gain)
                    self.logger.info("gain is set to %s.", self.gain)
                    self.gain_is_default_value = False

            else:
                self.gain = 1.0
                self.logger.warning(
                    (
                        "Gain has to be None, a numeric value or the FITS "
                        "header keyword, %s is given. It is set to 1."
                    ),
                    gain,
                )
                self.gain_is_default_value = True
        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:
                gain_keyword_matched = np.in1d(self.gain_keyword, self.header)

                if gain_keyword_matched.any():
                    self.gain = self.header[
                        self.gain_keyword[np.where(gain_keyword_matched)[0][0]]
                    ]
                    self.logger.info("gain is found to be %s.", self.gain)
                    self.gain_is_default_value = False

                else:
                    self.gain = 1.0
                    self.logger.warning(
                        "Gain value cannot be identified. It is set to 1."
                    )
                    self.gain_is_default_value = True

            else:
                self.gain = 1.0
                self.logger.warning(
                    "Header is not provide. Gain value is not provided. It "
                    "is set to 1."
                )
                self.gain_is_default_value = True

    # Get the Seeing
    def set_seeing(self, seeing: Union[float, str] = None):
        """
        Set the seeing of the image.

        Parameters
        ----------
        seeing: str, float, int or None (Default: None)
            If a string is provided, it will be treated as a header keyword
            for the seeing value. Float or int will be used as the
            seeing value. If None is provided, the header will be searched
            with the set of default seeing keywords.

        """

        if (seeing is not None) and (self.seeing is not None):
            if isinstance(seeing, str):
                # use the supplied keyword
                self.seeing = float(self.header[seeing])
                self.logger.info("seeing is found to be %s.", self.seeing)
                self.seeing_is_default_value = False

            elif isinstance(seeing, (float, int)) & (~np.isnan(seeing)):
                if seeing < 0:
                    pass

                else:
                    # use the given seeing value
                    self.seeing = float(seeing)
                    self.logger.info("seeing is set to %s.", self.seeing)
                    self.seeing_is_default_value = False

            else:
                self.seeing = 1.0
                self.logger.warning(
                    (
                        "Seeing has to be None, a numeric value or the FITS "
                        "header keyword, %s is given. It is set to 1."
                    ),
                    seeing,
                )
                self.seeing_is_default_value = True

        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:
                seeing_keyword_matched = np.in1d(
                    self.seeing_keyword, self.header
                )

                if seeing_keyword_matched.any():
                    self.seeing = self.header[
                        self.seeing_keyword[
                            np.where(seeing_keyword_matched)[0][0]
                        ]
                    ]
                    self.logger.info("seeing is found to be %s.", self.seeing)
                    self.seeing_is_default_value = False

                else:
                    self.seeing = 1.0
                    self.logger.warning(
                        "Seeing value cannot be identified. It is set to 1."
                    )
                    self.seeing_is_default_value = True

            else:
                self.seeing = 1.0
                self.logger.warning(
                    "Header is not provided. Seeing value is not provided. "
                    "It is set to 1."
                )
                self.seeing_is_default_value = True

    # Get the Exposure Time
    def set_exptime(self, exptime: Union[float, str] = None):
        """
        Set the exptime of the image.

        Parameters
        ----------
        exptime: str, float, int or None (Default: None)
            If a string is provided, it will be treated as a header keyword
            for the exptime value. Float or int will be used as the
            exptime value. If None is provided, the header will be searched
            with the set of default exptime keywords.

        """

        if (exptime is not None) and (self.exptime is not None):
            if isinstance(exptime, str):
                # use the supplied keyword
                self.exptime = float(self.header[exptime])
                self.logger.info("exptime is found to be %s.", self.exptime)
                self.exptime_is_default_value = False

            elif isinstance(exptime, (float, int)) & (~np.isnan(exptime)):
                if exptime < 0:
                    pass

                else:
                    # use the given exptime value
                    self.exptime = float(exptime)
                    self.logger.info("exptime is set to %s.", self.exptime)
                    self.exptime_is_default_value = False

            else:
                self.exptime = 1.0
                self.logger.warning(
                    (
                        "Exposure Time has to be None, a numeric value or the "
                        "FITS header keyword, %s is given. It is set to 1."
                    ),
                    exptime,
                )
                self.exptime_is_default_value = True

        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:
                exptime_keyword_matched = np.in1d(
                    self.exptime_keyword, self.header
                )

                if exptime_keyword_matched.any():
                    self.exptime = self.header[
                        self.exptime_keyword[
                            np.where(exptime_keyword_matched)[0][0]
                        ]
                    ]
                    self.logger.info(
                        "exptime is found to be %s.", self.exptime
                    )
                    self.exptime_is_default_value = False

                else:
                    self.exptime = 1.0
                    self.logger.warning(
                        "Exposure Time value cannot be identified. It is set "
                        "to 1."
                    )
                    self.exptime_is_default_value = True

            else:
                self.exptime = 1.0
                self.logger.warning(
                    "Header is not provided. Exposure Time value is not "
                    "provided. It is set to 1."
                )
                self.exptime_is_default_value = True

    # Get the Airmass
    def set_airmass(self, airmass: Union[float, str] = None):
        """
        Set the airmass of the image.

        Parameters
        ----------
        airmass: str, float, int or None (Default: None)
            If a string is provided, it will be treated as a header keyword
            for the airmass value. Float or int will be used as the
            airmass value. If None is provided, the header will be searched
            with the set of default airmass keywords.

        """

        if (airmass is not None) and (self.airmass is not None):
            if isinstance(airmass, str):
                # use the supplied keyword
                self.airmass = float(self.header[airmass])
                self.logger.info("Airmass is found to be %s.", self.airmass)
                self.airmass_is_default_value = False

            elif isinstance(airmass, (float, int)) & (~np.isnan(airmass)):
                if airmass < 0:
                    pass

                else:
                    # use the given airmass value
                    self.airmass = float(airmass)
                    self.logger.info("Airmass is set to %s.", self.airmass)
                    self.airmass_is_default_value = False

            else:
                self.logger.warning(
                    (
                        "Airmass has to be None, a numeric value or the FITS "
                        "header keyword, %s is given. It is set to 1."
                    ),
                    airmass,
                )
                self.airmass = 1.0
                self.airmass_is_default_value = True

        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:
                airmass_keyword_matched = np.in1d(
                    self.airmass_keyword, self.header
                )

                if airmass_keyword_matched.any():
                    self.airmass = self.header[
                        self.airmass_keyword[
                            np.where(airmass_keyword_matched)[0][0]
                        ]
                    ]
                    self.logger.info(
                        "Airmass is found to be %s.", self.airmass
                    )
                    self.airmass_is_default_value = False

                else:
                    self.airmass = 1.0
                    self.logger.warning(
                        "Airmass value cannot be identified. It is set to 1."
                    )
                    self.airmass_is_default_value = True

            else:
                self.airmass = 1.0
                self.logger.warning(
                    "Header is not provided. Airmass value is not provided. "
                    "It is set to 1."
                )
                self.airmass_is_default_value = True

    def add_bad_mask(
        self,
        bad_mask: Union[
            np.ndarray,
            fits.hdu.hdulist.HDUList,
            fits.hdu.image.PrimaryHDU,
            fits.hdu.image.ImageHDU,
            str,
        ] = None,
    ):
        """
        To provide a mask to ignore the bad pixels in the reduction.

        Parameters
        ----------
        bad_mask: numpy.ndarray, PrimaryHDU/ImageHDU, ImageReducer, str
            The bad pixel mask of the image, make sure it is of the same size
            as the image and the right orientation.

        """

        # If data provided is an numpy array
        if isinstance(bad_mask, np.ndarray):
            self.bad_mask = bad_mask

        # If it is a fits.hdu.hdulist.HDUList object
        elif isinstance(bad_mask, fits.hdu.hdulist.HDUList):
            self.bad_mask = bad_mask[0].data
            self.logger.warning(
                "An HDU list is provided, only the first HDU will be read."
            )

        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(bad_mask, fits.hdu.image.PrimaryHDU) or isinstance(
            bad_mask, fits.hdu.image.ImageHDU
        ):
            self.bad_mask = bad_mask.data

        # If a filepath is provided
        elif isinstance(bad_mask, str):
            # If HDU number is provided
            if bad_mask[-1] == "]":
                filepath, hdunum = bad_mask.split("[")
                hdunum = int(hdunum[:-1])

            # If not, assume the HDU idnex is 0
            else:
                filepath = bad_mask
                hdunum = 0

            # Load the file and dereference it afterwards
            fitsfile_tmp = fits.open(filepath)[hdunum]
            if isinstance(fitsfile_tmp, fits.hdu.hdulist.HDUList):
                fitsfile_tmp = fitsfile_tmp[0]
                self.logger.warning(
                    "An HDU list is provided, only the first HDU will be read."
                )
            fitsfile_tmp_shape = np.shape(fitsfile_tmp.data)

            # Normal case
            if len(fitsfile_tmp_shape) == 2:
                self.logger.debug("arc.data is 2 dimensional.")
                self.bad_mask = fitsfile_tmp.data

            # Try to trap common error when saving FITS file
            # Case with multiple image extensions, we only take the first one
            elif len(fitsfile_tmp_shape) == 3:
                self.logger.debug("arc.data is 3 dimensional.")
                self.bad_mask = fitsfile_tmp.data[0]

            # Case with an extra bracket when saving
            elif len(fitsfile_tmp_shape) == 1:
                self.logger.debug("arc.data is 1 dimensional.")
                # In case it in a multiple extension format, we take the
                # first one only
                if len(np.shape(fitsfile_tmp.data[0]) == 3):
                    self.bad_mask = fitsfile_tmp.data[0][0]

                else:
                    self.bad_mask = fitsfile_tmp.data[0]

            else:
                error_msg = (
                    "Please check the shape/dimension of the "
                    + "input light frame, it is probably empty "
                    + "or has an atypical output format."
                )
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

        else:
            error_msg = (
                "Please provide a numpy array, an "
                + "astropy.io.fits.hdu.image.PrimaryHDU object, an "
                + "astropy.io.fits.hdu.image.ImageHDU object, an "
                + "astropy.io.fits.HDUList object."
            )
            self.logger.critical(error_msg)
            raise TypeError(error_msg)

    def add_arc(
        self,
        arc: Union[
            np.ndarray,
            fits.hdu.hdulist.HDUList,
            fits.hdu.image.PrimaryHDU,
            fits.hdu.image.ImageHDU,
            CCDData,
            str,
        ],
        header: fits.Header = None,
    ):
        """
        To provide an arc image. Make sure left (small index) is blue,
        right (large index) is red.

        Parameters
        ----------
        arc: numpy.ndarray, PrimaryHDU/ImageHDU, ImageReducer, str
            The image of the arc image.
        header: FITS header (deafult: None)
            An astropy.io.fits.Header object. This is not used if arc is
            a PrimaryHDU or ImageHDU.

        """

        # If data provided is an numpy array
        if isinstance(arc, np.ndarray):
            self.arc = arc
            self.set_arc_header(header)

        # If it is a fits.hdu.hdulist.HDUList object
        elif isinstance(arc, fits.hdu.hdulist.HDUList):
            self.arc = arc[0].data
            self.set_arc_header(arc[0].header)
            self.logger.warning(
                "An HDU list is provided, only the first HDU will be read."
            )

        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(arc, fits.hdu.image.PrimaryHDU) or isinstance(
            arc, fits.hdu.image.ImageHDU
        ):
            self.arc = arc.data
            self.set_arc_header(arc.header)

        # If it is a CCDData
        elif isinstance(arc, CCDData):
            self.arc = arc.data
            if header is None:
                self.set_arc_header(arc.header)
            else:
                self.set_arc_header(header)
            self.logger.info("A CCDData is loaded as arc data.")

        # If a filepath is provided
        elif isinstance(arc, str):
            # If HDU number is provided
            if arc[-1] == "]":
                filepath, hdunum = arc.split("[")
                hdunum = int(hdunum[:-1])

            # If not, assume the HDU idnex is 0
            else:
                filepath = arc
                hdunum = 0

            # Load the file and dereference it afterwards
            fitsfile_tmp = fits.open(filepath)[hdunum]
            if isinstance(fitsfile_tmp, fits.hdu.hdulist.HDUList):
                fitsfile_tmp = fitsfile_tmp[0]
                self.logger.warning(
                    "An HDU list is provided, only the first HDU will be read."
                )

            fitsfile_tmp_shape = np.shape(fitsfile_tmp.data)

            # Normal case
            if len(fitsfile_tmp_shape) == 2:
                self.logger.debug("arc.data is 2 dimensional.")
                self.arc = fitsfile_tmp.data
                self.set_arc_header(fitsfile_tmp.header)

            # Try to trap common error when saving FITS file
            # Case with multiple image extensions, we only take the first one
            elif len(fitsfile_tmp_shape) == 3:
                self.logger.debug("arc.data is 3 dimensional.")
                self.arc = fitsfile_tmp.data[0]
                self.set_arc_header(fitsfile_tmp.header)

            # Case with an extra bracket when saving
            elif len(fitsfile_tmp_shape) == 1:
                self.logger.debug("arc.data is 1 dimensional.")
                # In case it in a multiple extension format, we take the
                # first one only
                if len(np.shape(fitsfile_tmp.data[0]) == 3):
                    self.arc = fitsfile_tmp.data[0][0]
                    self.set_arc_header(fitsfile_tmp[0].header)

                else:
                    self.arc = fitsfile_tmp.data[0]
                    self.set_arc_header(fitsfile_tmp[0].header)

            else:
                error_msg = (
                    "Please check the shape/dimension of the input light "
                    "frame, it is probably empty or has an atypical output "
                    "format."
                )
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

        else:
            error_msg = (
                "Please provide a numpy array, an "
                "astropy.io.fits.hdu.image.PrimaryHDU object, an "
                "astropy.io.fits.hdu.image.ImageHDU object, an "
                "astropy.io.fits.HDUList object, or an "
                "aspired.ImageReducer object."
            )
            self.logger.critical(error_msg)
            raise TypeError(error_msg)

        # Only compute if no error is raised
        self.arc_mean = np.nanmean(self.arc)
        self.arc_median = np.nanmedian(self.arc)
        self.arc_1_percentile = np.nanpercentile(self.arc, 1.0)

    def set_arc_header(self, header: fits.Header):
        """
        Adding the header for the arc.

        Parameters
        ----------
        header: FITS header (deafult: None)
            An astropy.io.fits.Header object. This is not used if arc is
            a PrimaryHDU or ImageHDU.

        """

        # If it is a fits.hdu.header.Header object
        if isinstance(header, fits.header.Header):
            self.arc_header = header

        elif isinstance(header, (list, tuple)):
            if isinstance(header[0], fits.header.Header):
                self.arc_header = header[0]
                self.logger.info("arc_header is set.")

            else:
                self.arc_header = None
                error_msg = (
                    "Please provide a valid "
                    + "astropy.io.fits.header.Header object. Process "
                    + "without storing the header of the arc file."
                )
                self.logger.warning(error_msg)

        else:
            self.arc_header = None
            error_msg = (
                "Please provide a valid "
                + "astropy.io.fits.header.Header object. Process "
                + "without storing the header of the arc file."
            )
            self.logger.warning(error_msg)

    def apply_mask_to_arc(self):
        """
        Apply both the spec_mask and spatial_mask that are already stroed in
        the object.

        """

        if self.transpose_applied is True:
            self.apply_transpose_to_arc()

        if self.flip_applied is True:
            self.apply_flip_to_arc()

        if np.shape(self.arc) == np.shape(self.img):
            pass

        else:
            self.apply_spec_mask_to_arc(self.spec_mask)
            self.apply_spatial_mask_to_arc(self.spatial_mask)

    def apply_spec_mask_to_arc(self, spec_mask: np.ndarray):
        """
        Apply to use only the valid x-range of the chip (i.e. dispersion
        direction)

        parameters
        ----------
        spec_mask: 1D numpy array (M)
            Mask in the dispersion direction, can be the indices of the pixels
            to be included (size <M) or a 1D numpy array of True/False (size M)
            (Default is (1,) i.e. keep everything)

        """

        if len(spec_mask) > 1:
            self.arc = self.arc[:, spec_mask]
            self.logger.info("spec_mask is applied to arc.")

        else:
            self.logger.info(
                "spec_mask has zero length, it cannot be applied to the arc."
            )

    def apply_spatial_mask_to_arc(self, spatial_mask: np.ndarray):
        """
        Apply to use only the valid y-range of the chip (i.e. spatial
        direction)

        parameters
        ----------
        spatial_mask: 1D numpy array (N)
            Mask in the spatial direction, can be the indices of the pixels
            to be included (size <N) or a 1D numpy array of True/False (size N)
            (Default is (1,) i.e. keep everything)

        """

        if len(spatial_mask) > 1:
            self.arc = self.arc[spatial_mask]
            self.logger.info("spatial_mask is applied to arc.")

        else:
            self.logger.info(
                "spatial_mask has zero length, it cannot be applied to the arc."
            )

    def apply_transpose_to_arc(self):
        """
        Apply transpose to arc.

        """

        self.arc = np.transpose(self.arc)

    def apply_flip_to_arc(self):
        """
        Apply flip to arc.

        """

        self.arc = np.flip(self.arc)

    def set_readnoise_keyword(
        self, keyword_list: list, append: bool = False, update: bool = True
    ):
        """
        Set the readnoise keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        append: bool (Default: False)
            Set to False to overwrite the current list.
        update: bool (Default: True)
            Set to True to search for the readnoise after the new list
            is provided.

        """

        if isinstance(keyword_list, str):
            keyword_list = [keyword_list]

        elif isinstance(keyword_list, list):
            pass

        elif isinstance(keyword_list, np.ndarray):
            keyword_list = list(keyword_list)

        else:
            self.logger.error(
                "Please provide the keyword list in str, list or numpy.ndarray."
            )

        if append:
            self.readnoise_keyword += keyword_list
            self.logger.info(
                "%s is appended to the readnoise_keyword list.", keyword_list
            )

        else:
            self.readnoise_keyword = keyword_list
            self.logger.info(
                "%s is used as the readnoise_keyword list.", keyword_list
            )

        if update:
            self.set_readnoise()

        else:
            self.logger.info(
                "readnoise_keyword list is updated, but it is "
                "opted not to update the readnoise automatically."
            )

    def set_gain_keyword(
        self, keyword_list: list, append: bool = False, update: bool = True
    ):
        """
        Set the gain keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        append: bool (Default: False)
            Set to False to overwrite the current list.
        update: bool (Default: True)
            Set to True to search for the readnoise after the new list
            is provided.

        """

        if isinstance(keyword_list, str):
            keyword_list = [keyword_list]

        elif isinstance(keyword_list, list):
            pass

        elif isinstance(keyword_list, np.ndarray):
            keyword_list = list(keyword_list)

        else:
            self.logger.error(
                "Please provide the keyword list in str, list or numpy.ndarray."
            )

        if append:
            self.gain_keyword += keyword_list
            self.logger.info(
                "%s is appended to the gain_keyword list.", keyword_list
            )

        else:
            self.gain_keyword = keyword_list
            self.logger.info(
                "%s is used as the gain_keyword list.", keyword_list
            )

        if update:
            self.set_gain()

        else:
            self.logger.info(
                "gain_keyword list is updated, but it is "
                "opted not to update the gain automatically."
            )

    def set_seeing_keyword(
        self, keyword_list: list, append: bool = False, update: bool = True
    ):
        """
        Set the seeing keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        append: bool (Default: False)
            Set to False to overwrite the current list.
        update: bool (Default: True)
            Set to True to search for the readnoise after the new list
            is provided.

        """

        if isinstance(keyword_list, str):
            keyword_list = [keyword_list]

        elif isinstance(keyword_list, list):
            pass

        elif isinstance(keyword_list, np.ndarray):
            keyword_list = list(keyword_list)

        else:
            self.logger.error(
                "Please provide the keyword list in str, list or numpy.ndarray."
            )

        if append:
            self.seeing_keyword += keyword_list
            self.logger.info(
                "%s is appended to the seeing_keyword list.", keyword_list
            )

        else:
            self.seeing_keyword = keyword_list
            self.logger.info(
                "%s is used as the seeing_keyword list.", keyword_list
            )

        if update:
            self.set_seeing()

        else:
            self.logger.info(
                "seeing_keyword list is updated, but it is "
                "opted not to update the seeing automatically."
            )

    def set_exptime_keyword(
        self, keyword_list: list, append: bool = False, update: bool = True
    ):
        """
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        append: bool (Default: False)
            Set to False to overwrite the current list.
        update: bool (Default: True)
            Set to True to search for the readnoise after the new list
            is provided.

        """

        if isinstance(keyword_list, str):
            keyword_list = [keyword_list]

        elif isinstance(keyword_list, list):
            pass

        elif isinstance(keyword_list, np.ndarray):
            keyword_list = list(keyword_list)

        else:
            self.logger.error(
                "Please provide the keyword list in str, list or numpy.ndarray."
            )

        if append:
            self.exptime_keyword += keyword_list
            self.logger.info(
                "%s is appended to the exptime_keyword list.", keyword_list
            )

        else:
            self.exptime_keyword = keyword_list
            self.logger.info(
                "%s is used as the exptime_keyword list.", keyword_list
            )

        if update:
            self.set_exptime()

        else:
            self.logger.info(
                "exptime_keyword list is updated, but it is "
                "opted not to update the exptime automatically."
            )

    def set_airmass_keyword(
        self, keyword_list: list, append: bool = False, update: bool = True
    ):
        """
        Set the airmass keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        append: bool (Default: False)
            Set to False to overwrite the current list.
        update: bool (Default: True)
            Set to True to search for the readnoise after the new list
            is provided.

        """

        if isinstance(keyword_list, str):
            keyword_list = [keyword_list]

        elif isinstance(keyword_list, list):
            pass

        elif isinstance(keyword_list, np.ndarray):
            keyword_list = list(keyword_list)

        else:
            self.logger.error(
                "Please provide the keyword list in str, list or numpy.ndarray."
            )

        if append:
            self.airmass_keyword += keyword_list
            self.logger.info(
                "%s is appended to the airmass_keyword list.", keyword_list
            )

        else:
            self.airmass_keyword = keyword_list
            self.logger.info(
                "%s is used as the airmass_keyword list.", keyword_list
            )

        if update:
            self.set_airmass()

        else:
            self.logger.info(
                "airmass_keyword list is updated, but it is "
                "opted not to update the airmass automatically."
            )

    def set_header(self, header: fits.Header):
        """
        Set/replace the header.

        Parameters
        ----------
        header: astropy.io.fits.header.Header
            FITS header from a single HDU.

        """

        if header is not None:
            # If it is a fits.hdu.header.Header object
            if isinstance(header, fits.header.Header):
                self.header = header

            elif isinstance(header[0], fits.header.Header):
                self.header = header[0]

            else:
                error_msg = (
                    "Please provide an "
                    + "astropy.io.fits.header.Header object."
                )
                self.logger.critical(error_msg)
                raise TypeError(error_msg)

        else:
            self.logger.info('The "header" provided is None. Doing nothing.')

        if self.exptime_is_default_value:
            self.set_exptime()

        if self.airmass_is_default_value:
            self.set_airmass()

        if self.seeing_is_default_value:
            self.set_seeing()

        if self.readnoise_is_default_value:
            self.set_readnoise()

        if self.gain_is_default_value:
            self.set_gain()

    def _build_line_spread_function(
        self, img, trace, trace_width=10.0, resample_factor=5.0, bounds=None
    ):
        # img_tmp is already upsampled
        line_spread_profile_upsampled = build_line_spread_profile(
            spectrum2D=ndimage.zoom(img, resample_factor),
            trace=ndimage.zoom(trace, resample_factor) * resample_factor,
            trace_width=trace_width * resample_factor,
        )
        self.logger.info(
            "The upsampled empirical line spread profile:"
            f" {line_spread_profile_upsampled}"
        )

        line_spread_profile = ndimage.zoom(
            line_spread_profile_upsampled, 1.0 / resample_factor
        )
        line_spread_profile -= np.nanmin(line_spread_profile)
        line_spread_profile /= np.nansum(line_spread_profile)
        self.logger.info(
            f"The empirical line spread profile: {line_spread_profile}"
        )

        fitted_profile_func = get_line_spread_function(
            trace=trace, line_spread_profile=line_spread_profile, bounds=bounds
        )

        return (
            line_spread_profile_upsampled,
            line_spread_profile,
            fitted_profile_func,
        )

    def ap_trace(
        self,
        nspec: int = 1,
        smooth: bool = False,
        nwindow: int = 20,
        spec_sep: int = 5,
        trace_width: int = 15,
        bounds: dict = None,
        resample_factor: int = 4,
        rescale: bool = False,
        scaling_min: float = 0.9995,
        scaling_max: float = 1.0005,
        scaling_step: float = 0.001,
        percentile: float = 5.0,
        shift_tol: int = 15,
        fit_deg: int = 3,
        ap_faint: float = 20.0,
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
        Aperture tracing by first using cross-correlation then the peaks are
        fitting with a polynomial with an order of floor(nwindow, 10) with a
        minimum order of 1. Nothing is returned unless return_jsonstring of the
        plotly graph is set to be returned.

        Each spectral slice is convolved with the adjacent one in the spectral
        direction. Basic tests show that the geometrical distortion from one
        end to the other in the dispersion direction is small. With LT/SPRAT, the
        linear distortion is less than 0.5%, thus, even provided as an option,
        the rescale option is set to False by default. Given how unlikely a
        geometrical distortion correction is needed, higher order correction
        options are not provided.

        A rough estimation on the background level is done by taking the
        n-th percentile of the slice, a rough guess can improve the
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
        smooth: bool (Default: False)
            Set to true to apply a 3x3 median filter before tracing. Not
            recommended for use with faint spectrum.
        nwindow: int
            Number of spectral slices (subspectra) to be produced for
            cross-correlation.
        spec_sep: int
            Minimum separation between spectra (only if there are multiple
            sources on the longslit).
        trace_width: int
            Distance from trace centre to be taken for profile fitting.
        bounds: dict
            Limits of the gaussian function: 'amplitude', 'mean' and 'stddev'.
            e.g. {'amplitude': [0.0, 100.0]}
        resample_factor: int
            Number of times the collapsed 1D slices in the spatial directions
            are to be upsampled.
        rescale: bool
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
        shift_tol: float
            Maximum allowed shift between neighbouring slices, this value is
            referring to native pixel size without the application of the
            resampling or rescaling. [pix]
        fit_deg: int
            Degree of the polynomial fit of the trace.
        ap_faint: float
            The percentile tolerance of Count aperture to be used for
            fitting the trace. Note that this percentile is of the Count,
            not of the number of subspectra.
        display: bool (Default: False)
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
        filename: str (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON-string if return_jsonstring is True, otherwise only an image is
        displayed

        """

        # Get the shape of the 2D spectrum and define upsampling ratio
        img_tmp = self.img.astype(float)
        img_tmp[np.isnan(img_tmp)] = 0.0
        img_tmp[
            img_tmp < np.nanpercentile(img_tmp, percentile)
        ] = np.nanpercentile(img_tmp, percentile)

        if smooth:
            img_tmp = signal.medfilt2d(img_tmp, kernel_size=3)

        self.resample_factor = resample_factor
        nresample = self.spatial_size * self.resample_factor
        img_tmp = ndimage.zoom(img_tmp, zoom=self.resample_factor)

        # split the spectrum into subspectra
        img_split = np.array_split(img_tmp, nwindow, axis=1)
        self.start_window_idx = nwindow // 2

        lines_ref_init = np.nanmedian(img_split[self.start_window_idx], axis=1)
        lines_ref_init[np.isnan(lines_ref_init)] = 0.0
        lines_ref_init -= np.nanmin(lines_ref_init)

        # linear scaling limits
        if rescale:
            scaling_range = np.arange(scaling_min, scaling_max, scaling_step)

        else:
            scaling_range = np.ones(1)

        # subtract the sky background level
        lines_ref = lines_ref_init - np.nanpercentile(
            lines_ref_init, percentile
        )

        shift_solution = np.zeros(nwindow)
        scale_solution = np.ones(nwindow)

        # maximum shift (SEMI-AMPLITUDE) from the neighbour (pixel)
        shift_tol_len = int(shift_tol * self.resample_factor)

        # line_spread_profile is the empirically measured profile
        # line_spread_function is the fitted profile
        spatial_profile = np.zeros((nwindow, nresample))

        pix = np.arange(nresample)

        # Scipy correlate method, ignore first and last window
        for i in chain(
            range(self.start_window_idx, nwindow),
            range(self.start_window_idx - 1, -1, -1),
        ):
            self.logger.info("Correlating the %s-th window.", i)

            # smooth by taking the median
            lines = np.nanmedian(img_split[i], axis=1)
            lines[np.isnan(lines)] = 0.0
            lines = signal.resample(lines, nresample)
            lines = lines - np.nanpercentile(lines, percentile)

            # cross-correlation values and indices
            corr_val = np.zeros(len(scaling_range))
            corr_idx = np.zeros(len(scaling_range))

            # upsample by the same amount as the reference
            for j, scale in enumerate(scaling_range):
                if scale == 1.0:
                    lines_ref_j = lines_ref

                else:
                    # Upsampling the reference lines
                    lines_ref_j = spectres(
                        np.arange(int(nresample * scale)) / scale,
                        np.arange(len(lines_ref)),
                        lines_ref,
                        fill=self.img_1_percentile,
                        verbose=False,
                    )

                # find the linear shift
                corr = signal.correlate(lines_ref_j, lines)

                # only consider the defined range of shift tolerance
                corr = corr[
                    nresample - 1 - shift_tol_len : nresample + shift_tol_len
                ]

                # Maximum corr position is the shift
                corr_val[j] = np.nanmax(corr)
                corr_idx[j] = np.nanargmax(corr) - shift_tol_len

            # Maximum corr_val position is the scaling
            shift_solution[i] = corr_idx[np.nanargmax(corr_val)]
            scale_solution[i] = scaling_range[np.nanargmax(corr_val)]

            # Align the spatial profile before stacking
            if i == (self.start_window_idx - 1):
                pix = np.arange(nresample)

            pix = pix * scale_solution[i] + shift_solution[i]

            spatial_profile_tmp = spectres(
                np.arange(nresample),
                np.array(pix).reshape(-1),
                np.array(lines).reshape(-1),
                fill=self.img_1_percentile,
                verbose=False,
            )
            spatial_profile_tmp[np.isnan(spatial_profile_tmp)] = np.nanmin(
                spatial_profile_tmp
            )
            spatial_profile[i] = copy.deepcopy(spatial_profile_tmp)

            # Update (increment) the reference line
            if i == nwindow - 1:
                lines_ref = lines_ref_init

            else:
                lines_ref = lines

        spatial_profile = np.nanmedian(spatial_profile, axis=0)
        nscaled = (nresample * scale_solution).astype("int")

        # Find the spectral position in the middle of the gram in the upsampled
        # pixel location location
        # FWHM cannot be smaller than 3 pixels for any real signal
        peaks = signal.find_peaks(
            spatial_profile, distance=spec_sep, width=3.0
        )

        # update the number of spectra if the number of peaks detected is less
        # than the number requested
        self.nspec_traced = min(len(peaks[0]), nspec)
        self.logger.info("%s spectra are identified.", self.nspec_traced)

        # Sort the positions by the prominences, and return to the original
        # scale (i.e. with subpixel position)
        spec_init = (
            np.sort(
                peaks[0][np.argsort(-peaks[1]["prominences"])][
                    : self.nspec_traced
                ]
                - self.resample_factor // 2
            )
            / self.resample_factor
        )

        # Create array to populate the spectral locations
        self.spec_idx = np.zeros((len(spec_init), len(img_split)))

        # Populate the initial values
        self.spec_idx[:, self.start_window_idx] = spec_init

        # Pixel positions of the mid point of each data_split (spectral)
        self.spec_pix = [len(i[0]) for i in img_split]
        self.spec_pix[0] -= self.spec_pix[0] // 2
        for i in range(1, len(self.spec_pix)):
            self.spec_pix[i] += self.spec_pix[i - 1]

        self.spec_pix = np.round(self.spec_pix).astype("int")

        # Looping through pixels larger than middle pixel
        for i in range(self.start_window_idx + 1, nwindow):
            self.spec_idx[:, i] = (
                self.spec_idx[:, i - 1]
                * self.resample_factor
                * nscaled[i]
                / nresample
                - shift_solution[i]
            ) / self.resample_factor

        # Looping through pixels smaller than middle pixel
        for i in range(self.start_window_idx - 1, -1, -1):
            self.spec_idx[:, i] = (
                (
                    self.spec_idx[:, i + 1] * self.resample_factor
                    - shift_solution[i]
                )
                / (np.round(nresample * scale_solution[i + 1]) / nresample)
                / self.resample_factor
            )

        for i, spec_i in enumerate(self.spec_idx):
            # Get the median of the subspectrum and then get the Count at the
            # central 5 pixels of the aperture
            ap_val = np.zeros(nwindow)

            for j in range(nwindow):
                # rounding
                idx = int(np.round(spec_i[j])) * resample_factor
                subspec_cleaned = sigma_clip(
                    img_split[j], sigma=3, masked=True
                ).data
                ap_val[j] = np.nansum(
                    np.nansum(subspec_cleaned, axis=1)[idx - 3 : idx + 3]
                ) / 7 - np.nanmedian(subspec_cleaned)

            # Mask out the faintest ap_faint percent of trace
            n_faint = int(np.round(len(ap_val) * ap_faint / 100))
            mask = np.argsort(ap_val)[n_faint:]
            self.logger.info(
                (
                    "The faintest %s subspectra are going to be ignored in the"
                    " tracing. They are %s."
                ),
                n_faint,
                np.argsort(ap_val)[:n_faint],
            )

            # fit the trace
            aper_p = np.polyfit(
                self.spec_pix[mask], spec_i[mask], int(fit_deg)
            )
            aper = np.polyval(
                aper_p, np.arange(self.spec_size) * self.resample_factor
            )
            self.logger.info("Only logging 10 evenly-spaced points.")
            self.logger.info(
                "The trace is found at %s.",
                [
                    (x, y)
                    for (x, y) in zip(
                        np.arange(self.spec_size)[:: self.spec_size // 10],
                        aper[:: self.spec_size // 10],
                    )
                ],
            )

            (
                line_spread_profile_upsampled,
                line_spread_profile,
                fitted_profile_func,
            ) = self._build_line_spread_function(
                img=self.img,
                trace=aper,
                trace_width=10.0,
                resample_factor=self.resample_factor,
                bounds=bounds,
            )

            ap_sigma = fitted_profile_func.stddev_0.value
            fitted_profile_func.slope_1 = 0.0
            fitted_profile_func.intercept_1 = 0.0

            self.logger.info(
                "Aperture is fitted with a Gaussian sigma of %s pix.",
                ap_sigma,
            )

            self.spectrum_list[i] = SpectrumOneD(
                spec_id=i,
                verbose=self.verbose,
                logger_name=self.logger_name,
                log_level=self.log_level,
                log_file_folder=self.log_file_folder,
                log_file_name=self.log_file_name,
            )
            self.spectrum_list[i].add_trace(list(aper), [ap_sigma] * len(aper))
            self.spectrum_list[i].add_line_spread_profile_upsampled(
                line_spread_profile_upsampled
            )
            self.spectrum_list[i].add_line_spread_profile(line_spread_profile)
            self.spectrum_list[i].add_profile_func(fitted_profile_func)
            self.spectrum_list[i].add_gain(self.gain)
            self.spectrum_list[i].add_readnoise(self.readnoise)
            self.spectrum_list[i].add_exptime(self.exptime)
            self.spectrum_list[i].add_seeing(self.seeing)
            self.spectrum_list[i].add_airmass(self.airmass)

        # Plot
        if save_fig or display or return_jsonstring:
            to_return = self.inspect_trace(
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

    def inspect_trace(
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
        Display the trace(s) over the image.

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
            set to True to return json str that can be rendered by Plotly
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
        JSON-string if return_jsonstring is True, otherwise only an image is
        displayed

        """

        fig = go.Figure(
            layout=dict(autosize=False, height=height, width=width)
        )

        fig.add_trace(
            go.Heatmap(
                z=np.log10(self.img),
                zmin=self.zmin,
                zmax=self.zmax,
                colorscale="Viridis",
                colorbar=dict(title="log( e- count )"),
            )
        )

        for i, spec_i in enumerate(self.spec_idx):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(self.spec_size),
                    y=self.spectrum_list[i].trace,
                    line=dict(color="black"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.spec_pix / self.resample_factor,
                    y=spec_i,
                    mode="markers",
                    marker=dict(color="grey"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(self.spec_size),
                    y=self.spectrum_list[i].trace,
                    line=dict(color="black"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.spec_pix / self.resample_factor,
                    y=spec_i,
                    mode="markers",
                    marker=dict(color="grey"),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=np.ones(len(self.spec_idx))
                * self.spec_pix[self.start_window_idx]
                / self.resample_factor,
                y=self.spec_idx[:, self.start_window_idx],
                mode="markers",
                marker=dict(color="firebrick"),
            )
        )
        fig.update_layout(
            yaxis_title="Spatial Direction / pixel",
            xaxis=dict(
                zeroline=False,
                showgrid=False,
                title="Dispersion Direction / pixel",
            ),
            bargap=0,
            hovermode="closest",
            showlegend=False,
        )

        if filename is None:
            filename = "ap_trace"

        if save_fig:
            fig_type_split = fig_type.split("+")

            for f_type in fig_type_split:
                if f_type == "iframe":
                    pio.write_html(
                        fig, filename + "." + f_type, auto_open=open_iframe
                    )

                elif f_type in ["jpg", "png", "svg", "pdf"]:
                    pio.write_image(fig, filename + "." + f_type)

                self.logger.info(
                    "Figure is saved to %s", filename + "." + f_type
                )

        if display:
            if renderer == "default":
                fig.show()

            else:
                fig.show(renderer)

        if return_jsonstring:
            return fig.to_json()

    def add_trace(
        self,
        trace: Union[list, np.ndarray],
        trace_sigma: Union[list, np.ndarray],
        spec_id: Union[int, list, np.ndarray] = None,
    ):
        """
        Add user-supplied trace. The trace has to have the size as the 2D
        spectral image in the dispersion direction.

        Parameters
        ----------
        trace: list or numpy.ndarray (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: list or numpy.ndarray (N)
            Standard deviation of the Gaussian profile of a trace
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object

        """

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        assert isinstance(spec_id, (int, list, np.ndarray)) or (
            spec_id is None
        ), "spec_id has to be an integer, None, list or array."

        if spec_id is None:
            if len(np.shape(trace)) == 1:
                spec_id = [0]

            elif len(np.shape(trace)) == 2:
                spec_id = list(np.arange(np.shape(trace)[0]))

        if isinstance(spec_id, np.ndarray):
            spec_id = list(spec_id)

        assert isinstance(
            trace, (list, np.ndarray)
        ), "trace has to be a list or an array."
        assert isinstance(
            trace_sigma, (list, np.ndarray)
        ), "trace_sigma has to be a list or an array."
        assert len(trace) == len(
            trace_sigma
        ), "trace and trace_sigma have to be the same size."

        for i in spec_id:
            if i in self.spectrum_list.keys():
                self.spectrum_list[i].add_trace(trace, trace_sigma)

            else:
                self.spectrum_list[i] = SpectrumOneD(
                    spec_id=i,
                    verbose=self.verbose,
                    logger_name=self.logger_name,
                    log_level=self.log_level,
                    log_file_folder=self.log_file_folder,
                    log_file_name=self.log_file_name,
                )
                self.spectrum_list[i].add_trace(trace, trace_sigma)

    def remove_trace(self, spec_id: int = None):
        """
        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum_oned object

        """

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if spec_id is not None:
            assert np.in1d(
                spec_id, list(self.spectrum_list.keys())
            ).all(), "Some spec_id provided are not in the spectrum_list."

        else:
            spec_id = list(self.spectrum_list.keys())

        for i in spec_id:
            self.spectrum_list[i].remove_trace()

    def get_rectification(
        self,
        upsample_factor: int = 1,
        bin_size: int = 6,
        n_bin: int = 15,
        spline_order: int = 3,
        order: int = 2,
        coeff: list = None,
        use_arc: bool = True,
        apply: bool = False,
        display: bool = False,
        renderer: str = "default",
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: str = False,
    ):
        """
        ONLY possible if there is ONE trace. If more than one trace is
        provided, only the first one (i.e. spec_id = 0) will get
        processed.

        The retification in the spatial direction depends on ONLY the
        trace, while that in the dispersion direction depends on the
        parameters provided here.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        upsample_factor: float (Default: 1)
            The upsampling rate for the correlation (10 times means
            precise to 1/10 of a pixel). The upsampling uses cubic
            spline that is adopted in the scipy.ndimage.zoom() function.
        bin_size: int (Default: 6)
            Number of rows in a slice.
        n_bin: int (Default: 10)
            Number of slices parallel to the trace to be correlated to to
            compute the distortion in the dispersion direction. (i.e.
            there are 10 // 2 = 5 slices below and above the trace.)
        spline_order: int (Default: 3)
            The order of spline for resampling.
        order: int (Default: 2)
            The order of polynomial to fit for the distortion in the
            dispersion direction
        coeff: list or numpy.ndarray (Default: None)
            The polynomial coefficients for aligned the dispersion direction
            as a function of distance from the trace.
        apply: bool (Default: False)
            Apply the rectification directly without checking.
        display: bool (Default: False)
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
        filename: str (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.

        """

        spec = self.spectrum_list[0]

        img_tmp = self.img.astype(float)

        if self.arc is None:
            self.logger.warning(
                "Arc frame is not available, only the data image "
                "will be rectified."
            )
            use_arc = False

        elif isinstance(self.arc, CCDData):
            arc_tmp = self.arc.data.astype(float)

        else:
            arc_tmp = self.arc.astype(float)

        y_tmp = np.array(spec.trace).copy()

        # Shift the spectrum to spatially aligned to the trace at ref
        ref = y_tmp[len(y_tmp) // 2]

        # The x/y-coordinates of the trace
        pix_x = np.arange(self.spec_size)
        pix_y = np.arange(self.spatial_size).astype("float")

        # This shifts the spectrum in the sptial direction using the trace
        for i in range(spec.len_trace):
            shift_i = y_tmp[i] - ref
            # resample here
            img_tmp[:, i] = spectres(
                pix_y + shift_i,
                pix_y,
                img_tmp[:, i],
                fill=self.img_1_percentile,
                verbose=False,
            )

            if self.arc is not None:
                arc_tmp[:, i] = spectres(
                    pix_y + shift_i,
                    pix_y,
                    arc_tmp[:, i],
                    fill=self.arc_1_percentile,
                    verbose=False,
                )

        # Now start working with the shift in the dispersion direction
        if coeff is not None:
            n_down = None
            n_up = None

            self.logger.info(
                (
                    "Polynomial coefficients for rectifying in the spatial "
                    "direction is given as: %s."
                ),
                coeff,
            )

        else:
            if isinstance(n_bin, (int, float)):
                n_down = int(n_bin // 2)
                n_up = int(n_bin // 2)

            elif isinstance(n_bin, (list, np.ndarray)):
                n_down = n_bin[0]
                n_up = n_bin[1]

            else:
                self.logger.error(
                    (
                        "The given n_bin is not numeric or a list/array of "
                        "size 2: %s. Using the default value to proceed."
                    ),
                    n_bin,
                )
                n_down = 5
                n_up = 5

            bin_half_size = bin_size / 2

            # s for "flattened signal of the slice"
            if use_arc:
                s = np.nanmedian(
                    [
                        img_tmp[
                            int(np.round(ref - bin_half_size)) : int(
                                np.round(ref + bin_half_size)
                            ),
                            i,
                        ]
                        + arc_tmp[
                            int(np.round(ref - bin_half_size)) : int(
                                np.round(ref + bin_half_size)
                            ),
                            i,
                        ]
                        for i in pix_x
                    ],
                    axis=1,
                )

            else:
                s = np.nanmedian(
                    [
                        img_tmp[
                            int(np.round(ref - bin_half_size)) : int(
                                np.round(ref + bin_half_size)
                            ),
                            i,
                        ]
                        for i in pix_x
                    ],
                    axis=1,
                )

            # Get the length of 10% of the dispersion direction
            # Do not use the first and last 10% for cross-correlation
            one_tenth = len(s) // 10

            s -= np.nanpercentile(s, 5.0)
            s -= min(s[one_tenth:-one_tenth])
            s /= max(s[one_tenth:-one_tenth])
            s_down = []
            s_up = []
            y_down = []
            y_up = []

            # Loop through the spectra below the trace
            for k in range(n_down):
                start = k * bin_half_size
                end = start + bin_size
                y_down.append(ref - (start + end) / 2.0)
                # Note the start and end are counting up, while the
                # indices are becoming smaller.
                if use_arc:
                    s_down.append(
                        np.nanmedian(
                            [
                                arc_tmp[
                                    int(np.round(ref - end)) : int(
                                        np.round(ref - start)
                                    ),
                                    i,
                                ]
                                + img_tmp[
                                    int(np.round(ref - end)) : int(
                                        np.round(ref - start)
                                    ),
                                    i,
                                ]
                                for i in pix_x
                            ],
                            axis=1,
                        )
                    )

                else:
                    s_down.append(
                        np.nanmedian(
                            [
                                img_tmp[
                                    int(np.round(ref - end)) : int(
                                        np.round(ref - start)
                                    ),
                                    i,
                                ]
                                for i in pix_x
                            ],
                            axis=1,
                        )
                    )

                s_down[k] -= np.nanpercentile(s_down[k], 5.0)
                s_down[k] -= min(s_down[k][one_tenth:-one_tenth])
                s_down[k] /= max(s_down[k][one_tenth:-one_tenth])

            # Loop through the spectra above the trace
            for k in range(n_up):
                start = k * bin_half_size
                end = start + bin_size
                y_up.append(ref + (start + end) / 2.0)
                if use_arc:
                    s_up.append(
                        np.nanmedian(
                            [
                                arc_tmp[
                                    int(np.round(ref + start)) : int(
                                        np.round(ref + end)
                                    ),
                                    i,
                                ]
                                + img_tmp[
                                    int(np.round(ref + start)) : int(
                                        np.round(ref + end)
                                    ),
                                    i,
                                ]
                                for i in pix_x
                            ],
                            axis=1,
                        )
                    )

                else:
                    s_up.append(
                        np.nanmedian(
                            [
                                img_tmp[
                                    int(np.round(ref + start)) : int(
                                        np.round(ref + end)
                                    ),
                                    i,
                                ]
                                for i in pix_x
                            ],
                            axis=1,
                        )
                    )
                s_up[k] -= np.nanpercentile(s_up[k], 5.0)
                s_up[k] -= min(s_up[k][one_tenth:-one_tenth])
                s_up[k] /= max(s_up[k][one_tenth:-one_tenth])

            s_all = s_down[::-1] + [s] + s_up
            y_all = y_down[::-1] + [ref] + y_up

            self.logger.info(
                "%s subspectra are used for cross-correlation.", len(s_all)
            )

            # correlate with the neighbouring slice to compute the shifts
            shift = np.zeros_like(y_all[:-1]).astype("float")

            for i in range(len(s_all) - 1):
                # Note: indice n_down is s
                corr = signal.correlate(
                    ndimage.zoom(
                        s_all[i][one_tenth:-one_tenth], upsample_factor
                    ),
                    ndimage.zoom(
                        s_all[i + 1][one_tenth:-one_tenth], upsample_factor
                    ),
                    mode="same",
                )
                shift[i:] += (
                    np.argmax(corr) / upsample_factor
                    - (self.spec_size - 2 * one_tenth) * 0.5
                )

            # Turn the shift to relative to the spectrum
            shift -= shift[n_down]

            self.logger.info(
                (
                    "The y-coordinates of subspectra are: %s and the "
                    "corresponding shifts are: %s."
                ),
                y_all,
                shift,
            )

            # fit the distortion in the dispersion direction as a function
            # of y-pixel.
            coeff = np.polynomial.polynomial.polyfit(
                (np.array(y_all)[:-1] + np.array(y_all)[1:]) / 2.0,
                shift,
                order,
            )
            self.logger.info(
                (
                    "Best fit polynomial for rectifying in the spatial"
                    " direction.is %s."
                ),
                coeff,
            )

        # shift in the dispersion direction, the shift is as a function
        # of distance from the trace at ref
        # For each row j (sort of a line of spectrum...)
        shifts = np.polynomial.polynomial.polyval(
            np.arange(self.spatial_size), coeff
        )

        pix_x = pix_x.astype("float")
        for j, shift_j in enumerate(shifts):
            shift_j = np.polynomial.polynomial.polyval(j, coeff)

            if j % 10 == 0:
                self.logger.info("The shift at line j = %s is %s.", j, shift_j)

            img_tmp[j] = spectres(
                pix_x - shift_j,
                pix_x,
                img_tmp[j],
                fill=self.img_1_percentile,
                verbose=False,
            )

            if self.arc is not None:
                arc_tmp[j] = spectres(
                    pix_x - shift_j,
                    pix_x,
                    arc_tmp[j],
                    fill=self.arc_1_percentile,
                    verbose=False,
                )

        self.rec_coeff = coeff
        self.rec_n_down = n_down
        self.rec_n_up = n_up
        self.rec_upsample_factor = upsample_factor
        self.rec_bin_size = bin_size
        self.rec_n_bin = n_bin
        self.rec_spline_order = spline_order
        self.rec_order = order
        self.img_rectified = img_tmp
        self.img_residual_rectified = copy.deepcopy(self.img_rectified)

        if self.arc is not None:
            self.arc_rectified = arc_tmp

        if apply:
            self.apply_rectification()

        if save_fig or display or return_jsonstring:
            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width)
            )

            # show the image on the top
            # the 3 is the show a little bit outside the extraction regions
            fig.add_trace(
                go.Heatmap(
                    z=np.log10(self.img_rectified),
                    colorscale="Viridis",
                    zmin=np.nanpercentile(np.log10(self.img_rectified), 5),
                    zmax=np.nanpercentile(np.log10(self.img_rectified), 95),
                    xaxis="x",
                    yaxis="y",
                    colorbar=dict(title="log( e- count )"),
                )
            )
            if self.arc_rectified is not None:
                fig.add_trace(
                    go.Heatmap(
                        z=np.log10(self.arc_rectified),
                        colorscale="Viridis",
                        zmin=np.nanpercentile(np.log10(self.arc_rectified), 5),
                        zmax=np.nanpercentile(
                            np.log10(self.arc_rectified), 95
                        ),
                        xaxis="x2",
                        yaxis="y2",
                    )
                )

                # Decorative stuff
                fig.update_layout(
                    yaxis=dict(
                        zeroline=False,
                        domain=[0.5, 1],
                        showgrid=False,
                        title="Spatial Direction / pixel",
                    ),
                    yaxis2=dict(
                        zeroline=False,
                        domain=[0, 0.5],
                        showgrid=False,
                        title="Spatial Direction / pixel",
                    ),
                    xaxis=dict(showticklabels=False),
                    xaxis2=dict(
                        title="Dispersion Direction / pixel",
                        anchor="y2",
                        matches="x",
                    ),
                    bargap=0,
                    hovermode="closest",
                )

            if filename is None:
                filename = "rectified_image"

            if save_fig:
                fig_type_split = fig_type.split("+")

                for f_type in fig_type_split:
                    if f_type == "iframe":
                        pio.write_html(
                            fig, filename + "." + f_type, auto_open=open_iframe
                        )

                    elif f_type in ["jpg", "png", "svg", "pdf"]:
                        pio.write_image(fig, filename + "." + f_type)

                    self.logger.info(
                        "Figure is saved as %s.", filename + "." + f_type
                    )

            if display:
                if renderer == "default":
                    fig.show()

                else:
                    fig.show(renderer)

        if return_jsonstring:
            return fig.to_json()

    def apply_rectification(self):
        """
        Accept the dispersion rectification computed.

        """

        if self.img_rectified is not None:
            self.img = self.img_rectified
            self.img_residual = self.img_residual_rectified
            self.logger.info("Image rectification is applied")

        else:
            self.logger.info(
                "Rectification is not computed, it cannot be "
                "applied to the image."
            )

        if self.arc_rectified is not None:
            self.arc = self.arc_rectified
            self.logger.info("Arc rectification is applied")

        else:
            self.logger.info(
                "Rectification is not computed, it cannot be "
                "applied to the arc."
            )

    def _fit_sky(
        self,
        extraction_slice: np.ndarray,
        extraction_bad_mask: np.ndarray,
        sky_sigma: float,
        sky_width_dn: int,
        sky_width_up: int,
        sky_polyfit_order: int,
    ):
        """
        Fit the sky background from the given extraction_slice and the aperture
        parameters.

        Parameters
        ----------
        extraction_slice: 1D numpy.ndarray
            The counts along the profile for extraction, including the sky
            regions to be fitted and subtracted from.
        extraction_bad_mask: 1D numpy.ndarray
            The mask of the bad pixels. They should be marked as 1 or True.
        sky_sigma: float
            Number of sigma to be clipped.
        sky_width_dn: int
            Number of pixels used for sky modelling on the lower side of the
            spectrum.
        sky_width_up: int
            Number of pixels used for sky modelling on the upper side of the
            spectrum.
        sky_polyfit_order: int
            The order of polynomial in fitting the sky background.

        Returns
        -------
        count_sky_extraction_slice: numpy.ndarray
            The sky count in each pixel of the extraction_slice.

        """

        if (sky_width_dn > 0) or (sky_width_up > 0):
            # get the sky region(s)
            sky_mask = np.zeros_like(extraction_slice, dtype=bool)
            sky_mask[0:sky_width_up] = True
            sky_mask[-(sky_width_dn + 1) : -1] = True

            sky_mask *= ~extraction_bad_mask
            sky_bad_mask = ~sigma_clip(
                extraction_slice[sky_mask], sigma=sky_sigma
            ).mask

            if sky_polyfit_order == 0:
                count_sky_extraction_slice = np.ones(
                    len(extraction_slice[sky_mask][sky_bad_mask])
                ) * np.nanmean(extraction_slice[sky_mask][sky_bad_mask])

            elif sky_polyfit_order > 0:
                # fit a polynomial to the sky in this column
                polyfit_coeff = np.polynomial.polynomial.polyfit(
                    np.arange(extraction_slice.size)[sky_mask][sky_bad_mask],
                    extraction_slice[sky_mask][sky_bad_mask],
                    sky_polyfit_order,
                )

                # evaluate the polynomial across the extraction_slice, and sum
                count_sky_extraction_slice = np.polynomial.polynomial.polyval(
                    np.arange(extraction_slice.size), polyfit_coeff
                )

            else:
                count_sky_extraction_slice = np.zeros_like(extraction_slice)

        else:
            # get the indexes of the sky regions
            count_sky_extraction_slice = np.zeros_like(extraction_slice)

        return count_sky_extraction_slice

    def ap_extract(
        self,
        apwidth: Union[int, list, np.ndarray] = 9,
        skysep: Union[int, list, np.ndarray] = 3,
        skywidth: Union[int, list, np.ndarray] = 5,
        skydeg: int = 1,
        sky_sigma: float = 3.0,
        optimal: bool = True,
        algorithm: str = "horne86",
        model: str = "gauss",
        bounds: dict = None,
        lowess_frac: float = 0.1,
        lowess_it: int = 3,
        lowess_delta: float = 0.0,
        tolerance: float = 1e-6,
        cosmicray_sigma: float = 4.5,
        max_iter: int = 99,
        forced: bool = False,
        variances: Union[list, np.ndarray] = None,
        npoly: int = 21,
        polyspacing: int = 1,
        pord: int = 5,
        qmode: str = "fast-linear",
        nreject: int = 100,
        display: bool = False,
        renderer: str = "default",
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
    ):
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
                            function along the entire spectrum. (Choose from
                            horne86 and marsh89 algorithm)

        Nothing is returned unless return_jsonstring of the plotly graph is
        set to be returned. The count, count_sky and count_err are stored as
        properties of the TwoDSpec object.

        count: 1-d array
            The summed count at each column about the trace.
        count_err: 1-d array
            the uncertainties of the count values.
        count_sky: 1-d array
            The integrated sky values along each column.

        Parameters
        ----------
        apwidth: int or list of int (Default: 7)
            Half the size of the aperature (fixed value for tophat extraction).
            If a list of two ints are provided, the first element is the
            lower half of the aperture and the second one is the upper half.
        skysep: int or list of int (Default: 3)
            The separation in pixels from the aperture to the sky window.
        skywidth: int or list of int (Default: 5)
            The width in pixels of the sky windows on either side of the
            aperture. Zero (0) means ignore sky subtraction.
        skydeg: int (Default: 1)
            The polynomial order to fit between the sky windows.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        optimal: bool (Default: True)
            Set optimal extraction. (Default is True)
        algorithm: str (Default: 'horne86')
            Available algorithms are horne86 and marsh89
        model: str (Default: 'lowess')
            Choice of model: 'lowess' and 'gauss'.
        bounds: dict
            Limits of the gaussian function: 'amplitude', 'mean' and 'stddev'.
            e.g. {'amplitude': [0.0, 100.0]}
            This is only used if the trace was provided manually.
        lowess_frac: float (Default: 0.1)
            Fraction of "good data" retained for LOWESS fit.
        lowess_it: int (Default: 3)
            Number of iteration in LOWESS fit -- the number of residual-based
            reweightings to perform.
        lowess_delta: float (Default: 0.0)
            The delta parameter in LOWESS fit -- distance within which to use
            linear-interpolation instead of weighted regression.
        tolerance: float (Default: 1e-6)
            The tolerance limit for the convergence of the optimal extraction
        cosmicray_sigma: float (Deafult: 4.5)
            Cosmic ray sigma clipping limit. This is for rejecting pixels when
            using horne87 and marsh89 optimal cleaning. Use sigclip in kwargs
            for configuring cosmicray cleaning with astroscrappy.
        max_iter: float (Default: 99)
            The maximum number of iterations before optimal extraction fails
            and return to standard tophot extraction
        forced: bool (Default: False)
            To perform forced optimal extraction by using the given aperture
            profile as it is without interation, the resulting uncertainty
            will almost certainly be wrong. This is an experimental feature.
        variances: list or numpy.ndarray (Default: None, only used if algorithm
                   is horne86)
            The weight function for forced extraction. It is only used if force
            is set to True.
        npoly: int (Default: 21, only used if algorithm is marsh89)
            Number of profile to be use for polynomial fitting to evaluate
            (Marsh's "K"). For symmetry, this should be odd.
        polyspacing: float (Default: 1, only used if algorithm is marsh89)
            Spacing between profile polynomials, in pixels. (Marsh's "S").
            A few cursory tests suggests that the extraction precision
            (in the high S/N case) scales as S^-2 -- but the code slows down
            as S^2.
        pord: int (Default: 5, only used if algorithm is marsh89)
            Order of profile polynomials; 1 = linear, etc.
        qmode: str (Default: 'fast-linear', only used if algorithm is marsh89)
            How to compute Marsh's Q-matrix. Valid inputs are 'fast-linear',
            'slow-linear', 'fast-nearest', and 'slow-nearest'. These select
            between various methods of integrating the nearest-neighbor or
            linear interpolation schemes as described by Marsh; the 'linear'
            methods are preferred for accuracy. Use 'slow' if you are
            running out of memory when using the 'fast' array-based methods.
        nreject: int (Default: 100, only used if algorithm is marsh89)
            Number of outlier-pixels to reject at each iteration.
        display: bool (Default: False)
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
        filename: str (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.

        """

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if spec_id is not None:
            assert np.in1d(
                spec_id, list(self.spectrum_list.keys())
            ).all(), "Some spec_id provided are not in the spectrum_list."

        else:
            spec_id = list(self.spectrum_list.keys())

        self.cosmicray_sigma = cosmicray_sigma

        to_return = []

        if isinstance(apwidth, int):
            # first do the aperture count
            width_dn = apwidth
            width_up = apwidth

        elif len(apwidth) == 2:
            width_dn = apwidth[0]
            width_up = apwidth[1]

        else:
            self.logger.error(
                "apwidth can only be an int or a list of two ints. It is "
                "set to the default value to continue the extraction."
            )
            width_dn = 7
            width_up = 7

        if isinstance(skysep, int):
            # first do the aperture count
            sep_dn = skysep
            sep_up = skysep

        elif len(skysep) == 2:
            sep_dn = skysep[0]
            sep_up = skysep[1]

        else:
            self.logger.error(
                "skysep can only be an int or a list of two ints. It is "
                "set to the default value to continue the extraction."
            )
            sep_dn = 3
            sep_up = 3

        if isinstance(skywidth, int):
            # first do the aperture count
            sky_width_dn = skywidth
            sky_width_up = skywidth

        elif len(skywidth) == 2:
            sky_width_dn = skywidth[0]
            sky_width_up = skywidth[1]

        else:
            self.logger.error(
                "skywidth can only be an int or a list of two ints. It "
                "is set to the default value to continue the extraction."
            )
            sky_width_dn = 5
            sky_width_up = 5

        offset = 0

        for j in spec_id:
            spec = self.spectrum_list[j]
            len_trace = len(spec.trace)
            count_sky = np.zeros(len_trace)
            count_err = np.zeros(len_trace)
            count = np.zeros(len_trace)
            var = (
                np.ones((len_trace, width_dn + width_up + 1))
                * self.readnoise**2.0
            )
            profile = np.zeros((len_trace, width_dn + width_up + 1))
            is_optimal = np.zeros(len_trace, dtype=bool)

            # This should only happen if a trace is provided manually
            if optimal & (spec.profile_func is None):
                (
                    line_spread_profile_upsampled,
                    line_spread_profile,
                    fitted_profile_func,
                ) = self._build_line_spread_function(
                    img=self.img,
                    trace=spec.trace,
                    trace_width=int(width_dn + width_up),
                    resample_factor=self.resample_factor,
                    bounds=bounds,
                )
                self.logger.info(fitted_profile_func)

                fitted_profile_func.slope_1 = 0.0
                fitted_profile_func.intercept_1 = 0.0

                spec.add_line_spread_profile_upsampled(
                    line_spread_profile_upsampled
                )
                spec.add_line_spread_profile(line_spread_profile)
                spec.add_profile_func(fitted_profile_func)

            # Sky extraction
            for i, pos in enumerate(spec.trace):
                itrace = int(pos)
                pix_frac = pos - itrace

                profile_start_idx = 0

                # fix width if trace is too close to the edge
                if itrace + width_up > self.spatial_size:
                    self.logger.info(
                        (
                            "Extraction is over the upper edge of the detector "
                            "plane. Fixing indices. width_up is changed from "
                            "%s to %s."
                        ),
                        width_up,
                        self.spatial_size - itrace - 1,
                    )
                    # ending at the last pixel
                    width_up = self.spatial_size - itrace - 3
                    sep_up = 0
                    sky_width_up = 0

                profile_end_idx = width_dn + width_up + 1

                if itrace - width_dn < 0:
                    self.logger.info(
                        "Extration is over the lower edge of "
                        "the detector plane. Fixing indices."
                    )
                    offset = width_dn - itrace
                    # starting at pixel row 0
                    width_dn = itrace - 1
                    sep_dn = 0
                    sky_width_dn = 0
                    profile_start_idx = offset
                    profile_end_idx = offset + width_dn + width_up + 1

                # Pixels where the source spectrum and the sky regions are
                source_pix = np.arange(
                    itrace - width_dn, itrace + width_up + 1
                )
                extraction_pix = np.arange(
                    itrace - width_dn - sep_dn - sky_width_dn,
                    itrace + width_up + sep_up + sky_width_up + 1,
                )

                # trace +/- aperture size
                source_slice = self.img[source_pix, i].copy()
                if self.bad_mask is not None:
                    source_bad_mask = self.bad_mask[source_pix, i]
                else:
                    source_bad_mask = np.zeros_like(source_slice, dtype="bool")

                # trace +/- aperture and sky region size
                extraction_slice = self.img[extraction_pix, i].copy()
                if self.bad_mask is not None:
                    extraction_bad_mask = self.bad_mask[extraction_pix, i]
                else:
                    extraction_bad_mask = np.zeros_like(
                        extraction_slice, dtype="bool"
                    )

                extraction_bad_mask = (
                    extraction_bad_mask
                    & ~np.isfinite(extraction_slice)
                    & ~np.isnan(extraction_slice)
                )

                count_sky_extraction_slice = self._fit_sky(
                    extraction_slice,
                    extraction_bad_mask,
                    sky_sigma,
                    sky_width_dn,
                    sky_width_up,
                    skydeg,
                )

                count_sky_source_slice = count_sky_extraction_slice[
                    source_pix - itrace
                ].copy()
                var_sky = np.nanvar(extraction_slice[source_pix - itrace])

                count_sky[i] = (
                    np.nansum(count_sky_source_slice)
                    - pix_frac * count_sky_source_slice[0]
                    - (1 - pix_frac) * count_sky_source_slice[-1]
                )

                self.img_residual[
                    source_pix, i
                ] = count_sky_source_slice.copy()

                self.logger.debug(
                    "count_sky at pixel %s is %s.", i, count_sky[i]
                )

                # if not optimal extraction or using marsh89, perform a
                # tophat extraction
                if not optimal or (optimal & (algorithm == "marsh89")):
                    (
                        count[i],
                        count_err[i],
                        is_optimal[i],
                    ) = tophat_extraction(
                        source_slice=source_slice,
                        sky_source_slice=count_sky_source_slice,
                        var_sky=var_sky,
                        pix_frac=pix_frac,
                        gain=self.gain,
                        sky_width_dn=sky_width_dn,
                        sky_width_up=sky_width_up,
                        width_dn=width_dn,
                        width_up=width_up,
                    )

                # Get the optimal signals
                if optimal & (algorithm == "horne86"):
                    self.logger.debug("Using Horne 1986 algorithm.")

                    # If the weights are given externally to perform forced
                    # extraction
                    if forced:
                        self.logger.debug("Using forced extraction.")

                        # Unit weighted
                        if np.ndim(variances) == 0:
                            if isinstance(variances, (int, float)):
                                var_i = (
                                    np.ones(width_dn + width_up + 1)
                                    * variances
                                )

                            else:
                                var_i = np.ones(len(source_pix))
                                self.logger.warning("Variances are set to 1.")

                        # A single LSF is given for the entire trace
                        elif np.ndim(variances) == 1:
                            if len(variances) == len(source_pix):
                                var_i = variances

                            elif len(variances) == len_trace:
                                var_i = np.ones(len(source_pix)) * variances[i]

                            else:
                                var_i = np.ones(len(source_pix))
                                self.logger.warning("Variances are set to 1.")

                        # A two dimensional LSF
                        elif np.ndim(variances) == 2:
                            var_i = variances[i]

                            # If the spectrum is outside of the frame
                            if itrace - apwidth < 0:
                                var_i = var_i[apwidth - width_dn :]

                            # If the spectrum is outside of the frame
                            elif itrace + apwidth > self.spatial_size:
                                var_i = var_i[: -(apwidth - width_up + 1)]

                            else:
                                pass

                        else:
                            var_i = np.ones(len(source_pix))
                            self.logger.warning("Variances are set to 1.")

                    else:
                        var_i = None

                    if model == "gauss":
                        # Number of pixels from the "centre" of the trace
                        # Only used in Horne86 extraction with gauss model
                        delta_trace = spec.profile_func.mean_0 - pos
                        # .left is the gaussian mode
                        _profile_func = spec.profile_func.left
                        _profile = _profile_func(source_pix + delta_trace)
                        _profile[_profile < 0] = 0.0
                        _profile /= np.nansum(_profile)
                        # _lower = (pos - min(source_pix)) / _profile_func.stddev
                        # _upper = (
                        #    max(source_pix) - pos + 1
                        # ) / _profile_func.stddev
                        # _profile *= np.diff(norm.cdf([-_lower, _upper]))

                    elif model == "lowess":
                        _profile = lowess(
                            source_slice - count_sky_source_slice,
                            source_pix,
                            frac=lowess_frac,
                            it=lowess_it,
                            delta=lowess_delta,
                            return_sorted=False,
                        )
                        _profile[_profile < 0] = 0.0
                        _profile /= np.nansum(_profile)

                    else:
                        self.logger.error(
                            "The provided model has to be gauss or lowess, "
                            f"{model} is given. lowess is used."
                        )
                        model = "lowess"

                    if (_profile == 0.0).all():
                        _profile = np.ones_like(_profile)

                        self.logger.warning(
                            "Optimal profile is all zeros. Unit weighting is"
                            " used instead."
                        )

                    # source_pix is the native pixel position
                    # pos is the trace at the native pixel position
                    (
                        count[i],
                        count_err[i],
                        is_optimal[i],
                        profile[i][profile_start_idx:profile_end_idx],
                        var_temp,
                    ) = optimal_extraction_horne86(
                        source_slice=source_slice,
                        sky=count_sky_source_slice,
                        profile=_profile,
                        tol=tolerance,
                        max_iter=max_iter,
                        readnoise=self.readnoise,
                        gain=self.gain,
                        cosmicray_sigma=self.cosmicray_sigma,
                        forced=forced,
                        variances=var_i,
                        bad_mask=source_bad_mask,
                    )
                    if var_i is None:
                        var[
                            i, offset : offset + width_dn + width_up + 1
                        ] = var_temp

                    else:
                        var[i] = var_i

            if optimal & (algorithm == "marsh89"):
                self.logger.debug("Using Marsh 1989 algorithm.")

                if variances is None:
                    variances = self.variance

                (
                    count,
                    count_err,
                    is_optimal,
                    profile,
                    var,
                ) = optimal_extraction_marsh89(
                    frame=self.img,
                    residual_frame=self.img_residual,
                    variance=variances,
                    trace=spec.trace,
                    spectrum=count,
                    readnoise=self.readnoise,
                    apwidth=(width_dn, width_up),
                    goodpixelmask=~self.bad_mask,
                    npoly=npoly,
                    polyspacing=polyspacing,
                    pord=pord,
                    cosmicray_sigma=self.cosmicray_sigma,
                    qmode=qmode,
                    nreject=nreject,
                )

            spec.add_aperture(
                width_dn, width_up, sep_dn, sep_up, sky_width_dn, sky_width_up
            )
            spec.add_count(list(count), list(count_err), list(count_sky))
            spec.add_variances(var)
            spec.add_profile(profile)
            spec.gain = self.gain
            spec.optimal_pixel = is_optimal
            spec.add_spectrum_header(self.header)

            self.logger.info("Spectrum extracted for spec_id: %s.", j)

            if optimal & (algorithm == "horne86"):
                spec.extraction_type = "OptimalHorne86"

            if optimal & (algorithm == "marsh89"):
                spec.extraction_type = "OptimalMarsh89"

            else:
                spec.extraction_type = "Tophat"

            # If more than a third of the spectrum is extracted suboptimally
            if np.sum(is_optimal) / len(is_optimal) < 0.333:
                self.logger.warning(
                    "Some signal extracted is likely to be suboptimal, it "
                    "is most likely happening at the red and/or blue ends "
                    "of the spectrum."
                )

        if save_fig or display or return_jsonstring:
            to_return = self.inspect_extraction(
                display=display,
                renderer=renderer,
                width=width,
                height=height,
                return_jsonstring=return_jsonstring,
                save_fig=save_fig,
                fig_type=fig_type,
                filename=filename,
                open_iframe=open_iframe,
                spec_id=spec_id,
            )

        if return_jsonstring:
            return to_return

    def inspect_extraction(
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
        spec_id: Union[int, list, np.ndarray] = None,
    ):
        """
        Display the extracted spectrum/a.

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
            set to True to return json str that can be rendered by Plotly
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

        """

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if spec_id is not None:
            assert np.in1d(
                spec_id, list(self.spectrum_list.keys())
            ).all(), "Some spec_id provided are not in the spectrum_list."

        else:
            spec_id = list(self.spectrum_list.keys())

        to_return = []

        for j in spec_id:
            spec = self.spectrum_list[j]

            width_dn = spec.widthdn
            width_up = spec.widthup

            sep_dn = spec.sepdn
            sep_up = spec.sepup

            sky_width_dn = spec.skywidthdn
            sky_width_up = spec.skywidthup

            offset = 0

            len_trace = len(spec.trace)

            spec_id = list(self.spectrum_list.keys())

            min_trace = int(min(spec.trace) + 0.5)
            max_trace = int(max(spec.trace) + 0.5)

            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width)
            )
            # the 3 is to show a little bit outside the extraction regions
            img_display = np.log10(
                self.img[
                    max(
                        0, min_trace - width_dn - sep_dn - sky_width_dn - 3
                    ) : min(
                        max_trace + width_up + sep_up + sky_width_up,
                        len(self.img[0]),
                    )
                    + 3,
                    :,
                ]
            )

            # show the image on the top
            # the 3 is the show a little bit outside the extraction regions
            fig.add_trace(
                go.Heatmap(
                    x=np.arange(len_trace),
                    y=np.arange(
                        max(
                            0,
                            min_trace - width_dn - sep_dn - sky_width_dn - 3,
                        ),
                        min(
                            max_trace + width_up + sep_up + sky_width_up + 3,
                            len(self.img[0]),
                        ),
                    ),
                    z=img_display,
                    colorscale="Viridis",
                    zmin=self.zmin,
                    zmax=self.zmax,
                    xaxis="x",
                    yaxis="y",
                    colorbar=dict(title="log( e- count )"),
                )
            )

            # Middle black box on the image
            fig.add_trace(
                go.Scatter(
                    x=list(
                        np.concatenate(
                            (
                                np.arange(len_trace),
                                np.arange(len_trace)[::-1],
                                np.zeros(1),
                            )
                        )
                    ),
                    y=list(
                        np.concatenate(
                            (
                                np.array(spec.trace) - width_dn - 1,
                                np.array(spec.trace[::-1]) + width_up + 1,
                                np.ones(1) * (spec.trace[0] - width_dn - 1),
                            )
                        )
                    ),
                    xaxis="x",
                    yaxis="y",
                    mode="lines",
                    line_color="black",
                    showlegend=False,
                )
            )

            # Lower red box on the image
            if offset == 0:
                lower_redbox_upper_bound = (
                    np.array(spec.trace) - width_dn - sep_dn - 1
                )
                lower_redbox_lower_bound = (
                    np.array(spec.trace)[::-1]
                    - width_dn
                    - sep_dn
                    - sky_width_dn
                )
                lower_redbox_lower_bound[lower_redbox_lower_bound < 0] = 1

                fig.add_trace(
                    go.Scatter(
                        x=list(
                            np.concatenate(
                                (
                                    np.arange(len_trace),
                                    np.arange(len_trace)[::-1],
                                    np.zeros(1),
                                )
                            )
                        ),
                        y=list(
                            np.concatenate(
                                (
                                    lower_redbox_upper_bound,
                                    lower_redbox_lower_bound,
                                    np.ones(1) * lower_redbox_upper_bound[0],
                                )
                            )
                        ),
                        line_color="red",
                        xaxis="x",
                        yaxis="y",
                        mode="lines",
                        showlegend=False,
                    )
                )

            # Upper red box on the image
            if sep_up + sky_width_up > 0:
                upper_redbox_upper_bound = (
                    np.array(spec.trace) + width_up + sep_up + sky_width_up
                )
                upper_redbox_lower_bound = (
                    np.array(spec.trace)[::-1] + width_up + sep_up + 1
                )

                upper_redbox_upper_bound[
                    upper_redbox_upper_bound > self.spatial_size
                ] = (self.spatial_size + 1)

                fig.add_trace(
                    go.Scatter(
                        x=list(
                            np.concatenate(
                                (
                                    np.arange(len_trace),
                                    np.arange(len_trace)[::-1],
                                    np.zeros(1),
                                )
                            )
                        ),
                        y=list(
                            np.concatenate(
                                (
                                    upper_redbox_upper_bound,
                                    upper_redbox_lower_bound,
                                    np.ones(1) * upper_redbox_upper_bound[0],
                                )
                            )
                        ),
                        xaxis="x",
                        yaxis="y",
                        mode="lines",
                        line_color="red",
                        showlegend=False,
                    )
                )

            # plot the SNR
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len_trace),
                    y=np.array(spec.count) / np.array(spec.count_err),
                    xaxis="x2",
                    yaxis="y3",
                    line=dict(color="slategrey"),
                    name="Signal-to-Noise Ratio",
                )
            )

            # extrated source, sky and uncertainty
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len_trace),
                    y=spec.count_sky,
                    xaxis="x2",
                    yaxis="y2",
                    line=dict(color="firebrick"),
                    name="Sky e- count",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len_trace),
                    y=spec.count_err,
                    xaxis="x2",
                    yaxis="y2",
                    line=dict(color="orange"),
                    name="Uncertainty e- count",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len_trace),
                    y=spec.count,
                    xaxis="x2",
                    yaxis="y2",
                    line=dict(color="royalblue"),
                    name="Target e- count",
                )
            )

            # Decorative stuff
            fig.update_layout(
                xaxis=dict(showticklabels=False),
                yaxis=dict(
                    zeroline=False,
                    domain=[0.5, 1],
                    showgrid=False,
                    title="Spatial Direction / pixel",
                ),
                yaxis2=dict(
                    range=[
                        min(
                            np.nanmin(
                                sigma_clip(spec.count, sigma=5.0, masked=False)
                            ),
                            np.nanmin(
                                sigma_clip(
                                    spec.count_err, sigma=5.0, masked=False
                                )
                            ),
                            np.nanmin(
                                sigma_clip(
                                    spec.count_sky, sigma=5.0, masked=False
                                )
                            ),
                            1,
                        ),
                        max(np.nanmax(spec.count), np.nanmax(spec.count_sky)),
                    ],
                    zeroline=False,
                    domain=[0, 0.5],
                    showgrid=True,
                    title=" e- count",
                ),
                yaxis3=dict(
                    title="S/N ratio",
                    anchor="x2",
                    overlaying="y2",
                    side="right",
                ),
                xaxis2=dict(
                    title="Dispersion Direction / pixel",
                    anchor="y2",
                    matches="x",
                ),
                legend=go.layout.Legend(
                    x=0,
                    y=0.45,
                    traceorder="normal",
                    font=dict(family="sans-serif", size=12, color="black"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                bargap=0,
                hovermode="closest",
                showlegend=True,
            )

            if filename is None:
                filename = "ap_extract"

            if save_fig:
                fig_type_split = fig_type.split("+")

                for t in fig_type_split:
                    save_path = filename + "_" + str(j) + "." + t

                    if t == "iframe":
                        pio.write_html(fig, save_path, auto_open=open_iframe)

                    elif t in ["jpg", "png", "svg", "pdf"]:
                        pio.write_image(fig, save_path)

                    self.logger.info(
                        "Figure is saved to %s for spec_id: %s.", save_path, j
                    )

            if display:
                if renderer == "default":
                    fig.show()

                else:
                    fig.show(renderer)

            if return_jsonstring:
                to_return.append(fig.to_json())

        if return_jsonstring:
            return to_return

    def inspect_line_spread_function(
        self,
        spec_id: Union[int, list, np.ndarray] = None,
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
        Call this method to inspect the line spread function used to extract
        the spectrum.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        display: bool
            Set to True to display disgnostic plot.
        renderer: str
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool
            set to True to return json str that can be rendered by Plotly
            in any support language.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool
            Open the iframe in the default browser if set to True.

        """

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if spec_id is not None:
            assert np.in1d(
                spec_id, list(self.spectrum_list.keys())
            ).all(), "Some spec_id provided are not in the spectrum_list."

        else:
            spec_id = list(self.spectrum_list.keys())

        to_return = []

        for j in spec_id:
            spec = self.spectrum_list[j]
            profile = self.spectrum_list[j].profile

            len_trace = len(spec.trace)
            # plot 10 LSFs
            lsf_dist = len_trace // 10
            lsf_idx = (
                np.arange(0, len_trace - lsf_dist + 1, lsf_dist)
                + lsf_dist // 2
            )

            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width)
            )

            for i in lsf_idx:
                # plot the SNR
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(profile[i])),
                        y=profile[i],
                        name=f"Pixel {i}",
                    )
                )

            # Decorative stuff
            fig.update_layout(
                yaxis=dict(
                    range=[
                        np.nanmin(profile),
                        np.nanmax(profile),
                    ],
                    zeroline=False,
                    domain=[0, 1.0],
                    showgrid=True,
                    title="Count / s",
                ),
                legend=go.layout.Legend(
                    traceorder="normal",
                    font=dict(family="sans-serif", size=12, color="black"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                bargap=0,
                hovermode="closest",
                showlegend=True,
            )

            if filename is None:
                filename = "extraction_profile"

            if save_fig:
                fig_type_split = fig_type.split("+")

                for f_type in fig_type_split:
                    save_path = filename + "_" + str(j) + "." + f_type

                    if f_type == "iframe":
                        pio.write_html(fig, save_path, auto_open=open_iframe)

                    elif f_type in ["jpg", "png", "svg", "pdf"]:
                        pio.write_image(fig, save_path)

                    self.logger.info(
                        "Figure is saved to %s for spec_id: %s.", save_path, j
                    )

            if display:
                if renderer == "default":
                    fig.show()

                else:
                    fig.show(renderer)

            if return_jsonstring:
                to_return.append(fig.to_json())

        if return_jsonstring:
            return to_return

    def inspect_extracted_spectrum(
        self,
        spec_id: Union[int, list, np.ndarray] = None,
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
        Call this method to inspect the extracted spectrum.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object
        display: bool
            Set to True to display disgnostic plot.
        renderer: str
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool
            set to True to return json str that can be rendered by Plotly
            in any support language.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool
            Open the iframe in the default browser if set to True.

        """

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if spec_id is not None:
            assert np.in1d(
                spec_id, list(self.spectrum_list.keys())
            ).all(), "Some spec_id provided are not in the spectrum_list."

        else:
            spec_id = list(self.spectrum_list.keys())

        to_return = []

        for j in spec_id:
            spec = self.spectrum_list[j]

            len_trace = len(spec.trace)
            count = np.array(spec.count)
            count_err = np.array(spec.count_err)
            count_sky = np.array(spec.count_sky)

            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width)
            )

            # plot the SNR
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len_trace),
                    y=count / count_err,
                    xaxis="x2",
                    yaxis="y3",
                    line=dict(color="slategrey"),
                    name="Signal-to-Noise Ratio",
                )
            )

            # extrated source, sky and uncertainty
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len_trace),
                    y=count_sky,
                    xaxis="x2",
                    yaxis="y2",
                    line=dict(color="firebrick"),
                    name="Sky e- count / s",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len_trace),
                    y=count_err,
                    xaxis="x2",
                    yaxis="y2",
                    line=dict(color="orange"),
                    name="Uncertainty e- count / s",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len_trace),
                    y=count,
                    xaxis="x2",
                    yaxis="y2",
                    line=dict(color="royalblue"),
                    name="Target e- count / s",
                )
            )

            # Decorative stuff
            fig.update_layout(
                yaxis2=dict(
                    range=[
                        min(
                            np.nanmin(
                                sigma_clip(count, sigma=5.0, masked=False)
                            ),
                            np.nanmin(
                                sigma_clip(count_err, sigma=5.0, masked=False)
                            ),
                            np.nanmin(
                                sigma_clip(count_sky, sigma=5.0, masked=False)
                            ),
                            1,
                        ),
                        max(np.nanmax(count), np.nanmax(count_sky)),
                    ],
                    zeroline=False,
                    domain=[0, 1.0],
                    showgrid=True,
                    title="Count / s",
                ),
                yaxis3=dict(
                    title="S/N ratio",
                    anchor="x2",
                    overlaying="y2",
                    side="right",
                ),
                xaxis2=dict(
                    title="Dispersion Direction / pixel",
                    anchor="y2",
                    matches="x",
                ),
                legend=go.layout.Legend(
                    x=0,
                    y=0.45,
                    traceorder="normal",
                    font=dict(family="sans-serif", size=12, color="black"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                bargap=0,
                hovermode="closest",
                showlegend=True,
            )

            if filename is None:
                filename = "extracted_spectrum"

            if save_fig:
                fig_type_split = fig_type.split("+")

                for f_type in fig_type_split:
                    save_path = filename + "_" + str(j) + "." + f_type

                    if f_type == "iframe":
                        pio.write_html(fig, save_path, auto_open=open_iframe)

                    elif f_type in ["jpg", "png", "svg", "pdf"]:
                        pio.write_image(fig, save_path)

                    self.logger.info(
                        "Figure is saved to %s for spec_id: %s.", save_path, j
                    )

            if display:
                if renderer == "default":
                    fig.show()

                else:
                    fig.show(renderer)

            if return_jsonstring:
                to_return.append(fig.to_json())

        if return_jsonstring:
            return to_return

    def inspect_residual(
        self,
        log: bool = True,
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
        Display the reduced image with a supported plotly renderer or export
        as json strings.

        Parameters
        ----------
        log: bool
            Log the ADU count per second in the display. Default is True.
        display: bool
            Set to True to display disgnostic plot.
        renderer: str
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
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
            Open the save_iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        """

        if log:
            fig = go.Figure(
                data=go.Heatmap(
                    z=np.log10(self.img_residual), colorscale="Viridis"
                )
            )
        else:
            fig = go.Figure(
                data=go.Heatmap(z=self.img_residual, colorscale="Viridis")
            )

        fig.update_layout(
            yaxis_title="Spatial Direction / pixel",
            xaxis=dict(
                zeroline=False,
                showgrid=False,
                title="dispersion direction / pixel",
            ),
            bargap=0,
            hovermode="closest",
            showlegend=False,
            autosize=False,
            height=height,
            width=width,
        )

        if filename is None:
            filename = "residual_image"

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

    def extract_arc_spec(
        self,
        spec_width: Union[float, int] = None,
        display: bool = False,
        renderer: str = "default",
        width: int = 1280,
        height: int = 720,
        return_jsonstring: bool = False,
        save_fig: bool = False,
        fig_type: str = "iframe+png",
        filename: str = None,
        open_iframe: bool = False,
        spec_id: Union[int, list, np.ndarray] = None,
    ):
        """
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
        spec_width: int (Default: None)
            The number of pixels in the spatial direction used to sum for the
            arc spectrum
        display: bool
            Set to True to display disgnostic plot.
        renderer: str
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool
            set to True to return json str that can be rendered by Plotly
            in any support language.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool
            Open the iframe in the default browser if set to True.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object

        """

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if spec_id is not None:
            if not set(spec_id).issubset(list(self.spectrum_list.keys())):
                error_msg = "The given spec_id does not exist."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:
            # if spec_id is None, all arc spectra are extracted
            spec_id = list(self.spectrum_list.keys())

        if self.arc is None:
            error_msg = (
                "arc is not provided. Please provide arc by "
                + "using add_arc() or with from_twodspec() before "
                + "executing find_arc_lines()."
            )
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        to_return = []

        for i in spec_id:
            spec = self.spectrum_list[i]

            len_trace = len(spec.trace)
            trace = np.nanmean(spec.trace)
            if spec_width is None:
                trace_width = np.nanmean(spec.trace_sigma) * 3.0

            else:
                trace_width = spec_width

            arc_trace = self.arc[
                max(0, int(trace - trace_width - 1)) : min(
                    int(trace + trace_width), len_trace
                ),
                :,
            ]
            arc_spec = np.nanmedian(arc_trace, axis=0)

            spec.add_arc_spec(list(arc_spec))
            spec.add_arc_header(self.arc_header)

        # note that the display is adjusted for the chip gaps
        if save_fig or display or return_jsonstring:
            to_return = self.inspect_arc_spec(
                display=display,
                renderer=renderer,
                width=width,
                height=height,
                return_jsonstring=return_jsonstring,
                save_fig=save_fig,
                fig_type=fig_type,
                filename=filename,
                open_iframe=open_iframe,
                spec_id=spec_id,
            )

        if return_jsonstring:
            return to_return

    def inspect_arc_spec(
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
        spec_id: Union[int, list, np.ndarray] = None,
    ):
        """
        Display the extracted arc spectrum.

        Parameters
        ----------
        display: bool
            Set to True to display disgnostic plot.
        renderer: str
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool
            set to True to return json str that can be rendered by Plotly
            in any support language.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool
            Open the iframe in the default browser if set to True.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum_oned object

        """

        if isinstance(spec_id, int):
            spec_id = [spec_id]

        if spec_id is not None:
            if not set(spec_id).issubset(list(self.spectrum_list.keys())):
                error_msg = (
                    (
                        "The given spec_id(s): %s do(es) not exist. The"
                        " twodspec object has %s."
                    ),
                    spec_id,
                    list(self.spectrum_list.keys()),
                )
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:
            # if spec_id is None, all arc spectra are extracted
            spec_id = list(self.spectrum_list.keys())

        if self.arc is None:
            error_msg = (
                "arc is not provided. Please provide arc by "
                + "using add_arc() or with from_twodspec() before "
                + "executing find_arc_lines()."
            )
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        to_return = []

        for i in spec_id:
            spec = self.spectrum_list[i]

            len_trace = len(spec.trace)

            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width)
            )

            fig.add_trace(
                go.Scatter(
                    x=np.arange(len_trace),
                    y=spec.arc_spec,
                    mode="lines",
                    line=dict(color="royalblue", width=1),
                )
            )

            fig.update_layout(
                xaxis=dict(
                    zeroline=False,
                    range=[0, len_trace],
                    title="Dispersion Direction / pixel",
                ),
                yaxis=dict(
                    zeroline=False,
                    range=[0, max(spec.arc_spec)],
                    title="e- count / s",
                ),
                hovermode="closest",
                showlegend=False,
            )

            if filename is None:
                filename = f"arc_spec_{i}"

            if save_fig:
                fig_type_split = fig_type.split("+")

                for f_type in fig_type_split:
                    save_path = filename + "_" + str(i) + "." + f_type

                    if f_type == "iframe":
                        pio.write_html(fig, save_path, auto_open=open_iframe)

                    elif f_type in ["jpg", "png", "svg", "pdf"]:
                        pio.write_image(fig, save_path)

                    self.logger.info(
                        "Figure is saved to %s for spec_id: %s.", save_path, i
                    )

            if display:
                if renderer == "default":
                    fig.show()

                else:
                    fig.show(renderer)

            if return_jsonstring:
                to_return.append(fig.to_json())

        if return_jsonstring:
            return to_return

    def create_fits(
        self,
        output: str = "*",
        recreate: bool = False,
        empty_primary_hdu: bool = True,
    ):
        """
        Parameters
        ----------
        output: String (Default: "*")
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strs are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 3 HDUs
                    Count, uncertainty, and sky (pixel)
                weight_map: 1 HDU
                    Weight (pixel)
                arc_spec: 1 HDU
                    1D arc spectrum

        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank

        """

        if output == "*":
            output = "trace+count+weight_map+arc_spec"

        for i in output.split("+"):
            if i not in ["trace", "count", "weight_map", "arc_spec"]:
                error_msg = f"{i} is not a valid output."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        # Save each trace as a separate FITS file
        for i, spec_i in self.spectrum_list.items():
            spec_i.create_fits(
                output=output,
                recreate=recreate,
                empty_primary_hdu=empty_primary_hdu,
            )
            self.logger.info("FITS file is created for spec_id: %s.", i)

    def save_fits(
        self,
        output: str = "*",
        filename: str = None,
        overwrite: bool = False,
        recreate: bool = False,
        empty_primary_hdu: bool = True,
    ):
        """
        Save the reduced image to disk. Each trace is saved into a separate
        file.

        Parameters
        ----------
        output: String (Default: "*")
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strs are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 3 HDUs
                    Count, uncertainty, and sky (pixel)
                weight_map: 1 HDU
                    Weight (pixel)
                arc_spec: 1 HDU
                    1D arc spectrum

        filename: str (Default: TwoDSpecExtracted)
            Filename for the output, all of them will share the same name but
            will have different extension.
        overwrite: bool
            Default is False.
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank

        """

        if output == "*":
            output = "trace+count+weight_map+arc_spec"

        if filename is not None:
            filename = os.path.splitext(filename)[0]

        else:
            filename = "TwoDSpecExtracted_" + output

        for i in output.split("+"):
            if i not in ["trace", "count", "weight_map", "arc_spec"]:
                error_msg = f"{i} is not a valid output."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        # Save each trace as a separate FITS file
        for i, spec_i in self.spectrum_list.items():
            filename_i = filename + "_" + str(i)

            spec_i.save_fits(
                output=output,
                filename=filename_i,
                overwrite=overwrite,
                recreate=recreate,
                empty_primary_hdu=empty_primary_hdu,
            )
            self.logger.info(
                "FITS file is saved to %s for spec_id: %s.", filename_i, i
            )

    def save_csv(
        self,
        output: str = "*",
        filename: str = None,
        overwrite: bool = False,
        recreate: bool = False,
    ):
        """
        Save the reduced image to disk. Each trace is saved into a separate
        file.

        Parameters
        ----------
        output: String (Default: "*")
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strs are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 3 HDUs
                    Count, uncertainty, and sky (pixel)
                weight_map: 1 HDU
                    Weight (pixel)
                arc_spec: 1 HDU
                    1D arc spectrum

        filename: str (Default: TwoDSpecExtracted)
            Filename for the output, all of them will share the same name but
            will have different extension.
        overwrite: bool
            Default is False.
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header before exporting
            as CSV.

        """

        if output == "*":
            output = "trace+count+weight_map+arc_spec"

        if filename is not None:
            filename = os.path.splitext(filename)[0]

        else:
            filename = "TwoDSpecExtracted_" + output

        for i in output.split("+"):
            if i not in ["trace", "count", "weight_map", "arc_spec"]:
                error_msg = f"{i} is not a valid output."
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        # Save each trace as a separate FITS file
        for i, spec_i in self.spectrum_list.items():
            filename_i = filename + "_" + str(i)

            spec_i.save_csv(
                output=output,
                filename=filename_i,
                overwrite=overwrite,
                recreate=recreate,
            )
            self.logger.info(
                "CSV file is saved to %s for spec_id: %s.", filename_i, i
            )
