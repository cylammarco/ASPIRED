# -*- coding: utf-8 -*-
import copy
import datetime
import logging
import os
from itertools import chain

import numpy as np
from astropy.nddata import CCDData
from astropy.io import fits
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
from plotly import graph_objects as go
from plotly import io as pio
from scipy import ndimage
from scipy import signal
from scipy.optimize import curve_fit
from spectres import spectres
from statsmodels.nonparametric.smoothers_lowess import lowess

from .image_reduction import ImageReduction
from .spectrum1D import Spectrum1D
from .util import create_bad_pixel_mask, bfixpix

__all__ = ["TwoDSpec"]


class TwoDSpec:
    """
    This is a class for processing a 2D spectral image.

    """

    def __init__(
        self,
        data=None,
        header=None,
        verbose=True,
        logger_name="TwoDSpec",
        log_level="INFO",
        log_file_folder="default",
        log_file_name=None,
        **kwargs
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

        self.img = None
        self.img_residual = None
        self.header = None
        self.arc = None
        self.arc_header = None
        self.bad_mask = None

        self.saxis = 1
        self.waxis = 0

        self.spatial_mask = (1,)
        self.spec_mask = (1,)
        self.flip = False
        self.cosmicray = False
        self.fsmode = None
        self.psfmodel = None

        self.spatial_mask_applied = False
        self.spec_mask_applied = False
        self.transpose_applied = False
        self.flip_applied = False

        # Default values if not supplied
        self.airmass = None
        self.readnoise = None
        self.gain = None
        self.seeing = None
        self.exptime = None

        self.verbose = verbose
        self.logger_name = logger_name
        self.log_level = log_level
        self.log_file_folder = log_file_folder
        self.log_file_name = log_file_name

        # Default keywords to be searched in the order in the list
        self.readnoise_keyword = ["RDNOISE", "RNOISE", "RN"]
        self.gain_keyword = ["GAIN"]
        self.seeing_keyword = [
            "SEEING",
            "L1SEEING",
            "ESTSEE",
            "DIMMSEE",
            "SEEDIMM",
            "DSEEING",
        ]
        self.exptime_keyword = [
            "XPOSURE",
            "EXPOSURE",
            "EXPTIME",
            "EXPOSED",
            "TELAPSED",
            "ELAPSED",
        ]
        self.airmass_keyword = ["AIRMASS", "AMASS", "AIRM", "AIR"]

        self.img_rectified = None
        self.arc_rectified = None

        self.add_data(data, header)
        self.spectrum_list = {}
        self.set_properties(**kwargs)

        if self.arc is not None:

            self.apply_mask_to_arc()

    def add_data(self, data, header=None):
        """
        Adding the 2D image data to be processed. The data can be a 2D numpy
        array, an AstroPy ImageHDU/Primary HDU object or an ImageReduction
        object.

        parameters
        ----------
        data: 2D numpy array (M x N) OR astropy.io.fits object
            2D spectral image in either format
        header: FITS header (deafult: None)
            THIS WILL OVERRIDE the header from the astropy.io.fits object

        """

        # If data provided is an numpy array
        if isinstance(data, np.ndarray):

            self.img = data
            self.logger.info("An numpy array is loaded as data.")
            self.header = header
            self.bad_mask = create_bad_pixel_mask(self.img)[0]

        # If it is a fits.hdu.hdulist.HDUList object
        elif isinstance(data, fits.hdu.hdulist.HDUList):

            self.img = data[0].data
            self.header = data[0].header
            self.bad_mask = create_bad_pixel_mask(self.img)[0]
            self.logger.warning(
                "An HDU list is provided, only the first " "HDU will be read."
            )

        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(data, fits.hdu.image.PrimaryHDU) or isinstance(
            data, fits.hdu.image.ImageHDU
        ):

            self.img = data.data
            self.header = data.header
            self.bad_mask = create_bad_pixel_mask(self.img)[0]
            self.logger.info("A PrimaryHDU is loaded as data.")

        # If it is an ImageReduction object
        elif isinstance(data, ImageReduction):

            # If the data is not reduced, reduce it here. Error handling is
            # done by the ImageReduction class
            if data.image_fits is None:

                data._create_image_fits()

            self.img = data.image_fits.data
            self.header = data.image_fits.header

            if data.arc_main is not None:

                self.arc = data.arc_main
                self.arc_header = data.arc_header[0]

            else:

                self.logger.warning(
                    "Arc frame is not in the ImageReduction "
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
            self.header = copy.deepcopy(fitsfile_tmp.header)
            logging.info(
                "Loaded data from: {}, with hdunum: {}".format(
                    filepath, hdunum
                )
            )

            fitsfile_tmp = None

        elif data is None:

            pass

        else:

            error_msg = (
                "Please provide a numpy array, an "
                + "astropy.io.fits.hdu.image.PrimaryHDU object "
                + "or an ImageReduction object."
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

    def set_properties(
        self,
        saxis=None,
        variance=None,
        spatial_mask=None,
        spec_mask=None,
        flip=None,
        cosmicray=None,
        gain=-1,
        readnoise=-1,
        fsmode=None,
        psfmodel=None,
        seeing=-1,
        exptime=-1,
        airmass=-1,
        verbose=None,
        **kwargs
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
            Spectral direction, 0 for vertical, 1 for horizontal.
        variance: 2D numpy array (M, N)
            The per-pixel-variance of the frame.
        spatial_mask: 1D numpy array (size: N. Default is (1,))
            Mask in the spatial direction, can be the indices of the pixels
            to be included (size <N) or a 1D numpy array of True/False (size N)
        spec_mask: 1D numpy array (Size: M. Default: (1,))
            Mask in the spectral direction, can be the indices of the pixels
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
                    "saxis can only be 0 or 1, {} is ".format(saxis)
                    + "given. It is set to 0."
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

            self.logger.info(
                "Removing cosmic rays in mode: {}.".format(psfmodel)
            )

            if self.fsmode == "convolve":

                if psfmodel == "gaussyx":

                    self.img = detect_cosmics(
                        self.img / self.gain,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode="convolve",
                        psfmodel="gaussy",
                        **kwargs
                    )[1]

                    self.img = detect_cosmics(
                        self.img / self.gain,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode="convolve",
                        psfmodel="gaussx",
                        **kwargs
                    )[1]

                elif psfmodel == "gaussxy":

                    self.img = detect_cosmics(
                        self.img / self.gain,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode="convolve",
                        psfmodel="gaussx",
                        **kwargs
                    )[1]

                    self.img = detect_cosmics(
                        self.img / self.gain,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode="convolve",
                        psfmodel="gaussy",
                        **kwargs
                    )[1]

                else:

                    self.img = detect_cosmics(
                        self.img / self.gain,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode="convolve",
                        psfmodel=self.psfmodel,
                        **kwargs
                    )[1]

            else:

                self.img = detect_cosmics(
                    self.img / self.gain,
                    gain=self.gain,
                    readnoise=self.readnoise,
                    fsmode=self.fsmode,
                    psfmodel=self.psfmodel,
                    **kwargs
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

            # the valid x-range of the chip (i.e. spectral direction)
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
        self.logger.info("spec_size is found to be {}.".format(self.spec_size))
        self.logger.info(
            "spatial_size is found to be " "{}.".format(self.spatial_size)
        )

    def _get_image_zminmax(self):

        # set the 2D histogram z-limits
        img_log = np.log10(self.img)
        img_log_finite = img_log[np.isfinite(img_log)]
        self.zmin = np.nanpercentile(img_log_finite, 5)
        self.zmax = np.nanpercentile(img_log_finite, 95)
        self.logger.info("zmin is set to {}.".format(self.zmin))
        self.logger.info("zmax is set to {}.".format(self.zmax))

    # Get the readnoise
    def set_readnoise(self, readnoise=None):
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
                    "readnoise is found to be {}.".format(self.readnoise)
                )

            elif isinstance(readnoise, (float, int)) & (~np.isnan(readnoise)):

                if readnoise < 0:

                    pass

                else:

                    # use the given readnoise value
                    self.readnoise = float(readnoise)
                    self.logger.info(
                        "readnoise is set to {}.".format(self.readnoise)
                    )

            else:

                self.readnoise = 0.0
                self.logger.warning(
                    "readnoise has to be None, a numeric value or the "
                    + "FITS header keyword, "
                    + str(readnoise)
                    + " is "
                    + "given. It is set to 0."
                )

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
                        "readnoise is found to be {}.".format(self.readnoise)
                    )

                else:

                    self.readnoise = 0.0
                    self.logger.warning(
                        "Readnoise value cannot be identified. "
                        + "It is set to 0."
                    )

            else:

                self.readnoise = 0.0
                self.logger.warning(
                    "Header is not provided. Readnoise value "
                    + "is not provided. It is set to 0."
                )

    # Get the gain
    def set_gain(self, gain=None):
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
                self.logger.info("gain is found to be {}.".format(self.gain))

            elif isinstance(gain, (float, int)) & (~np.isnan(gain)):

                if gain < 0:

                    pass

                else:

                    # use the given gain value
                    self.gain = float(gain)
                    self.logger.info("gain is set to {}.".format(self.gain))

            else:

                self.gain = 1.0
                self.logger.warning(
                    "Gain has to be None, a numeric value or the FITS "
                    + "header keyword, "
                    + str(gain)
                    + " is given. It is "
                    + "set to 1."
                )
        else:

            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:

                gain_keyword_matched = np.in1d(self.gain_keyword, self.header)

                if gain_keyword_matched.any():

                    self.gain = self.header[
                        self.gain_keyword[np.where(gain_keyword_matched)[0][0]]
                    ]
                    self.logger.info(
                        "gain is found to be {}.".format(self.gain)
                    )

                else:

                    self.gain = 1.0
                    self.logger.warning(
                        "Gain value cannot be identified. " + "It is set to 1."
                    )

            else:

                self.gain = 1.0
                self.logger.warning(
                    "Header is not provide. Gain value is not "
                    + "provided. It is set to 1."
                )

    # Get the Seeing
    def set_seeing(self, seeing=None):
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
                self.logger.info(
                    "seeing is found to be {}.".format(self.seeing)
                )

            elif isinstance(seeing, (float, int)) & (~np.isnan(seeing)):

                if seeing < 0:

                    pass

                else:

                    # use the given seeing value
                    self.seeing = float(seeing)
                    self.logger.info(
                        "seeing is set to {}.".format(self.seeing)
                    )

            else:

                self.seeing = 1.0
                self.logger.warning(
                    "Seeing has to be None, a numeric value or the FITS "
                    + "header keyword, "
                    + str(seeing)
                    + " is given. It is "
                    + "set to 1."
                )

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
                    self.logger.info(
                        "seeing is found to be {}.".format(self.seeing)
                    )

                else:

                    self.seeing = 1.0
                    self.logger.warning(
                        "Seeing value cannot be identified. "
                        + "It is set to 1."
                    )

            else:

                self.seeing = 1.0
                self.logger.warning(
                    "Header is not provided. Seeing value is "
                    + "not provided. It is set to 1."
                )

    # Get the Exposure Time
    def set_exptime(self, exptime=None):
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
                self.logger.info(
                    "exptime is found to be {}.".format(self.exptime)
                )

            elif isinstance(exptime, (float, int)) & (~np.isnan(exptime)):

                if exptime < 0:

                    pass

                else:

                    # use the given exptime value
                    self.exptime = float(exptime)
                    self.logger.info(
                        "exptime is set to {}.".format(self.exptime)
                    )

            else:

                self.exptime = 1.0
                self.logger.warning(
                    "Exposure Time has to be None, a numeric value or the "
                    + "FITS header keyword, "
                    + str(exptime)
                    + " is given. "
                    + "It is set to 1."
                )

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
                        "exptime is found to be {}.".format(self.exptime)
                    )

                else:

                    self.exptime = 1.0
                    self.logger.warning(
                        "Exposure Time value cannot be identified. "
                        + "It is set to 1."
                    )

            else:

                self.exptime = 1.0
                self.logger.warning(
                    "Header is not provided. "
                    + "Exposure Time value is not provided. "
                    + "It is set to 1."
                )

    # Get the Exposure Time
    def set_airmass(self, airmass=None):
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
                self.logger.info(
                    "exptime is found to be {}.".format(self.exptime)
                )

            elif isinstance(airmass, (float, int)) & (~np.isnan(airmass)):

                if airmass < 0:

                    pass

                else:

                    # use the given airmass value
                    self.airmass = float(airmass)
                    self.logger.info(
                        "airmass is set to {}.".format(self.airmass)
                    )

            else:

                self.logger.warning(
                    "Exposure Time has to be None, a numeric value or the "
                    + "FITS header keyword, "
                    + str(airmass)
                    + " is "
                    + "given. It is set to 1."
                )

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
                        "exptime is found to be {}.".format(self.airmass)
                    )

                else:

                    self.airmass = 1.0
                    self.logger.warning(
                        "Exposure Time value cannot be identified. "
                        + "It is set to 1."
                    )

            else:

                self.airmass = 1.0
                self.logger.warning(
                    "Header is not provided. "
                    + "Exposure Time value is not provided. "
                    + "It is set to 1."
                )

    def add_bad_mask(self, bad_mask=None):
        """
        To provide a mask to ignore the bad pixels in the reduction.

        Parameters
        ----------
        bad_mask: numpy.ndarray, PrimaryHDU/ImageHDU, ImageReduction, str
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
                "An HDU list is provided, only the first " "HDU will be read."
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
            if type(fitsfile_tmp) == "astropy.io.fits.hdu.hdulist.HDUList":

                fitsfile_tmp = fitsfile_tmp[0]
                self.logger.warning(
                    "An HDU list is provided, only the first "
                    "HDU will be read."
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

    def add_arc(self, arc, header=None):
        """
        To provide an arc image. Make sure left (small index) is blue,
        right (large index) is red.

        Parameters
        ----------
        arc: numpy.ndarray, PrimaryHDU/ImageHDU, ImageReduction, str
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
                "An HDU list is provided, only the first " "HDU will be read."
            )

        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(arc, fits.hdu.image.PrimaryHDU) or isinstance(
            arc, fits.hdu.image.ImageHDU
        ):

            self.arc = arc.data
            self.set_arc_header(arc.header)

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
            if type(fitsfile_tmp) == "astropy.io.fits.hdu.hdulist.HDUList":

                fitsfile_tmp = fitsfile_tmp[0]
                self.logger.warning(
                    "An HDU list is provided, only the first "
                    "HDU will be read."
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
                + "astropy.io.fits.HDUList object, or an "
                + "aspired.ImageReduction object."
            )
            self.logger.critical(error_msg)
            raise TypeError(error_msg)

    def set_arc_header(self, header):
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

    def apply_spec_mask_to_arc(self, spec_mask):
        """
        Apply to use only the valid x-range of the chip (i.e. dispersion
        direction)

        parameters
        ----------
        spec_mask: 1D numpy array (M)
            Mask in the spectral direction, can be the indices of the pixels
            to be included (size <M) or a 1D numpy array of True/False (size M)
            (Default is (1,) i.e. keep everything)

        """

        if len(spec_mask) > 1:

            self.arc = self.arc[:, spec_mask]
            self.logger.info("spec_mask is applied to arc.")

        else:

            self.logger.info(
                "spec_mask has zero length, it cannot be "
                "applied to the arc."
            )

    def apply_spatial_mask_to_arc(self, spatial_mask):
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
                "spatial_mask has zero length, it cannot be "
                "applied to the arc."
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

    def set_readnoise_keyword(self, keyword_list, append=False, update=True):
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
                "Please provide the keyword list in str, list or "
                "numpy.ndarray."
            )

        if append:

            self.readnoise_keyword += keyword_list
            self.logger.info(
                "{} is appended to ".format(keyword_list)
                + "the readnoise_keyword list."
            )

        else:

            self.readnoise_keyword = keyword_list
            self.logger.info(
                "{} is used as ".format(keyword_list)
                + "the readnoise_keyword list."
            )

        if update:

            self.set_readnoise()

        else:

            self.logger.info(
                "readnoise_keyword list is updated, but it is "
                "opted not to update the readnoise automatically."
            )

    def set_gain_keyword(self, keyword_list, append=False, update=True):
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
                "Please provide the keyword list in str, list or "
                "numpy.ndarray."
            )

        if append:

            self.gain_keyword += keyword_list
            self.logger.info(
                "{} is appended to ".format(keyword_list)
                + "the gain_keyword list."
            )

        else:

            self.gain_keyword = keyword_list
            self.logger.info(
                "{} is used as ".format(keyword_list)
                + "the gain_keyword list."
            )

        if update:

            self.set_gain()

        else:

            self.logger.info(
                "gain_keyword list is updated, but it is "
                "opted not to update the gain automatically."
            )

    def set_seeing_keyword(self, keyword_list, append=False, update=True):
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
                "Please provide the keyword list in str, list or "
                "numpy.ndarray."
            )

        if append:

            self.seeing_keyword += keyword_list
            self.logger.info(
                "{} is appended to ".format(keyword_list)
                + "the seeing_keyword list."
            )

        else:

            self.seeing_keyword = keyword_list
            self.logger.info(
                "{} is used as ".format(keyword_list)
                + "the seeing_keyword list."
            )

        if update:

            self.set_seeing()

        else:

            self.logger.info(
                "seeing_keyword list is updated, but it is "
                "opted not to update the seeing automatically."
            )

    def set_exptime_keyword(self, keyword_list, append=False, update=True):
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
                "Please provide the keyword list in str, list or "
                "numpy.ndarray."
            )

        if append:

            self.exptime_keyword += keyword_list
            self.logger.info(
                "{} is appended to ".format(keyword_list)
                + "the exptime_keyword list."
            )

        else:

            self.exptime_keyword = keyword_list
            self.logger.info(
                "{} is used as ".format(keyword_list)
                + "the exptime_keyword list."
            )

        if update:

            self.set_exptime()

        else:

            self.logger.info(
                "exptime_keyword list is updated, but it is "
                "opted not to update the exptime automatically."
            )

    def set_airmass_keyword(self, keyword_list, append=False, update=True):
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
                "Please provide the keyword list in str, list or "
                "numpy.ndarray."
            )

        if append:

            self.airmass_keyword += keyword_list
            self.logger.info(
                "{} is appended to ".format(keyword_list)
                + "the airmass_keyword list."
            )

        else:

            self.airmass_keyword = keyword_list
            self.logger.info(
                "{} is used as ".format(keyword_list)
                + "the airmass_keyword list."
            )

        if update:

            self.set_airmass()

        else:

            self.logger.info(
                "airmass_keyword list is updated, but it is "
                "opted not to update the airmass automatically."
            )

    def set_header(self, header):
        """
        Set/replace the header.

        Parameters
        ----------
        header: astropy.io.fits.header.Header
            FITS header from a single HDU.

        """

        # If it is a fits.hdu.header.Header object
        if isinstance(header, fits.header.Header):

            self.header = header

        elif isinstance(header[0], fits.header.Header):

            self.header = header[0]

        else:

            error_msg = (
                "Please provide an " + "astropy.io.fits.header.Header object."
            )
            self.logger.critical(error_msg)
            raise TypeError(error_msg)

    def _gaus(self, x, a, b, x0, sigma):
        """
        Simple Gaussian function.

        Parameters
        ----------
        x: float or 1-d numpy array
            The data to evaluate the Gaussian over
        a: float
            the amplitude
        b: float
            the constant offset
        x0: float
            the center of the Gaussian
        sigma: float
            the width of the Gaussian

        Returns
        -------
        Array or float of same type as input (x).

        """

        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + b

    def ap_trace(
        self,
        nspec=1,
        smooth=False,
        nwindow=20,
        spec_sep=5,
        trace_width=15,
        resample_factor=4,
        rescale=False,
        scaling_min=0.9995,
        scaling_max=1.0005,
        scaling_step=0.001,
        percentile=5,
        shift_tol=10,
        fit_deg=3,
        ap_faint=20,
        display=False,
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
        Aperture tracing by first using cross-correlation then the peaks are
        fitting with a polynomial with an order of floor(nwindow, 10) with a
        minimum order of 1. Nothing is returned unless return_jsonstring of the
        plotly graph is set to be returned.

        Each spectral slice is convolved with the adjacent one in the spectral
        direction. Basic tests show that the geometrical distortion from one
        end to the other in the spectral direction is small. With LT/SPRAT, the
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

        nresample = self.spatial_size * resample_factor
        img_tmp = ndimage.zoom(img_tmp, zoom=resample_factor)

        # split the spectrum into subspectra
        img_split = np.array_split(img_tmp, nwindow, axis=1)
        start_window_idx = nwindow // 2

        lines_ref_init = np.nanmedian(img_split[start_window_idx], axis=1)
        lines_ref_init[np.isnan(lines_ref_init)] = 0.0

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
        shift_tol_len = int(shift_tol * resample_factor)

        spec_spatial = np.zeros(nresample)

        pix = np.arange(nresample)

        # Scipy correlate method, ignore first and last window
        for i in chain(
            range(start_window_idx, nwindow),
            range(start_window_idx - 1, -1, -1),
        ):

            self.logger.info("Correlating the {}-th window.".format(i))

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

                # Upsampling the reference lines
                lines_ref_j = spectres(
                    np.arange(int(nresample * scale)) / scale,
                    np.arange(len(lines_ref)),
                    lines_ref,
                    fill=0.0,
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
            if i == (start_window_idx - 1):

                pix = np.arange(nresample)

            pix = pix * scale_solution[i] + shift_solution[i]

            spec_spatial_tmp = spectres(
                np.arange(nresample),
                np.array(pix).reshape(-1),
                np.array(lines).reshape(-1),
                fill=0.0,
                verbose=False,
            )
            spec_spatial_tmp[np.isnan(spec_spatial_tmp)] = np.nanmin(
                spec_spatial_tmp
            )
            spec_spatial += spec_spatial_tmp

            # Update (increment) the reference line
            if i == nwindow - 1:

                lines_ref = lines_ref_init

            else:

                lines_ref = lines

        nscaled = (nresample * scale_solution).astype("int")

        # Find the spectral position in the middle of the gram in the upsampled
        # pixel location location
        # FWHM cannot be smaller than 3 pixels for any real signal
        peaks = signal.find_peaks(spec_spatial, distance=spec_sep, width=3.0)

        # update the number of spectra if the number of peaks detected is less
        # than the number requested
        self.nspec_traced = min(len(peaks[0]), nspec)
        self.logger.info(
            "{} spectra are identified.".format(self.nspec_traced)
        )

        # Sort the positions by the prominences, and return to the original
        # scale (i.e. with subpixel position)
        spec_init = (
            np.sort(
                peaks[0][np.argsort(-peaks[1]["prominences"])][
                    : self.nspec_traced
                ]
            )
            / resample_factor
        )

        # Create array to populate the spectral locations
        spec_idx = np.zeros((len(spec_init), len(img_split)))

        # Populate the initial values
        spec_idx[:, start_window_idx] = spec_init

        # Pixel positions of the mid point of each data_split (spectral)
        spec_pix = [len(i[0]) for i in img_split]
        spec_pix[0] -= spec_pix[0] // 2
        for i in range(1, len(spec_pix)):
            spec_pix[i] += spec_pix[i - 1]

        spec_pix = np.array(spec_pix).astype("int")

        # Looping through pixels larger than middle pixel
        for i in range(start_window_idx + 1, nwindow):

            spec_idx[:, i] = (
                spec_idx[:, i - 1] * resample_factor * nscaled[i] / nresample
                - shift_solution[i]
            ) / resample_factor

        # Looping through pixels smaller than middle pixel
        for i in range(start_window_idx - 1, -1, -1):

            spec_idx[:, i] = (
                (spec_idx[:, i + 1] * resample_factor - shift_solution[i])
                / (int(nresample * scale_solution[i + 1]) / nresample)
                / resample_factor
            )

        for i in range(len(spec_idx)):

            # Get the median of the subspectrum and then get the Count at the
            # central 5 pixels of the aperture
            ap_val = np.zeros(nwindow)

            for j in range(nwindow):

                # rounding
                idx = int(np.round(spec_idx[i][j] + 0.5))
                subspec_cleaned = sigma_clip(img_split[j], sigma=3)
                ap_val[j] = np.sum(
                    np.nansum(subspec_cleaned, axis=1)[idx - 2 : idx + 2]
                ) / 5 - np.nanmedian(subspec_cleaned)

            # Mask out the faintest ap_faint percent of trace
            n_faint = int(np.round(len(ap_val) * ap_faint / 100))
            mask = np.argsort(ap_val)[n_faint:]
            self.logger.info(
                "The faintest {} subspectra are ".format(n_faint)
                + "going to be ignored in the tracing. They are {}.".format(
                    np.argsort(ap_val)[:n_faint]
                )
            )

            # fit the trace
            ap_p = np.polyfit(spec_pix[mask], spec_idx[i][mask], int(fit_deg))
            ap = np.polyval(ap_p, np.arange(self.spec_size) * resample_factor)
            self.logger.info(
                "The trace is found at {}.".format(
                    [(x, y) for (x, y) in zip(ap_p, ap)]
                )
            )

            # Get the centre of the upsampled spectrum
            ap_centre_idx = ap[start_window_idx] * resample_factor

            # Get the indices for the trace_width-pixels on the left and right
            # of the spectrum, and apply the resampling factor.
            start_idx = int(ap_centre_idx - trace_width * resample_factor)
            end_idx = int(start_idx + 2 * trace_width * resample_factor + 1)

            start_idx = max(0, start_idx)
            end_idx = min(self.spatial_size * resample_factor, end_idx)

            if start_idx == end_idx:

                ap_sigma = np.nan
                continue

            # compute ONE sigma for each trace
            pguess = [
                np.nanmax(spec_spatial[start_idx:end_idx]),
                np.nanpercentile(spec_spatial, 10),
                ap_centre_idx,
                3.0,
            ]

            non_nan_mask = np.isfinite(spec_spatial[start_idx:end_idx])

            popt, _ = curve_fit(
                self._gaus,
                np.arange(start_idx, end_idx)[non_nan_mask],
                spec_spatial[start_idx:end_idx][non_nan_mask],
                p0=pguess,
            )
            ap_sigma = abs(popt[3]) / resample_factor

            self.logger.info(
                "Aperture is fitted with a Gaussian sigma of "
                "{} pix.".format(ap_sigma)
            )

            self.spectrum_list[i] = Spectrum1D(
                spec_id=i,
                verbose=self.verbose,
                logger_name=self.logger_name,
                log_level=self.log_level,
                log_file_folder=self.log_file_folder,
                log_file_name=self.log_file_name,
            )
            self.spectrum_list[i].add_trace(list(ap), [ap_sigma] * len(ap))
            self.spectrum_list[i].add_gain(self.gain)
            self.spectrum_list[i].add_readnoise(self.readnoise)
            self.spectrum_list[i].add_exptime(self.exptime)
            self.spectrum_list[i].add_seeing(self.seeing)
            self.spectrum_list[i].add_airmass(self.airmass)

        # Plot
        if save_fig or display or return_jsonstring:

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

            for i in range(len(spec_idx)):

                fig.add_trace(
                    go.Scatter(
                        x=np.arange(self.spec_size),
                        y=self.spectrum_list[i].trace,
                        line=dict(color="black"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=spec_pix / resample_factor,
                        y=spec_idx[i],
                        mode="markers",
                        marker=dict(color="grey"),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=np.ones(len(spec_idx))
                    * spec_pix[start_window_idx]
                    / resample_factor,
                    y=spec_idx[:, start_window_idx],
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

                for t in fig_type_split:

                    if t == "iframe":

                        pio.write_html(
                            fig, filename + "." + t, auto_open=open_iframe
                        )

                    elif t in ["jpg", "png", "svg", "pdf"]:

                        pio.write_image(fig, filename + "." + t)

                    self.logger.info(
                        "Figure is saved to {} for the ".format(
                            filename + "." + t
                        )
                        + "science_spectrum_list for spec_id: {}.".format(i)
                    )

            if display:

                if renderer == "default":

                    fig.show()

                else:

                    fig.show(renderer)

            if return_jsonstring:

                return fig.to_json()

    def add_trace(self, trace, trace_sigma, spec_id=None):
        """
        Add user-supplied trace. The trace has to have the size as the 2D
        spectral image in the spectral direction.

        Parameters
        ----------
        trace: list or numpy.ndarray (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: list or numpy.ndarray (N)
            Standard deviation of the Gaussian profile of a trace
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object

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
        assert len(trace) == len(trace_sigma), "trace and trace_sigma have to "
        "be the same size."

        for i in spec_id:

            if i in self.spectrum_list.keys():

                self.spectrum_list[i].add_trace(trace, trace_sigma)

            else:

                self.spectrum_list[i] = Spectrum1D(
                    spec_id=i,
                    verbose=self.verbose,
                    logger_name=self.logger_name,
                    log_level=self.log_level,
                    log_file_folder=self.log_file_folder,
                    log_file_name=self.log_file_name,
                )
                self.spectrum_list[i].add_trace(trace, trace_sigma)

    def remove_trace(self, spec_id=None):
        """
        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object

        """

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if spec_id is not None:

            assert np.in1d(
                spec_id, list(self.spectrum_list.keys())
            ).all(), "Some "
            "spec_id provided are not in the spectrum_list."

        else:

            spec_id = list(self.spectrum_list.keys())

        for i in spec_id:

            self.spectrum_list[i].remove_trace()

    def get_rectification(
        self,
        upsample_factor=5,
        bin_size=6,
        n_bin=15,
        spline_order=3,
        order=2,
        coeff=None,
        use_arc=True,
        apply=False,
        display=False,
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
        ONLY possible if there is ONE trace. If more than one trace is
        provided, only the first one (i.e. spec_id = 0) will get
        processed.

        The retification in the spatial direction depends on ONLY the
        trace, while that in the dispersion direction depends on the
        parameters provided here.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        upsample_factor: float (Default: 10)
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
        spec_size_tmp = spec.len_trace * upsample_factor

        # Upsample and shift in the dispersion direction
        img_tmp = ndimage.zoom(
            self.img.astype(float), zoom=upsample_factor, order=spline_order
        )
        y_tmp = (
            ndimage.zoom(
                np.array(spec.trace), zoom=upsample_factor, order=spline_order
            )
            * upsample_factor
        )

        if self.arc is None:

            self.logger.warning(
                "Arc frame is not available, only the data image "
                "will be rectified."
            )

            if use_arc:

                use_arc = False

        elif isinstance(self.arc, CCDData):

            arc_tmp = ndimage.zoom(
                self.arc.data.astype(float),
                zoom=upsample_factor,
                order=spline_order,
            )
            self.logger.info("The arc frame is upsampled.")

        else:

            arc_tmp = ndimage.zoom(
                self.arc.astype(float),
                zoom=upsample_factor,
                order=spline_order,
            )
            self.logger.info("The arc frame is upsampled.")

        # Shift the spectrum to spatially aligned to the trace at ref
        ref = y_tmp[len(y_tmp) // 2]
        for i in range(self.spec_size * upsample_factor):

            shift_i = int(np.round(y_tmp[i] - ref))

            img_tmp[:, i] = np.roll(img_tmp[:, i], -shift_i)

            if self.arc is not None:

                arc_tmp[:, i] = np.roll(arc_tmp[:, i], -shift_i)

        # Now start working with the shift in the spectral direction
        if coeff is not None:

            n_down = None
            n_up = None

            self.logger.info(
                "Polynomial coefficients for rectifying in the spatial "
                "direction is given as: {}.".format(coeff)
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
                    "The given n_bin is not numeric or a list/array of "
                    "size 2: {}. Using the default value to proceed.".format(
                        n_bin
                    )
                )
                n_down = 5
                n_up = 5

            bin_half_size = bin_size / 2 * upsample_factor

            # The x-coordinates of the trace (of length len_trace)
            x = np.arange(spec.len_trace * upsample_factor).astype("int")

            # s for "flattened signal of the slice"
            if use_arc:

                s = [
                    np.nansum(
                        [
                            arc_tmp[
                                int(np.round(ref - bin_half_size)) : int(
                                    np.round(ref + bin_half_size) + 1
                                ),
                                i,
                            ]
                            for i in x
                        ],
                        axis=1,
                    )
                ]

            else:

                s = [
                    np.nansum(
                        [
                            img_tmp[
                                int(np.round(ref - bin_half_size)) : int(
                                    np.round(ref + bin_half_size) + 1
                                ),
                                i,
                            ]
                            for i in x
                        ],
                        axis=1,
                    )
                ]

            one_tenth = len(s[0]) // 10

            s[0] -= lowess(
                s[0], np.arange(spec_size_tmp), frac=0.05, return_sorted=False
            )
            s[0] -= min(s[0][one_tenth:-one_tenth])
            s[0] /= max(s[0][one_tenth:-one_tenth])
            s_down = []
            s_up = []

            # Loop through the spectra below the trace
            for k in range(n_down):
                start = k * bin_half_size
                end = start + bin_size * upsample_factor + 1
                # Note the start and end are counting up, while the
                # indices are becoming smaller.
                if use_arc:

                    s_down.append(
                        np.nansum(
                            [
                                arc_tmp[
                                    int(np.round(ref - end)) : int(
                                        np.round(ref - start)
                                    ),
                                    i,
                                ]
                                for i in x
                            ],
                            axis=1,
                        )
                    )

                else:

                    s_down.append(
                        np.nansum(
                            [
                                img_tmp[
                                    int(np.round(ref - end)) : int(
                                        np.round(ref - start)
                                    ),
                                    i,
                                ]
                                for i in x
                            ],
                            axis=1,
                        )
                    )

                s_down[k] -= lowess(
                    s_down[k],
                    np.arange(spec_size_tmp),
                    frac=0.05,
                    return_sorted=False,
                )
                s_down[k] -= min(s_down[k][one_tenth:-one_tenth])
                s_down[k] /= max(s_down[k][one_tenth:-one_tenth])

            # Loop through the spectra above the trace
            for k in range(n_up):
                start = k * bin_half_size
                end = start + bin_size * upsample_factor + 1

                if use_arc:

                    s_up.append(
                        np.nansum(
                            [
                                arc_tmp[
                                    int(np.round(ref + start)) : int(
                                        np.round(ref + end)
                                    ),
                                    i,
                                ]
                                for i in x
                            ],
                            axis=1,
                        )
                    )

                else:

                    s_up.append(
                        np.nansum(
                            [
                                img_tmp[
                                    int(np.round(ref + start)) : int(
                                        np.round(ref + end)
                                    ),
                                    i,
                                ]
                                for i in x
                            ],
                            axis=1,
                        )
                    )
                s_up[k] -= lowess(
                    s_up[k],
                    np.arange(spec_size_tmp),
                    frac=0.05,
                    return_sorted=False,
                )
                s_up[k] -= min(s_up[k][one_tenth:-one_tenth])
                s_up[k] /= max(s_up[k][one_tenth:-one_tenth])

            s_all = s_down[::-1] + s + s_up

            self.logger.info(
                "{} subspectra is used for cross-correlation.".format(s_all)
            )

            y_trace_upsampled = (
                np.arange(-n_down + 1, n_up + 1) * bin_half_size + ref
            )

            # correlate with the neighbouring slice to compute the shifts
            shift_upsampled = np.zeros_like(y_trace_upsampled)

            for i in range(1, len(s_all)):

                # Note: indice n_down is s
                corr = signal.correlate(
                    10.0 ** s_all[i][one_tenth:-one_tenth],
                    10.0 ** s_all[i - 1][one_tenth:-one_tenth],
                )
                shift_upsampled[i - 1 :] += (
                    spec_size_tmp
                    - 2 * one_tenth
                    - np.argwhere(corr == corr[np.argmax(corr)])[0]
                    - 1
                )

            # Turn the shift to relative to the spectrum
            shift_upsampled -= shift_upsampled[n_down]

            self.logger.info(
                "The upsampled y-coordinates of subspectra "
                "are: {} ".format(y_trace_upsampled)
                + "and the corresponding upsampled shifts "
                "are: {}.".format(shift_upsampled)
            )

            self.logger.info(
                "The y-coordinates of subspectra "
                "are: {} ".format(y_trace_upsampled / upsample_factor)
                + "and the corresponding shifts "
                "are: {}.".format(shift_upsampled / upsample_factor)
            )

            # fit the distortion in the spectral direction as a function
            # of y-pixel. The coeff is in the upsampled resolution
            coeff = np.polynomial.polynomial.polyfit(
                y_trace_upsampled,
                lowess(
                    shift_upsampled, y_trace_upsampled, return_sorted=False
                ),
                order,
            )
            self.logger.info(
                "Best fit polynomial for rectifying in the spatial direction."
                "is {}.".format(coeff)
            )

        # shift in the spectral direction, the shift is as a function
        # of distance from the trace at ref
        # For each row j (sort of a line of spectrum...)
        for j in range(len(img_tmp)):

            shift_j = np.polynomial.polynomial.polyval(j, coeff)

            if j % 10 == 0:
                self.logger.info(
                    "The shift at line j = {} is {}.".format(j, shift_j)
                )

            img_tmp[j] = np.roll(img_tmp[j], int(np.round(shift_j)))

            if self.arc is not None:

                arc_tmp[j] = np.roll(arc_tmp[j], int(np.round(shift_j)))

        self.rec_coeff = coeff
        self.rec_n_down = n_down
        self.rec_n_up = n_up
        self.rec_upsample_factor = upsample_factor
        self.rec_bin_size = bin_size
        self.rec_n_bin = n_bin
        self.rec_spline_order = spline_order
        self.rec_order = order
        self.img_rectified = ndimage.zoom(
            img_tmp, zoom=1.0 / upsample_factor, order=spline_order
        )
        if self.arc is not None:

            self.arc_rectified = ndimage.zoom(
                arc_tmp, zoom=1.0 / upsample_factor, order=spline_order
            )

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
                    zmin=np.nanpercentile(np.log10(self.img_rectified), 10),
                    zmax=np.nanpercentile(np.log10(self.img_rectified), 90),
                    xaxis="x",
                    yaxis="y",
                    colorbar=dict(title="log( e- count / s)"),
                )
            )
            if self.arc_rectified is not None:
                fig.add_trace(
                    go.Heatmap(
                        z=np.log10(self.arc_rectified),
                        colorscale="Viridis",
                        zmin=np.nanpercentile(
                            np.log10(self.arc_rectified), 10
                        ),
                        zmax=np.nanpercentile(
                            np.log10(self.arc_rectified), 90
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

                for t in fig_type_split:

                    if t == "iframe":

                        pio.write_html(
                            fig, filename + "." + t, auto_open=open_iframe
                        )

                    elif t in ["jpg", "png", "svg", "pdf"]:

                        pio.write_image(fig, filename + "." + t)

                    self.logger.info(
                        "Figure is saved to {} for the ".format(
                            filename + "." + t
                        )
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
        extraction_slice,
        extraction_bad_mask,
        sky_sigma,
        sky_width_dn,
        sky_width_up,
        sky_polyfit_order,
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

                self.logger.warning(
                    "sky_polyfit_order cannot be negative. sky "
                    "background is set to zero."
                )
                count_sky_extraction_slice = np.zeros_like(extraction_slice)

            self.logger.debug(
                "Background sky flux is "
                "{}.".format(count_sky_extraction_slice)
            )

        else:

            # get the indexes of the sky regions
            count_sky_extraction_slice = np.zeros_like(extraction_slice)
            self.logger.debug(
                "Sky region is not provided, backgound is set " "to zero."
            )

        return count_sky_extraction_slice

    def ap_extract(
        self,
        apwidth=7,
        skysep=3,
        skywidth=5,
        skydeg=1,
        sky_sigma=3.0,
        spec_id=None,
        optimal=True,
        algorithm="horne86",
        model="lowess",
        lowess_frac=0.1,
        lowess_it=3,
        lowess_delta=0.0,
        tolerance=1e-6,
        cosmicray_sigma=4.0,
        max_iter=99,
        forced=False,
        variances=None,
        npoly=21,
        polyspacing=1,
        pord=5,
        qmode="fast-linear",
        nreject=100,
        display=False,
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
            The summed count at each column about the trace. Note: is not
            sky subtracted!
        count_err: 1-d array
            the uncertainties of the count values
        count_sky: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract

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
            The ID corresponding to the spectrum1D object
        optimal: bool (Default: True)
            Set optimal extraction. (Default is True)
        algorithm: str (Default: 'horne86')
            Available algorithms are horne86 and marsh89
        model: str (Default: 'lowess')
            Choice of model: 'lowess' and 'gauss'.
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
        cosmicray_sigma: float (Deafult: 4.0)
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
            ).all(), "Some "
            "spec_id provided are not in the spectrum_list."

        else:

            spec_id = list(self.spectrum_list.keys())

        self.cosmicray_sigma = cosmicray_sigma

        to_return = []

        for j in spec_id:

            if isinstance(apwidth, int):

                # first do the aperture count
                width_dn = apwidth
                width_up = apwidth

            elif len(apwidth) == 2:

                width_dn = apwidth[0]
                width_up = apwidth[1]

            else:

                self.logger.error(
                    "apwidth can only be an int or a list "
                    + "of two ints. It is set to the default "
                    + "value to continue the extraction."
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
                    "skysep can only be an int or a list of "
                    + "two ints. It is set to the default "
                    + "value to continue the extraction."
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
                    "skywidth can only be an int or a list of "
                    + "two ints. It is set to the default value "
                    + "to continue the extraction."
                )
                sky_width_dn = 5
                sky_width_up = 5

            offset = 0

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

            # Sky extraction
            for i, pos in enumerate(spec.trace):

                itrace = int(pos)
                pix_frac = pos - itrace

                profile_start_idx = 0

                # fix width if trace is too close to the edge
                if itrace + width_up > self.spatial_size:

                    self.logger.info(
                        "Extration is over the upper edge of the detector "
                        "plane. Fixing indices. width_up is changed "
                        "from {} to {}.".format(
                            width_up, self.spatial_size - itrace - 1
                        )
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

                count_sky[i] = (
                    np.nansum(count_sky_source_slice)
                    - pix_frac * count_sky_source_slice[0]
                    - (1 - pix_frac) * count_sky_source_slice[-1]
                )

                self.img_residual[
                    source_pix, i
                ] = count_sky_source_slice.copy()

                self.logger.debug(
                    "count_sky at pixel {} is {}.".format(i, count_sky[i])
                )

                # if not optimal extraction or using marsh89, perform a
                # tophat extraction
                if not optimal or (optimal & (algorithm == "marsh89")):

                    (
                        count[i],
                        count_err[i],
                        is_optimal[i],
                    ) = self._tophat_extraction(
                        source_slice=source_slice,
                        sky_source_slice=count_sky_source_slice,
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

                    # source_pix is the native pixel position
                    # pos is the trace at the native pixel position
                    (
                        count[i],
                        count_err[i],
                        is_optimal[i],
                        profile[i][profile_start_idx:profile_end_idx],
                        var_temp,
                    ) = self._optimal_extraction_horne86(
                        pix=source_pix,
                        source_slice=source_slice,
                        sky=count_sky_source_slice,
                        mu=pos,
                        sigma=spec.trace_sigma[i],
                        tol=tolerance,
                        max_iter=max_iter,
                        readnoise=self.readnoise,
                        gain=self.gain,
                        cosmicray_sigma=self.cosmicray_sigma,
                        forced=forced,
                        variances=var_i,
                        model=model,
                        lowess_frac=lowess_frac,
                        lowess_it=lowess_it,
                        lowess_delta=lowess_delta,
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
                ) = self._optimal_extraction_marsh89(
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

            # All the extraction methods return signal and noise in the
            # same format
            count /= self.exptime
            count_err /= self.exptime
            count_sky /= self.exptime
            var /= self.exptime

            spec.add_aperture(
                width_dn, width_up, sep_dn, sep_up, sky_width_dn, sky_width_up
            )
            spec.add_count(list(count), list(count_err), list(count_sky))
            spec.add_variances(var)
            spec.add_profile(profile)
            spec.gain = self.gain
            spec.optimal_pixel = is_optimal
            spec.add_spectrum_header(self.header)

            self.logger.info("Spectrum extracted for spec_id: {}.".format(j))

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
                                min_trace
                                - width_dn
                                - sep_dn
                                - sky_width_dn
                                - 3,
                            ),
                            min(
                                max_trace
                                + width_up
                                + sep_up
                                + sky_width_up
                                + 3,
                                len(self.img[0]),
                            ),
                        ),
                        z=img_display,
                        colorscale="Viridis",
                        zmin=self.zmin,
                        zmax=self.zmax,
                        xaxis="x",
                        yaxis="y",
                        colorbar=dict(title="log( e- count / s )"),
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
                                    np.ones(1)
                                    * (spec.trace[0] - width_dn - 1),
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
                                        np.ones(1)
                                        * lower_redbox_upper_bound[0],
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
                                        np.ones(1)
                                        * upper_redbox_upper_bound[0],
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
                                    sigma_clip(count, sigma=5.0, masked=False)
                                ),
                                np.nanmin(
                                    sigma_clip(
                                        count_err, sigma=5.0, masked=False
                                    )
                                ),
                                np.nanmin(
                                    sigma_clip(
                                        count_sky, sigma=5.0, masked=False
                                    )
                                ),
                                1,
                            ),
                            max(np.nanmax(count), np.nanmax(count_sky)),
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

                            pio.write_html(
                                fig, save_path, auto_open=open_iframe
                            )

                        elif t in ["jpg", "png", "svg", "pdf"]:

                            pio.write_image(fig, save_path)

                        self.logger.info(
                            "Figure is saved to {} ".format(save_path)
                            + "for spec_id: {}.".format(j)
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

    def _tophat_extraction(
        self,
        source_slice,
        sky_source_slice,
        pix_frac,
        gain,
        sky_width_dn,
        sky_width_up,
        width_dn,
        width_up,
        source_bad_mask=None,
        sky_source_bad_mask=None,
    ):
        """
        Make sure the counts are the number of photoelectrons or an equivalent
        detector unit, and not counts per second.

        Parameters
        ----------
        source_slice: 1-d numpy array (N)
            The counts along the profile for aperture extraction.
        sky_source_slice: 1-d numpy array (M)
            Count of the fitted sky along the pix, has to be the same
            length as pix
        pix_frac: float
            The decimal places of the centroid.
        gain: float
            Detector gain, in electrons per ADU
        sky_width_dn: int
            Number of pixels used for sky modelling on the lower side of the
            spectrum.
        sky_width_up: int
            Number of pixels used for sky modelling on the upper side of the
            spectrum.
        width_dn: int
            Number of pixels used for aperture extraction on the lower side
            of the spectrum.
        width_up: int
            Number of pixels used for aperture extraction on the upper side
            of the spectrum.
        source_bad_mask: 1-d numpy array (N, default: None)
            Masking the unusable pixels for extraction.
        sky_source_bad_mask: 1-d numpy array (M, default: None)
            Masking the unusable pixels for sky subtraction.

        """

        if source_bad_mask is not None:

            source_slice = source_slice[source_bad_mask]

        if source_bad_mask is not None:

            sky_source_slice = source_slice[sky_source_bad_mask]

        # Get the total count
        source_plus_sky = (
            np.nansum(source_slice)
            - pix_frac * source_slice[0]
            - (1 - pix_frac) * source_slice[-1]
        )

        # finally, compute the error in this pixel
        # standarddev in the background data
        sigB = np.nanstd(sky_source_slice)
        sky = (
            np.nansum(sky_source_slice)
            - pix_frac * sky_source_slice[0]
            - (1 - pix_frac) * sky_source_slice[-1]
        )

        # number of bkgd pixels
        nB = sky_width_dn + sky_width_up - np.sum(np.isnan(sky_source_slice))
        # number of aperture pixels
        nA = width_dn + width_up - np.sum(np.isnan(source_slice))

        # Based on aperture phot err description by F. Masci,
        # Caltech:
        # http://wise2.ipac.caltech.edu/staff/fmasci/
        #   ApPhotUncert.pdf
        # All the counts are in per second already, so need to
        # multiply by the exposure time when computing the
        # uncertainty
        signal = source_plus_sky - sky
        noise = np.sqrt(signal / gain + (nA + nA**2.0 / nB) * (sigB**2.0))

        self.logger.debug(
            "The signal and noise from the tophat extraction "
            "are {} and {}.".format(signal, noise)
        )

        return signal, noise, False

    def _optimal_extraction_horne86(
        self,
        pix,
        source_slice,
        sky,
        mu,
        sigma,
        tol=1e-6,
        max_iter=99,
        gain=1.0,
        readnoise=0.0,
        cosmicray_sigma=5.0,
        forced=False,
        variances=None,
        model="lowess",
        lowess_frac=0.1,
        lowess_it=3,
        lowess_delta=0.0,
        bad_mask=None,
    ):
        """
        Make sure the counts are the number of photoelectrons or an equivalent
        detector unit, and not counts per second or ADU.

        Iterate to get the optimal signal. Following the algorithm on
        Horne, 1986, PASP, 98, 609 (1986PASP...98..609H). The 'steps' in the
        inline comments are in reference to this article.

        The LOWESS setting can be found at:
        https://www.statsmodels.org/dev/generated/
            statsmodels.nonparametric.smoothers_lowess.lowess.html

        Parameters
        ----------
        pix: 1D numpy.ndarray (N)
            pixel number along the spatial direction
        source_slice: 1D numpy.ndarray (N)
            The counts along the profile for extraction, including the sky
            regions to be fitted and subtracted from. (NOT count per second)
        sky: 1D numpy.ndarray (N)
            Count of the fitted sky along the pix, has to be the same
            length as pix
        mu: float
            The center of the Gaussian
        sigma: float
            The width of the Gaussian
        tol: float
            The tolerance limit for the covergence
        max_iter: int
            The maximum number of iteration in the optimal extraction
        gain: float (Default: 1.0)
            Detector gain, in electrons per ADU
        readnoise: float
            Detector readnoise, in electrons.
        cosmicray_sigma: int (Default: 5)
            Sigma-clipping threshold for cleaning & cosmic ray rejection.
        forced: bool
            Forced extraction with the given weights.
        variances: 1D numpy.ndarray (N)
            The 1/weights of used for optimal extraction, has to be the
            same length as the pix. Only used if forced is True.
        model: str (Default: 'lowess')
            Choice of 'gauss' and 'lowess' for gaussian profile and a LOWESS
            local regression fitting.
        lowess_frac: float (Default: 0.1)
            The fraction of the data used when estimating each y-value.
        lowess_it: int (Default: 3)
            The number of residual-based reweightings to perform.
        lowess_delta: float (Default: 0.0)
            Distance within which to use linear-interpolation instead of
            weighted regression.
        bad_mask: list or None (Default: None)
            Mask of the bad or usable pixels.

        Returns
        -------
        signal: float
            The optimal signal.
        noise: float
            The noise associated with the optimal signal.
        is_optimal: bool
            List indicating whether the extraction at that pixel was
            optimal or not. True = optimal, False = suboptimal.
        P: numpy array
            The line spread function of the extraction
        var_f: float
            The variance in the extraction.

        """

        # step 2 - initial variance estimates
        var1 = readnoise**2.0 + np.abs(source_slice) / gain

        # step 4a - extract standard spectrum
        f = source_slice - sky
        f[f < 0] = 0.0
        f1 = np.nansum(f)

        # step 4b - variance of standard spectrum
        v1 = 1.0 / np.nansum(1.0 / var1)

        # step 5 - construct the spatial profile
        if not np.in1d(model, ["gauss", "lowess"]):

            self.logger.error(
                "The provided model has to be gauss or lowess, "
                "{} is given. lowess is used.".format(model)
            )
            model = "lowess"

        f_diff = 1
        v_diff = 1
        i = 0
        is_optimal = True

        while (f_diff > tol) | (v_diff > tol):

            if model == "gauss":

                P = self._gaus(pix, 1.0, 0.0, mu, sigma)

            else:

                P = lowess(
                    f,
                    pix,
                    frac=lowess_frac,
                    it=lowess_it,
                    delta=lowess_delta,
                    return_sorted=False,
                )

            P[P < 0] = 0.0
            P /= np.nansum(P)

            mask_cr = np.ones(len(P), dtype=bool)
            mask_cr = mask_cr & ~bad_mask.astype(bool)

            if forced:

                var_f = variances

            f0 = f1
            v0 = v1

            # step 6 - revise variance estimates
            # var_f is the V in Horne87
            if not forced:

                var_f = readnoise**2.0 + np.abs(P * f0 + sky) / gain

            # step 7 - cosmic ray mask, only start considering after the
            # 2nd iteration. 1 pixel is masked at a time until convergence,
            # once the pixel is masked, it will stay masked.
            if i > 1:

                ratio = (cosmicray_sigma**2.0 * var_f) / (f - P * f0) ** 2.0

                if (ratio > 1).any():

                    mask_cr[np.argmax(ratio)] = False

            denom = np.nansum((P**2.0 / var_f)[mask_cr])

            # step 8a - extract optimal signal
            f1 = np.nansum((P * f / var_f)[mask_cr]) / denom

            # step 8b - variance of optimal signal
            v1 = np.nansum(P[mask_cr]) / denom

            f_diff = abs((f1 - f0) / f0)
            v_diff = abs((v1 - v0) / v0)

            i += 1

            if i == int(max_iter):

                is_optimal = False
                break

        signal = f1
        noise = np.sqrt(v1)

        self.logger.debug(
            "The signal and noise from the tophat extraction "
            "are {} and {}.".format(signal, noise)
        )

        return signal, noise, is_optimal, P, var_f

    def _optimal_extraction_marsh89(
        self,
        frame,
        residual_frame,
        variance,
        trace,
        spectrum=None,
        readnoise=0.0,
        apwidth=7,
        goodpixelmask=None,
        npoly=21,
        polyspacing=1,
        pord=2,
        cosmicray_sigma=5,
        qmode="slow-nearest",
        nreject=100,
    ):
        """
        Optimally extract curved spectra taken and updated from
        Ian Crossfield's code

        https://people.ucsc.edu/~ianc/python/_modules/spec.html#superExtract,
        following Marsh 1989.

        Parameters
        ----------
        frame: 2-d Numpy array (M, N)
            The calibrated frame from which to extract spectrum. In units
            of electrons count.
        residual_frame: 2-d Numpy array (M, N)
            The sky background only frame.
        variance: 2-d Numpy array (M, N)
            Variances of pixel values in 'frame'.
        trace: 1-d numpy array (N)
            :ocation of spectral trace.
        spectrum: 1-d numpy array (M) (Default: None)
            The extracted spectrum for initial guess.
        gain: float (Default: 1.0)
            Detector gain, in electrons per ADU
        readnoise: float (Default: 0.0)
            Detector readnoise, in electrons.
        apwidth: int or list of int (default: 7)
            The size of the aperture for extraction.
        goodpixelmask : 2-d numpy array (M, N) (Default: None)
            Equals 0 for bad pixels, 1 for good pixels
        npoly: int (Default: 21)
            Number of profile to be use for polynomial fitting to evaluate
            (Marsh's "K"). For symmetry, this should be odd.
        polyspacing: float (Default: 1)
            Spacing between profile polynomials, in pixels. (Marsh's "S").
            A few cursory tests suggests that the extraction precision
            (in the high S/N case) scales as S^-2 -- but the code slows down
            as S^2.
        pord: int (Default: 2)
            Order of profile polynomials; 1 = linear, etc.
        cosmicray_sigma: int (Default: 5)
            Sigma-clipping threshold for cleaning & cosmic-ray rejection.
        qmode: str (Default: 'slow-nearest')
            How to compute Marsh's Q-matrix. Valid inputs are 'fast-linear',
            'slow-linear', 'fast-nearest', and 'slow-nearest'. These select
            between various methods of integrating the nearest-neighbor or
            linear interpolation schemes as described by Marsh; the 'linear'
            methods are preferred for accuracy. Use 'slow' if you are
            running out of memory when using the 'fast' array-based methods.
        nreject: int (Default: 100)
            Number of outlier-pixels to reject at each iteration.

        Returns
        -------
        spectrum_marsh:
            The optimal signal.
        spectrum_err_marsh:
            The noise associated with the optimal signal.
        is_optimal:
            List indicating whether the extraction at that pixel was
            optimal or not (this list is always all optimal).
        profile:
            The line spread functions of the extraction
        variance0:
            The variance in the extraction.

        """

        frame = frame.transpose()
        residual_frame = residual_frame.transpose()
        variance = variance.transpose()

        if isinstance(apwidth, (float, int)):

            # first do the aperture count
            width_dn = apwidth
            width_up = apwidth

        elif len(apwidth) == 2:

            width_dn = apwidth[0]
            width_up = apwidth[1]

        else:

            self.logger.error(
                "apwidth can only be an int or a list "
                + "of two ints. It is set to the default "
                + "value to continue the extraction."
            )
            width_dn = 7
            width_up = 7

        if goodpixelmask is not None:
            goodpixelmask = goodpixelmask.transpose()
            goodpixelmask = np.array(goodpixelmask, copy=True).astype(bool)
        else:
            goodpixelmask = np.ones_like(frame, dtype=bool)

        goodpixelmask *= np.isfinite(frame) * np.isfinite(variance)

        variance[~goodpixelmask] = frame[goodpixelmask].max() * 1e9
        spectral_size, spatial_size = frame.shape

        # (my 3a: mask any bad values)
        bad_residual_frame_mask = ~np.isfinite(residual_frame)
        residual_frame[bad_residual_frame_mask] = 0.0
        if np.any(bad_residual_frame_mask.nonzero()):
            self.logger.warning(
                "Found bad residual_frame values at: {}".format(
                    bad_residual_frame_mask.nonzero()
                )
            )

        skysubFrame = frame - residual_frame
        """
        # Interpolate and fix bad pixels for extraction of standard
        # spectrum -- otherwise there can be 'holes' in the spectrum from
        # ill-placed bad pixels.
        fixSkysubFrame = bfixpix(skysubFrame, ~goodpixelmask, n=8, retdat=True)
        """

        # Define new indices (in Marsh's appendix):
        N = pord + 1
        mm = np.tile(np.arange(N).reshape(N, 1), (npoly)).ravel()
        nn = mm.copy()
        ll = np.tile(np.arange(npoly), N)
        kk = ll.copy()
        pp = N * ll + mm
        qq = N * kk + nn

        ii = np.arange(spatial_size)  # column (i.e., spatial direction)
        jjnorm = np.linspace(-1, 1, spectral_size)  # normalized X-coordinate
        jjnorm_pow = jjnorm.reshape(1, 1, spectral_size) ** (
            np.arange(2 * N - 1).reshape(2 * N - 1, 1, 1)
        )

        # Marsh eq. 9, defining centers of each polynomial:
        constant = 0.0  # What is it for???
        poly_centers = (
            np.array(trace).reshape(spectral_size, 1)
            + polyspacing * np.arange(-npoly / 2 + 1, npoly / 2 + 1)
            + constant
        )

        # Marsh eq. 11, defining Q_kij    (via nearest-neighbor interpolation)
        #    Q_kij =  max(0, min(S, (S+1)/2 - abs(x_kj - i)))
        if qmode == "fast-nearest":  # Array-based nearest-neighbor mode.
            Q = np.array(
                [
                    np.zeros((npoly, spatial_size, spectral_size)),
                    np.array(
                        [
                            polyspacing
                            * np.ones((npoly, spatial_size, spectral_size)),
                            0.5 * (polyspacing + 1)
                            - np.abs(
                                (
                                    poly_centers
                                    - ii.reshape(spatial_size, 1, 1)
                                ).transpose(2, 0, 1)
                            ),
                        ]
                    ).min(0),
                ]
            ).max(0)

        elif qmode == "slow-linear":  # Code is a mess, but it works.
            invs = 1.0 / polyspacing
            poly_centers_over_s = poly_centers / polyspacing
            xps_mat = poly_centers + polyspacing
            xms_mat = poly_centers - polyspacing
            Q = np.zeros((npoly, spatial_size, spectral_size))
            for i in range(spatial_size):
                ip05 = i + 0.5
                im05 = i - 0.5
                for j in range(spectral_size):
                    for k in range(npoly):
                        xkj = poly_centers[j, k]
                        xkjs = poly_centers_over_s[j, k]
                        # xkj + polyspacing
                        xps = xps_mat[j, k]
                        # xkj - polyspacing
                        xms = xms_mat[j, k]

                        if (ip05 <= xms) or (im05 >= xps):
                            qval = 0.0
                        elif (im05) > xkj:
                            lim1 = im05
                            lim2 = min(ip05, xps)
                            qval = (lim2 - lim1) * (
                                1.0 + xkjs - 0.5 * invs * (lim1 + lim2)
                            )
                        elif (ip05) < xkj:
                            lim1 = max(im05, xms)
                            lim2 = ip05
                            qval = (lim2 - lim1) * (
                                1.0 - xkjs + 0.5 * invs * (lim1 + lim2)
                            )
                        else:
                            lim1 = max(im05, xms)
                            lim2 = min(ip05, xps)
                            qval = (
                                lim2
                                - lim1
                                + invs
                                * (
                                    xkj * (-xkj + lim1 + lim2)
                                    - 0.5 * (lim1 * lim1 + lim2 * lim2)
                                )
                            )
                        Q[k, i, j] = max(0, qval)

        # Code is a mess, but it's faster than 'slow' mode
        elif qmode == "fast-linear":
            invs = 1.0 / polyspacing
            xps_mat = poly_centers + polyspacing
            Q = np.zeros((npoly, spatial_size, spectral_size))
            for j in range(spectral_size):
                xkj_vec = np.tile(
                    poly_centers[j, :].reshape(npoly, 1), (1, spatial_size)
                )
                xps_vec = np.tile(
                    xps_mat[j, :].reshape(npoly, 1), (1, spatial_size)
                )
                xms_vec = xps_vec - 2 * polyspacing

                ip05_vec = np.tile(np.arange(spatial_size) + 0.5, (npoly, 1))
                im05_vec = ip05_vec - 1
                ind00 = (ip05_vec <= xms_vec) + (im05_vec >= xps_vec)
                ind11 = (im05_vec > xkj_vec) * ~ind00
                ind22 = (ip05_vec < xkj_vec) * ~ind00
                ind33 = ~(ind00 + ind11 + ind22)
                ind11 = ind11.nonzero()
                ind22 = ind22.nonzero()
                ind33 = ind33.nonzero()

                n_ind11 = len(ind11[0])
                n_ind22 = len(ind22[0])
                n_ind33 = len(ind33[0])

                if n_ind11 > 0:
                    ind11_3d = ind11 + (np.ones(n_ind11, dtype=int) * j,)
                    lim2_ind11 = np.array(
                        (ip05_vec[ind11], xps_vec[ind11])
                    ).min(0)
                    Q[ind11_3d] = (
                        (lim2_ind11 - im05_vec[ind11])
                        * invs
                        * (
                            polyspacing
                            + xkj_vec[ind11]
                            - 0.5 * (im05_vec[ind11] + lim2_ind11)
                        )
                    )

                if n_ind22 > 0:
                    ind22_3d = ind22 + (np.ones(n_ind22, dtype=int) * j,)
                    lim1_ind22 = np.array(
                        (im05_vec[ind22], xms_vec[ind22])
                    ).max(0)
                    Q[ind22_3d] = (
                        (ip05_vec[ind22] - lim1_ind22)
                        * invs
                        * (
                            polyspacing
                            - xkj_vec[ind22]
                            + 0.5 * (ip05_vec[ind22] + lim1_ind22)
                        )
                    )

                if n_ind33 > 0:
                    ind33_3d = ind33 + (np.ones(n_ind33, dtype=int) * j,)
                    lim1_ind33 = np.array(
                        (im05_vec[ind33], xms_vec[ind33])
                    ).max(0)
                    lim2_ind33 = np.array(
                        (ip05_vec[ind33], xps_vec[ind33])
                    ).min(0)
                    Q[ind33_3d] = (lim2_ind33 - lim1_ind33) + invs * (
                        xkj_vec[ind33]
                        * (-xkj_vec[ind33] + lim1_ind33 + lim2_ind33)
                        - 0.5
                        * (lim1_ind33 * lim1_ind33 + lim2_ind33 * lim2_ind33)
                    )

        # 'slow' Loop-based nearest-neighbor mode: requires less memory
        else:
            Q = np.zeros((npoly, spatial_size, spectral_size))
            for k in range(npoly):
                for i in range(spatial_size):
                    for j in range(spectral_size):
                        Q[k, i, j] = max(
                            0,
                            min(
                                polyspacing,
                                0.5 * (polyspacing + 1)
                                - np.abs(poly_centers[j, k] - i),
                            ),
                        )

        # Some quick math to find out which dat columns are important, and
        # which contain no useful spectral information:
        Qmask = Q.sum(0).transpose() > 0
        Qind = Qmask.transpose().nonzero()
        Q_cols = [Qind[0].min(), Qind[0].max()]
        Qsm = Q[:, Q_cols[0] : Q_cols[1] + 1, :]

        # Prepar to iteratively clip outliers
        self.logger.info("Looking for bad pixel outliers.")
        newBadPixels = True
        i = -1
        while newBadPixels:
            i += 1
            self.logger.debug("Beginning iteration {}.".format(i))

            # Compute pixel fractions (Marsh Eq. 5):
            #     (Note that values outside the desired polynomial region
            #     have Q=0, and so do not contribute to the fit)
            invEvariance = (
                np.array(spectrum).reshape(spectral_size, 1) ** 2 / variance
            ).transpose()
            weightedE = (
                skysubFrame
                * np.array(spectrum).reshape(spectral_size, 1)
                / variance
            ).transpose()  # E / var_E
            invEvariance_subset = invEvariance[Q_cols[0] : Q_cols[1] + 1, :]

            # Define X vector (Marsh Eq. A3):
            X = np.zeros(N * npoly)
            for q in qq:
                X[q] = (
                    weightedE[Q_cols[0] : Q_cols[1] + 1, :]
                    * Qsm[kk[q], :, :]
                    * jjnorm_pow[nn[q]]
                ).sum()
            """
            # The unoptimised way to compute the X vector:
            X2 = np.zeros(N * npoly)
            for n in nn:
                for k in kk:
                    q = N * k + n
                    xtot = 0.
                    for i in ii:
                        for j in jj:
                            xtot += E[i, j] * Q[k, i, j] * (
                                jjnorm[j]**n) / Evariance[i, j]
                    X2[q] = xtot
            """

            # Define C matrix (Marsh Eq. A3)
            C = np.zeros((N * npoly, N * npoly))

            # C-matrix computation buffer (to be sure we don't miss any pixels)
            buffer = 1.1

            # Compute *every* element of C (though most equal zero!)
            for p in pp:
                qp = Qsm[ll[p], :, :]
                for q in qq:
                    #  Check that we need to compute C:
                    if np.abs(kk[q] - ll[p]) <= (1.0 / polyspacing + buffer):
                        if q >= p:
                            # Only compute over non-zero columns:
                            C[q, p] = (
                                Qsm[kk[q], :, :]
                                * qp
                                * jjnorm_pow[nn[q] + mm[p]]
                                * invEvariance_subset
                            ).sum()
                        if q > p:
                            C[p, q] = C[q, p]

            # Solve for the profile-polynomial coefficients (Marsh Eq. A4):
            if np.abs(np.linalg.det(C)) < 1e-10:
                Bsoln = np.dot(np.linalg.pinv(C), X)
            else:
                Bsoln = np.linalg.solve(C, X)

            Asoln = Bsoln.reshape(N, npoly).transpose()

            # Define G_kj, the profile-defining polynomial profiles
            # (Marsh Eq. 8)
            Gsoln = np.zeros((npoly, spectral_size))
            for n in range(npoly):
                Gsoln[n] = np.polyval(
                    Asoln[n, ::-1], jjnorm
                )  # reorder polynomial coef.

            # Compute the profile (Marsh eq. 6) and normalize it:
            profile = np.zeros((spatial_size, spectral_size))
            for i in range(spatial_size):
                profile[i, :] = (Q[:, i, :] * Gsoln).sum(0)

            self.logger.debug(profile)
            if profile.min() < 0:
                profile[profile < 0] = 0.0
            profile /= np.nansum(profile, axis=0)
            profile[~np.isfinite(profile)] = 0.0

            # Step6: Revise variance estimates
            modelSpectrum = (
                np.array(spectrum).reshape(spectral_size, 1)
                * profile.transpose()
            )
            modelData = modelSpectrum + residual_frame
            variance0 = np.abs(modelData) + readnoise**2
            variance = variance0 / (
                goodpixelmask + 1e-9
            )  # De-weight bad pixels, avoiding infinite variance

            outlierVariances = (frame - modelData) ** 2 / variance

            if outlierVariances.max() > cosmicray_sigma**2:
                newBadPixels = True
                # nreject-counting on pixels within the spectral trace
                maxRejectedValue = max(
                    cosmicray_sigma**2,
                    np.sort(outlierVariances[Qmask])[-nreject],
                )
                worstOutliers = (
                    outlierVariances >= maxRejectedValue
                ).nonzero()
                goodpixelmask[worstOutliers] = False
                numberRejected = len(worstOutliers[0])
            else:
                newBadPixels = False
                numberRejected = 0

            self.logger.info(
                "Rejected {} pixels in this iteration.".format(numberRejected)
            )

            # Optimal Spectral Extraction: (Horne, Step 8)
            spectrum_marsh = np.zeros(spectral_size)
            spectrum_err_marsh = np.zeros(spectral_size)
            is_optimal = np.zeros(spectral_size)

            for i in range(spectral_size):
                aperture = np.arange(
                    int(trace[i]) - width_dn, int(trace[i]) + width_up + 1
                ).astype(int)

                # Horne86 notation
                P = profile[aperture, i]
                V = variance0[i, aperture]
                D = skysubFrame[i, aperture]

                denom = np.nansum(P**2.0 / V)

                if denom == 0:
                    spectrum_marsh[i] = 0.0
                    spectrum_err_marsh[i] = 9e9
                else:
                    spectrum_marsh[i] = np.nansum(P / V * D) / denom
                    spectrum_err_marsh[i] = np.sqrt(np.nansum(P) / denom)
                    is_optimal[i] = True

        spectrum_marsh = spectrum_marsh
        spectrum_err_marsh = spectrum_err_marsh

        self.logger.debug(
            "The signal and noise from the tophat extraction "
            "are {} and {}.".format(spectrum_marsh, spectrum_err_marsh)
        )

        return (
            spectrum_marsh,
            spectrum_err_marsh,
            is_optimal,
            profile,
            variance0,
        )

    def inspect_extracted_spectrum(
        self,
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
        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
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
            ).all(), "Some "
            "spec_id provided are not in the spectrum_list."

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

                for t in fig_type_split:

                    save_path = filename + "_" + str(j) + "." + t

                    if t == "iframe":

                        pio.write_html(fig, save_path, auto_open=open_iframe)

                    elif t in ["jpg", "png", "svg", "pdf"]:

                        pio.write_image(fig, save_path)

                    self.logger.info(
                        "Figure is saved to {} ".format(save_path)
                        + "for spec_id: {}.".format(j)
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
        log=True,
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
                title="Spectral Direction / pixel",
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

    def extract_arc_spec(
        self,
        spec_id=None,
        spec_width=None,
        display=False,
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
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
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

        """

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if spec_id is not None:

            if spec_id not in list(self.spectrum_list.keys()):

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

                fig = go.Figure(
                    layout=dict(autosize=False, height=height, width=width)
                )

                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len_trace),
                        y=arc_spec,
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
                        range=[0, max(arc_spec)],
                        title="e- count / s",
                    ),
                    hovermode="closest",
                    showlegend=False,
                )

                if filename is None:

                    filename = "arc_spec"

                if save_fig:

                    fig_type_split = fig_type.split("+")

                    for t in fig_type_split:

                        save_path = filename + "_" + str(i) + "." + t

                        if t == "iframe":

                            pio.write_html(
                                fig, save_path, auto_open=open_iframe
                            )

                        elif t in ["jpg", "png", "svg", "pdf"]:

                            pio.write_image(fig, save_path)

                        self.logger.info(
                            "Figure is saved to {} ".format(save_path)
                            + "for spec_id: {}.".format(i)
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

    def create_fits(self, output, recreate=False, empty_primary_hdu=True):
        """
        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strs are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 3 HDUs
                    Count, uncertainty, and sky (pixel)
                weight_map: 1 HDU
                    Weight (pixel)
                arc_spec: 3 HDUs
                    1D arc spectrum, arc line pixels, and arc line effective
                    pixels

        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank

        """

        for i in output.split("+"):

            if i not in ["trace", "count"]:

                error_msg = "{} is not a valid output.".format(i)
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        # Save each trace as a separate FITS file
        for i in range(len(self.spectrum_list)):

            self.spectrum_list[i].create_fits(
                output=output,
                recreate=recreate,
                empty_primary_hdu=empty_primary_hdu,
            )
            self.logger.info("FITS file is created for spec_id: {}.".format(i))

    def save_fits(
        self,
        output="trace+count",
        filename="TwoDSpecExtracted",
        overwrite=False,
        recreate=False,
        empty_primary_hdu=True,
    ):
        """
        Save the reduced image to disk.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strs are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 3 HDUs
                    Count, uncertainty, and sky (pixel)
                weight_map: 1 HDU
                    Weight (pixel)
                arc_spec: 3 HDUs
                    1D arc spectrum, arc line pixels, and arc line effective
                    pixels

        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        overwrite: bool
            Default is False.
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank

        """

        filename = os.path.splitext(filename)[0]

        for i in output.split("+"):

            if i not in ["trace", "count"]:

                error_msg = "{} is not a valid output.".format(i)
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        # Save each trace as a separate FITS file
        for i in range(len(self.spectrum_list)):

            filename_i = filename + "_" + output + "_" + str(i)

            self.spectrum_list[i].save_fits(
                output=output,
                filename=filename_i,
                overwrite=overwrite,
                recreate=recreate,
                empty_primary_hdu=empty_primary_hdu,
            )
            self.logger.info(
                "FITS file is saved to {} ".format(filename_i)
                + "for spec_id: {}.".format(i)
            )
