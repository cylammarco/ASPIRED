import datetime
import logging
import os
from itertools import chain

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astroscrappy import detect_cosmics
from plotly import graph_objects as go
from plotly import io as pio
from scipy import signal
from scipy.optimize import curve_fit
from spectres import spectres

from .image_reduction import ImageReduction
from .spectrum1D import Spectrum1D

__all__ = ['TwoDSpec']


class TwoDSpec:
    '''
    This is a class for processing a 2D spectral image.

    '''
    def __init__(self,
                 data=None,
                 header=None,
                 verbose=True,
                 logger_name='TwoDSpec',
                 log_level='WARNING',
                 log_file_folder='default',
                 log_file_name='default',
                 **kwargs):
        '''
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
        verbose: boolean (Default: True)
            Set to False to suppress all verbose warnings, except for
            critical failure.
        logger_name: str (Default: TwoDSpec)
            This will set the name of the logger, if the name is used already,
            it will reference to the existing logger. This will be the
            first part of the default log file name unless log_file_name is
            provided.
        log_level: str (Default: WARNING)
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
        log_file_name: None or str (Default: "default")
            File name of the log, set to None to print to screen only.
        **kwargs: keyword arguments (Default: see set_properties())
            see set_properties().

        '''

        # Set-up logger
        logger = logging.getLogger()
        if (log_level == "CRITICAL") or (not verbose):
            logging.basicConfig(level=logging.CRITICAL)
        elif log_level == "ERROR":
            logging.basicConfig(level=logging.ERROR)
        elif log_level == "WARNING":
            logging.basicConfig(level=logging.WARNING)
        elif log_level == "INFO":
            logging.basicConfig(level=logging.INFO)
        elif log_level == "DEBUG":
            logging.basicConfig(level=logging.DEBUG)
        else:
            raise ValueError('Unknonw logging level.')
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] '
            '%(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S')

        if log_file_name is None:
            # Only print log to screen
            handler = logging.StreamHandler()
        else:
            if log_file_name == 'default':
                log_file_name = '{}_{}.log'.format(
                    logger_name,
                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            # Save log to file
            if log_file_folder == 'default':
                log_file_folder = ''

            handler = logging.FileHandler(
                os.path.join(log_file_folder, log_file_name), 'a+')

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.add_data(data, header)
        self.spectrum_list = {}

        self.saxis = 1
        self.waxis = 0

        self.spatial_mask = (1, )
        self.spec_mask = (1, )
        self.flip = False
        self.cosmicray = False
        self.cosmicray_sigma = 5.

        # Default values if not supplied or cannot be automatically identified
        # from the header
        self.readnoise = 0.
        self.gain = 1.
        self.seeing = 1.
        self.exptime = 1.

        self.verbose = verbose
        self.logger_name = logger_name
        self.log_level = log_level
        self.log_file_folder = log_file_folder
        self.log_file_name = log_file_name

        # Default keywords to be searched in the order in the list
        self.readnoise_keyword = ['RDNOISE', 'RNOISE', 'RN']
        self.gain_keyword = ['GAIN']
        self.seeing_keyword = ['SEEING', 'L1SEEING', 'ESTSEE']
        self.exptime_keyword = [
            'XPOSURE', 'EXPOSURE', 'EXPTIME', 'EXPOSED', 'TELAPSED', 'ELAPSED'
        ]

        self.set_properties(**kwargs)

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
            logging.info('An numpy array is loaded as data.')
            self.header = header

        # If it is a fits.hdu.hdulist.HDUList object
        elif isinstance(data, fits.hdu.hdulist.HDUList):

            self.img = data[0].data
            self.header = data[0].header
            logging.warning('An HDU list is provided, only the first '
                            'HDU will be read.')

        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(data, fits.hdu.image.PrimaryHDU) or isinstance(
                data, fits.hdu.image.ImageHDU):

            self.img = data.data
            self.header = data.header
            logging.info('A PrimaryHDU is loaded as data.')

        # If it is an ImageReduction object
        elif isinstance(data, ImageReduction):

            # If the data is not reduced, reduce it here. Error handling is
            # done by the ImageReduction class
            if data.image_fits is None:

                data._create_image_fits()

            self.img = data.image_fits.data
            self.header = data.image_fits.header
            self.arc = data.arc_master
            self.arc_header = data.arc_header

        # If a filepath is provided
        elif isinstance(data, str):

            # If HDU number is provided
            if data[-1] == ']':

                filepath, hdunum = data.split('[')
                hdunum = hdunum[:-1]

            # If not, assume the HDU idnex is 0
            else:

                filepath = data
                hdunum = 0

            # Load the file and dereference it afterwards
            fitsfile_tmp = fits.open(filepath)[hdunum]
            self.img = fitsfile_tmp.data
            self.header = fitsfile_tmp.header
            fitsfile_tmp = None

        elif data is None:

            self.img = None
            self.header = None

        else:

            error_msg = 'Please provide a numpy array, an ' +\
                'astropy.io.fits.hdu.image.PrimaryHDU object ' +\
                'or an ImageReduction object.'
            logging.critical(error_msg)
            raise TypeError(error_msg)

        self.img_residual = self.img.copy()

    def set_properties(self,
                       saxis=None,
                       variance=None,
                       spatial_mask=None,
                       spec_mask=None,
                       flip=None,
                       cosmicray=None,
                       cosmicray_sigma=None,
                       readnoise=None,
                       gain=None,
                       seeing=None,
                       exptime=None,
                       verbose=None):
        '''
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
        |   top  | bottom |   0   | False |
        +--------+--------+-------+-------+
        | bottom |   top  |   0   |  True |
        +--------+--------+-------+-------+

        Spectra are sorted by their brightness. If there are multiple spectra
        on the image, and the target is not the brightest source, use at least
        the number of spectra visible to eye and pick the one required later.
        The default automated outputs is the brightest one, which is the
        most common case for images from a long-slit spectrograph.

        Parameters
        ----------
        saxis: int
            Spectral direction, 0 for vertical, 1 for horizontal.
            (Default is 1)
        variance: 2D numpy array (M, N)
            The per-pixel-variance of the frame.
        spatial_mask: 1D numpy array (N)
            Mask in the spatial direction, can be the indices of the pixels
            to be included (size <N) or a 1D numpy array of True/False (size N)
            (Default is (1,) i.e. keep everything)
        spec_mask: 1D numpy array (M)
            Mask in the spectral direction, can be the indices of the pixels
            to be included (size <M) or a 1D numpy array of True/False (size M)
            (Default is (1,) i.e. keep everything)
        flip: boolean
            If the frame has to be left-right flipped, set to True.
            (Deafult is False)
        cosmicray: boolean
            Set to True to apply cosmic ray rejection by sigma clipping with
            astroscrappy if available, otherwise a 2D median filter of size 5
            would be used. (default is True)
        cosmicray_sigma: float
            Cosmic ray sigma clipping limit (Deafult is 5.0)
        readnoise: float
            Readnoise of the detector, not important if noise estimation is
            not needed.
            (Deafult is None, which will be replaced with 0.0)
        gain: float
            Gain of the detector, not important if noise estimation is
            not needed.
            (Deafult is None, which will be replaced with 1.0)
        seeing: float
            Seeing in unit of arcsec, use as the first guess of the line
            spread function of the spectra.
            (Deafult is None, which will be replaced with 1.0)
        exptime: float
            Esposure time for the observation, not important if absolute flux
            calibration is not needed.
            (Deafult is None, which will be replaced with 1.0)
        verbose: boolean
            Set to False to suppress all verbose warnings, except for
            critical failure.

        '''

        if saxis is not None:

            self.saxis = saxis

            if self.saxis == 1:

                self.waxis = 0

            elif self.saxis == 0:

                self.waxis = 1

            else:

                self.saxis = 0
                logging.error(
                    "saxis can only be 0 or 1, {} is ".format(saxis) +
                    "given. It is set to 0.")

        if spatial_mask is not None:

            self.spatial_mask = spatial_mask

        if spec_mask is not None:

            self.spec_mask = spec_mask

        if flip is not None:

            self.flip = flip

        # cosmic ray rejection
        if cosmicray is not None:

            self.cosmicray = cosmicray

            self.img = detect_cosmics(self.img,
                                      sigclip=self.cosmicray_sigma,
                                      readnoise=self.readnoise,
                                      gain=self.gain,
                                      fsmode='convolve',
                                      psfmodel='gaussy',
                                      psfsize=31,
                                      psffwhm=self.seeing)[1]

        if cosmicray_sigma is not None:

            self.cosmicray_sigma = cosmicray_sigma

        # Get the Read Noise
        if readnoise is not None:

            self.readnoise = readnoise

            if isinstance(readnoise, str):

                # use the supplied keyword
                self.readnoise = float(self.header[readnoise])

            elif np.isfinite(readnoise):

                # use the given readnoise value
                self.readnoise = float(readnoise)

            else:

                logging.warning(
                    'readnoise has to be None, a numeric value or the ' +
                    'FITS header keyword, ' + str(readnoise) + ' is ' +
                    'given. It is set to 0.')

        else:

            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:

                readnoise_keyword_matched = np.in1d(self.readnoise_keyword,
                                                    self.header)

                if readnoise_keyword_matched.any():

                    self.readnoise = self.header[self.readnoise_keyword[
                        np.where(readnoise_keyword_matched)[0][0]]]

                else:

                    logging.warning('Read Noise value cannot be identified. ' +
                                    'It is set to 0.')

            else:

                logging.warning('Header is not provided. ' +
                                'Read Noise value is not provided. ' +
                                'It is set to 0.')

        # Get the gain
        if gain is not None:

            self.gain = gain

            if isinstance(gain, str):

                # use the supplied keyword
                self.gain = float(self.header[gain])

            elif np.isfinite(gain):

                # use the given gain value
                self.gain = float(gain)

            else:

                logging.warning(
                    'Gain has to be None, a numeric value or the ' +
                    'FITS header keyword, ' + str(gain) + ' is ' +
                    'given. It is set to 1.')
        else:

            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:

                gain_keyword_matched = np.in1d(self.gain_keyword, self.header)

                if gain_keyword_matched.any():

                    self.gain = self.header[self.gain_keyword[np.where(
                        gain_keyword_matched)[0][0]]]

                else:

                    logging.warning('Gain value cannot be identified. ' +
                                    'It is set to 1.')

            else:

                logging.warning('Header is not provide. ' +
                                'Gain value is not provided. ' +
                                'It is set to 1.')

        # Get the Seeing
        if seeing is not None:

            self.seeing = seeing

            if isinstance(seeing, str):

                # use the supplied keyword
                self.seeing = float(self.header[seeing])

            elif np.isfinite(gain):

                # use the given gain value
                self.seeing = float(seeing)

            else:

                logging.warning(
                    'Seeing has to be None, a numeric value or the ' +
                    'FITS header keyword, ' + str(seeing) + ' is ' +
                    'given. It is set to 1.')

        else:

            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:

                seeing_keyword_matched = np.in1d(self.seeing_keyword,
                                                 self.header)

                if seeing_keyword_matched.any():

                    self.seeing = self.header[self.seeing_keyword[np.where(
                        seeing_keyword_matched)[0][0]]]

                else:

                    logging.warning('Seeing value cannot be identified. ' +
                                    'It is set to 1.')

            else:

                logging.warning('Header is not provide. ' +
                                'Seeing value is not provided. ' +
                                'It is set to 1.')

        # Get the Exposure Time
        if exptime is not None:

            self.exptime = exptime

            if isinstance(exptime, str):

                # use the supplied keyword
                self.exptime = float(self.header[exptime])

            elif np.isfinite(gain):

                # use the given gain value
                self.exptime = float(exptime)

            else:

                logging.warning(
                    'Exposure Time has to be None, a numeric value or the ' +
                    'FITS header keyword, ' + str(exptime) + ' is ' +
                    'given. It is set to 1.')

        else:

            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if self.header is not None:

                exptime_keyword_matched = np.in1d(self.exptime_keyword,
                                                  self.header)

                if exptime_keyword_matched.any():

                    self.exptime = self.header[self.exptime_keyword[np.where(
                        exptime_keyword_matched)[0][0]]]

                else:

                    logging.warning(
                        'Exposure Time value cannot be identified. ' +
                        'It is set to 1.')

            else:

                logging.warning('Header is not provide. ' +
                                'Exposure Time value is not provided. ' +
                                'It is set to 1.')

        if verbose is not None:

            self.verbose = verbose

        if self.img is not None:

            self.img = self.img / self.exptime
            self.img_residual = self.img_residual / self.exptime

            # the valid y-range of the chip (i.e. spatial direction)
            if (len(self.spatial_mask) > 1):

                if self.saxis == 1:

                    self.img = self.img[self.spatial_mask]
                    self.img_residual = self.img_residual[self.spatial_mask]

                else:

                    self.img = self.img[:, self.spatial_mask]
                    self.img_residual = self.img_residual[:, self.spatial_mask]

            # the valid x-range of the chip (i.e. spectral direction)
            if (len(self.spec_mask) > 1):

                if self.saxis == 1:

                    self.img = self.img[:, self.spec_mask]
                    self.img_residual = self.img_residual[:, self.spec_mask]

                else:

                    self.img = self.img[self.spec_mask]
                    self.img_residual = self.img_residual[self.spec_mask]

            # get the length in the spectral and spatial directions
            self.spec_size = np.shape(self.img)[self.saxis]
            self.spatial_size = np.shape(self.img)[self.waxis]

            if self.saxis == 0:

                self.img = np.transpose(self.img)
                self.img_residual = np.transpose(self.img_residual)

            if self.flip:

                self.img = np.flip(self.img)
                self.img_residual = np.flip(self.img_residual)

            # set the 2D histogram z-limits
            img_log = np.log10(self.img)
            img_log_finite = img_log[np.isfinite(img_log)]
            self.zmin = np.nanpercentile(img_log_finite, 5)
            self.zmax = np.nanpercentile(img_log_finite, 95)

        else:

            logging.critical(
                'Please add the data when initialising the '
                'TwoDSpec() or by using add_data(), before setting the '
                'properties.')

        if variance is not None:

            self.variance = variance

        else:

            self.variance = np.abs(self.img) + self.readnoise**2

    def add_arc(self, arc, header=None):
        '''
        To provide an arc image. Make sure left (small index) is blue,
        right (large index) is red.

        Parameters
        ----------
        arc: 2D numpy array, PrimaryHDU/ImageHDU or ImageReduction object
            The image of the arc image.
        header: FITS header (deafult: None)
            An astropy.io.fits.Header object. This is not used if arc is
            a PrimaryHDU or ImageHDU.

        '''

        # If data provided is an numpy array
        if isinstance(arc, np.ndarray):

            self.arc = arc
            self.set_arc_header(header)

        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(arc, fits.hdu.image.PrimaryHDU) or isinstance(
                arc, fits.hdu.image.ImageHDU):
            self.arc = arc.data
            self.set_arc_header(header)

        # If it is an ImageReduction object
        elif isinstance(arc, ImageReduction):
            if arc.saxis == 1:
                self.arc = arc.arc_master
                if arc.arc_header is not []:
                    self.set_arc_header(arc.arc_header[0])
                else:
                    self.set_arc_header(None)
            else:
                self.arc = np.transpose(arc.arc_master)
                if arc.arc_header is not []:
                    self.set_arc_header(arc.arc_header[0])
                else:
                    self.set_arc_header(None)

        # If a filepath is provided
        elif isinstance(arc, str):

            # If HDU number is provided
            if arc[-1] == ']':

                filepath, hdunum = arc.split('[')
                hdunum = hdunum[:-1]

            # If not, assume the HDU idnex is 0
            else:

                filepath = arc
                hdunum = 0

            # Load the file and dereference it afterwards
            fitsfile_tmp = fits.open(filepath)[hdunum]
            if type(fitsfile_tmp) == 'astropy.io.fits.hdu.hdulist.HDUList':
                fitsfile_tmp = fitsfile_tmp[0]
                logging.warning('An HDU list is provided, only the first '
                                'HDU will be read.')
            fitsfile_tmp_shape = np.shape(fitsfile_tmp.data)

            # Normal case
            if len(fitsfile_tmp_shape) == 2:
                logging.debug('arc.data is 2 dimensional.')
                self.arc = fitsfile_tmp.data
                self.set_arc_header(fitsfile_tmp.header)
            # Try to trap common error when saving FITS file
            # Case with multiple image extensions, we only take the first one
            elif len(fitsfile_tmp_shape) == 3:
                logging.debug('arc.data is 3 dimensional.')
                self.arc = fitsfile_tmp.data[0]
                self.set_arc_header(fitsfile_tmp.header)
            # Case with an extra bracket when saving
            elif len(fitsfile_tmp_shape) == 1:
                logging.debug('arc.data is 1 dimensional.')
                # In case it in a multiple extension format, we take the
                # first one only
                if len(np.shape(fitsfile_tmp.data[0]) == 3):
                    self.arc = fitsfile_tmp.data[0][0]
                    self.set_arc_header(fitsfile_tmp[0].header)
                else:
                    self.arc = fitsfile_tmp.data[0]
                    self.set_arc_header(fitsfile_tmp[0].header)
            else:
                error_msg = 'Please check the shape/dimension of the ' +\
                            'input light frame, it is probably empty ' +\
                            'or has an atypical output format.'
                logging.critical(error_msg)
                raise RuntimeError(error_msg)

        else:

            error_msg = 'Please provide a numpy array, an ' +\
                'astropy.io.fits.hdu.image.PrimaryHDU object or an ' +\
                'aspired.ImageReduction object.'
            logging.critical(error_msg)
            raise TypeError(error_msg)

    def set_arc_header(self, header):
        """
        Adding the header for the arc.

        """

        # If it is a fits.hdu.header.Header object
        if isinstance(header, fits.header.Header):

            self.arc_header = header

        else:

            self.arc_header = None
            error_msg = 'Please provide a valid ' +\
                'astropy.io.fits.header.Header object. Process continues ' +\
                'without storing the header of the arc file.'
            logging.warning(error_msg)

    def apply_twodspec_mask_to_arc(self):
        '''
        **EXPERIMENTAL, as of 17 Jan 2021**
        Apply both the spec_mask and spatial_mask that are already stroed in
        the object.

        '''

        if self.saxis == 0:

            self.arc = np.transpose(self.arc)

        if self.flip:

            self.arc = np.flip(self.arc)

        self.apply_spec_mask_to_arc(self.spec_mask)
        self.apply_spatial_mask_to_arc(self.spatial_mask)

    def apply_spec_mask_to_arc(self, spec_mask):
        '''
        **EXPERIMENTAL, as of 17 Jan 2021**
        Apply to use only the valid x-range of the chip (i.e. dispersion
        direction)

        parameters
        ----------
        spec_mask: 1D numpy array (M)
            Mask in the spectral direction, can be the indices of the pixels
            to be included (size <M) or a 1D numpy array of True/False (size M)
            (Default is (1,) i.e. keep everything)

        '''

        if (len(spec_mask) > 1):

            self.arc = self.arc[:, spec_mask]

    def apply_spatial_mask_to_arc(self, spatial_mask):
        '''
        **EXPERIMENTAL, as of 17 Jan 2021**
        Apply to use only the valid y-range of the chip (i.e. spatial
        direction)

        parameters
        ----------
        spatial_mask: 1D numpy array (N)
            Mask in the spatial direction, can be the indices of the pixels
            to be included (size <N) or a 1D numpy array of True/False (size N)
            (Default is (1,) i.e. keep everything)

        '''

        if (len(spatial_mask) > 1):

            self.arc = self.arc[spatial_mask]

    def set_readnoise_keyword(self, keyword_list, append=False):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        append: boolean (Default: False)
            Set to False to overwrite the current list.

        '''

        if append:

            self.readnoise_keyword += list(keyword_list)

        else:

            self.readnoise_keyword = list(keyword_list)

    def set_gain_keyword(self, keyword_list, append=False):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        append: boolean (Default: False)
            Set to False to overwrite the current list.

        '''

        if append:

            self.gain_keyword += list(keyword_list)

        else:

            self.gain_keyword = list(keyword_list)

    def set_seeing_keyword(self, keyword_list, append=False):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        append: boolean (Default: False)
            Set to False to overwrite the current list.

        '''

        if append:

            self.seeing_keyword += list(keyword_list)

        else:

            self.seeing_keyword = list(keyword_list)

    def set_exptime_keyword(self, keyword_list, append=False):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        append: boolean (Default: False)
            Set to False to overwrite the current list.

        '''

        if append:

            self.exptime_keyword += list(keyword_list)

        else:

            self.exptime_keyword = list(keyword_list)

    def set_header(self, header):
        '''
        Set/replace the header.

        Parameters
        ----------
        header: astropy.io.fits.header.Header
            FITS header from a single HDU.

        '''

        # If it is a fits.hdu.header.Header object
        if isinstance(header, fits.header.Header):

            self.header = header

        else:

            error_msg = 'Please provide an ' +\
                'astropy.io.fits.header.Header object.'
            logging.critical(error_msg)
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

        return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b

    def _identify_spectra(self, f_height, display, renderer, width, height,
                          return_jsonstring, save_iframe, filename,
                          open_iframe):
        """
        Identify peaks assuming the spatial and spectral directions are
        aligned with the X and Y direction within a few degrees.

        Parameters
        ----------
        f_height: float
            The minimum intensity as a fraction of maximum height.
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

        """

        ydata = np.arange(self.spec_size)
        ztot = np.nanmedian(self.img, axis=1)

        # get the height thershold
        peak_height = np.nanmax(ztot) * f_height

        # identify peaks
        peaks_y, heights_y = signal.find_peaks(ztot, height=peak_height)
        heights_y = heights_y['peak_heights']

        # sort by strength
        mask = np.argsort(heights_y)
        peaks_y = peaks_y[mask][::-1]
        heights_y = heights_y[mask][::-1]

        self.peak = peaks_y
        self.peak_height = heights_y

        if display or save_iframe or return_jsonstring:

            # set a side-by-side subplot
            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width))

            # show the image on the left
            fig.add_trace(
                go.Heatmap(z=np.log10(self.img),
                           colorscale="Viridis",
                           xaxis='x',
                           yaxis='y'))

            # plot the integrated count and the detected peaks on the right
            fig.add_trace(
                go.Scatter(x=ztot,
                           y=ydata,
                           line=dict(color='black'),
                           xaxis='x2'))
            fig.add_trace(
                go.Scatter(x=heights_y,
                           y=ydata[peaks_y],
                           marker=dict(color='firebrick'),
                           xaxis='x2'))
            fig.update_layout(yaxis_title='Spatial Direction / pixel',
                              xaxis=dict(zeroline=False,
                                         domain=[0, 0.5],
                                         showgrid=False,
                                         title='Spectral Direction / pixel'),
                              xaxis2=dict(zeroline=False,
                                          domain=[0.5, 1],
                                          showgrid=True,
                                          title='Integrated Count'),
                              bargap=0,
                              hovermode='closest',
                              showlegend=False)

            if save_iframe:

                if filename is None:

                    pio.write_html(fig,
                                   'identify_spectra.html',
                                   auto_open=open_iframe)

                else:

                    pio.write_html(fig,
                                   filename + '.html',
                                   auto_open=open_iframe)

            # display disgnostic plot
            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if return_jsonstring:

                return fig.to_json()

    def _bfixpix(self, data, badmask, n=4, retdat=False):
        """
        Replace pixels flagged as nonzero in a bad-pixel mask with the
        average of their nearest four good neighboring pixels.

        Based on Ian Crossfield's code
        https://www.lpl.arizona.edu/~ianc/python/_modules/nsdata.html#bfixpix

        Parameters
        ----------
        data : numpy array (two-dimensional)

        badmask : numpy array (same shape as data)

        n : int
            number of nearby, good pixels to average over

        retdat : bool
            If True, return an array instead of replacing-in-place and do
            _not_ modify input array `data`.  This is always True if a 1D
            array is input!

        Returns
        -------
        another numpy array (if retdat is True)

        """

        if data.ndim == 1:
            data = np.tile(data, (3, 1))
            badmask = np.tile(badmask, (3, 1))
            ret = self._bfixpix(data, badmask, n=2, retdat=True)
            return ret[1]

        nx, ny = data.shape

        badx, bady = np.nonzero(badmask)
        nbad = len(badx)

        if retdat:
            data = np.array(data, copy=True)

        for ii in range(nbad):
            rad = 0
            numNearbyGoodPixels = 0

            while numNearbyGoodPixels < n:
                rad += 1
                xmin = max(0, badx[ii] - rad)
                xmax = min(nx, badx[ii] + rad)
                ymin = max(0, bady[ii] - rad)
                ymax = min(ny, bady[ii] + rad)
                x = np.arange(nx)[xmin:xmax + 1]
                y = np.arange(ny)[ymin:ymax + 1]
                yy, xx = np.meshgrid(y, x)

                rr = abs(xx + 1j * yy) * (
                    1. - badmask[xmin:xmax + 1, ymin:ymax + 1])
                numNearbyGoodPixels = (rr > 0).sum()

            closestDistances = np.unique(np.sort(rr[rr > 0])[0:n])
            numDistances = len(closestDistances)
            localSum = 0.
            localDenominator = 0.
            for jj in range(numDistances):
                localSum += data[xmin:xmax + 1, ymin:ymax +
                                 1][rr == closestDistances[jj]].sum()
                localDenominator += (rr == closestDistances[jj]).sum()

            data[badx[ii], bady[ii]] = 1.0 * localSum / localDenominator

        if retdat:
            ret = data
        else:
            ret = None

        return ret

    def ap_trace(self,
                 nspec=1,
                 nwindow=25,
                 spec_sep=5,
                 resample_factor=10,
                 rescale=False,
                 scaling_min=0.9995,
                 scaling_max=1.0005,
                 scaling_step=0.001,
                 percentile=5,
                 shift_tol=3,
                 fit_deg=3,
                 ap_faint=10,
                 display=False,
                 renderer='default',
                 width=1280,
                 height=720,
                 return_jsonstring=False,
                 save_iframe=False,
                 filename=None,
                 open_iframe=False):
        '''
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
        n-th percentile percentile of the slice, a rough guess can improve the
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
        nwindow: int
            Number of spectral slices (subspectra) to be produced for
            cross-correlation.
        spec_sep: int
            Minimum separation between spectra (only if there are multiple
            sources on the longslit).
        resample_factor: int
            Number of times the collapsed 1D slices in the spatial directions
            are to be upsampled.
        rescale: boolean
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
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

        Returns
        -------
        json string if return_jsonstring is True, otherwise only an image is
        displayed

        '''

        # Get the shape of the 2D spectrum and define upsampling ratio
        nwave = len(self.img[0])
        nspatial = len(self.img)

        nresample = nspatial * resample_factor

        # window size
        w_size = nwave // nwindow
        img_split = np.array_split(self.img, nwindow, axis=1)
        start_window_idx = nwindow // 2

        lines_ref_init = np.nanmedian(img_split[start_window_idx], axis=1)
        lines_ref_init[np.isnan(lines_ref_init)] = 0.
        lines_ref_init_resampled = signal.resample(lines_ref_init, nresample)

        # linear scaling limits
        if rescale:

            scaling_range = np.arange(scaling_min, scaling_max, scaling_step)

        else:

            scaling_range = np.ones(1)

        # estimate the n-th percentile as the sky background level
        lines_ref = lines_ref_init_resampled - np.percentile(
            lines_ref_init_resampled, percentile)

        shift_solution = np.zeros(nwindow)
        scale_solution = np.ones(nwindow)

        # maximum shift (SEMI-AMPLITUDE) from the neighbour (pixel)
        shift_tol_len = int(shift_tol * resample_factor)

        spec_spatial = np.zeros(nresample)

        pix_init = np.arange(nresample)
        pix_resampled = pix_init

        # Scipy correlate method
        for i in chain(range(start_window_idx, nwindow),
                       range(start_window_idx - 1, -1, -1)):

            # smooth by taking the median
            lines = np.nanmedian(img_split[i], axis=1)
            lines[np.isnan(lines)] = 0.
            lines = signal.resample(lines, nresample)
            lines = lines - np.nanpercentile(lines, percentile)

            # cross-correlation values and indices
            corr_val = np.zeros(len(scaling_range))
            corr_idx = np.zeros(len(scaling_range))

            # upsample by the same amount as the reference
            for j, scale in enumerate(scaling_range):

                # Upsampling the reference lines
                lines_ref_j = signal.resample(lines_ref,
                                              int(nresample * scale))

                # find the linear shift
                corr = signal.correlate(lines_ref_j, lines)

                # only consider the defined range of shift tolerance
                corr = corr[nresample - 1 - shift_tol_len:nresample +
                            shift_tol_len]

                # Maximum corr position is the shift
                corr_val[j] = np.nanmax(corr)
                corr_idx[j] = np.nanargmax(corr) - shift_tol_len

            # Maximum corr_val position is the scaling
            shift_solution[i] = corr_idx[np.nanargmax(corr_val)]
            scale_solution[i] = scaling_range[np.nanargmax(corr_val)]

            # Align the spatial profile before stacking
            if i == (start_window_idx - 1):

                pix_resampled = pix_init

            pix_resampled = pix_resampled * scale_solution[i] + shift_solution[
                i]

            spec_spatial += spectres(np.arange(nresample),
                                     np.array(pix_resampled).reshape(-1),
                                     np.array(lines).reshape(-1),
                                     verbose=False)

            # Update (increment) the reference line
            if (i == nwindow - 1):

                lines_ref = lines_ref_init_resampled

            else:

                lines_ref = lines

        nscaled = (nresample * scale_solution).astype('int')

        # Find the spectral position in the middle of the gram in the upsampled
        # pixel location location
        peaks = signal.find_peaks(spec_spatial,
                                  distance=spec_sep,
                                  prominence=0)

        # update the number of spectra if the number of peaks detected is less
        # than the number requested
        self.nspec_traced = min(len(peaks[0]), nspec)

        # Sort the positions by the prominences, and return to the original
        # scale (i.e. with subpixel position)
        spec_init = np.sort(peaks[0][np.argsort(-peaks[1]['prominences'])]
                            [:self.nspec_traced]) / resample_factor

        # Create array to populate the spectral locations
        spec_idx = np.zeros((len(spec_init), len(img_split)))

        # Populate the initial values
        spec_idx[:, start_window_idx] = spec_init

        # Pixel positions of the mid point of each data_split (spectral)
        spec_pix = np.arange(len(img_split)) * w_size + w_size / 2.

        # Looping through pixels larger than middle pixel
        for i in range(start_window_idx + 1, nwindow):

            spec_idx[:,
                     i] = (spec_idx[:, i - 1] * resample_factor * nscaled[i] /
                           nresample - shift_solution[i]) / resample_factor

        # Looping through pixels smaller than middle pixel
        for i in range(start_window_idx - 1, -1, -1):

            spec_idx[:, i] = (spec_idx[:, i + 1] * resample_factor -
                              shift_solution[i + 1]) / (
                                  int(nresample * scale_solution[i + 1]) /
                                  nresample) / resample_factor

        for i in range(len(spec_idx)):

            # Get the median of the subspectrum and then get the Count at the
            # centre of the aperture
            ap_val = np.zeros(nwindow)

            for j in range(nwindow):

                # rounding
                idx = int(spec_idx[i][j] + 0.5)
                ap_val[j] = np.nanmedian(img_split[j], axis=1)[idx]

            # Mask out the faintest ap_faint percentile
            mask = (ap_val > np.nanpercentile(ap_val, ap_faint))

            # fit the trace
            ap_p = np.polyfit(spec_pix[mask], spec_idx[i][mask], int(fit_deg))
            ap = np.polyval(ap_p, np.arange(nwave))

            # Get the centre of the upsampled spectrum
            ap_centre_idx = ap[start_window_idx] * resample_factor

            # Get the indices for the 10 pixels on the left and right of the
            # spectrum, and apply the resampling factor.
            start_idx = int(ap_centre_idx - 10 * resample_factor + 0.5)
            end_idx = start_idx + 20 * resample_factor + 1

            start_idx = max(0, start_idx)
            end_idx = min(nspatial * resample_factor, end_idx)

            if start_idx == end_idx:

                ap_sigma = np.nan
                continue

            # compute ONE sigma for each trace
            pguess = [
                np.nanmax(spec_spatial[start_idx:end_idx]),
                np.nanpercentile(spec_spatial, 10), ap_centre_idx, 3.
            ]

            non_nan_mask = np.isfinite(spec_spatial[start_idx:end_idx])
            popt, _ = curve_fit(self._gaus,
                                np.arange(start_idx, end_idx)[non_nan_mask],
                                spec_spatial[start_idx:end_idx][non_nan_mask],
                                p0=pguess)
            ap_sigma = popt[3] / resample_factor

            self.spectrum_list[i] = Spectrum1D(
                spec_id=i,
                verbose=self.verbose,
                logger_name=self.logger_name,
                log_level=self.log_level,
                log_file_folder=self.log_file_folder,
                log_file_name=self.log_file_name)
            self.spectrum_list[i].add_trace(list(ap), [ap_sigma] * len(ap))
            self.spectrum_list[i].add_gain(self.gain)
            self.spectrum_list[i].add_readnoise(self.readnoise)
            self.spectrum_list[i].add_exptime(self.exptime)

        # Plot
        if save_iframe or display or return_jsonstring:

            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width))

            fig.add_trace(
                go.Heatmap(z=np.log10(self.img),
                           zmin=self.zmin,
                           zmax=self.zmax,
                           colorscale="Viridis",
                           colorbar=dict(title='log( e- / s )')))

            for i in range(len(spec_idx)):

                fig.add_trace(
                    go.Scatter(x=np.arange(nwave),
                               y=self.spectrum_list[i].trace,
                               line=dict(color='black')))
                fig.add_trace(
                    go.Scatter(x=spec_pix,
                               y=spec_idx[i],
                               mode='markers',
                               marker=dict(color='grey')))
            fig.add_trace(
                go.Scatter(x=np.ones(len(spec_idx)) *
                           spec_pix[start_window_idx],
                           y=spec_idx[:, start_window_idx],
                           mode='markers',
                           marker=dict(color='firebrick')))
            fig.update_layout(yaxis_title='Spatial Direction / pixel',
                              xaxis=dict(zeroline=False,
                                         showgrid=False,
                                         title='Spectral Direction / pixel'),
                              bargap=0,
                              hovermode='closest',
                              showlegend=False)

            if save_iframe:

                if filename is None:

                    pio.write_html(fig, 'ap_trace.html', auto_open=open_iframe)

                else:

                    pio.write_html(fig,
                                   filename + '.html',
                                   auto_open=open_iframe)

            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if return_jsonstring:

                return fig.to_json()

    def add_trace(self, trace, trace_sigma, spec_id=None):
        '''
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

        '''

        assert isinstance(spec_id, (int, list, np.ndarray)) or (
            spec_id is
            None), 'spec_id has to be an integer, None, list or array.'

        if spec_id is None:

            if len(np.shape(trace)) == 1:

                spec_id = [0]

            elif len(np.shape(trace)) == 2:

                spec_id = list(np.arange(np.shape(trace)[0]))

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if isinstance(spec_id, np.ndarray):

            spec_id = list(spec_id)

        assert isinstance(
            trace, (list, np.ndarray)), 'trace has to be a list or an array.'
        assert isinstance(
            trace_sigma,
            (list, np.ndarray)), 'trace_sigma has to be a list or an array.'
        assert len(trace) == len(trace_sigma), 'trace and trace_sigma have to '
        'be the same size.'

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
                    log_file_name=self.log_file_name)
                self.spectrum_list[i].add_trace(trace, trace_sigma)

    def remove_trace(self, spec_id):
        '''
        Parameters
        ----------
        spec_id: int
            The ID corresponding to the spectrum1D object

        '''

        if spec_id in self.spectrum_list:

            self.spectrum_list[spec_id].remove_trace()

        else:

            error_msg = "{spec_id: %s} is not in the list of " +\
                "spectra." % spec_id
            logging.critical(error_msg)
            raise ValueError(error_msg)

    def _fit_sky(self, extraction_slice, sky_width_dn, sky_width_up,
                 sky_polyfit_order):
        """
        Fit the sky background from the given extraction_slice and the aperture
        parameters.

        Parameters
        ----------
        extraction_slice: 1-d numpy array
            The counts along the profile for extraction, including the sky
            regions to be fitted and subtracted from.
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
            sky_mask[0:sky_width_dn] = True
            sky_mask[-(sky_width_up + 1):-1] = True

            if (sky_polyfit_order == 0):

                count_sky_extraction_slice = np.ones(
                    len(extraction_slice[sky_mask])) * np.nanmean(
                        extraction_slice[sky_mask])

            elif (sky_polyfit_order > 0):

                # fit a polynomial to the sky in this column
                polyfit_coeff = np.polynomial.polynomial.polyfit(
                    np.arange(extraction_slice.size)[sky_mask],
                    extraction_slice[sky_mask], sky_polyfit_order)

                # evaluate the polynomial across the extraction_slice, and sum
                count_sky_extraction_slice = np.polynomial.polynomial.polyval(
                    np.arange(extraction_slice.size), polyfit_coeff)

            else:

                logging.warning('sky_polyfit_order cannot be negative. sky '
                                'background is set to zero.')
                count_sky_extraction_slice = np.zeros_like(extraction_slice)

        else:

            # get the indexes of the sky regions
            count_sky_extraction_slice = np.zeros_like(extraction_slice)

        return count_sky_extraction_slice

    def ap_extract(self,
                   apwidth=7,
                   skysep=3,
                   skywidth=5,
                   skydeg=1,
                   spec_id=None,
                   optimal=True,
                   algorithm='horne86',
                   tolerance=1e-6,
                   max_iter=99,
                   forced=False,
                   variances=None,
                   npoly=21,
                   polyspacing=1,
                   pord=2,
                   qmode='fast-linear',
                   nreject=100,
                   display=False,
                   renderer='default',
                   width=1280,
                   height=720,
                   return_jsonstring=False,
                   save_iframe=False,
                   filename=None,
                   open_iframe=False):
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
        optimal: boolean (Default: True)
            Set optimal extraction. (Default is True)
        algorithm: str (Default: 'horne86')
            Available algorithms are horne86 and marsh89.
        tolerance: float (Default: 1e-6)
            The tolerance limit for the convergence of the optimal extraction
        max_iter: float (Default: 99)
            The maximum number of iterations before optimal extraction fails
            and return to standard tophot extraction
        forced: boolean (Default: False)
            To perform forced optimal extraction  by using the given aperture
            profile as it is without interation, the resulting uncertainty
            will almost certainly be wrong. This is an experimental feature.
        variances: list or numpy.ndarray (Default: None, only used if algorithm
                   is 'horne86')
            The weight function for forced extraction. It is only used if force
            is set to True.
        npoly: int (Default: 21, only used if algorithm is marsh89)
            Number of profile to be use for polynomial fitting to evaluate
            (Marsh's "K"). For symmetry, this should be odd.
        polyspacing: float (Default: 1)
            Spacing between profile polynomials, in pixels. (Marsh's "S").
            A few cursory tests suggests that the extraction precision
            (in the high S/N case) scales as S^-2 -- but the code slows down
            as S^2.
        pord: int (Default: 2)
            Order of profile polynomials; 1 = linear, etc.
        qmode: str (Default: 'fast-linear')
            How to compute Marsh's Q-matrix. Valid inputs are 'fast-linear',
            'slow-linear', 'fast-nearest', and 'slow-nearest'. These select
            between various methods of integrating the nearest-neighbor or
            linear interpolation schemes as described by Marsh; the 'linear'
            methods are preferred for accuracy. Use 'slow' if you are
            running out of memory when using the 'fast' array-based methods.
        nreject: int (Default: 100)
            Number of outlier-pixels to reject at each iteration.
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

        """

        if spec_id is not None:

            assert np.in1d(spec_id,
                           list(self.spectrum_list.keys())).all(), 'Some '
            'spec_id provided are not in the spectrum_list.'

        else:

            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        for j in spec_id:

            spec = self.spectrum_list[j]
            len_trace = len(spec.trace)
            count_sky = np.zeros(len_trace)
            count_err = np.zeros(len_trace)
            count = np.zeros(len_trace)
            var = np.ones((len_trace, 2 * apwidth + 1)) * self.readnoise**2.
            is_optimal = np.zeros(len_trace, dtype=bool)

            if isinstance(apwidth, int):

                # first do the aperture count
                width_dn = apwidth
                width_up = apwidth

            elif len(apwidth) == 2:

                width_dn = apwidth[0]
                width_up = apwidth[1]

            else:

                logging.error('apwidth can only be an int or a list ' +
                              'of two ints. It is set to the default ' +
                              'value to continue the extraction.')
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

                logging.error('skysep can only be an int or a list of ' +
                              'two ints. It is set to the default ' +
                              'value to continue the extraction.')
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

                logging.error('skywidth can only be an int or a list of ' +
                              'two ints. It is set to the default value ' +
                              'to continue the extraction.')
                sky_width_dn = 5
                sky_width_up = 5

            # Sky extraction
            for i, pos in enumerate(spec.trace):

                itrace = int(pos)
                pix_frac = pos - itrace

                # fix width if trace is too close to the edge
                if (itrace + width_up > self.spatial_size):

                    # ending at the last pixel
                    width_up = self.spatial_size - itrace - 1

                if (itrace - width_dn < 0):

                    # starting at pixel row 0
                    width_dn = itrace - 1

                # Pixels where the source spectrum and the sky regions are
                source_pix = np.arange(itrace - width_dn,
                                       itrace + width_up + 1)
                extraction_pix = np.arange(
                    itrace - width_dn - sep_dn - sky_width_dn,
                    itrace + width_up + sep_up + sky_width_up + 1)

                # trace +/- aperture size
                source_slice = self.img[source_pix, i].copy()

                # trace +/- aperture and sky region size
                extraction_slice = self.img[extraction_pix, i].copy()

                count_sky_extraction_slice = self._fit_sky(
                    extraction_slice, sky_width_dn, sky_width_up, skydeg)
                logging.debug('Sky count (including wings): {}'.format(
                    count_sky_extraction_slice))

                count_sky_source_slice = count_sky_extraction_slice[
                    source_pix - itrace].copy()
                logging.debug('Sky count: {}'.format(count_sky_source_slice))

                count_sky[i] = np.nansum(
                    count_sky_source_slice
                ) - pix_frac * count_sky_source_slice[0] - (
                    1 - pix_frac) * count_sky_source_slice[-1]

                self.img_residual[source_pix, i] = count_sky_source_slice

                logging.debug('count_sky at pixel {} is {}.'.format(
                    i, count_sky[i]))

                # if not optimal extraction or using marsh89, perform a
                # tophat extraction
                if not optimal or algorithm == 'marsh89':

                    count[i], count_err[i], is_optimal[i] =\
                        self._tophat_extraction(
                            source_slice=source_slice * self.exptime,
                            sky_source_slice=(
                                count_sky_source_slice * self.exptime),
                            pix_frac=pix_frac,
                            gain=self.gain,
                            sky_width_dn=sky_width_dn,
                            sky_width_up=sky_width_up,
                            width_dn=width_dn,
                            width_up=width_up)

            if optimal:

                if algorithm == 'marsh89':

                    count, count_err, is_optimal, profile, var =\
                        self._optimal_extraction_marsh89(
                            frame=self.img,
                            residual_frame=self.img_residual,
                            variance=self.variance,
                            trace=spec.trace,
                            spectrum=count,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            apwidth=(width_dn, width_up),
                            goodpixelmask=None,
                            npoly=npoly,
                            polyspacing=polyspacing,
                            pord=pord,
                            cosmicray_sigma=self.cosmicray_sigma,
                            qmode=qmode,
                            nreject=nreject)

                # Get the optimal signals
                elif algorithm == 'horne86':

                    for i, pos in enumerate(spec.trace):

                        # If the weights are given externally to perform forced
                        # extraction
                        if forced:

                            # Unit weighted
                            if np.ndim(variances) == 0:

                                if np.isfinite(variances):

                                    var_i = np.ones(width_dn + width_up +
                                                    1) * variances

                                else:

                                    var_i = np.ones(len(source_pix))
                                    logging.warning('Variances are set to 1.')

                            # A single LSF is given for the entire trace
                            elif np.ndim(variances) == 1:
                                if len(variances) == len(source_pix):

                                    var_i = variances

                                elif len(variances) == len_trace:

                                    var_i = np.ones(
                                        len(source_pix)) * variances[i]

                                else:

                                    var_i = np.ones(len(source_pix))
                                    logging.warning('Variances are set to 1.')

                            # A two dimensional LSF
                            elif np.ndim(variances) == 2:

                                var_i = variances[i]

                                # If the spectrum is outside of the frame
                                if itrace - apwidth < 0:

                                    var_i = var_i[apwidth - width_dn:]

                                # If the spectrum is outside of the frame
                                elif itrace + apwidth > self.spatial_size:

                                    var_i = var_i[:-(apwidth - width_up + 1)]

                                else:

                                    pass

                            else:

                                var_i = np.ones(len(source_pix))
                                logging.warning('Variances are set to 1.')

                        else:

                            var_i = None

                        # source_pix is the native pixel position
                        # pos is the trace at the native pixel position
                        (count[i], count_err[i], is_optimal[i], profile,
                         var_temp) =\
                            self._optimal_extraction_horne86(
                                pix=source_pix,
                                source_slice=source_slice * self.exptime,
                                sky=count_sky_source_slice * self.exptime,
                                mu=pos,
                                sigma=spec.trace_sigma[i],
                                tol=tolerance,
                                max_iter=max_iter,
                                readnoise=self.readnoise,
                                gain=self.gain,
                                cosmicray_sigma=self.cosmicray_sigma,
                                forced=forced,
                                variances=var_i)

                        if var_i is None:
                            var[i] = var_temp
                        else:
                            var[i] = var_i

            # All the extraction methods return signal and noise in the
            # same format
            count[i] /= self.exptime
            count_err[i] /= self.exptime

            spec.add_aperture(width_dn, width_up, sep_dn, sep_up, sky_width_dn,
                              sky_width_up)
            spec.add_count(list(count), list(count_err), list(count_sky))
            spec.add_variances(var)
            spec.gain = self.gain
            spec.optimal_pixel = is_optimal
            spec.add_spectrum_header(self.header)

            if optimal:

                spec.extraction_type = "Optimal"

            else:

                spec.extraction_type = "Aperture"

            # If more than a third of the spectrum is extracted suboptimally
            if np.sum(is_optimal) / len(is_optimal) < 0.333:

                logging.warning(
                    'Signal extracted is likely to be suboptimal, please '
                    'try a longer iteration, larger tolerance or revert '
                    'to top-hat extraction.')

            if save_iframe or display or return_jsonstring:

                min_trace = int(min(spec.trace) + 0.5)
                max_trace = int(max(spec.trace) + 0.5)

                fig = go.Figure(
                    layout=dict(autosize=False, height=height, width=width))
                # the 3 is to show a little bit outside the extraction regions
                img_display = np.log10(self.img[
                    max(0, min_trace - width_dn - sep_dn - sky_width_dn -
                        3):min(max_trace + width_up + sep_up +
                               sky_width_up, len(self.img[0])) + 3, :])

                # show the image on the top
                # the 3 is the show a little bit outside the extraction regions
                fig.add_trace(
                    go.Heatmap(x=np.arange(len_trace),
                               y=np.arange(
                                   max(
                                       0, min_trace - width_dn - sep_dn -
                                       sky_width_dn - 3),
                                   min(
                                       max_trace + width_up + sep_up +
                                       sky_width_up + 3, len(self.img[0]))),
                               z=img_display,
                               colorscale="Viridis",
                               zmin=self.zmin,
                               zmax=self.zmax,
                               xaxis='x',
                               yaxis='y',
                               colorbar=dict(title='log( e- / s )')))

                # Middle black box on the image
                fig.add_trace(
                    go.Scatter(x=list(
                        np.concatenate(
                            (np.arange(len_trace), np.arange(len_trace)[::-1],
                             np.zeros(1)))),
                               y=list(
                                   np.concatenate(
                                       (np.array(spec.trace) - width_dn - 1,
                                        np.array(spec.trace[::-1]) + width_up +
                                        1, np.ones(1) *
                                        (spec.trace[0] - width_dn - 1)))),
                               xaxis='x',
                               yaxis='y',
                               mode='lines',
                               line_color='black',
                               showlegend=False))

                # Lower red box on the image
                lower_redbox_upper_bound = np.array(
                    spec.trace) - width_dn - sep_dn - 1
                lower_redbox_lower_bound = np.array(
                    spec.trace)[::-1] - width_dn - sep_dn - sky_width_dn
                lower_redbox_lower_bound[lower_redbox_lower_bound < 0] = 1

                fig.add_trace(
                    go.Scatter(x=list(
                        np.concatenate(
                            (np.arange(len_trace), np.arange(len_trace)[::-1],
                             np.zeros(1)))),
                               y=list(
                                   np.concatenate(
                                       (lower_redbox_upper_bound,
                                        lower_redbox_lower_bound, np.ones(1) *
                                        lower_redbox_upper_bound[0]))),
                               line_color='red',
                               xaxis='x',
                               yaxis='y',
                               mode='lines',
                               showlegend=False))

                # Upper red box on the image
                upper_redbox_upper_bound = np.array(
                    spec.trace) + width_up + sep_up + sky_width_up
                upper_redbox_lower_bound = np.array(
                    spec.trace)[::-1] + width_up + sep_up + 1

                upper_redbox_upper_bound[upper_redbox_upper_bound > self.
                                         spatial_size] = self.spatial_size + 1

                fig.add_trace(
                    go.Scatter(x=list(
                        np.concatenate(
                            (np.arange(len_trace), np.arange(len_trace)[::-1],
                             np.zeros(1)))),
                               y=list(
                                   np.concatenate(
                                       (upper_redbox_upper_bound,
                                        upper_redbox_lower_bound, np.ones(1) *
                                        upper_redbox_upper_bound[0]))),
                               xaxis='x',
                               yaxis='y',
                               mode='lines',
                               line_color='red',
                               showlegend=False))

                # plot the SNR
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=count / count_err,
                               xaxis='x2',
                               yaxis='y3',
                               line=dict(color='slategrey'),
                               name='Signal-to-Noise Ratio'))

                # extrated source, sky and uncertainty
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=count_sky,
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='firebrick'),
                               name='Sky count / (e- / s)'))
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=count_err,
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='orange'),
                               name='Uncertainty count / (e- / s)'))
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=count,
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='royalblue'),
                               name='Target count / (e- / s)'))

                # Decorative stuff
                fig.update_layout(
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(zeroline=False,
                               domain=[0.5, 1],
                               showgrid=False,
                               title='Spatial Direction / pixel'),
                    yaxis2=dict(
                        range=[
                            min(
                                np.nanmin(
                                    sigma_clip(count, sigma=5., masked=False)),
                                np.nanmin(
                                    sigma_clip(count_err,
                                               sigma=5.,
                                               masked=False)),
                                np.nanmin(
                                    sigma_clip(count_sky,
                                               sigma=5.,
                                               masked=False)), 1),
                            max(np.nanmax(count), np.nanmax(count_sky))
                        ],
                        zeroline=False,
                        domain=[0, 0.5],
                        showgrid=True,
                        title='Count / s',
                    ),
                    yaxis3=dict(title='S/N ratio',
                                anchor="x2",
                                overlaying="y2",
                                side="right"),
                    xaxis2=dict(title='Spectral Direction / pixel',
                                anchor="y2",
                                matches="x"),
                    legend=go.layout.Legend(x=0,
                                            y=0.45,
                                            traceorder="normal",
                                            font=dict(family="sans-serif",
                                                      size=12,
                                                      color="black"),
                                            bgcolor='rgba(0,0,0,0)'),
                    bargap=0,
                    hovermode='closest',
                    showlegend=True)

                if save_iframe:

                    if filename is None:

                        pio.write_html(fig,
                                       'ap_extract_' + str(j) + '.html',
                                       auto_open=open_iframe)

                    else:

                        pio.write_html(fig,
                                       filename + '_' + str(j) + '.html',
                                       auto_open=open_iframe)

                if display:

                    if renderer == 'default':

                        fig.show()

                    else:

                        fig.show(renderer)

                if return_jsonstring:

                    return fig.to_json()

    def _tophat_extraction(self, source_slice, sky_source_slice, pix_frac,
                           gain, sky_width_dn, sky_width_up, width_dn,
                           width_up):
        """
        Make sure the counts are the number of photoelectrons or an equivalent
        detector unit, and not counts per second.

        Parameters
        ----------
        source_slice: 1-d numpy array (N)
            The counts along the profile for aperture extraction.
        sky_source_slice: 1-d numpy array (N)
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

        """
        # Get the total count
        source_plus_sky = np.nansum(source_slice) - pix_frac * source_slice[
            0] - (1 - pix_frac) * source_slice[-1]

        # finally, compute the error in this pixel
        # standarddev in the background data
        sigB = np.nanstd(sky_source_slice)
        sky = np.nansum(sky_source_slice) - pix_frac * sky_source_slice[0] - (
            1 - pix_frac) * sky_source_slice[-1]

        # number of bkgd pixels
        nB = sky_width_dn + sky_width_up
        # number of aperture pixels
        nA = width_dn + width_up

        # Based on aperture phot err description by F. Masci,
        # Caltech:
        # http://wise2.ipac.caltech.edu/staff/fmasci/
        #   ApPhotUncert.pdf
        # All the counts are in per second already, so need to
        # multiply by the exposure time when computing the
        # uncertainty
        signal = source_plus_sky - sky
        noise = np.sqrt(signal / gain + (nA + nA**2. / nB) * (sigB**2.))

        return signal, noise, False

    def _optimal_extraction_horne86(self,
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
                                    variances=None):
        """
        Make sure the counts are the number of photoelectrons or an equivalent
        detector unit, and not counts per second or ADU.

        Iterate to get the optimal signal. Following the algorithm on
        Horne, 1986, PASP, 98, 609 (1986PASP...98..609H). The 'steps' in the
        inline comments are in reference to this article.

        Parameters
        ----------
        pix: 1-d numpy array (N)
            pixel number along the spatial direction
        source_slice: 1-d numpy array (N)
            The counts along the profile for extraction, including the sky
            regions to be fitted and subtracted from.
        sky: 1-d numpy array (N)
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
        forced: boolean
            Forced extraction with the given weights.
        variances: 2-d numpy array (N)
            The 1/weights of used for optimal extraction, has to be the
            same length as the pix. Only used if forced is True.

        Returns
        -------
        signal: float
            The optimal signal.
        noise: float
            The noise associated with the optimal signal.
        is_optimal: boolean
            List indicating whether the extraction at that pixel was
            optimal or not. True = optimal, False = suboptimal.
        P: numpy array
            The line spread function of the extraction
        var_f: float
            The variance in the extraction.

        """

        # step 2 - initial variance estimates
        var1 = readnoise + np.abs(source_slice) / gain

        # step 4a - extract standard spectrum
        f = source_slice - sky
        f[f < 0] = 0.
        f1 = np.nansum(f)

        # step 4b - variance of standard spectrum
        v1 = 1. / np.nansum(1. / var1)

        # step 5 - construct the spatial profile
        P = self._gaus(pix, 1., 0., mu, sigma)
        P /= np.nansum(P)

        f_diff = 1
        v_diff = 1
        i = 0
        is_optimal = True

        mask_cr = np.ones(len(P), dtype=bool)

        if forced:

            var_f = variances

        while (f_diff > tol) | (v_diff > tol):

            f0 = f1
            v0 = v1

            # step 6 - revise variance estimates
            if not forced:

                var_f = readnoise + np.abs(P * f0 + sky) / gain

            # step 7 - cosmic ray mask, only start considering after the
            # 2nd iteration. 1 pixel is masked at a time until convergence,
            # once the pixel is masked, it will stay masked.
            if i > 1:

                ratio = (cosmicray_sigma**2. * var_f) / (f - P * f0)**2.
                outlier = np.sum(ratio > 1)

                if outlier > 0:

                    mask_cr[np.argmax(ratio)] = False

            denom = np.nansum((P**2. / var_f)[mask_cr])

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

        logging.debug('The signal, noise, is_optimal flag and variance '
                      'are {}, {}, {} and {}.'.format(signal, noise,
                                                      is_optimal, var_f))

        return signal, noise, is_optimal, P, var_f

    def _optimal_extraction_marsh89(self,
                                    frame,
                                    residual_frame,
                                    variance,
                                    trace,
                                    spectrum=None,
                                    gain=1.0,
                                    readnoise=0.0,
                                    apwidth=7,
                                    goodpixelmask=None,
                                    npoly=21,
                                    polyspacing=1,
                                    pord=2,
                                    cosmicray_sigma=5,
                                    qmode='fast-linear',
                                    nreject=100):
        """
        Optimally extract curved spectra taken and updated from
        https://people.ucsc.edu/~ianc/python/_modules/spec.html,
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
        qmode: str (Default: 'fast-linear')
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

        if isinstance(apwidth, int):

            # first do the aperture count
            width_dn = apwidth
            width_up = apwidth

        elif len(apwidth) == 2:

            width_dn = apwidth[0]
            width_up = apwidth[1]

        else:

            logging.error('apwidth can only be an int or a list ' +
                          'of two ints. It is set to the default ' +
                          'value to continue the extraction.')
            width_dn = 7
            width_up = 7

        if goodpixelmask is not None:
            goodpixelmask = np.array(goodpixelmask, copy=True).astype(bool)
        else:
            goodpixelmask = np.ones(frame.shape, dtype=bool)

        goodpixelmask *= (np.isfinite(frame) * np.isfinite(variance))

        variance[~goodpixelmask] = frame[goodpixelmask].max() * 1e9
        spectral_size, spatial_size = frame.shape

        # (my 3a: mask any bad values)
        bad_residual_frame_mask = ~np.isfinite(residual_frame)
        residual_frame[bad_residual_frame_mask] = 0.
        if np.any(bad_residual_frame_mask.nonzero()):
            logging.warning("Found bad residual_frame values at: {}".format(
                bad_residual_frame_mask.nonzero()))

        skysubFrame = frame - residual_frame

        # Interpolate and fix bad pixels for extraction of standard
        # spectrum -- otherwise there can be 'holes' in the spectrum from
        # ill-placed bad pixels.
        fixSkysubFrame = self._bfixpix(skysubFrame,
                                       ~goodpixelmask,
                                       n=8,
                                       retdat=True)

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
        jjnorm_pow = jjnorm.reshape(
            1, 1,
            spectral_size)**(np.arange(2 * N - 1).reshape(2 * N - 1, 1, 1))

        # Marsh eq. 9, defining centers of each polynomial:
        constant = 0.  # What is it for???
        poly_centers = np.array(trace).reshape(
            spectral_size, 1) + polyspacing * np.arange(
                -npoly / 2 + 1, npoly / 2 + 1) + constant

        # Marsh eq. 11, defining Q_kij    (via nearest-neighbor interpolation)
        #    Q_kij =  max(0, min(S, (S+1)/2 - abs(x_kj - i)))
        if qmode == 'fast-nearest':  # Array-based nearest-neighbor mode.
            Q = np.array([
                np.zeros((npoly, spatial_size, spectral_size)),
                np.array([
                    polyspacing * np.ones(
                        (npoly, spatial_size, spectral_size)),
                    0.5 * (polyspacing + 1) - np.abs(
                        (poly_centers -
                         ii.reshape(spatial_size, 1, 1)).transpose(2, 0, 1))
                ]).min(0)
            ]).max(0)

        elif qmode == 'slow-linear':  # Code is a mess, but it works.
            invs = 1. / polyspacing
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
                            qval = 0.
                        elif (im05) > xkj:
                            lim1 = im05
                            lim2 = min(ip05, xps)
                            qval = (lim2 - lim1) * (1. + xkjs - 0.5 * invs *
                                                    (lim1 + lim2))
                        elif (ip05) < xkj:
                            lim1 = max(im05, xms)
                            lim2 = ip05
                            qval = (lim2 - lim1) * (1. - xkjs + 0.5 * invs *
                                                    (lim1 + lim2))
                        else:
                            lim1 = max(im05, xms)
                            lim2 = min(ip05, xps)
                            qval = lim2 - lim1 + invs * (
                                xkj * (-xkj + lim1 + lim2) - 0.5 *
                                (lim1 * lim1 + lim2 * lim2))
                        Q[k, i, j] = max(0, qval)

        # Code is a mess, but it's faster than 'slow' mode
        elif qmode == 'fast-linear':
            invs = 1. / polyspacing
            xps_mat = poly_centers + polyspacing
            Q = np.zeros((npoly, spatial_size, spectral_size))
            for j in range(spectral_size):
                xkj_vec = np.tile(poly_centers[j, :].reshape(npoly, 1),
                                  (1, spatial_size))
                xps_vec = np.tile(xps_mat[j, :].reshape(npoly, 1),
                                  (1, spatial_size))
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
                    ind11_3d = ind11 + (np.ones(n_ind11, dtype=int) * j, )
                    lim2_ind11 = np.array(
                        (ip05_vec[ind11], xps_vec[ind11])).min(0)
                    Q[ind11_3d] = ((lim2_ind11 - im05_vec[ind11]) * invs *
                                   (polyspacing + xkj_vec[ind11] - 0.5 *
                                    (im05_vec[ind11] + lim2_ind11)))

                if n_ind22 > 0:
                    ind22_3d = ind22 + (np.ones(n_ind22, dtype=int) * j, )
                    lim1_ind22 = np.array(
                        (im05_vec[ind22], xms_vec[ind22])).max(0)
                    Q[ind22_3d] = ((ip05_vec[ind22] - lim1_ind22) * invs *
                                   (polyspacing - xkj_vec[ind22] + 0.5 *
                                    (ip05_vec[ind22] + lim1_ind22)))

                if n_ind33 > 0:
                    ind33_3d = ind33 + (np.ones(n_ind33, dtype=int) * j, )
                    lim1_ind33 = np.array(
                        (im05_vec[ind33], xms_vec[ind33])).max(0)
                    lim2_ind33 = np.array(
                        (ip05_vec[ind33], xps_vec[ind33])).min(0)
                    Q[ind33_3d] = (
                        (lim2_ind33 - lim1_ind33) + invs *
                        (xkj_vec[ind33] *
                         (-xkj_vec[ind33] + lim1_ind33 + lim2_ind33) - 0.5 *
                         (lim1_ind33 * lim1_ind33 + lim2_ind33 * lim2_ind33)))

        # 'slow' Loop-based nearest-neighbor mode: requires less memory
        else:
            Q = np.zeros((npoly, spatial_size, spectral_size))
            for k in range(npoly):
                for i in range(spatial_size):
                    for j in range(spectral_size):
                        Q[k, i, j] = max(
                            0,
                            min(
                                polyspacing, 0.5 * (polyspacing + 1) -
                                np.abs(poly_centers[j, k] - i)))

        # Some quick math to find out which dat columns are important, and
        # which contain no useful spectral information:
        Qmask = Q.sum(0).transpose() > 0
        Qind = Qmask.transpose().nonzero()
        Q_cols = [Qind[0].min(), Qind[0].max()]
        Qsm = Q[:, Q_cols[0]:Q_cols[1] + 1, :]

        # Prepar to iteratively clip outliers
        logging.info("Looking for bad pixel outliers.")
        newBadPixels = True
        i = -1
        while newBadPixels:
            i += 1
            logging.debug("Beginning iteration {}.".format(i))

            # Compute pixel fractions (Marsh Eq. 5):
            #     (Note that values outside the desired polynomial region
            #     have Q=0, and so do not contribute to the fit)
            invEvariance = (np.array(spectrum).reshape(spectral_size, 1)**2 /
                            variance).transpose()
            weightedE = (skysubFrame *
                         np.array(spectrum).reshape(spectral_size, 1) /
                         variance).transpose()  # E / var_E
            invEvariance_subset = invEvariance[Q_cols[0]:Q_cols[1] + 1, :]

            # Define X vector (Marsh Eq. A3):
            X = np.zeros(N * npoly)
            for q in qq:
                X[q] = (weightedE[Q_cols[0]:Q_cols[1] + 1, :] *
                        Qsm[kk[q], :, :] * jjnorm_pow[nn[q]]).sum()
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
                    if np.abs(kk[q] - ll[p]) <= (1. / polyspacing + buffer):
                        if q >= p:
                            # Only compute over non-zero columns:
                            C[q, p] = (Qsm[kk[q], :, :] * qp *
                                       jjnorm_pow[nn[q] + mm[p]] *
                                       invEvariance_subset).sum()
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
                Gsoln[n] = np.polyval(Asoln[n, ::-1],
                                      jjnorm)  # reorder polynomial coef.

            # Compute the profile (Marsh eq. 6) and normalize it:
            profile = np.zeros((spatial_size, spectral_size))
            for i in range(spatial_size):
                profile[i, :] = (Q[:, i, :] * Gsoln).sum(0)

            logging.debug(profile)
            if profile.min() < 0:
                profile[profile < 0] = 0.
            profile /= np.nansum(profile, axis=0)
            profile[~np.isfinite(profile)] = 0.

            # Step6: Revise variance estimates
            modelSpectrum = np.array(spectrum).reshape(
                spectral_size, 1) * profile.transpose()
            modelData = modelSpectrum + residual_frame
            variance0 = np.abs(modelData) + readnoise**2
            variance = variance0 / (
                goodpixelmask + 1e-9
            )  # De-weight bad pixels, avoiding infinite variance

            outlierVariances = (frame - modelData)**2 / variance

            if outlierVariances.max() > cosmicray_sigma**2:
                newBadPixels = True
                # nreject-counting on pixels within the spectral trace
                maxRejectedValue = max(
                    cosmicray_sigma**2,
                    np.sort(outlierVariances[Qmask])[-nreject])
                worstOutliers = (outlierVariances >=
                                 maxRejectedValue).nonzero()
                goodpixelmask[worstOutliers] = False
                numberRejected = len(worstOutliers[0])
            else:
                newBadPixels = False
                numberRejected = 0

            logging.info(
                "Rejected {} pixels in this iteration.".format(numberRejected))

            # Optimal Spectral Extraction: (Horne, Step 8)
            fixSkysubFrame = self._bfixpix(skysubFrame,
                                           ~goodpixelmask,
                                           n=8,
                                           retdat=True)

            spectrum_marsh = np.zeros(spectral_size)
            spectrum_err_marsh = np.zeros(spectral_size)
            is_optimal = np.zeros(spectral_size)

            for i in range(spectral_size):
                aperture = np.arange(
                    int(trace[i]) - width_dn,
                    int(trace[i]) + width_up + 1).astype(int)

                # Horne86 notation
                P = profile[aperture, i]
                V = variance0[i, aperture]
                D = fixSkysubFrame[i, aperture]

                denom = np.nansum(P**2. / V)

                if denom == 0:
                    spectrum_marsh[i] = 0.
                    spectrum_err_marsh[i] = 9e9
                else:
                    spectrum_marsh[i] = np.nansum(P / V * D) / denom
                    spectrum_err_marsh[i] = np.sqrt(np.nansum(P) / denom)
                    is_optimal[i] = True

        spectrum_marsh = spectrum_marsh
        spectrum_err_marsh = spectrum_err_marsh

        return (spectrum_marsh, spectrum_err_marsh, is_optimal, profile,
                variance0)

    def inspect_extracted_spectrum(self,
                                   spec_id=None,
                                   display=True,
                                   renderer='default',
                                   width=1280,
                                   height=720,
                                   return_jsonstring=False,
                                   save_iframe=False,
                                   filename=None,
                                   open_iframe=False):
        """
        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

        """

        if spec_id is not None:

            assert np.in1d(spec_id,
                           list(self.spectrum_list.keys())).all(), 'Some '
            'spec_id provided are not in the spectrum_list.'

        else:

            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        for j in spec_id:

            spec = self.spectrum_list[j]

            len_trace = len(spec.trace)
            count = spec.count
            count_err = spec.count_err
            count_sky = spec.count_sky

            fig = go.Figure(
                layout=dict(autosize=False, height=height, width=width))

            # plot the SNR
            fig.add_trace(
                go.Scatter(x=np.arange(len_trace),
                           y=count / count_err,
                           xaxis='x2',
                           yaxis='y3',
                           line=dict(color='slategrey'),
                           name='Signal-to-Noise Ratio'))

            # extrated source, sky and uncertainty
            fig.add_trace(
                go.Scatter(x=np.arange(len_trace),
                           y=count_sky,
                           xaxis='x2',
                           yaxis='y2',
                           line=dict(color='firebrick'),
                           name='Sky count / (e- / s)'))
            fig.add_trace(
                go.Scatter(x=np.arange(len_trace),
                           y=count_err,
                           xaxis='x2',
                           yaxis='y2',
                           line=dict(color='orange'),
                           name='Uncertainty count / (e- / s)'))
            fig.add_trace(
                go.Scatter(x=np.arange(len_trace),
                           y=count,
                           xaxis='x2',
                           yaxis='y2',
                           line=dict(color='royalblue'),
                           name='Target count / (e- / s)'))

            # Decorative stuff
            fig.update_layout(yaxis2=dict(
                range=[
                    min(
                        np.nanmin(sigma_clip(count, sigma=5., masked=False)),
                        np.nanmin(sigma_clip(count_err, sigma=5.,
                                             masked=False)),
                        np.nanmin(sigma_clip(count_sky, sigma=5.,
                                             masked=False)), 1),
                    max(np.nanmax(count), np.nanmax(count_sky))
                ],
                zeroline=False,
                domain=[0, 1.0],
                showgrid=True,
                title='Count / s',
            ),
                              yaxis3=dict(title='S/N ratio',
                                          anchor="x2",
                                          overlaying="y2",
                                          side="right"),
                              xaxis2=dict(title='Spectral Direction / pixel',
                                          anchor="y2",
                                          matches="x"),
                              legend=go.layout.Legend(x=0,
                                                      y=0.45,
                                                      traceorder="normal",
                                                      font=dict(
                                                          family="sans-serif",
                                                          size=12,
                                                          color="black"),
                                                      bgcolor='rgba(0,0,0,0)'),
                              bargap=0,
                              hovermode='closest',
                              showlegend=True)

            if save_iframe:

                if filename is None:

                    pio.write_html(fig,
                                   'ap_extract_' + str(j) + '.html',
                                   auto_open=open_iframe)

                else:

                    pio.write_html(fig,
                                   filename + '_' + str(j) + '.html',
                                   auto_open=open_iframe)

            if display:

                if renderer == 'default':

                    fig.show()

                else:

                    fig.show(renderer)

            if return_jsonstring:

                return fig.to_json()

    def extract_arc_spec(self,
                         spec_id=None,
                         display=False,
                         renderer='default',
                         width=1280,
                         height=720,
                         return_jsonstring=False,
                         save_iframe=False,
                         filename=None,
                         open_iframe=False):
        '''
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
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

        '''

        if spec_id is not None:

            if spec_id not in list(self.spectrum_list.keys()):

                error_msg = 'The given spec_id does not exist.'
                logging.critical(error_msg)
                raise ValueError(error_msg)

        else:

            # if spec_id is None, all arc spectra are extracted
            spec_id = list(self.spectrum_list.keys())

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if self.arc is None:

            error_msg = 'arc is not provided. Please provide arc by ' +\
                'using add_arc() or with from_twodspec() before ' +\
                'executing find_arc_lines().'
            logging.critical(error_msg)
            raise ValueError(error_msg)

        for i in spec_id:

            spec = self.spectrum_list[i]

            len_trace = len(spec.trace)
            trace = np.nanmean(spec.trace)
            trace_width = np.nanmean(spec.trace_sigma) * 3.

            arc_trace = self.arc[
                max(0, int(trace - trace_width -
                           1)):min(int(trace + trace_width), len_trace), :]
            arc_spec = np.nanmedian(arc_trace, axis=0)

            spec.add_arc_spec(list(arc_spec))
            spec.add_arc_header(self.arc_header)

            # note that the display is adjusted for the chip gaps
            if save_iframe or display or return_jsonstring:

                fig = go.Figure(
                    layout=dict(autosize=False, height=height, width=width))

                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=arc_spec,
                               mode='lines',
                               line=dict(color='royalblue', width=1)))

                fig.update_layout(xaxis=dict(
                    zeroline=False,
                    range=[0, len_trace],
                    title='Spectral Direction / pixel'),
                                  yaxis=dict(zeroline=False,
                                             range=[0, max(arc_spec)],
                                             title='e- / s'),
                                  hovermode='closest',
                                  showlegend=False)

                if save_iframe:

                    if filename is None:

                        pio.write_html(fig,
                                       'arc_spec_' + str(i) + '.html',
                                       auto_open=open_iframe)

                    else:

                        pio.write_html(fig,
                                       filename + '_' + str(i) + '.html',
                                       auto_open=open_iframe)

                if display:

                    if renderer == 'default':

                        fig.show()

                    else:

                        fig.show(renderer)

                if return_jsonstring:

                    return fig.to_json()

    def create_fits(self,
                    output,
                    recreate=False,
                    empty_primary_hdu=True,
                    return_hdu_list=False):
        '''
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

        recreate: boolean (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: boolean (Default: True)
            Set to True to leave the Primary HDU blank (Default: True)
        return_hdu_list: boolean (Default: False)
            Set to True to return the HDU List

        '''

        for i in output.split('+'):

            if i not in ['trace', 'count']:

                error_msg = '{} is not a valid output.'.format(i)
                logging.critical(error_msg)
                raise ValueError(error_msg)

        # Save each trace as a separate FITS file
        for i in range(len(self.spectrum_list)):

            self.spectrum_list[i].create_fits(
                output=output,
                recreate=recreate,
                empty_primary_hdu=empty_primary_hdu,
                return_hdu_list=return_hdu_list)

    def save_fits(self,
                  output='trace+count',
                  filename='TwoDSpecExtracted',
                  overwrite=False,
                  recreate=False,
                  empty_primary_hdu=True):
        '''
        Save the reduced image to disk.

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

        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        overwrite: boolean
            Default is False.
        recreate: boolean (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: boolean (Default: True)
            Set to True to leave the Primary HDU blank (Default: True)

        '''

        filename = os.path.splitext(filename)[0]

        for i in output.split('+'):

            if i not in ['trace', 'count']:

                error_msg = '{} is not a valid output.'.format(i)
                logging.critical(error_msg)
                raise ValueError(error_msg)

        # Save each trace as a separate FITS file
        for i in range(len(self.spectrum_list)):

            filename_i = filename + '_' + output + '_' + str(i)

            self.spectrum_list[i].save_fits(
                output=output,
                filename=filename_i,
                overwrite=overwrite,
                recreate=recreate,
                empty_primary_hdu=empty_primary_hdu)
