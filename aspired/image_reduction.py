import copy
import datetime
import logging
import os

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astroscrappy import detect_cosmics
from ccdproc import Combiner
from plotly import graph_objects as go
from plotly import io as pio

from .util import bfixpix, create_bad_pixel_mask, create_cutoff_mask

__all__ = ['ImageReduction']


class ImageReduction:
    '''
    This class is not intented for quality data reduction, it exists for
    completeness such that users can produce a minimal pipeline with
    a single pacakge. Users should preprocess calibration frames, for
    example, we cannot properly combine arc frames taken with long and
    short exposures for wavelength calibration with both bright and faint
    lines; fringing correction with flat frames; light frames with various
    exposure times.

    '''
    def __init__(self,
                 verbose=True,
                 logger_name='ImageReduction',
                 log_level='WARNING',
                 log_file_folder='default',
                 log_file_name='default'):
        '''
        The initialisation only sets up the logger.

        Parameters
        ----------
        verbose: boolean (Default: True)
            Set to False to suppress all verbose warnings, except for
            critical failure.
        logger_name: str (Default: ImageReduction)
            This will set the name of the logger, if the name is used already,
            it will reference to the existing logger. This will be the
            first part of the default log file name unless log_file_name is
            provided.
        log_level: str (Default: WARNING)
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
        log_file_name: None or str (Default: "default")
            File name of the log, set to None to print to screen only.

        '''

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

            raise ValueError('Unknonw logging level.')

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] '
            '%(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S')

        if log_file_name is None:

            # Only print log to screen
            self.handler = logging.StreamHandler()

        else:

            if log_file_name == 'default':

                log_file_name = '{}_{}.log'.format(
                    logger_name,
                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

            # Save log to file
            if log_file_folder == 'default':

                log_file_folder = ''

            self.handler = logging.FileHandler(
                os.path.join(log_file_folder, log_file_name), 'a+')

        self.handler.setFormatter(formatter)
        if (self.logger.hasHandlers()):

            self.logger.handlers.clear()

        self.logger.addHandler(self.handler)

        # FITS keyword standard recommends XPOSURE, but most observatories
        # use EXPTIME for supporting iraf. Also included a few other keywords
        # which are the proxy-exposure times at best. ASPIRED will use the
        # first keyword found on the list, if all failed, an exposure time of
        # 1 second will be applied. A warning will be promted.
        self.exptime_keyword = [
            'XPOSURE', 'EXPOSURE', 'EXPTIME', 'EXPOSED', 'TELAPSED', 'ELAPSED'
        ]

        self.saxis = 1

        self.combinetype_light = 'median'
        self.sigma_clipping_light = True
        self.clip_low_light = 5
        self.clip_high_light = 5
        self.exptime_light = None
        self.exptime_light_keyword = None

        self.combinetype_dark = 'median'
        self.sigma_clipping_dark = True
        self.clip_low_dark = 5
        self.clip_high_dark = 5
        self.exptime_dark = None
        self.exptime_dark_keyword = None

        self.combinetype_flat = 'median'
        self.sigma_clipping_flat = True
        self.clip_low_flat = 5
        self.clip_high_flat = 5
        self.exptime_flat = None
        self.exptime_flat_keyword = None

        self.combinetype_bias = 'median'
        self.sigma_clipping_bias = False
        self.clip_low_bias = 5
        self.clip_high_bias = 5

        self.combinetype_arc = 'median'
        self.sigma_clipping_arc = False
        self.clip_low_arc = 5
        self.clip_high_arc = 5

        self.cosmicray = False
        self.gain = 1.0
        self.readnoise = 0.0
        self.fsmode = 'convolve'
        self.psfmodel = 'gaussy'
        self.cr_kwargs = None

        self.cutoff = 60000.0
        self.grow = False
        self.iterations = 1
        self.diagonal = False
        self.pixel_healed = False

        self.bias_list = []
        self.dark_list = []
        self.flat_list = []
        self.arc_list = []
        self.light = []

        self.bias_main = None
        self.dark_main = None
        self.flat_main = None
        self.arc_main = None
        self.light_main = None

        self.flat_reduced = None
        self.light_reduced = None

        self.image_fits = None
        self.bad_pixel_mask = None
        self.bad_pixels = False
        self.saturation_mask = None
        self.saturated = False
        self.bad_mask = None

        self.light_filename = []
        self.arc_filename = []
        self.dark_filename = []
        self.flat_filename = []
        self.bias_filename = []

        self.light_CCDData = []
        self.arc_CCDData = []
        self.dark_CCDData = []
        self.flat_CCDData = []
        self.bias_CCDData = []

        self.light_header = []
        self.arc_header = []
        self.dark_header = []
        self.flat_header = []
        self.bias_header = []

        self.light_time = []
        self.arc_time = []
        self.dark_time = []
        self.flat_time = []

    def add_filelist(self, filelist, ftype='csv', delimiter=None):
        '''
        If a field-flattened 2D spectrum is already avaialble, it can be
        the only listed item. Set it as a 'light' frame.

        Parameters
        ----------
        filelist: str
            File location, does not support URL
        ftype: str (Default: "csv")
            One of csv, tsv and ascii. Default is csv.
        delimiter: str (Default: None)
            Set the delimiter. This overrides ftype.

        '''

        if delimiter is not None:

            self.delimiter = delimiter
            self.ftype = None

        else:

            self.ftype = ftype

            if ftype == 'csv':

                self.delimiter = ','

            elif ftype == 'tsv':

                self.delimiter = '\t'

            elif ftype == 'ascii':

                self.delimiter = ' '

        # import file with first column as image type and second column as
        # file path
        if isinstance(filelist, str):

            if os.path.isabs(filelist):

                self.filelist = filelist

            else:

                self.filelist = os.path.abspath(filelist)

            self.logger.debug('The filelist is: {}'.format(self.filelist))

            # Check if running on Windows
            if os.name == 'nt':

                self.filelist_abspath = self.filelist.rsplit('\\', 1)[0]

            else:

                self.filelist_abspath = self.filelist.rsplit('/', 1)[0]

            self.logger.debug(
                'The absolute path of the filelist is: {}'.format(
                    self.filelist_abspath))
            self.logger.info('Loading filelist from {}.'.format(self.filelist))
            self.filelist = np.loadtxt(self.filelist,
                                       delimiter=self.delimiter,
                                       dtype='U',
                                       ndmin=2)

            if np.shape(self.filelist)[1] == 3:

                self.logger.debug('filelist contains 3 columns.')

            elif np.shape(self.filelist)[1] == 2:

                self.logger.debug('filelist contains 2 column.')

            else:

                error_msg = 'Please provide a text file with 2 or 3 ' +\
                    'columns: where the first column is the image type ' +\
                    'and the second column is the file path, and optional ' +\
                    'third column being the #HDU.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

        elif isinstance(filelist, np.ndarray):

            self.filelist = filelist
            self.filelist_abspath = ''
            self.logger.info('Loading filelist from an numpy.ndarray.')
            if np.shape(self.filelist)[1] == 3:

                self.logger.debug('filelist contains 3 columns.')

            elif np.shape(self.filelist)[1] == 2:

                self.logger.debug('filelist contains 2 columns.')

            else:

                error_msg = 'Please provide a numpy.ndarray with at ' +\
                    'least 2 columns: where the first column is the ' +\
                    'image type and the second column is the file ' +\
                    'path, and optional third column being the #HDU.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

        else:

            error_msg = 'Please provide a file path to the file list.'
            self.logger.critical(error_msg)
            raise TypeError(error_msg)

        if np.shape(self.filelist)[1] == 3:

            self.logger.debug('filelist contains 3 columns.')
            self.imtype = self.filelist[:, 0].astype('object')
            self.impath = self.filelist[:, 1].astype('object')
            self.hdunum = self.filelist[:, 2].astype('int')

        else:

            self.logger.debug('filelist contains 2 columns.')
            self.imtype = self.filelist[:, 0].astype('object')
            self.impath = self.filelist[:, 1].astype('object')
            self.hdunum = np.zeros(len(self.imtype)).astype('int')

        for i, im in enumerate(self.impath):

            if not os.path.isabs(im):

                self.impath[i] = os.path.join(self.filelist_abspath,
                                              im.strip())

            self.logger.debug(self.impath[i])

        # If there are multiple rows, which is in most of the cases
        self.bias_list = self.impath[self.imtype == 'bias']
        self.dark_list = self.impath[self.imtype == 'dark']
        self.flat_list = self.impath[self.imtype == 'flat']
        self.arc_list = self.impath[self.imtype == 'arc']
        self.light_list = self.impath[self.imtype == 'light']

        self.bias_hdunum = self.hdunum[self.imtype == 'bias']
        self.dark_hdunum = self.hdunum[self.imtype == 'dark']
        self.flat_hdunum = self.hdunum[self.imtype == 'flat']
        self.arc_hdunum = self.hdunum[self.imtype == 'arc']
        self.light_hdunum = self.hdunum[self.imtype == 'light']

    def set_saxis(self, saxis=None):

        if saxis is None:

            if 'SAXIS' in self.light_CCDData[0].header:

                self.saxis = int(self.light_CCDData[0].header['SAXIS'])

            else:

                self.saxis = 1

        elif np.in1d(saxis, [0, 1]).any():

            self.saxis = saxis

        else:

            self.saxis = 1

        self.logger.info('Saxis is found/set to be {}.'.format(self.saxis))

    def set_properties(self,
                       saxis=None,
                       combinetype_light='median',
                       sigma_clipping_light=True,
                       clip_low_light=5,
                       clip_high_light=5,
                       exptime_light=None,
                       exptime_light_keyword=None,
                       combinetype_arc='median',
                       sigma_clipping_arc=False,
                       clip_low_arc=5,
                       clip_high_arc=5,
                       combinetype_dark='median',
                       sigma_clipping_dark=True,
                       clip_low_dark=5,
                       clip_high_dark=5,
                       exptime_dark=None,
                       exptime_dark_keyword=None,
                       combinetype_bias='median',
                       sigma_clipping_bias=False,
                       clip_low_bias=5,
                       clip_high_bias=5,
                       combinetype_flat='median',
                       sigma_clipping_flat=True,
                       clip_low_flat=5,
                       clip_high_flat=5,
                       exptime_flat=None,
                       exptime_flat_keyword=None,
                       cosmicray=False,
                       gain=1.0,
                       readnoise=0.0,
                       fsmode='convolve',
                       psfmodel='gaussy',
                       cutoff=60000.,
                       grow=False,
                       iterations=1,
                       diagonal=False,
                       **kwargs):
        '''
        Parameters
        ----------
        sxais: int, 0 or 1 (Default: None)
            OVERRIDE the SAXIS value in the FITS header, or to provide the
            SAXIS if it does not exist
        saxis_keyword: str (Default: None)
            Supply the saxis keyword used in the header.
        combinetype_light: str (Default: 'median')
            'average' or 'median' for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_light: bool (Default: True)
            Perform sigma clipping if set to True. sigma is computed with the
            numpy.ma.std method
        clip_low_light: float (Default: 5)
            Set the lower threshold of the sigma clipping
        clip_high_light: float (Default: 5)
            Set the upper threshold of the sigma clipping
        exptime_light: float (Default: None)
            OVERRIDE the exposure time value in the FITS header, or to provide
            one if the keyword does not exist
        exptime_light_keyword: str (Default: None)
            HDU keyword for the exposure time of the light frame
        combinetype_dark: str (Default: 'median')
            'average' or 'median' for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_dark: bool (Default: True)
            Perform sigma clipping if set to True. sigma is computed with the
            numpy.ma.std method
        clip_low_dark: float (Default: 5)
            Set the lower threshold of the sigma clipping
        clip_high_dark: float (Default: 5)
            Set the upper threshold of the sigma clipping
        exptime_dark: float (Default: None)
            OVERRIDE the exposure time value in the FITS header, or to provide
            one if the keyword does not exist
        exptime_dark_keyword: str (Default: None)
            HDU keyword for the exposure time of the dark frame
        combinetype_bias: str (Default: 'median')
            'average' or 'median' for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_bias: bool (Default: False)
            Perform sigma clipping if set to True. sigma is computed with the
            numpy.ma.std method
        clip_low_bias: float (Default: 5)
            Set the lower threshold of the sigma clipping
        clip_high_bias: float (Default: 5)
            Set the upper threshold of the sigma clipping
        combinetype_flat: str (Default: 'median')
            'average' or 'median' for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_flat: bool (Default: True)
            Perform sigma clipping if set to True. sigma is computed with the
            numpy.ma.std method
        clip_low_flat: float (Default: 5)
            Set the lower threshold of the sigma clipping
        clip_high_flat: float (Default: 5)
            Set the upper threshold of the sigma clipping
        exptime_flat: float (Default: None)
            OVERRIDE the exposure time value in the FITS header, or to provide
            one if the keyword does not exist
        exptime_flat_keyword: str (Default: None)
            HDU keyword for the exposure time of the flat frame
        cosmicray: bool (Default: False)
            Set to True to remove cosmic rays, this directly alter the reduced
            image data. We only explicitly include the 4 most important
            parameters in this function: `gain`, `readnoise`, `fsmode`, and
            `psfmodel`, the rest can be configured with kwargs.
        gain: float (Default: 1.0)
            Gain of the image (electrons / ADU). We always need to work in
            electrons for cosmic ray detection.
        readnoise: float (Default: 0.0)
            Read noise of the image (electrons). Used to generate the noise
            model of the image.
        fsmode: str (Default: 'convolve')
            Method to build the fine structure image: `median`: Use the median
            filter in the standard LA Cosmic algorithm. `convolve`: Convolve
            the image with the psf kernel to calculate the fine structure
            image.
        psfmodel: str (Default: 'gaussy')
            Model to use to generate the psf kernel if fsmode is `convolve`
            and psfk is None. The current choices are Gaussian and Moffat
            profiles. `gauss` and 'moffat' produce circular PSF kernels. The
            `gaussx` and `gaussy` produce Gaussian kernels in the x and y
            directions respectively. `gaussxy` and `gaussyx` apply the
            Gaussian kernels in the x then the y direction, and first y then
            x direction, respectively.
        cutoff: float (Default: 60000.)
            This sets the (lower and) upper limit of electron count.
        grow: bool (Default: False)
            Set to True to grow the mask, see `grow_mask()`
        iterations: int (Default: 1)
            The number of pixel growth along the Cartesian axes directions.
        diagonal: bool (Default: False)
            Set to True to grow in the diagonal directions.
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

        '''

        # FITS keyword standard recommends XPOSURE, but most observatories
        # use EXPTIME for supporting iraf. Also included a few other keywords
        # which are the proxy-exposure times at best. ASPIRED will use the
        # first keyword found on the list, if all failed, an exposure time of
        # 1 second will be applied. A warning will be promted.
        self.exptime_keyword = [
            'XPOSURE', 'EXPOSURE', 'EXPTIME', 'EXPOSED', 'TELAPSED', 'ELAPSED'
        ]

        self.saxis = saxis

        self.set_light_properties(combinetype_light=combinetype_light,
                                  sigma_clipping_light=sigma_clipping_light,
                                  clip_low_light=clip_low_light,
                                  clip_high_light=clip_high_light,
                                  exptime_light=exptime_light,
                                  exptime_light_keyword=exptime_light_keyword)

        self.set_dark_properties(combinetype_dark=combinetype_dark,
                                 sigma_clipping_dark=sigma_clipping_dark,
                                 clip_low_dark=clip_low_dark,
                                 clip_high_dark=clip_high_dark,
                                 exptime_dark=exptime_dark,
                                 exptime_dark_keyword=exptime_dark_keyword)

        self.set_flat_properties(combinetype_flat=combinetype_flat,
                                 sigma_clipping_flat=sigma_clipping_flat,
                                 clip_low_flat=clip_low_flat,
                                 clip_high_flat=clip_high_flat,
                                 exptime_flat=exptime_flat,
                                 exptime_flat_keyword=exptime_flat_keyword)

        self.set_bias_properties(combinetype_bias=combinetype_bias,
                                 sigma_clipping_bias=sigma_clipping_bias,
                                 clip_low_bias=clip_low_bias,
                                 clip_high_bias=clip_high_bias)

        self.set_arc_properties(combinetype_arc=combinetype_arc,
                                sigma_clipping_arc=sigma_clipping_arc,
                                clip_low_arc=clip_low_arc,
                                clip_high_arc=clip_high_arc)

        self.cosmicray = cosmicray
        self.gain = gain
        self.readnoise = readnoise
        self.fsmode = fsmode
        self.psfmodel = psfmodel
        self.cr_kwargs = kwargs

        self.cutoff = cutoff
        self.grow = grow
        self.iterations = iterations
        self.diagonal = diagonal

    def set_light_properties(self,
                             combinetype_light=-1,
                             sigma_clipping_light=-1,
                             clip_low_light=-1,
                             clip_high_light=-1,
                             exptime_light=-1,
                             exptime_light_keyword=-1):

        if combinetype_light is None:

            self.combinetype_light = 'median'
            self.logger.warning(
                'Unknown combinetype_light, it is set as median.')

        else:

            if isinstance(combinetype_light, (float, int)):

                pass

            elif isinstance(combinetype_light, str):

                if combinetype_light in ['average', 'median']:

                    # use the given readnoise value
                    self.combinetype_light = combinetype_light
                    self.logger.info('combinetype_light is set to {}.'.format(
                        self.combinetype_light))

                else:

                    self.combinetype_light = 'median'
                    self.logger.warning(
                        'Unknown combinetype_light, it is set as median.')

            else:

                self.combinetype_light = 'median'
                self.logger.warning(
                    'Unknown combinetype_light, it is set as median.')

        if sigma_clipping_light is None:

            self.sigma_clipping_light = True
            self.logger.warning(
                'Unknown sigma_clipping_light, it is set to True.')

        else:

            if isinstance(sigma_clipping_light, (float, int)):

                pass

            elif isinstance(sigma_clipping_light, bool):

                self.sigma_clipping_light = sigma_clipping_light

            else:

                self.sigma_clipping_light = True
                self.logger.warning(
                    'Unknown sigma_clipping_light, it is set to True.')

        if clip_low_light is None:

            self.clip_low_light = 5
            self.logger.warning(
                'Unknown sigma_clipping_light, it is set to 5.')

        else:

            if isinstance(clip_low_light, (float, int)):

                if clip_low_light > 0:

                    self.clip_low_light = clip_low_light

                else:

                    pass

            else:

                self.clip_low_light = 5
                self.logger.warning('Unknown clip_low_light, it is set to 5.')

        if clip_high_light is None:

            self.clip_high_light = 5
            self.logger.warning(
                'Unknown sigma_clipping_light, it is set to 5.')

        else:

            if isinstance(clip_high_light, (float, int)):

                if clip_high_light > 0:

                    self.clip_high_light = clip_high_light

                else:

                    pass

            else:

                self.clip_high_light = 5
                self.logger.warning('Unknown clip_high_light, it is set to 5.')

        if exptime_light is None:

            self.exptime_light = 5
            self.logger.warning(
                'Unknown sigma_clipping_light, it is set to 5.')

        else:

            if isinstance(exptime_light, (float, int)):

                if exptime_light > 0:

                    self.exptime_light = exptime_light

                else:

                    pass

            else:

                self.exptime_light = exptime_light
                self.logger.warning('Unknown exptime_light, it is set to 1.')

        if exptime_light_keyword is None:

            self.exptime_light_keyword = self.exptime_keyword
            self.logger.warning(
                'Unknown exptime_light_keyword, it is set to {}.'.format(
                    self.exptime_keyword))

        else:

            if isinstance(exptime_light_keyword, (float, int)):

                pass

            elif isinstance(exptime_light_keyword, str):

                self.exptime_light_keyword

            else:

                self.exptime_light_keyword = self.exptime_keyword
                self.logger.warning(
                    'Unknown exptime_light_keyword, it is set to {}.'.format(
                        self.exptime_keyword))

    def set_dark_properties(self,
                            combinetype_dark=-1,
                            sigma_clipping_dark=-1,
                            clip_low_dark=-1,
                            clip_high_dark=-1,
                            exptime_dark=-1,
                            exptime_dark_keyword=-1):

        if combinetype_dark is None:

            self.combinetype_dark = 'median'
            self.logger.warning(
                'Unknown combinetype_dark, it is set as median.')

        else:

            if isinstance(combinetype_dark, (float, int)):

                pass

            elif isinstance(combinetype_dark, str):

                if combinetype_dark in ['average', 'median']:

                    # use the given readnoise value
                    self.combinetype_dark = combinetype_dark
                    self.logger.info('combinetype_dark is set to {}.'.format(
                        self.combinetype_dark))

                else:

                    self.combinetype_dark = 'median'
                    self.logger.warning(
                        'Unknown combinetype_dark, it is set as median.')

            else:

                self.combinetype_dark = 'median'
                self.logger.warning(
                    'Unknown combinetype_dark, it is set as median.')

        if sigma_clipping_dark is None:

            self.sigma_clipping_dark = True
            self.logger.warning(
                'Unknown sigma_clipping_dark, it is set to True.')

        else:

            if isinstance(sigma_clipping_dark, (float, int)):

                pass

            elif isinstance(sigma_clipping_dark, bool):

                self.sigma_clipping_dark = sigma_clipping_dark

            else:

                self.sigma_clipping_dark = True
                self.logger.warning(
                    'Unknown sigma_clipping_dark, it is set to True.')

        if clip_low_dark is None:

            self.clip_low_dark = 5
            self.logger.warning('Unknown sigma_clipping_dark, it is set to 5.')

        else:

            if isinstance(clip_low_dark, (float, int)):

                if clip_low_dark > 0:

                    self.clip_low_dark = clip_low_dark

                else:

                    pass

            else:

                self.clip_low_dark = 5
                self.logger.warning('Unknown clip_low_dark, it is set to 5.')

        if clip_high_dark is None:

            self.clip_high_dark = 5
            self.logger.warning('Unknown sigma_clipping_dark, it is set to 5.')

        else:

            if isinstance(clip_high_dark, (float, int)):

                if clip_high_dark > 0:

                    self.clip_high_dark = clip_high_dark

                else:

                    pass

            else:

                self.clip_high_dark = 5
                self.logger.warning('Unknown clip_high_dark, it is set to 5.')

        if exptime_dark is None:

            self.exptime_dark = 5
            self.logger.warning('Unknown sigma_clipping_dark, it is set to 5.')

        else:

            if isinstance(exptime_dark, (float, int)):

                if exptime_dark > 0:

                    self.exptime_dark = exptime_dark

                else:

                    pass

            else:

                self.exptime_dark = exptime_dark
                self.logger.warning('Unknown exptime_dark, it is set to 1.')

        if exptime_dark_keyword is None:

            self.exptime_dark_keyword = self.exptime_keyword
            self.logger.warning(
                'Unknown exptime_dark_keyword, it is set to {}.'.format(
                    self.exptime_keyword))

        else:

            if isinstance(exptime_dark_keyword, (float, int)):

                pass

            elif isinstance(exptime_dark_keyword, str):

                self.exptime_dark_keyword

            else:

                self.exptime_dark_keyword = self.exptime_keyword
                self.logger.warning(
                    'Unknown exptime_dark_keyword, it is set to {}.'.format(
                        self.exptime_keyword))

    def set_flat_properties(self,
                            combinetype_flat=-1,
                            sigma_clipping_flat=-1,
                            clip_low_flat=-1,
                            clip_high_flat=-1,
                            exptime_flat=-1,
                            exptime_flat_keyword=-1):

        if combinetype_flat is None:

            self.combinetype_flat = 'median'
            self.logger.warning(
                'Unknown combinetype_flat, it is set as median.')

        else:

            if isinstance(combinetype_flat, (float, int)):

                pass

            elif isinstance(combinetype_flat, str):

                if combinetype_flat in ['average', 'median']:

                    # use the given readnoise value
                    self.combinetype_flat = combinetype_flat
                    self.logger.info('combinetype_flat is set to {}.'.format(
                        self.combinetype_flat))

                else:

                    self.combinetype_flat = 'median'
                    self.logger.warning(
                        'Unknown combinetype_flat, it is set as median.')

            else:

                self.combinetype_flat = 'median'
                self.logger.warning(
                    'Unknown combinetype_flat, it is set as median.')

        if sigma_clipping_flat is None:

            self.sigma_clipping_flat = True
            self.logger.warning(
                'Unknown sigma_clipping_flat, it is set to True.')

        else:

            if isinstance(sigma_clipping_flat, (float, int)):

                pass

            elif isinstance(sigma_clipping_flat, bool):

                self.sigma_clipping_flat = sigma_clipping_flat

            else:

                self.sigma_clipping_flat = True
                self.logger.warning(
                    'Unknown sigma_clipping_flat, it is set to True.')

        if clip_low_flat is None:

            self.clip_low_flat = 5
            self.logger.warning('Unknown sigma_clipping_flat, it is set to 5.')

        else:

            if isinstance(clip_low_flat, (float, int)):

                if clip_low_flat > 0:

                    self.clip_low_flat = clip_low_flat

                else:

                    pass

            else:

                self.clip_low_flat = 5
                self.logger.warning('Unknown clip_low_flat, it is set to 5.')

        if clip_high_flat is None:

            self.clip_high_flat = 5
            self.logger.warning('Unknown sigma_clipping_flat, it is set to 5.')

        else:

            if isinstance(clip_high_flat, (float, int)):

                if clip_high_flat > 0:

                    self.clip_high_flat = clip_high_flat

                else:

                    pass

            else:

                self.clip_high_flat = 5
                self.logger.warning('Unknown clip_high_flat, it is set to 5.')

        if exptime_flat is None:

            self.exptime_flat = 5
            self.logger.warning('Unknown sigma_clipping_flat, it is set to 5.')

        else:

            if isinstance(exptime_flat, (float, int)):

                if exptime_flat > 0:

                    self.exptime_flat = exptime_flat

                else:

                    pass

            else:

                self.exptime_flat = exptime_flat
                self.logger.warning('Unknown exptime_flat, it is set to 1.')

        if exptime_flat_keyword is None:

            self.exptime_flat_keyword = self.exptime_keyword
            self.logger.warning(
                'Unknown exptime_flat_keyword, it is set to {}.'.format(
                    self.exptime_keyword))

        else:

            if isinstance(exptime_flat_keyword, (float, int)):

                pass

            elif isinstance(exptime_flat_keyword, str):

                self.exptime_flat_keyword

            else:

                self.exptime_flat_keyword = self.exptime_keyword
                self.logger.warning(
                    'Unknown exptime_flat_keyword, it is set to {}.'.format(
                        self.exptime_keyword))

    def set_bias_properties(self,
                            combinetype_bias=-1,
                            sigma_clipping_bias=-1,
                            clip_low_bias=-1,
                            clip_high_bias=-1):

        if combinetype_bias is None:

            self.combinetype_bias = 'median'
            self.logger.warning(
                'Unknown combinetype_bias, it is set as median.')

        else:

            if isinstance(combinetype_bias, (float, int)):

                pass

            elif isinstance(combinetype_bias, str):

                if combinetype_bias in ['average', 'median']:

                    # use the given readnoise value
                    self.combinetype_bias = combinetype_bias
                    self.logger.info('combinetype_dark is set to {}.'.format(
                        self.combinetype_bias))

                else:

                    self.combinetype_bias = 'median'
                    self.logger.warning(
                        'Unknown combinetype_bias, it is set as median.')

            else:

                self.combinetype_bias = 'median'
                self.logger.warning(
                    'Unknown combinetype_bias, it is set as median.')

        if sigma_clipping_bias is None:

            self.sigma_clipping_bias = True
            self.logger.warning(
                'Unknown sigma_clipping_bias, it is set to True.')

        else:

            if isinstance(sigma_clipping_bias, (float, int)):

                pass

            elif isinstance(sigma_clipping_bias, bool):

                self.sigma_clipping_bias = sigma_clipping_bias

            else:

                self.sigma_clipping_bias = True
                self.logger.warning(
                    'Unknown sigma_clipping_bias, it is set to True.')

        if clip_low_bias is None:

            self.clip_low_bias = 5
            self.logger.warning('Unknown sigma_clipping_bias, it is set to 5.')

        else:

            if isinstance(clip_low_bias, (float, int)):

                if clip_low_bias > 0:

                    self.clip_low_bias = clip_low_bias

                else:

                    pass

            else:

                self.clip_low_bias = 5
                self.logger.warning('Unknown clip_low_bias, it is set to 5.')

        if clip_high_bias is None:

            self.clip_high_bias = 5
            self.logger.warning('Unknown sigma_clipping_bias, it is set to 5.')

        else:

            if isinstance(clip_high_bias, (float, int)):

                if clip_high_bias > 0:

                    self.clip_high_bias = clip_high_bias

                else:

                    pass

            else:

                self.clip_high_bias = 5
                self.logger.warning('Unknown clip_high_bias, it is set to 5.')

    def set_arc_properties(self,
                           combinetype_arc=-1,
                           sigma_clipping_arc=-1,
                           clip_low_arc=-1,
                           clip_high_arc=-1):

        if combinetype_arc is None:

            self.combinetype_arc = 'median'
            self.logger.warning(
                'Unknown combinetype_arc, it is set as median.')

        else:

            if isinstance(combinetype_arc, (float, int)):

                pass

            elif isinstance(combinetype_arc, str):

                if combinetype_arc in ['average', 'median']:

                    # use the given readnoise value
                    self.combinetype_arc = combinetype_arc
                    self.logger.info('combinetype_dark is set to {}.'.format(
                        self.combinetype_arc))

                else:

                    self.combinetype_arc = 'median'
                    self.logger.warning(
                        'Unknown combinetype_arc, it is set as median.')

            else:

                self.combinetype_arc = 'median'
                self.logger.warning(
                    'Unknown combinetype_arc, it is set as median.')

        if sigma_clipping_arc is None:

            self.sigma_clipping_arc = True
            self.logger.warning(
                'Unknown sigma_clipping_arc, it is set to True.')

        else:

            if isinstance(sigma_clipping_arc, (float, int)):

                pass

            elif isinstance(sigma_clipping_arc, bool):

                self.sigma_clipping_arc = sigma_clipping_arc

            else:

                self.sigma_clipping_arc = True
                self.logger.warning(
                    'Unknown sigma_clipping_arc, it is set to True.')

        if clip_low_arc is None:

            self.clip_low_arc = 5
            self.logger.warning('Unknown sigma_clipping_arc, it is set to 5.')

        else:

            if isinstance(clip_low_arc, (float, int)):

                if clip_low_arc > 0:

                    self.clip_low_arc = clip_low_arc

                else:

                    pass

            else:

                self.clip_low_arc = 5
                self.logger.warning('Unknown clip_low_arc, it is set to 5.')

        if clip_high_arc is None:

            self.clip_high_arc = 5
            self.logger.warning('Unknown sigma_clipping_arc, it is set to 5.')

        else:

            if isinstance(clip_high_arc, (float, int)):

                if clip_high_arc > 0:

                    self.clip_high_arc = clip_high_arc

                else:

                    pass

            else:

                self.clip_high_arc = 5
                self.logger.warning('Unknown clip_high_arc, it is set to 5.')

    def load_data(self):

        # If there is no science frames, nothing to process.
        assert (self.light_list.size > 0), 'There is no light frame.'

        # Only load the science data, other types of image data are loaded by
        # separate methods.
        for i in range(self.light_list.size):

            # Open all the light frames
            self.logger.debug('Loading light frame: {}.'.format(
                self.light_list[i]))
            light = fits.open(self.light_list[i])[self.light_hdunum[i]]

            data, header, exposure_time = self._get_data_and_header(light)
            self.add_light(data, header, exposure_time)

            # Cosmic ray cleaning
            if self.cosmicray:

                self.logger.info('Removing cosmic rays in mode: {}.'.format(
                    self.psfmodel))

                if self.fsmode == 'convolve':

                    if self.psfmodel == 'gaussyx':

                        self.light_CCDData[i].data = detect_cosmics(
                            self.light_CCDData[i].data,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            fsmode='convolve',
                            psfmodel='gaussy',
                            **self.cr_kwargs)[1]

                        self.light_CCDData[i].data = detect_cosmics(
                            self.light_CCDData[i].data,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            fsmode='convolve',
                            psfmodel='gaussx',
                            **self.cr_kwargs)[1]

                    elif self.psfmodel == 'gaussxy':

                        self.light_CCDData[i].data = detect_cosmics(
                            self.light_CCDData[i].data,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            fsmode='convolve',
                            psfmodel='gaussx',
                            **self.cr_kwargs)[1]

                        self.light_CCDData[i].data = detect_cosmics(
                            self.light_CCDData[i].data,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            fsmode='convolve',
                            psfmodel='gaussy',
                            **self.cr_kwargs)[1]

                    else:

                        self.light_CCDData[i].data = detect_cosmics(
                            self.light_CCDData[i].data,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            fsmode='convolve',
                            psfmodel=self.psfmodel,
                            **self.cr_kwargs)[1]

                else:

                    self.light_CCDData[i].data = detect_cosmics(
                        self.light_CCDData[i].data,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode=self.fsmode,
                        psfmodel=self.psfmodel,
                        **self.cr_kwargs)[1]

            self.logger.debug('Light frame header: {}.'.format(
                self.light_header[i]))

            self.logger.debug('Appending light filename: {}.'.format(
                self.light_list[i].split('/')[-1]))
            self.light_filename.append(self.light_list[i].split('/')[-1])

            saturation_mask, saturated = create_cutoff_mask(
                self.light_CCDData[i].data, self.cutoff, self.grow,
                self.iterations, self.diagonal)

            if self.saturation_mask is None:

                self.saturation_mask = saturation_mask
                self.saturated = saturated

            else:

                self.saturation_mask *= saturation_mask
                self.saturated *= saturated

            self.light_CCDData[i].data /= self.light_time[i]

        self.light_main = self.combine_light()

        # FITS keyword standard for the spectral direction, if FITS header
        # does not contain SAXIS, the image in assumed to have the spectra
        # going across (left to right corresponds to blue to red). All frames
        # get rotated in the anti-clockwise direction if the first light frame
        # has a verticle spectrum (top to bottom corresponds to blue to red).
        self.set_saxis(self.saxis)

        # Get the arcs if available
        if len(self.arc_list) > 0:

            # Combine the arcs
            for i in range(self.arc_list.size):

                # Open all the light frames
                arc = fits.open(self.arc_list[i])[self.arc_hdunum[i]]

                data, header, _ = self._get_data_and_header(arc)
                self.add_arc(data, header)

                self.logger.debug('Arc frame header: {}.'.format(
                    self.arc_header[i]))

                self.logger.debug('Appending arc filename: {}.'.format(
                    self.arc_list[i].split('/')[-1]))
                self.arc_filename.append(self.arc_list[i].split('/')[-1])

            # combine the arc frames
            self.arc_main = self.combine_arc()

        # Get the darks if available
        if len(self.dark_list) > 0:

            for i in range(self.dark_list.size):

                # Open all the dark frames
                dark = fits.open(self.dark_list[i])[self.dark_hdunum[i]]

                data, header, exposure_time = self._get_data_and_header(dark)
                self.add_dark(data, header, exposure_time)

                self.logger.debug('Dark frame header: {}.'.format(
                    self.dark_header[i]))

                self.logger.debug('Appending dark filename: {}.'.format(
                    self.dark_list[i].split('/')[-1]))
                self.dark_filename.append(self.dark_list[i].split('/')[-1])

                self.dark_CCDData[i].data /= self.dark_time[i]

            # combine the arc frames
            self.dark_main = self.combine_dark()

        # Get the flats if available
        if len(self.flat_list) > 0:

            for i in range(self.flat_list.size):

                # Open all the flatfield frames
                flat = fits.open(self.flat_list[i])[self.flat_hdunum[i]]

                data, header, exposure_time = self._get_data_and_header(flat)
                self.add_flat(data, header, exposure_time)

                self.logger.debug('Flat frame header: {}.'.format(
                    self.flat_header[i]))

                self.logger.debug('Appending flat filename: {}.'.format(
                    self.flat_list[i].split('/')[-1]))
                self.flat_filename.append(self.flat_list[i].split('/')[-1])

                self.flat_CCDData[i].data /= self.flat_time[i]

            # combine the arc frames
            self.flat_main = self.combine_flat()

        # Get the biases if available
        if len(self.bias_list) > 0:

            for i in range(self.bias_list.size):

                # Open all the flatfield frames
                bias = fits.open(self.bias_list[i])[self.bias_hdunum[i]]

                data, header, _ = self._get_data_and_header(bias)
                self.add_bias(data, header)

                self.logger.debug('Flat frame header: {}.'.format(
                    self.bias_header[i]))

                self.logger.debug('Appending flat filename: {}.'.format(
                    self.bias_list[i].split('/')[-1]))
                self.bias_filename.append(self.bias_list[i].split('/')[-1])

            # combine the arc frames
            self.bias_main = self.combine_bias()

    def add_light(self, light, header, exposure_time):

        if type(light) == np.ndarray:

            light = CCDData(light, unit=u.ct)

        self.light_CCDData.append(light)
        self.light_header.append(header)
        self.light_time.append(exposure_time)

    def add_arc(self, arc, header):

        if type(arc) == np.ndarray:

            arc = CCDData(arc, unit=u.ct)

        self.arc_CCDData.append(arc)
        self.arc_header.append(header)

    def add_flat(self, flat, header, exposure_time):

        if type(flat) == np.ndarray:

            flat = CCDData(flat, unit=u.ct)

        self.flat_CCDData.append(flat)
        self.flat_header.append(header)
        self.flat_time.append(exposure_time)

    def add_dark(self, dark, header, exposure_time):

        if type(dark) == np.ndarray:

            dark = CCDData(dark, unit=u.ct)

        self.dark_CCDData.append(dark)
        self.dark_header.append(header)
        self.dark_time.append(exposure_time)

    def add_bias(self, bias, header):

        if type(bias) == np.ndarray:

            bias = CCDData(bias, unit=u.ct)

        self.bias_CCDData.append(bias)
        self.bias_header.append(header)

    def _get_data_and_header(self, input):

        if type(input) == 'astropy.io.fits.hdu.hdulist.HDUList':

            input = input[0]
            self.logger.warning('An HDU list is provided, only the first '
                                'HDU will be read.')
        input_shape = np.shape(input.data)
        self.logger.debug('data.data has shape {}.'.format(input_shape))

        # Normal case
        if len(input_shape) == 2:

            self.logger.debug('input.data is 2 dimensional.')
            input_CCDData = CCDData(input.data.astype('float'), unit=u.ct)
            input_header = input.header

        # Try to trap common error when saving FITS file
        # Case with multiple image extensions, we only take the first one
        elif len(input_shape) == 3:

            self.logger.debug('input.data is 3 dimensional.')
            input_CCDData = CCDData(input.data[0].astype('float'), unit=u.ct)
            input_header = input.header

        # Case with an extra bracket when saving
        elif len(input_shape) == 4:

            self.logger.debug('input.data is 4 dimensional, there is most '
                              'likely an extra bracket, attempting to go in '
                              'one level.')

            # In case it in a multiple extension format, we take the
            # first one only
            if len(np.shape(input.data[0])) == 3:

                input_CCDData = CCDData(input.data[0][0].astype('float'),
                                        unit=u.ct)
                input_header = input.header

            else:

                input_CCDData = CCDData(input.data[0].astype('float'),
                                        unit=u.ct)
                input_header = input.header

        else:

            error_msg = 'Please check the shape/dimension of the ' +\
                        'input input frame, it is probably empty ' +\
                        'or has an atypical format. The shape of the ' +\
                        'data is: {}.'.format(input_shape) + '. The ' +\
                        'data type is: {}'.format(type(input)) + '.'
            self.logger.critical(error_msg)
            raise RuntimeError(error_msg)

        # Get the exposure time for the dark frames
        if np.in1d(self.exptime_keyword, input_header).any():
            # Get the exposure time for the light frames
            exptime_keyword_idx = int(
                np.where(np.in1d(self.exptime_keyword, input_header))[0][0])
            exptime_keyword = self.exptime_keyword[exptime_keyword_idx]
            exposure_time = input_header[exptime_keyword]

        else:

            # If exposure time cannot be found from the header and
            # user failed to supply the exposure time, use 1 second
            self.logger.warning('Exposure time cannot be found. '
                                '1 second is used as the exposure time.')
            exposure_time = 1.0

        return input_CCDData, input_header, exposure_time

    def combine_light(self,
                      combinetype_light=None,
                      sigma_clipping_light=None,
                      clip_low_light=None,
                      clip_high_light=None):

        if combinetype_light is not None:

            self.combinetype_light = combinetype_light

        if sigma_clipping_light is not None:

            self.sigma_clipping_light = sigma_clipping_light

        if clip_low_light is not None:

            self.clip_low_light = clip_low_light

        if clip_high_light is not None:

            self.clip_high_light = clip_high_light

        return self._combine(self.light_CCDData, self.combinetype_light,
                             self.sigma_clipping_light, self.clip_low_light,
                             self.clip_high_light)

    def combine_arc(self,
                    combinetype_arc=None,
                    sigma_clipping_arc=None,
                    clip_low_arc=None,
                    clip_high_arc=None):

        if combinetype_arc is not None:

            self.combinetype_arc = combinetype_arc

        if sigma_clipping_arc is not None:

            self.sigma_clipping_arc = sigma_clipping_arc

        if clip_low_arc is not None:

            self.clip_low_arc = clip_low_arc

        if clip_high_arc is not None:

            self.clip_high_arc = clip_high_arc

        return self._combine(self.arc_CCDData, self.combinetype_arc,
                             self.sigma_clipping_arc, self.clip_low_arc,
                             self.clip_high_arc)

    def combine_flat(self,
                     combinetype_flat=None,
                     sigma_clipping_flat=None,
                     clip_low_flat=None,
                     clip_high_flat=None):

        if combinetype_flat is not None:

            self.combinetype_flat = combinetype_flat

        if sigma_clipping_flat is not None:

            self.sigma_clipping_flat = sigma_clipping_flat

        if clip_low_flat is not None:

            self.clip_low_flat = clip_low_flat

        if clip_high_flat is not None:

            self.clip_high_flat = clip_high_flat

        return self._combine(self.flat_CCDData, self.combinetype_flat,
                             self.sigma_clipping_flat, self.clip_low_flat,
                             self.clip_high_flat)

    def combine_dark(self,
                     combinetype_dark=None,
                     sigma_clipping_dark=None,
                     clip_low_dark=None,
                     clip_high_dark=None):

        if combinetype_dark is not None:

            self.combinetype_dark = combinetype_dark

        if sigma_clipping_dark is not None:

            self.sigma_clipping_dark = sigma_clipping_dark

        if clip_low_dark is not None:

            self.clip_low_dark = clip_low_dark

        if clip_high_dark is not None:

            self.clip_high_dark = clip_high_dark

        return self._combine(self.dark_CCDData, self.combinetype_dark,
                             self.sigma_clipping_dark, self.clip_low_dark,
                             self.clip_high_dark)

    def combine_bias(self,
                     combinetype_bias=None,
                     sigma_clipping_bias=None,
                     clip_low_bias=None,
                     clip_high_bias=None):

        if combinetype_bias is not None:

            self.combinetype_bias = combinetype_bias

        if sigma_clipping_bias is not None:

            self.sigma_clipping_bias = sigma_clipping_bias

        if clip_low_bias is not None:

            self.clip_low_bias = clip_low_bias

        if clip_high_bias is not None:

            self.clip_high_bias = clip_high_bias

        return self._combine(self.bias_CCDData, self.combinetype_bias,
                             self.sigma_clipping_bias, self.clip_low_bias,
                             self.clip_high_bias)

    def _combine(self, CCDData, combine_type, sigma_clipping, clip_low,
                 clip_high):

        # Put data into a Combiner
        combiner = Combiner(CCDData)

        # Apply sigma clipping
        if sigma_clipping:

            combiner.sigma_clipping(low_thresh=clip_low,
                                    high_thresh=clip_high,
                                    func=np.ma.median)

        # Image combine by median or average
        if combine_type == 'median':

            combined_CCDdata = combiner.median_combine()

        elif combine_type == 'average':

            combined_CCDdata = combiner.average_combine()

        else:

            self.logger.error('Unknown combinetype.')
            raise RuntimeError('Unknown combinetype: {}.'.format(combine_type))

        # Free memory
        combiner = None

        return combined_CCDdata

    def _bias_subtract(self):
        '''
        Perform bias subtraction if bias frames are available.

        '''

        # Put data into a Combiner
        self.bias_main = self.combine_bias()

        # Bias subtract
        if self.bias_main is None:

            self.logger.error('Main flat is not available, frame will '
                              'not be flattened.')

        else:

            self.light_redcued = self.light_reduced.subtract(self.bias_main)

    def _dark_subtract(self):
        '''
        Perform dark subtraction if dark frames are available

        '''

        self.dark_main = self.combine_dark()

        if self.dark_filename != []:

            # Dark subtraction adjusted for exposure time
            self.light_reduced =\
                self.light_reduced.subtract(self.dark_main)
            self.logger.info('Light frame is dark subtracted.')

    def _flatfield(self):
        '''
        Perform field flattening if flat frames are available

        '''

        self.flat_main = self.combine_flat()

        # Field-flattening
        if self.flat_main is None:

            self.logger.warning('Main flat is not available, frame will '
                                'not be flattened.')

        else:

            self.flat_reduced = copy.deepcopy(self.flat_main)

            # Dark subtract the flat field
            if self.dark_main is None:

                self.logger.warning('Main dark is not available, main '
                                    'flat will not be dark subtracted.')

            else:

                self.flat_reduced =\
                    self.flat_reduced.subtract(self.dark_main)
                self.logger.info('Flat frame is flat subtracted.')

            # Bias subtract the flat field
            if self.bias_main is None:

                self.logger.warning('Main bias is not available, main '
                                    'flat will not be bias subtracted.')

            else:

                self.flat_redcued = self.flat_reduced.subtract(self.bias_main)
                self.logger.info('Flat frame is bias subtracted.')

            self.flat_reduced = self.flat_reduced / np.nanmean(
                self.flat_reduced)

            # Flattenning the light frame
            self.light_reduced = self.light_reduced / self.flat_reduced
            self.logger.info('Light frame is flattened.')

    def create_bad_pixel_mask(self,
                              grow=False,
                              iterations=1,
                              diagonal=False,
                              create_bad_mask=True):
        """
        Heal the bad pixels by taking the average of their n-nearest
        good neighboring pixels. See more at util.bfixpix(). If you
        wish to have fine control of the the bad mask creation, please
        use the util.create_cutoff_mask() and util.create_bad_mask()
        manually; or supply your own the bad masks.

        Parameters
        ----------
        grow: bool (Default: False)
            Set to True to grow the mask, see `grow_mask()`
        iterations: int (Default: 1)
            The number of pixel growth along the Cartesian axes directions.
        diagonal: bool (Default: False)
            Set to True to grow in the diagonal directions.
        create_bad_mask: bool (Deafult: True)
            If set to True, combine the the bad_pixel_mask and saturation_mask

        """

        self.bad_pixel_mask, self.bad_pixels = create_bad_pixel_mask(
            self.light_reduced, grow, iterations, diagonal)

        if create_bad_mask:

            self.create_bad_mask()

    def create_bad_mask(self):

        if self.bad_pixel_mask is None:

            self.create_bad_pixel_mask(create_bad_mask=False)

        if self.saturation_mask is None:

            saturation_mask, saturated = create_cutoff_mask(
                self.light_reduced, self.cutoff, self.grow, self.iterations,
                self.diagonal)

            if self.saturation_mask is None:

                self.saturation_mask = saturation_mask
                self.saturated = saturated

            else:

                self.saturation_mask *= saturation_mask
                self.saturated *= saturated

        self.bad_mask = self.saturation_mask | self.bad_pixel_mask

    def reduce(self):
        '''
        Perform data reduction using the frames provided.

        '''

        if self.light_main is None:

            self.light_main = self.combine_light()

        self.light_reduced = self.light_main

        # Process the arc
        if len(self.arc_list) > 0:

            if self.arc_main is None:

                self.arc_main = self.combine_arc()

        # Bias subtraction
        if len(self.bias_list) > 0:

            if self.bias_main is None:

                self.bias_main = self.combine_bias()

            self._bias_subtract()

        else:

            self.logger.warning('No bias frames. Bias subtraction is not '
                                'performed.')

        # Dark subtraction
        if len(self.dark_list) > 0:

            if self.dark_main is None:

                self.dark_main = self.combine_dark()

            self._dark_subtract()

        else:

            self.logger.warning('No dark frames. Dark subtraction is not '
                                'performed.')

        # Field flattening
        if len(self.flat_list) > 0:

            if self.flat_main is None:

                self.combine_flat = self.combine_flat()

            self._flatfield()

        else:

            self.logger.warning('No flat frames. Field-flattening is not '
                                'performed.')

        # Construct a FITS object of the reduced frame
        self.light_reduced = np.array((self.light_reduced))

        # Create bad pixel mask
        self.heal_bad_pixels()

        # rotate the frame by 90 degrees anti-clockwise if saxis is 0
        if self.saxis == 0:

            self.light_reduced = np.rot90(self.light_reduced)
            self.bad_mask = np.rot90(self.bad_mask)
            self.bad_pixel_mask = np.rot90(self.bad_pixel_mask)
            self.saturation_mask = np.rot90(self.saturation_mask)
            self.arc_main = np.rot90(self.arc_main)

    def heal_bad_pixels(self, bad_mask=None, n=4):
        """
        Heal the bad pixels by taking the average of their n-nearest
        good neighboring pixels. See more at util.bfixpix(). By default,
        this is not used, becase any automatic tampering of data is
        not a good idea.

        Parameters
        ----------
        bad_mask: numpy.ndarray
            The bad pixel mask for healing. If it is not provided, it
            will be computed from the reduced data.
        n: int
            Number of nearest, good pixels to average over.

        """

        if bad_mask is None:

            if self.bad_mask is None:

                self.create_bad_mask()
                self.logger.info('Created a bad_mask using the reduced data.')

        elif np.shape(bad_mask) == np.shape(self.light_reduced):

            self.bad_mask = bad_mask
            self.logger.info('Bad_mask is set to the supplied mask.')

        else:

            err_msg = 'The bad_mask provided has to be the ' +\
                'same shape as the data.'
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)

        bfixpix(self.light_reduced, self.bad_mask, n=n)
        self.pixel_healed = True

    def _create_image_fits(self):
        '''
        Put the reduced data in FITS format with an image header.

        '''

        self.image_fits = fits.ImageHDU(self.light_reduced)

        self.logger.info('Appending the header from the first light frame.')
        self.image_fits.header = self.light_header[0]

        # Add the names of all the light frames to header
        if len(self.light_filename) > 0:

            for i in range(len(self.light_filename)):

                self.logger.debug('Light frame: {} is added to the header.'
                                  ''.format(self.light_filename[i]))
                self.image_fits.header.set(keyword='light' + str(i + 1),
                                           value=self.light_filename[i],
                                           comment='Light frames')

        # Add the names of all the biad frames to header
        if len(self.bias_filename) > 0:

            for i in range(len(self.bias_filename)):

                self.logger.debug('Bias frame: {} is added to the header.'
                                  ''.format(self.bias_filename[i]))
                self.image_fits.header.set(keyword='bias' + str(i + 1),
                                           value=self.bias_filename[i],
                                           comment='Bias frames')

        # Add the names of all the dark frames to header
        if len(self.dark_filename) > 0:

            for i in range(len(self.dark_filename)):

                self.logger.debug('Dark frame: {} is added to the header.'
                                  ''.format(self.dark_filename[i]))
                self.image_fits.header.set(keyword='dark' + str(i + 1),
                                           value=self.dark_filename[i],
                                           comment='Dark frames')

        # Add the names of all the flat frames to header
        if len(self.flat_filename) > 0:

            for i in range(len(self.flat_filename)):

                self.logger.debug('Flat frame: {} is added to the header.'
                                  ''.format(self.flat_filename[i]))
                self.image_fits.header.set(keyword='flat' + str(i + 1),
                                           value=self.flat_filename[i],
                                           comment='Flat frames')

        # Add all the other keywords
        self.image_fits.header.set(
            keyword='COMBTYPE',
            value=self.combinetype_light,
            comment='Type of image combine of the light frames.')
        self.image_fits.header.set(
            keyword='SIGCLIP',
            value=self.sigma_clipping_light,
            comment='True if the light frames are sigma clipped.')
        self.image_fits.header.set(
            keyword='CLIPLOW',
            value=self.clip_low_light,
            comment='Lower threshold of sigma clipping of the light frames.')
        self.image_fits.header.set(
            keyword='CLIPHIG',
            value=self.clip_high_light,
            comment='Higher threshold of sigma clipping of the light frames.')
        self.image_fits.header.set(
            keyword='XPOSURE',
            value=sum(self.light_time),
            comment='Total exposure time of the light frames.')
        self.image_fits.header.set(
            keyword='KEYWORD',
            value=self.exptime_light_keyword,
            comment='Automatically identified exposure time keyword of the '
            'light frames.')
        self.image_fits.header.set(
            keyword='DCOMTYPE',
            value=self.combinetype_dark,
            comment='Type of image combine of the dark frames.')
        self.image_fits.header.set(
            keyword='DSIGCLIP',
            value=self.sigma_clipping_dark,
            comment='True if the dark frames are sigma clipped.')
        self.image_fits.header.set(
            keyword='DCLIPLOW',
            value=self.clip_low_dark,
            comment='Lower threshold of sigma clipping of the dark frames.')
        self.image_fits.header.set(
            keyword='DCLIPHIG',
            value=self.clip_high_dark,
            comment='Higher threshold of sigma clipping of the dark frames.')
        self.image_fits.header.set(
            keyword='DXPOSURE',
            value=sum(self.dark_time),
            comment='Total exposure time of the dark frames.')
        self.image_fits.header.set(
            keyword='DKEYWORD',
            value=self.exptime_dark_keyword,
            comment='Automatically identified exposure time keyword of the ' +
            'dark frames.')
        self.image_fits.header.set(
            keyword='BCOMTYPE',
            value=self.combinetype_bias,
            comment='Type of image combine of the bias frames.')
        self.image_fits.header.set(
            keyword='BSIGCLIP',
            value=self.sigma_clipping_bias,
            comment='True if the dark frames are sigma clipped.')
        self.image_fits.header.set(
            keyword='BCLIPLOW',
            value=self.clip_low_bias,
            comment='Lower threshold of sigma clipping of the bias frames.')
        self.image_fits.header.set(
            keyword='BCLIPHIG',
            value=self.clip_high_bias,
            comment='Higher threshold of sigma clipping of the bias frames.')
        self.image_fits.header.set(
            keyword='FCOMTYPE',
            value=self.combinetype_flat,
            comment='Type of image combine of the flat frames.')
        self.image_fits.header.set(
            keyword='FSIGCLIP',
            value=self.sigma_clipping_flat,
            comment='True if the flat frames are sigma clipped.')
        self.image_fits.header.set(
            keyword='FCLIPLOW',
            value=self.clip_low_flat,
            comment='Lower threshold of sigma clipping of the flat frames.')
        self.image_fits.header.set(
            keyword='FCLIPHIG',
            value=self.clip_high_flat,
            comment='Higher threshold of sigma clipping of the flat frames.')
        self.image_fits.header.set(
            keyword='CCLEANED',
            value=self.cosmicray,
            comment='Indicate if cosmic ray cleaning was performed.')
        self.image_fits.header.set(
            keyword='CGAIN',
            value=self.gain,
            comment='Gain value used for cosmic ray cleaning.')
        self.image_fits.header.set(
            keyword='CRDNOISE',
            value=self.readnoise,
            comment='Readnoise value used for cosmic ray cleaning.')
        self.image_fits.header.set(
            keyword='CFSMODE',
            value=self.fsmode,
            comment='The fine structure mode for cosmic ray cleaning.')
        self.image_fits.header.set(
            keyword='CPSFMOD',
            value=self.psfmodel,
            comment='The PSF model used for cosmic ray cleaning.')
        self.image_fits.header.set(
            keyword='CUTOFF',
            value=self.cutoff,
            comment='The upper and lower limit of the good pixel values.')
        self.image_fits.header.set(
            keyword='GROW',
            value=self.grow,
            comment='Indicate if the bad pixel mask is grown outward.')
        self.image_fits.header.set(
            keyword='ITERATE',
            value=self.iterations,
            comment='The number of pixels the bad pixel mask is grown.')
        self.image_fits.header.set(
            keyword='DIAGONAL',
            value=self.diagonal,
            comment='If False, ITERATE is the number of pixels grown '
            'in Mahattan distance.')
        self.image_fits.header.set(
            keyword='HEALED',
            value=self.pixel_healed,
            comment='Indicate if the pixels are healed (i.e. altered!).')
        if self.cr_kwargs is not None:
            for key, value in self.cr_kwargs.items():
                self.image_fits.header.set(keyword=key,
                                           value=value,
                                           comment=key)

    def save_fits(self,
                  filename='reduced_image',
                  extension='fits',
                  overwrite=False):
        '''
        Save the reduced image to disk.

        Parameters
        ----------
        filename: str
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        extension: str
            File extension without the dot.
        overwrite: bool
            Default is False.

        '''

        if filename[-5:] == '.fits':
            filename = filename[:-5]
        if filename[-4:] == '.fit':
            filename = filename[:-4]

        self._create_image_fits()
        self.image_fits = fits.PrimaryHDU(self.image_fits.data,
                                          self.image_fits.header)
        # Save file to disk
        self.image_fits.writeto(filename + '.' + extension,
                                overwrite=overwrite)
        self.logger.info('FITS file saved to {}.'.format(filename + '.' +
                                                         extension))

    def save_masks(self,
                   filename='reduced_image_mask',
                   extension='fits',
                   overwrite=False):
        '''
        Save the reduced image to disk.

        Parameters
        ----------
        filename: str
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        extension: str
            File extension without the dot.
        overwrite: bool
            Default is False.

        '''

        if filename[-5:] == '.fits':
            filename = filename[:-5]
        if filename[-4:] == '.fit':
            filename = filename[:-4]

        bad_mask_fits = fits.PrimaryHDU(self.bad_mask.astype('int'))
        bad_pixel_mask_fits = fits.ImageHDU(self.bad_pixel_mask.astype('int'))
        saturation_mask_fits = fits.ImageHDU(
            self.saturation_mask.astype('int'))

        output_HDU = fits.HDUList(
            [bad_mask_fits, bad_pixel_mask_fits, saturation_mask_fits])

        # Save file to disk
        output_HDU.writeto(filename + '.' + extension, overwrite=overwrite)
        self.logger.info('FITS file saved to {}.'.format(filename + '.' +
                                                         extension))

    def inspect(self,
                log=True,
                display=True,
                renderer='default',
                width=1280,
                height=720,
                return_jsonstring=False,
                save_fig=False,
                fig_type='iframe+png',
                filename=None,
                open_iframe=False):
        '''
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

        '''

        if log:

            fig = go.Figure(data=go.Heatmap(z=np.log10(self.light_reduced),
                                            colorscale="Viridis"))
        else:

            fig = go.Figure(
                data=go.Heatmap(z=self.light_reduced, colorscale="Viridis"))

        fig.update_layout(
            yaxis_title='Spatial Direction / pixel',
            xaxis=dict(zeroline=False,
                       showgrid=False,
                       title='Spectral Direction / pixel'),
            bargap=0,
            hovermode='closest',
            showlegend=False,
            autosize=False,
            height=height,
            width=width,
        )

        if filename is None:

            filename = 'reduced_image'

        if save_fig:

            fig_type_split = fig_type.split('+')

            for t in fig_type_split:

                if t == 'iframe':

                    pio.write_html(fig,
                                   filename + '.' + t,
                                   auto_open=open_iframe)

                elif t in ['jpg', 'png', 'svg', 'pdf']:

                    pio.write_image(fig, filename + '.' + t)

        if display:

            if renderer == 'default':

                fig.show()

            else:

                fig.show(renderer)

        if return_jsonstring:

            return fig.to_json()

    def list_files(self):
        '''
        Print the file input list.

        '''

        for i in self.filelist:
            print(i)
