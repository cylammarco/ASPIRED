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
    def __init__(self,
                 filelist,
                 ftype='csv',
                 delimiter=None,
                 saxis=None,
                 saxis_keyword=None,
                 combinetype_light='median',
                 sigma_clipping_light=True,
                 clip_low_light=5,
                 clip_high_light=5,
                 exptime_light=None,
                 exptime_light_keyword=None,
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
                 verbose=True,
                 logger_name='ImageReduction',
                 log_level='WARNING',
                 log_file_folder='default',
                 log_file_name='default',
                 **kwargs):
        '''
        This class is not intented for quality data reduction, it exists for
        completeness such that users can produce a minimal pipeline with
        a single pacakge. Users should preprocess calibration frames, for
        example, we cannot properly combine arc frames taken with long and
        short exposures for wavelength calibration with both bright and faint
        lines; fringing correction with flat frames; light frames with various
        exposure times.

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
        sxais: int, 0 or 1 (Default: None)
            OVERRIDE the SAXIS value in the FITS header, or to provide the
            SAXIS if it does not exist
        saxis_keyword: str (Default: None)
            Supply the saxis keyword used in the header.
        combinetype_light: str (Default: 'median')
            Average of median for CCDproc.Combiner.average_combine() and
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
            average of median for CCDproc.Combiner.average_combine() and
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
            Average of median for CCDproc.Combiner.average_combine() and
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
            Average of median for CCDproc.Combiner.average_combine() and
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

        # FITS keyword standard recommends XPOSURE, but most observatories
        # use EXPTIME for supporting iraf. Also included a few other keywords
        # which are the proxy-exposure times at best. ASPIRED will use the
        # first keyword found on the list, if all failed, an exposure time of
        # 1 second will be applied. A warning will be promted.
        self.exptime_keyword = [
            'XPOSURE', 'EXPOSURE', 'EXPTIME', 'EXPOSED', 'TELAPSED', 'ELAPSED'
        ]

        self.combinetype_light = combinetype_light
        self.sigma_clipping_light = sigma_clipping_light
        self.clip_low_light = clip_low_light
        self.clip_high_light = clip_high_light
        self.exptime_light = exptime_light
        self.exptime_light_keyword = exptime_light_keyword

        self.combinetype_dark = combinetype_dark
        self.sigma_clipping_dark = sigma_clipping_dark
        self.clip_low_dark = clip_low_dark
        self.clip_high_dark = clip_high_dark
        self.exptime_dark = exptime_dark
        self.exptime_dark_keyword = exptime_dark_keyword

        self.combinetype_bias = combinetype_bias
        self.sigma_clipping_bias = sigma_clipping_bias
        self.clip_low_bias = clip_low_bias
        self.clip_high_bias = clip_high_bias

        self.combinetype_flat = combinetype_flat
        self.sigma_clipping_flat = sigma_clipping_flat
        self.clip_low_flat = clip_low_flat
        self.clip_high_flat = clip_high_flat
        self.exptime_flat = exptime_flat
        self.exptime_flat_keyword = exptime_flat_keyword

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
        self.pixel_healed = False

        self.bias_list = None
        self.dark_list = None
        self.flat_list = None
        self.arc_list = None
        self.light = None

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

        self.bias_filename = []
        self.dark_filename = []
        self.flat_filename = []
        self.arc_filename = []
        self.light_filename = []

        self.light_header = []
        self.arc_header = []
        self.dark_header = []
        self.flat_header = []

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

        # If there is no science frames, nothing to process.
        assert (self.light_list.size > 0), 'There is no light frame.'

        # Only load the science data, other types of image data are loaded by
        # separate methods.
        light_CCDData = []
        light_time = []

        for i in range(self.light_list.size):

            # Open all the light frames
            self.logger.debug('Loading light frame: {}.'.format(
                self.light_list[i]))
            light = fits.open(self.light_list[i])[self.light_hdunum[i]]

            if type(light) == 'astropy.io.fits.hdu.hdulist.HDUList':

                light = light[0]
                self.logger.warning('An HDU list is provided, only the first '
                                    'HDU will be read.')
            light_shape = np.shape(light.data)
            self.logger.debug('light.data has shape {}.'.format(light_shape))

            # Normal case
            if len(light_shape) == 2:

                self.logger.debug('light.data is 2 dimensional.')
                light_CCDData.append(
                    CCDData(light.data.astype('float'), unit=u.ct))
                self.light_header.append(light.header)

            # Try to trap common error when saving FITS file
            # Case with multiple image extensions, we only take the first one
            elif len(light_shape) == 3:

                self.logger.debug('light.data is 3 dimensional.')
                light_CCDData.append(
                    CCDData(light.data[0].astype('float'), unit=u.ct))
                self.light_header.append(light.header)

            # Case with an extra bracket when saving
            elif len(light_shape) == 4:

                self.logger.debug(
                    'light.data is 4 dimensional, there is most '
                    'likely an extra bracket, attempting to go in '
                    'one level.')

                # In case it in a multiple extension format, we take the
                # first one only
                if len(np.shape(light.data[0])) == 3:

                    light_CCDData.append(
                        CCDData(light.data[0][0].astype('float'), unit=u.ct))
                    self.light_header.append(light.header)

                else:

                    light_CCDData.append(
                        CCDData(light.data[0].astype('float'), unit=u.ct))
                    self.light_header.append(light.header)

            else:

                error_msg = 'Please check the shape/dimension of the ' +\
                            'input light frame, it is probably empty ' +\
                            'or has an atypical format. The shape of the ' +\
                            'data is: {}.'.format(light_shape) + '. The ' +\
                            'data type is: {}'.format(type(light)) + '.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            # Cosmic ray cleaning
            if self.cosmicray:

                self.logger.info(
                    'Removing cosmic rays in mode: {}.'.format(psfmodel))

                if self.fsmode == 'convolve':

                    if psfmodel == 'gaussyx':

                        light_CCDData[i].data = detect_cosmics(
                            light_CCDData[i].data,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            fsmode='convolve',
                            psfmodel='gaussy',
                            **kwargs)[1]

                        light_CCDData[i].data = detect_cosmics(
                            light_CCDData[i].data,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            fsmode='convolve',
                            psfmodel='gaussx',
                            **kwargs)[1]

                    elif psfmodel == 'gaussxy':

                        light_CCDData[i].data = detect_cosmics(
                            light_CCDData[i].data,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            fsmode='convolve',
                            psfmodel='gaussx',
                            **kwargs)[1]

                        light_CCDData[i].data = detect_cosmics(
                            light_CCDData[i].data,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            fsmode='convolve',
                            psfmodel='gaussy',
                            **kwargs)[1]

                    else:

                        light_CCDData[i].data = detect_cosmics(
                            light_CCDData[i].data,
                            gain=self.gain,
                            readnoise=self.readnoise,
                            fsmode='convolve',
                            psfmodel=self.psfmodel,
                            **kwargs)[1]

                else:

                    light_CCDData[i].data = detect_cosmics(
                        light_CCDData[i].data,
                        gain=self.gain,
                        readnoise=self.readnoise,
                        fsmode=self.fsmode,
                        psfmodel=self.psfmodel,
                        **kwargs)[1]

            self.logger.debug('Light frame header: {}.'.format(
                self.light_header[i]))

            self.logger.debug('Appending light filename: {}.'.format(
                self.light_list[i].split('/')[-1]))
            self.light_filename.append(self.light_list[i].split('/')[-1])

            # Get the exposure time for the light frames
            if exptime_light is None:

                if np.in1d(self.exptime_keyword, self.light_header[i]).any():
                    # Get the exposure time for the light frames
                    exptime_keyword_idx = int(
                        np.where(
                            np.in1d(self.exptime_keyword,
                                    self.light_header[i]))[0][0])

                    if np.isfinite(exptime_keyword_idx):

                        exptime_keyword = self.exptime_keyword[
                            exptime_keyword_idx]
                        light_time.append(
                            self.light_header[i][exptime_keyword])

                else:

                    # If exposure time cannot be found from the header and
                    # user failed to supply the exposure time, use 1 second
                    logging.warning(
                        'Light frame exposure time cannot be found. '
                        '1 second is used as the exposure time.')
                    light_time.append(1.0)

            else:

                light_time.append(exptime_light)

            self.logger.debug('The light frame exposure time is {}.'.format(
                light_time[i]))

            saturation_mask, saturated = create_cutoff_mask(
                light_CCDData[i].data, cutoff, grow, iterations, diagonal)

            if self.saturation_mask is None:

                self.saturation_mask = saturation_mask
                self.saturated = saturated

            else:

                self.saturation_mask *= saturation_mask
                self.saturated *= saturated

            light_CCDData[i].data /= light_time[i]

        # Put data into a Combiner
        light_combiner = Combiner(light_CCDData)
        self.logger.debug('Combiner for the light frames is created.')

        # Free memory
        light_CCDData = None
        self.logger.debug('light_CCDData is deleted.')

        # Apply sigma clipping
        if self.sigma_clipping_light:
            light_combiner.sigma_clipping(low_thresh=self.clip_low_light,
                                          high_thresh=self.clip_high_light,
                                          func=np.ma.median)
            self.logger.info('Sigma clipping with a lower and upper threshold '
                             'of {} and {} sigma.'.format(
                                 self.clip_low_light, self.clip_high_light))
        # Image combine by median or average
        if self.combinetype_light == 'median':
            self.light_main = light_combiner.median_combine()
            self.exptime_light = np.nanmedian(light_time)
            self.logger.info('light frames are median_combined.')
        elif self.combinetype_light == 'average':
            self.light_main = light_combiner.average_combine()
            self.exptime_light = np.nanmean(light_time)
            self.logger.info('light frames are mean_combined.')
        else:
            error_msg = 'Unknown combinetype for light frames.'
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        # Free memory
        light_combiner = None

        # FITS keyword standard for the spectral direction, if FITS header
        # does not contain SAXIS, the image in assumed to have the spectra
        # going across (left to right corresponds to blue to red). All frames
        # get rotated in the anti-clockwise direction if the first light frame
        # has a verticle spectrum (top to bottom corresponds to blue to red).
        if saxis is None:

            if 'SAXIS' in light.header:

                self.saxis = int(light.header['SAXIS'])

            else:

                self.saxis = 1

        else:

            self.saxis = saxis

        self.logger.info('Saxis is found/set to be {}.'.format(self.saxis))

        # Collect and stack the arcs
        if len(self.arc_list) > 0:
            # Combine the arcs
            arc_CCDData = []
            for i in range(self.arc_list.size):
                # Open all the light frames
                arc = fits.open(self.arc_list[i])[self.arc_hdunum[i]]
                if type(arc) == 'astropy.io.fits.hdu.hdulist.HDUList':
                    arc = arc[0]
                    self.logger.warning(
                        'A HDU list is provided, only the first '
                        'HDU will be read.')

                arc_shape = np.shape(arc)

                # Normal case
                if len(arc_shape) == 2:
                    self.logger.debug('arc.data is 2 dimensional.')
                    arc_CCDData.append(
                        CCDData(arc.data.astype('float'), unit=u.ct))
                    self.arc_header.append(arc.header)
                # Try to trap common error when saving FITS file
                # Case with multiple extensions, we only take the first one
                elif len(arc_shape) == 3:
                    self.logger.debug('arc.data is 3 dimensional.')
                    arc_CCDData.append(
                        CCDData(arc.data[0].astype('float'), unit=u.ct))
                    self.arc_header.append(arc.header)
                # Case with an extra bracket when saving
                elif len(arc_shape) == 4:
                    self.logger.debug(
                        'arc.data is 4 dimensional, there is most '
                        'likely an extra bracket, attempting to go in '
                        'one level.')
                    # In case it in a multiple extension format, we take the
                    # first one only
                    if len(np.shape(arc.data[0])) == 3:
                        arc_CCDData.append(
                            CCDData(arc.data[0][0].astype('float'), unit=u.ct))
                        self.arc_header.append(arc.header)
                    else:
                        arc_CCDData.append(
                            CCDData(arc.data[0].astype('float'), unit=u.ct))
                        self.arc_header.append(arc.header)
                else:
                    error_msg = 'Please check the shape/dimension of the ' +\
                                'input arc frame, it is probably empty ' +\
                                'or has an atypical output format.'
                    self.logger.critical(error_msg)
                    raise RuntimeError(error_msg)

                self.arc_filename.append(self.arc_list[i].split('/')[-1])

                self.logger.debug('Arc frame header: {}.'.format(
                    self.arc_header[i]))

            # combine the arc frames
            arc_combiner = Combiner(arc_CCDData)
            self.arc_main = arc_combiner.median_combine()

            # Free memory
            arc_CCDData = None
            arc_combiner = None

    def _bias_subtract(self):
        '''
        Perform bias subtraction if bias frames are available.

        '''

        bias_CCDData = []

        for i in range(self.bias_list.size):
            # Open all the bias frames
            bias = fits.open(self.bias_list[i])[self.bias_hdunum[i]]
            if type(bias) == 'astropy.io.fits.hdu.hdulist.HDUList':
                bias = bias[0]
                self.logger.warning('An HDU list is provided, only the first '
                                    'HDU will be read.')
            bias_shape = np.shape(bias)

            # Normal case
            if len(bias_shape) == 2:
                self.logger.debug('bias.data is 2 dimensional.')
                bias_CCDData.append(
                    CCDData(bias.data.astype('float'), unit=u.ct))
            # Try to trap common error when saving FITS file
            # Case with multiple image extensions, we only take the first one
            elif len(bias_shape) == 3:
                self.logger.debug('bias.data is 3 dimensional.')
                bias_CCDData.append(
                    CCDData(bias.data[0].astype('float'), unit=u.ct))
            # Case with an extra bracket when saving
            elif len(bias_shape) == 4:
                self.logger.debug(
                    'bias.data is 4 dimensional, there is most '
                    'likely an extra bracket, attempting to go in '
                    'one level.')
                # In case it in a multiple extension format, we take the
                # first one only
                if len(np.shape(bias.data[0])) == 3:
                    bias_CCDData.append(
                        CCDData(bias.data[0][0].astype('float'), unit=u.ct))
                else:
                    bias_CCDData.append(
                        CCDData(bias.data[0].astype('float'), unit=u.ct))
            else:
                error_msg = 'Please check the shape/dimension of the ' +\
                            'input bias frame, it is probably empty ' +\
                            'or has an atypical output format.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            self.bias_filename.append(self.bias_list[i].split('/')[-1])

        # Put data into a Combiner
        bias_combiner = Combiner(bias_CCDData)

        # Apply sigma clipping
        if self.sigma_clipping_bias:
            bias_combiner.sigma_clipping(low_thresh=self.clip_low_bias,
                                         high_thresh=self.clip_high_bias,
                                         func=np.ma.median)

        # Image combine by median or average
        if self.combinetype_bias == 'median':
            self.bias_main = bias_combiner.median_combine()
        elif self.combinetype_bias == 'average':
            self.bias_main = bias_combiner.average_combine()
        else:
            self.bias_filename = []
            self.logger.error('Unknown combinetype for bias frames, main '
                              'bias cannot be created. Process continues '
                              'without bias subtraction.')

        # Bias subtract
        if self.bias_main is None:

            self.logger.error('Main flat is not available, frame will '
                              'not be flattened.')

        else:

            self.light_redcued = self.light_reduced.subtract(self.bias_main)

        # Free memory
        bias_CCDData = None
        bias_combiner = None

    def _dark_subtract(self):
        '''
        Perform dark subtraction if dark frames are available

        '''

        dark_CCDData = []
        dark_time = []

        for i in range(self.dark_list.size):
            # Open all the dark frames
            dark = fits.open(self.dark_list[i])[self.dark_hdunum[i]]
            if type(dark) == 'astropy.io.fits.hdu.hdulist.HDUList':
                dark = dark[0]
                self.logger.warning('An HDU list is provided, only the first '
                                    'HDU will be read.')
            dark_shape = np.shape(dark)

            # Normal case
            if len(dark_shape) == 2:
                self.logger.debug('dark.data is 2 dimensional.')
                dark_CCDData.append(
                    CCDData(dark.data.astype('float'), unit=u.ct))
                self.dark_header.append(dark.header)
            # Try to trap common error when saving FITS file
            # Case with multiple image extensions, we only take the first one
            elif len(dark_shape) == 3:
                self.logger.debug('dark.data is 3 dimensional.')
                dark_CCDData.append(
                    CCDData(dark.data[0].astype('float'), unit=u.ct))
                self.dark_header.append(dark.header)
            # Case with an extra bracket when saving
            elif len(dark_shape) == 4:
                self.logger.debug(
                    'dark.data is 4 dimensional, there is most '
                    'likely an extra bracket, attempting to go in '
                    'one level.')
                # In case it in a multiple extension format, we take the
                # first one only
                if len(np.shape(dark.data[0])) == 3:
                    dark_CCDData.append(
                        CCDData(dark.data[0][0].astype('float'), unit=u.ct))
                    self.dark_header.append(dark.header)
                else:
                    dark_CCDData.append(
                        CCDData(dark.data[0].astype('float'), unit=u.ct))
                    self.dark_header.append(dark.header)
            else:
                error_msg = 'Please check the shape/dimension of the ' +\
                            'input dark frame, it is probably empty ' +\
                            'or has an atypical output format.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            self.logger.debug('Dark frame header: {}.'.format(
                self.dark_header[i]))

            self.dark_filename.append(self.dark_list[i].split('/')[-1])

            # Get the exposure time for the light frames
            if self.exptime_dark is None:

                if np.in1d(self.exptime_keyword, self.dark_header[i]).any():
                    # Get the exposure time for the light frames
                    exptime_keyword_idx = int(
                        np.where(
                            np.in1d(self.exptime_keyword,
                                    self.dark_header[i]))[0][0])

                    if np.isfinite(exptime_keyword_idx):

                        exptime_keyword = self.exptime_keyword[
                            exptime_keyword_idx]
                        dark_time.append(self.dark_header[i][exptime_keyword])

                else:

                    # If exposure time cannot be found from the header and
                    # user failed to supply the exposure time, use 1 second
                    self.logger.warning(
                        'Dark frame exposure time cannot be found. '
                        '1 second is used as the exposure time.')
                    dark_time.append(1.0)

            else:

                dark_time.append(self.exptime_dark)

            dark_CCDData[i].data /= dark_time[i]

        # Put data into a Combiner
        dark_combiner = Combiner(dark_CCDData)

        # Apply sigma clipping
        if self.sigma_clipping_dark:
            dark_combiner.sigma_clipping(low_thresh=self.clip_low_dark,
                                         high_thresh=self.clip_high_dark,
                                         func=np.ma.median)
        # Image combine by median or average
        if self.combinetype_dark == 'median':
            self.dark_main = dark_combiner.median_combine()
            self.exptime_dark = np.nanmedian(dark_time)
        elif self.combinetype_dark == 'average':
            self.dark_main = dark_combiner.average_combine()
            self.exptime_dark = np.nanmean(dark_time)
        else:
            self.dark_filename = []
            self.logger.error('Unknown combinetype for dark frames, main '
                              'dark cannot be created. Process continues '
                              'without dark subtraction.')

        if self.dark_filename != []:
            # Dark subtraction adjusted for exposure time
            self.light_reduced =\
                self.light_reduced.subtract(self.dark_main)
            self.logger.info('Light frame is dark subtracted.')

        # Free memory
        dark_CCDData = None
        dark_combiner = None

    def _flatfield(self):
        '''
        Perform field flattening if flat frames are available

        '''

        flat_CCDData = []
        flat_time = []

        for i in range(self.flat_list.size):
            # Open all the flatfield frames
            flat = fits.open(self.flat_list[i])[self.flat_hdunum[i]]
            if type(flat) == 'astropy.io.fits.hdu.hdulist.HDUList':
                flat = flat[0]
                self.logger.warning('An HDU list is provided, only the first '
                                    'HDU will be read.')

            flat_shape = np.shape(flat)

            # Normal case
            if len(flat_shape) == 2:
                self.logger.debug('flat.data is 2 dimensional.')
                flat_CCDData.append(
                    CCDData(flat.data.astype('float'), unit=u.ct))
                self.flat_header.append(flat.header)
            # Try to trap common error when saving FITS file
            # Case with multiple image extensions, we only take the first one
            elif len(flat_shape) == 3:
                self.logger.debug('flat.data is 3 dimensional.')
                flat_CCDData.append(
                    CCDData(flat.data[0].astype('float'), unit=u.ct))
                self.flat_header.append(flat.header)
            # Case with an extra bracket when saving
            elif len(flat_shape) == 4:
                self.logger.debug(
                    'flat.data is 4 dimensional, there is most '
                    'likely an extra bracket, attempting to go in '
                    'one level.')
                # In case it in a multiple extension format, we take the
                # first one only
                if len(np.shape(flat.data[0])) == 3:
                    flat_CCDData.append(
                        CCDData(flat.data[0][0].astype('float'), unit=u.ct))
                    self.flat_header.append(flat.header)
                else:
                    flat_CCDData.append(
                        CCDData(flat.data[0].astype('float'), unit=u.ct))
                    self.flat_header.append(flat.header)
            else:
                error_msg = 'Please check the shape/dimension of the ' +\
                            'input flat frame, it is probably empty ' +\
                            'or has an atypical output format.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            self.flat_filename.append(self.flat_list[i].split('/')[-1])

            # Get the exposure time for the flat frames
            if self.exptime_flat is None:

                if np.in1d(self.exptime_keyword, self.flat_header[i]).any():
                    # Get the exposure time for the light frames
                    exptime_keyword_idx = int(
                        np.where(
                            np.in1d(self.exptime_keyword,
                                    self.flat_header[i]))[0][0])

                    if np.isfinite(exptime_keyword_idx):

                        exptime_keyword = self.exptime_keyword[
                            exptime_keyword_idx]
                        flat_time.append(self.flat_header[i][exptime_keyword])

                else:

                    # If exposure time cannot be found from the header and
                    # user failed to supply the exposure time, use 1 second
                    self.logger.warning(
                        'Dark frame exposure time cannot be found. '
                        '1 second is used as the exposure time.')
                    flat_time.append(1.0)

            else:

                flat_time.append(self.exptime_flat)

            flat_CCDData[i].data /= flat_time[i]

        # Put data into a Combiner
        flat_combiner = Combiner(flat_CCDData)

        # Apply sigma clipping
        if self.sigma_clipping_flat:
            flat_combiner.sigma_clipping(low_thresh=self.clip_low_flat,
                                         high_thresh=self.clip_high_flat,
                                         func=np.ma.median)

        # Image combine by median or average
        if self.combinetype_flat == 'median':
            self.flat_main = flat_combiner.median_combine()
        elif self.combinetype_flat == 'average':
            self.flat_main = flat_combiner.average_combine()
        else:
            self.flat_filename = []
            self.logger.error('Unknown combinetype for flat frames, main '
                              'flat cannot be created. Process continues '
                              'without flatfielding.')

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

        # Free memory
        flat_CCDData = None
        flat_combiner = None

    def create_bad_pixel_mask(self,
                              cutoff=60000.,
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
        cutoff: float (Default: 60000)
            This sets the (lower and) upper limit of electron count.
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
            self.light_reduced.data, grow, iterations, diagonal)

        if create_bad_mask:

            self.create_bad_mask()

    def create_bad_mask(self):

        if self.bad_pixel_mask is None:

            self.create_bad_pixel_mask(create_bad_mask=False)

        self.bad_mask = self.saturation_mask | self.bad_pixel_mask

        return self.bad_mask

    def reduce(self):
        '''
        Perform data reduction using the frames provided.

        '''

        self.light_reduced = self.light_main

        # Bias subtraction
        if self.bias_list.size > 0:
            self._bias_subtract()
        else:
            self.logger.warning('No bias frames. Bias subtraction is not '
                                'performed.')

        # Dark subtraction
        if self.dark_list.size > 0:
            self._dark_subtract()
        else:
            self.logger.warning('No dark frames. Dark subtraction is not '
                                'performed.')

        # Field flattening
        if self.flat_list.size > 0:
            self._flatfield()
        else:
            self.logger.warning('No flat frames. Field-flattening is not '
                                'performed.')

        # Construct a FITS object of the reduced frame
        self.light_reduced = np.array((self.light_reduced))

        # Create bad pixel mask
        self.create_bad_mask()

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
            value=self.exptime_light,
            comment='Average exposure time of the light frames.')
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
            value=self.exptime_dark,
            comment='Average exposure time of the dark frames.')
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
