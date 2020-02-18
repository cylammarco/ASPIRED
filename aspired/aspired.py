import os
import sys
import difflib
import warnings
from functools import partial
from itertools import chain

from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from ccdproc import Combiner
import numpy as np
from scipy import signal
from scipy import stats
from scipy import interpolate as itp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
sys.path.insert(0, "/users/marcolam/git/rascal")
from rascal.calibrator import Calibrator
from rascal.util import load_calibration_lines
from rascal.util import refine_peaks

try:
    from astroscrappy import detect_cosmics
except ImportError:
    warn(
        AstropyWarning('astroscrappy is not present, so ap_trace will clean ' +
                       'cosmic rays with a 2D-median filter of size 5.'))
    detect_cosmics = partial(signal.medfilt2d, kernel_size=5)
try:
    from spectres import spectres
    spectres_imported = True
except ImportError:
    warnings.warn(
        'spectres is not present, spectral resampling cannot be performed. '
        'Flux calibration is suboptimal. Flux is not conserved.')
    spectres_imported = False
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    plotly_imported = True
except ImportError:
    warnings.warn(
        'plotly is not present, diagnostic plots cannot be generated.')

from aspired.standard_list import *


def _check_files(paths):
    '''
    Go through the filelist provided and check if all files exist.
    '''

    for filepath in paths:
        try:
            os.path.isfile(filepath)
        except:
            ValueError('File ' + filepath + ' does not exist.')


class ImageReduction:
    def __init__(self,
                 filelistpath,
                 ftype='csv',
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
                 sigma_clipping_bias=True,
                 clip_low_bias=5,
                 clip_high_bias=5,
                 combinetype_flat='median',
                 sigma_clipping_flat=True,
                 clip_low_flat=5,
                 clip_high_flat=5,
                 silence=False):
        '''
        This class is not intented for quality data reduction, it exists for
        completeness such that users can produce a minimal pipeline with
        a single pacakge. Users should preprocess calibration frames, for
        example, arc frames taken with long and short exposures for
        wavelength calibration with both bright and faint lines; fringing
        correction of flat frames; light frames with various exposure times.

        If a field-flattening 2D spectrum is already avaialble, it can be
        the only listed item. Set it as a 'light' frame.

        Parameters
        ----------
        filelistpath: string
            file location, does not support URL
        ftype: string
            one of csv, tsv and ascii. Default is csv.
        Sxais: int, 0 or 1
            OVERRIDE the SAXIS value in the FITS header, or to provide the
            SAXIS if it does not exist
        saxis_keyword: string
            HDU keyword for the spectral axis direction
        combinetype_light: string
            average of median for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_light: boolean
            perform sigma clipping if set to True. sigma is computed with the
            numpy.ma.std method
        clip_low_light: float
            lower threshold of the sigma clipping
        clip_high_light: float
            upper threshold of the sigma clipping
        exptime_light: float
            OVERRIDE the exposure time value in the FITS header, or to provide
            one if the keyword does not exist
        exptime_light_keyword: string
            HDU keyword for the exposure time of the light frame
        combinetype_dark: string
            average of median for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_dark: boolean
            perform sigma clipping if set to True. sigma is computed with the
            numpy.ma.std method
        clip_low_dark: float
            lower threshold of the sigma clipping
        clip_high_dark: float
            upper threshold of the sigma clipping
        exptime_dark: float
            OVERRIDE the exposure time value in the FITS header, or to provide
            one if the keyword does not exist
        exptime_dark_keyword: string
            HDU keyword for the exposure time of the dark frame
        combinetype_bias: string
            average of median for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_bias: boolean
            perform sigma clipping if set to True. sigma is computed with the
            numpy.ma.std method
        clip_low_bias: float
            lower threshold of the sigma clipping
        clip_high_bias: float
            upper threshold of the sigma clipping
        combinetype_flat: string
            average of median for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_flat: boolean
            perform sigma clipping if set to True. sigma is computed with the
            numpy.ma.std method
        clip_low_flat: float
            lower threshold of the sigma clipping
        clip_high_flat: float
            upper threshold of the sigma clipping
        silence: boolean
            set to suppress all messages
        '''

        self.filelistpath = filelistpath
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
            'XPOSURE', 'EXPTIME', 'EXPOSED', 'TELAPSED', 'ELAPSED'
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

        self.silence = silence

        self.bias_list = None
        self.dark_list = None
        self.flat_list = None
        self.arc_list = None
        self.light = None

        self.bias_master = None
        self.dark_master = None
        self.flat_master = None
        self.arc_master = None
        self.light_master = None

        self.bias_filename = []
        self.dark_filename = []
        self.flat_filename = []
        self.arc_filename = []
        self.light_filename = []

        # import file with first column as image type and second column as
        # file path
        self.filelist = np.genfromtxt(self.filelistpath,
                                      delimiter=self.delimiter,
                                      dtype='str',
                                      autostrip=True)
        self.imtype = self.filelist[:, 0]
        self.impath = self.filelist[:, 1]
        try:
            self.hdunum = self.filelist[:, 2].astype('int')
        except:
            self.hdunum = np.zeros(len(self.impath)).astype('int')

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

        # Check if all files exist
        _check_files(self.impath)

        # FITS keyword standard for the spectral direction, if FITS header
        # does not contain SAXIS, the image in assumed to have the spectra
        # going across (left to right corresponds to blue to red). All frames
        # get rotated in the anti-clockwise direction if the first light frame
        # has a verticle spectrum (top to bottom corresponds to blue to red).
        if saxis is None:
            if saxis_keyword is None:
                self.saxis_keyword = 'SAXIS'
            else:
                self.saxis_keyword = saxis_keyword
            try:
                self.saxis = int(light.header[self.saxis_keyword])
            except:
                if not self.silence:
                    warnings.warn('saxis keyword "' + self.saxis_keyword +
                                  '" is not in the header. saxis is set to 1.')
                self.saxis = 1
        else:
            self.saxis = saxis

        # Only load the science data, other types of image data are loaded by
        # separate methods.
        light_CCDData = []
        light_time = []

        for i in range(self.light_list.size):
            # Open all the light frames
            light = fits.open(self.light_list[i])[self.light_hdunum[i]]
            light_CCDData.append(CCDData(light.data, unit=u.adu))

            self.light_filename.append(self.light_list[i].split('/')[-1])

            # Get the exposure time for the light frames
            if exptime_light is None:
                if exptime_light_keyword is not None:
                    # add line to check exptime_light_keyword is string
                    light_time.append(light.header[exptime_light_keyword])
                else:
                    # check if the exposure time keyword exists
                    exptime_keyword_matched =\
                    np.in1d(self.exptime_keyword, light.header)
                    if exptime_keyword_matched.any():
                        light_time.append(light.header[self.exptime_keyword[
                            np.where(exptime_keyword_matched)[0][0]]])
                    else:
                        pass
            else:
                assert (exptime_light > 0), 'Exposure time has to be positive.'
                light_time = exptime_light

        # Put data into a Combiner
        light_combiner = Combiner(light_CCDData)
        # Free memory
        del light_CCDData

        # Apply sigma clipping
        if self.sigma_clipping_light:
            light_combiner.sigma_clipping(low_thresh=self.clip_low_light,
                                          high_thresh=self.clip_high_light,
                                          func=np.ma.median)

        # Image combine by median or average
        if self.combinetype_light == 'median':
            self.light_master = light_combiner.median_combine()
            self.exptime_light = np.median(light_time)
        elif self.combinetype_light == 'average':
            self.light_master = light_combiner.average_combine()
            self.exptime_light = np.mean(light_time)
        else:
            raise ValueError('ASPIRED: Unknown combinetype.')

        # Free memory
        del light_combiner

        # If exposure time cannot be found from the header and user failed
        # to supply the exposure time, use 1 second
        if len(light_time) == 0:
            self.light_time = 1.
            if not self.silence:
                warnings.warn('Light frame exposure time cannot be found. '
                              '1 second is used as the exposure time.')

        # Combine the arcs
        arc_CCDData = []
        for i in range(self.arc_list.size):
            # Open all the light frames
            arc = fits.open(self.arc_list[i])[self.arc_hdunum[i]]
            arc_CCDData.append(CCDData(arc.data, unit=u.adu))

            self.arc_filename.append(self.arc_list[i].split('/')[-1])

        # combine the arc frames
        arc_combiner = Combiner(arc_CCDData)
        self.arc_master = arc_combiner.median_combine()

        # Free memory
        del arc_CCDData
        del arc_combiner

    def _bias_subtract(self):
        '''
        Perform bias subtraction if bias frames are available.
        '''

        bias_CCDData = []

        for i in range(self.bias_list.size):
            # Open all the bias frames
            bias = fits.open(self.bias_list[i])[self.bias_hdunum[i]]
            bias_CCDData.append(CCDData(bias.data, unit=u.adu))

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
            self.bias_master = bias_combiner.median_combine()
        elif self.combinetype_bias == 'average':
            self.bias_master = bias_combiner.average_combine()
        else:
            self.bias_filename = []
            raise ValueError('ASPIRED: Unknown combinetype.')

        # Bias subtract
        self.light_master = self.light_master.subtract(self.bias_master)

        # Free memory
        del bias_CCDData
        del bias_combiner

    def _dark_subtract(self):
        '''
        Perform dark subtraction if dark frames are available
        '''

        dark_CCDData = []
        dark_time = []

        for i in range(self.dark_list.size):
            # Open all the dark frames
            dark = fits.open(self.dark_list[i])[self.dark_hdunum[i]]
            dark_CCDData.append(CCDData(dark.data, unit=u.adu))

            self.dark_filename.append(self.dark_list[i].split('/')[-1])

            # Get the exposure time for the dark frames
            for exptime in self.exptime_keyword:
                try:
                    dark_time.append(dark.header[exptime])
                    break
                except:
                    continue

        # Put data into a Combiner
        dark_combiner = Combiner(dark_CCDData)

        # Apply sigma clipping
        if self.sigma_clipping_dark:
            dark_combiner.sigma_clipping(low_thresh=self.clip_low_dark,
                                         high_thresh=self.clip_high_dark,
                                         func=np.ma.median)
        # Image combine by median or average
        if self.combinetype_dark == 'median':
            self.dark_master = dark_combiner.median_combine()
            self.exptime_dark = np.median(dark_time)
        elif self.combinetype_dark == 'average':
            self.dark_master = dark_combiner.average_combine()
            self.exptime_dark = np.mean(dark_time)
        else:
            self.dark_filename = []
            raise ValueError('ASPIRED: Unknown combinetype.')

        # If exposure time cannot be found from the header, use 1 second
        if len(dark_time) == 0:
            if not self.silence:
                warnings.warn('Dark frame exposure time cannot be found. '
                              '1 second is used as the exposure time.')
            self.exptime_dark = 1.

        # Frame in unit of ADU per second
        self.light_master =\
            self.light_master.subtract(
                self.dark_master.multiply(
                    self.exptime_light / self.exptime_dark)
            )

        # Free memory
        del dark_CCDData
        del dark_combiner

    def _flatfield(self):
        '''
        Perform field flattening if flat frames are available
        '''

        flat_CCDData = []

        for i in range(self.flat_list.size):
            # Open all the flatfield frames
            flat = fits.open(self.flat_list[i])[self.flat_hdunum[i]]
            flat_CCDData.append(CCDData(flat.data, unit=u.adu))

            self.flat_filename.append(self.flat_list[i].split('/')[-1])

        # Put data into a Combiner
        flat_combiner = Combiner(flat_CCDData)

        # Apply sigma clipping
        if self.sigma_clipping_flat:
            flat_combiner.sigma_clipping(low_thresh=self.clip_low_flat,
                                         high_thresh=self.clip_high_flat,
                                         func=np.ma.median)

        # Image combine by median or average
        if self.combinetype_flat == 'median':
            self.flat_master = flat_combiner.median_combine()
        elif self.combinetype_flat == 'average':
            self.flat_master = flat_combiner.average_combine()
        else:
            self.flat_filename = []
            raise ValueError('ASPIRED: Unknown combinetype.')

        # Field-flattening
        self.light_master = self.light_master.divide(self.flat_master)

        # Free memory
        del flat_CCDData
        del flat_combiner

    def reduce(self):
        '''
        Perform data reduction using the frames provided.
        '''

        # Bias subtraction
        if self.bias_list.size > 0:
            self._bias_subtract()
        else:
            if not self.silence:
                warnings.warn('No bias frames. Bias subtraction is not '
                              'performed.')

        # Dark subtraction
        if self.dark_list.size > 0:
            self._dark_subtract()
        else:
            if not self.silence:
                warnings.warn('No dark frames. Dark subtraction is not '
                              'performed.')

        # Field flattening
        if self.flat_list.size > 0:
            self._flatfield()
        else:
            if not self.silence:
                warnings.warn('No flat frames. Field-flattening is not '
                              'performed.')

        # rotate the frame by 90 degrees anti-clockwise if saxis is 0
        if self.saxis is 0:
            self.light_master = np.rot(self.light_master)

        # Construct a FITS object of the reduced frame
        self.light_master = np.array((self.light_master))

        # Put the reduced data in FITS format with a primary header
        self.fits_data = fits.PrimaryHDU(self.light_master)

        # Add the names of all the light frames to header
        if len(self.light_filename) > 0:
            for i in range(len(self.light_filename)):
                self.fits_data.header.set(keyword='light' + str(i + 1),
                                          value=self.light_filename[i],
                                          comment='Light frames')

        # Add the names of all the biad frames to header
        if len(self.bias_filename) > 0:
            for i in range(len(self.bias_filename)):
                self.fits_data.header.set(keyword='bias' + str(i + 1),
                                          value=self.bias_filename[i],
                                          comment='Bias frames')

        # Add the names of all the dark frames to header
        if len(self.dark_filename) > 0:
            for i in range(len(self.dark_filename)):
                self.fits_data.header.set(keyword='dark' + str(i + 1),
                                          value=self.dark_filename[i],
                                          comment='Dark frames')

        # Add the names of all the flat frames to header
        if len(self.flat_filename) > 0:
            for i in range(len(self.flat_filename)):
                self.fits_data.header.set(keyword='flat' + str(i + 1),
                                          value=self.flat_filename[i],
                                          comment='Flat frames')

        # Add all the other keywords
        self.fits_data.header.set(keyword='FILELIST',
                                  value=self.filelistpath,
                                  comment='File location of the frames used.')
        self.fits_data.header.set(
            keyword='COMBTYPE',
            value=self.combinetype_light,
            comment='Type of image combine of the light frames.')
        self.fits_data.header.set(
            keyword='SIGCLIP',
            value=self.sigma_clipping_light,
            comment='True if the light frames are sigma clipped.')
        self.fits_data.header.set(
            keyword='CLIPLOW',
            value=self.clip_low_light,
            comment='Lower threshold of sigma clipping of the light frames.')
        self.fits_data.header.set(
            keyword='CLIPHIG',
            value=self.clip_high_light,
            comment='Higher threshold of sigma clipping of the light frames.')
        self.fits_data.header.set(
            keyword='XPOSURE',
            value=self.exptime_light,
            comment='Average exposure time of the light frames.')
        self.fits_data.header.set(
            keyword='KEYWORD',
            value=self.exptime_light_keyword,
            comment='Automatically identified exposure time keyword of the '
            'light frames.')
        self.fits_data.header.set(
            keyword='DCOMTYPE',
            value=self.combinetype_dark,
            comment='Type of image combine of the dark frames.')
        self.fits_data.header.set(
            keyword='DSIGCLIP',
            value=self.sigma_clipping_dark,
            comment='True if the dark frames are sigma clipped.')
        self.fits_data.header.set(
            keyword='DCLIPLOW',
            value=self.clip_low_dark,
            comment='Lower threshold of sigma clipping of the dark frames.')
        self.fits_data.header.set(
            keyword='DCLIPHIG',
            value=self.clip_high_dark,
            comment='Higher threshold of sigma clipping of the dark frames.')
        self.fits_data.header.set(
            keyword='DXPOSURE',
            value=self.exptime_dark,
            comment='Average exposure time of the dark frames.')
        self.fits_data.header.set(
            keyword='DKEYWORD',
            value=self.exptime_dark_keyword,
            comment='Automatically identified exposure time keyword of the ' +
            'dark frames.')
        self.fits_data.header.set(
            keyword='BCOMTYPE',
            value=self.combinetype_bias,
            comment='Type of image combine of the bias frames.')
        self.fits_data.header.set(
            keyword='BSIGCLIP',
            value=self.sigma_clipping_bias,
            comment='True if the dark frames are sigma clipped.')
        self.fits_data.header.set(
            keyword='BCLIPLOW',
            value=self.clip_low_bias,
            comment='Lower threshold of sigma clipping of the bias frames.')
        self.fits_data.header.set(
            keyword='BCLIPHIG',
            value=self.clip_high_bias,
            comment='Higher threshold of sigma clipping of the bias frames.')
        self.fits_data.header.set(
            keyword='FCOMTYPE',
            value=self.combinetype_flat,
            comment='Type of image combine of the flat frames.')
        self.fits_data.header.set(
            keyword='FSIGCLIP',
            value=self.sigma_clipping_flat,
            comment='True if the flat frames are sigma clipped.')
        self.fits_data.header.set(
            keyword='FCLIPLOW',
            value=self.clip_low_flat,
            comment='Lower threshold of sigma clipping of the flat frames.')
        self.fits_data.header.set(
            keyword='FCLIPHIG',
            value=self.clip_high_flat,
            comment='Higher threshold of sigma clipping of the flat frames.')

    def savefits(self, filepath='reduced_image.fits', overwrite=False):
        '''
        Save the reduced image to disk.

        Parameters
        ----------
        filepath: String
            Disk location to be written to. Default is at where the Python
            process/subprocess is execuated.
        overwrite: boolean
            Default is False.

        '''

        # Save file to disk
        self.fits_data.writeto(filepath, overwrite=overwrite)

    def inspect(self,
                log=True,
                renderer='default',
                jsonstring=False,
                iframe=False):
        '''
        Display the reduced image with a supported plotly renderer or export
        as json strings.

        Parameters
        ----------
        log: boolean
            Log the ADU count per second in the display. Default is True.
        renderer: string
            plotly renderer: jpg, png
        jsonstring: boolean
            set to True to return json string that can be rendered by Plot.ly
            in any support language

        Returns
        -------
        json string if jsonstring is True, otherwise only an image is displayed

        '''

        if plotly_imported:
            if log:
                fig = go.Figure(data=go.Heatmap(z=np.log10(self.light_master),
                                                colorscale="Viridis"))
            else:
                fig = go.Figure(
                    data=go.Heatmap(z=self.light_master, colorscale="Viridis"))
            if jsonstring:
                return fig.to_json()
            if iframe:
                pio.write_html(fig, 'reduced_image.html')
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)
        else:
            if not self.silence:
                warnings.warn('plotly is not present, diagnostic plots cannot '
                              'be generated.')

    def list_files(self):
        '''
        Print the file input list.
        '''

        print(self.filelist)


class TwoDSpec:
    def __init__(self,
                 data,
                 header=None,
                 saxis=1,
                 spatial_mask=(1, ),
                 spec_mask=(1, ),
                 flip=False,
                 cr=False,
                 cr_sigma=5.,
                 rn=None,
                 gain=None,
                 seeing=None,
                 exptime=None,
                 silence=False):
        '''
        This is a class for processing a 2D spectral image, the read noise,
        detector gain, seeing and exposure time will be automatically extracted
        from the FITS header if it conforms with the IAUFWG FITS standard.

        Currently, there is no automated way to decide if a flip is needed.

        The supplied file should contain 2 or 3 columns with the following
        structure:

            column 1: one of bias, dark, flat or light
            column 2: file location
            column 3: HDU number (default to 0 if not given)

        If the 2D spectrum is
        +--------+--------+-------+-------+
        |  blue  |   red  | saxis |  flip |
        +--------+--------+-------+-------+
        |  left  |  right |   1   | False |
        |  right |  left  |   1   |  True |
        |   top  | bottom |   0   | False |
        | bottom |   top  |   0   |  True |
        +--------+--------+-------+-------+

        Spectra are sorted by their brightness. If there are multiple spectra
        on the image, and the target is not the brightest source, use at least
        the number of spectra visible to eye and pick the one required later.
        The default automated outputs is the brightest one, which is the
        most common case for images from a long-slit spectrograph.

        Parameters
        ----------
        data: 2D numpy array (M x N) OR astropy.io.fits object
            2D spectral image in either format
        header: FITS header
            THIS WILL OVERRIDE the header from the astropy.io.fits object
        saxis: int
            Spectral direction, 0 for vertical, 1 for horizontal.
            (Default is 1)
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
        cr: boolean
            Set to True to apply cosmic ray rejection by sigma clipping with
            astroscrappy if available, otherwise a 2D median filter of size 5
            would be used. (default is True)
        cr_sigma: float
            Cosmic ray sigma clipping limit (Deafult is 5.0)
        rn: float
            Readnoise of the detector, not important if noise estimation is
            not needed.
            (Deafult is None, which will be replaced with 1.0)
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
        silence: boolean
            Set to True to suppress all verbose output.
        '''

        # If data provided is an numpy array
        if isinstance(data, np.ndarray):
            img = data
            self.header = header
        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(data, fits.hdu.image.PrimaryHDU):
            img = data.data
            self.header = data.header
        # If it is an ImageReduction object
        elif isinstance(data, ImageReduction):
            img = data.fits_data.data
            self.header = data.fits_data.header
        # If a filepath is provided
        elif isinstace(data, str):
            # If HDU number is provided
            if data[-1] == ']':
                filepath, hdunum = data.split('[')
                hdunum = hdunum[:-1]
            # If not, assume the HDU idnex is 0
            else:
                filepath = data
                hdunum = 0

            # Check if file exists
            _check_files(filepath)

            # Load the file and dereference it afterwards
            fitsfile_tmp = fits.open(filepath)[hdunum]
            img = fitsfile_tmp.data
            self.header = fitsfile_tmp.header
            fitsfile_tmp = None
        else:
            TypeError('Please provide a numpy array, an ' +
                      'astropy.io.fits.hdu.image.PrimaryHDU object or an ' +
                      'ImageReduction object.')

        self.saxis = saxis
        if self.saxis is 1:
            self.waxis = 0
        else:
            self.waxis = 1
        self.spatial_mask = spatial_mask
        self.spec_mask = spec_mask
        self.flip = flip
        self.cr_sigma = cr_sigma

        # Default values if not supplied or cannot be automatically identified
        # from the header
        self.rn = 0.
        self.gain = 1.
        self.seeing = 1.
        self.exptime = 1.

        # Default keywords to be searched in the order in the list
        self._set_default_rn_keyword(['RDNOISE', 'RNOISE', 'RN'])
        self._set_default_gain_keyword(['GAIN'])
        self._set_default_seeing_keyword(['SEEING', 'L1SEEING'])
        self._set_default_exptime_keyword(
            ['XPOSURE', 'EXPTIME', 'EXPOSED', 'TELAPSED', 'ELAPSED'])

        # Get the Read Noise
        if rn is not None:
            if isinstance(rn, str):
                # use the supplied keyword
                self.rn = float(data.header[rn])
            elif np.isfinite(rn):
                # use the given rn value
                self.rn = float(rn)
            else:
                warnings.warn('rn has to be None, a numeric value or the ' +
                              'FITS header keyword, ' + str(rn) + ' is ' +
                              'given. It is set to 0.')
        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if header is not None:
                rn_keyword_matched = np.in1d(self.rn_keyword, header)
                if rn_keyword_matched.any():
                    self.rn = data.header[self.rn_keyword[np.where(
                        rn_keyword_matched)[0][0]]]
                else:
                    warnings.warn('Read Noise value cannot be identified. ' +
                                  'It is set to 0.')
            else:
                warnings.warn('Header is not provided. ' +
                              'Read Noise value is not provided. ' +
                              'It is set to 0.')

        # Get the Gain
        if gain is not None:
            if isinstance(gain, str):
                # use the supplied keyword
                self.gain = float(data.header[gain])
            elif np.isfinite(gain):
                # use the given gain value
                self.gain = float(gain)
            else:
                warnings.warn('Gain has to be None, a numeric value or the ' +
                              'FITS header keyword, ' + str(gain) + ' is ' +
                              'given. It is set to 1.')
        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if header is not None:
                gain_keyword_matched = np.in1d(self.gain_keyword, header)
                if gain_keyword_matched.any():
                    self.gain = data.header[self.gain_keyword[np.where(
                        gain_keyword_matched)[0][0]]]
                else:
                    warnings.warn('Gain value cannot be identified. ' +
                                  'It is set to 1.')
            else:
                warnings.warn('Header is not provide. ' +
                              'Gain value is not provided. ' +
                              'It is set to 1.')

        # Get the Seeing
        if seeing is not None:
            if isinstance(seeing, str):
                # use the supplied keyword
                self.seeing = float(data.header[seeing])
            elif np.isfinite(gain):
                # use the given gain value
                self.seeing = float(seeing)
            else:
                warnings.warn(
                    'Seeing has to be None, a numeric value or the ' +
                    'FITS header keyword, ' + str(seeing) + ' is ' +
                    'given. It is set to 1.')
        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if header is not None:
                seeing_keyword_matched = np.in1d(self.seeing_keyword, header)
                if seeing_keyword_matched.any():
                    self.seeing = data.header[self.seeing_keyword[np.where(
                        seeing_keyword_matched)[0][0]]]
                else:
                    warnings.warn('Seeing value cannot be identified. ' +
                                  'It is set to 1.')
            else:
                warnings.warn('Header is not provide. ' +
                              'Seeing value is not provided. ' +
                              'It is set to 1.')

        # Get the Exposure Time
        if exptime is not None:
            if isinstance(exptime, str):
                # use the supplied keyword
                self.exptime = float(data.header[exptime])
            elif isfinite(gain):
                # use the given gain value
                self.exptime = float(exptime)
            else:
                warnings.warn(
                    'Exposure Time has to be None, a numeric value or the ' +
                    'FITS header keyword, ' + str(exptime) + ' is ' +
                    'given. It is set to 1.')
        else:
            # if None is given and header is provided, check if the read noise
            # keyword exists in the default list.
            if header is not None:
                exptime_keyword_matched = np.in1d(self.exptime_keyword, header)
                if exptime_keyword_matched.any():
                    self.exptime = data.header[self.exptime_keyword[np.where(
                        exptime_keyword_matched)[0][0]]]
                else:
                    warnings.warn(
                        'Exposure Time value cannot be identified. ' +
                        'It is set to 1.')
            else:
                warnings.warn('Header is not provide. ' +
                              'Exposure Time value is not provided. ' +
                              'It is set to 1.')

        self.silence = silence

        # cosmic ray rejection
        if cr:
            img = detect_cosmics(img,
                                 sigclip=self.cr_sigma,
                                 readnoise=self.rn,
                                 gain=self.gain,
                                 fsmode='convolve',
                                 psfmodel='gaussy',
                                 psfsize=31,
                                 psffwhm=self.seeing)[1]

        # the valid y-range of the chip (i.e. spatial direction)
        if (len(self.spatial_mask) > 1):
            if self.saxis is 1:
                img = img[self.spatial_mask]
            else:
                img = img[:, self.spatial_mask]

        # the valid x-range of the chip (i.e. spectral direction)
        if (len(self.spec_mask) > 1):
            if self.saxis is 1:
                img = img[:, self.spec_mask]
            else:
                img = img[self.spec_mask]

        # get the length in the spectral and spatial directions
        self.spec_size = np.shape(img)[self.waxis]
        self.spatial_size = np.shape(img)[self.saxis]
        if self.saxis is 0:
            self.img = np.transpose(img)
            img = None
        else:
            self.img = img
            img = None

        if self.flip:
            self.img = np.flip(self.img)

        # set the 2D histogram z-limits
        self.zmin = np.nanpercentile(np.log10(self.img), 5)
        self.zmax = np.nanpercentile(np.log10(self.img), 95)

    def _set_default_rn_keyword(self, keyword_list):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        '''

        self.rn_keyword = list(keyword_list)

    def _set_default_gain_keyword(self, keyword_list):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        '''

        self.gain_keyword = list(keyword_list)

    def _set_default_seeing_keyword(self, keyword_list):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        '''

        self.seeing_keyword = list(keyword_list)

    def _set_default_exptime_keyword(self, keyword_list):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        '''

        self.exptime_keyword = list(keyword_list)

    def set_rn_keyword(self, keyword_list, append=False):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
        '''

        if append:
            self.rn_keyword += list(keyword_list)
        else:
            self.rn_keyword = list(keyword_list)

    def set_gain_keyword(self, keyword_list, append=False):
        '''
        Set the exptime keyword list.

        Parameters
        ----------
        keyword_list: list
            List of keyword (string).
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
            self.header = data.header
        else:
            TypeError(
                'Please provide an astropy.io.fits.header.Header object.')

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

    def _identify_spectra(self, f_height, display, renderer, jsonstring,
                          iframe):
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
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        peaks_y :
            Array or float of the pixel values of the detected peaks
        heights_y :
            Array or float of the integrated counts at the peaks
        """

        ydata = np.arange(self.spec_size)
        ztot = np.nanmedian(self.img, axis=self.saxis)

        # get the height thershold
        height = np.nanmax(ztot) * f_height

        # identify peaks
        peaks_y, heights_y = signal.find_peaks(ztot, height=height)
        heights_y = heights_y['peak_heights']

        # sort by strength
        mask = np.argsort(heights_y)
        peaks_y = peaks_y[mask][::-1]
        heights_y = heights_y[mask][::-1]

        # display disgnostic plot
        if display:
            # set a side-by-side subplot
            fig = go.Figure()

            # show the image on the left
            if self.saxis == 1:
                fig.add_trace(
                    go.Heatmap(z=np.log10(self.img),
                               colorscale="Viridis",
                               xaxis='x',
                               yaxis='y'))
            else:
                fig.add_trace(
                    go.Heatmap(z=np.log10(np.transpose(self.img)),
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
            fig.update_layout(autosize=True,
                              yaxis_title='Spatial Direction / pixel',
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

            if jsonstring:
                return fig.to_json()
            if iframe:
                pio.write_html(fig, 'identify_spectra.html')
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

        self.peak = peaks_y
        self.peak_height = heights_y

    def _optimal_signal(self, pix, xslice, sky, mu, sigma):
        """
        Iterate to get the optimal signal. Following the algorithm on
        Horne, 1986, PASP, 98, 609 (1986PASP...98..609H).

        Parameters
        ----------
        pix: 1-d numpy array
            pixel number along the spatial direction
        xslice: 1-d numpy array
            ADU along the pix
        sky: 1-d numpy array
            ADU of the fitted sky along the pix
        mu: float
            The center of the Gaussian
        sigma: float
            The width of the Gaussian

        Returns
        -------
        signal: float
            The optimal signal.
        noise: float
            The noise associated with the optimal signal.
        """

        # construct the Gaussian model
        P = self._gaus(pix, 1., 0., mu, sigma)
        P /= np.nansum(P)

        #
        signal = xslice - sky
        signal[signal < 0] = 0.
        # weight function and initial values
        signal1 = np.nansum(signal)
        var1 = self.rn + np.abs(xslice) / self.gain
        variance1 = 1. / np.nansum(P**2. / var1)

        signal_diff = 1
        variance_diff = 1
        i = 0

        while ((signal_diff > 0.0001) or (variance_diff > 0.0001)):

            signal0 = signal1
            var0 = var1
            variance0 = variance1

            # cosmic ray mask, only start considering after the 2nd iteration
            if i > 1:
                mask_cr = ((signal - P * signal0)**2. <
                           self.cr_sigma**2. * var0)
            else:
                mask_cr = True

            # compute signal and noise
            signal1 = np.nansum((P * (signal) / var0)[mask_cr]) / \
                np.nansum((P**2. / var0)[mask_cr])
            var1 = self.rn + np.abs(P * signal1 + sky) / self.gain
            variance1 = 1. / np.nansum((P**2. / var1)[mask_cr])

            signal_diff = (signal1 - signal0) / signal0
            variance_diff = (variance1 - variance0) / variance0

            P = signal / signal1
            P[P < 0.] = 0.
            P /= np.nansum(P)

            if i == 999:
                print(
                    'Unable to obtain optimal signal, please try a longer ' +
                    'iteration or revert to unit-weighted extraction. Values '
                    + 'returned (if at all) are sub-optimal at best.')
                break

        signal = signal1
        noise = np.sqrt(variance1)

        return signal, noise

    def ap_trace(self,
                 nspec=1,
                 nwindow=25,
                 spec_sep=5,
                 resample_factor=10,
                 rescale=False,
                 scaling_min=0.975,
                 scaling_max=1.025,
                 scaling_step=0.005,
                 percentile=5,
                 tol=3,
                 polydeg=3,
                 ap_faint=10,
                 display=False,
                 renderer='default',
                 jsonstring=False,
                 iframe=False):
        '''
        Aperture tracing by first using cross-correlation then the peaks are
        fitting with a polynomial with an order of floor(nwindow, 10) with a
        minimum order of 1. Nothing is returned unless jsonstring of the
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
            Minimum separation between sky lines.
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
            background sky level to the first order. [ADU]
        tol: float
            Maximum allowed shift between neighbouring slices, this value is
            referring to native pixel size without the application of the
            resampling or rescaling. [pix]
        polydeg: int
            Degree of the polynomial fit of the trace.
        ap_faint: float
            The percentile threshold of ADU aperture to be used for fitting
            the trace. Note that this percentile is of the ADU, not of the
            number of subspectra.
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        json string if jsonstring is True, otherwise only an image is displayed
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
        tol_len = int(tol * resample_factor)

        spec_spatial = np.zeros(nresample)

        # Scipy correlate method
        for i in chain(range(start_window_idx, nwindow),
                       range(start_window_idx - 1, -1, -1)):

            # smooth by taking the median
            lines = np.nanmedian(img_split[i], axis=1)
            lines = signal.resample(lines, nresample)
            lines = lines - np.percentile(lines, percentile)

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
                corr = corr[nresample - 1 - tol_len:nresample + tol_len]

                # Maximum corr position is the shift
                corr_val[j] = np.nanmax(corr)
                corr_idx[j] = np.nanargmax(corr) - tol_len

            # Maximum corr_val position is the scaling
            shift_solution[i] = corr_idx[np.nanargmax(corr_val)]
            scale_solution[i] = scaling_range[np.nanargmax(corr_val)]

            # Align the spatial profile before stacking
            pix_resampled = np.arange(nresample) * resample_factor *\
                nresample * scale_solution[i] / nresample - shift_solution[i]

            spec_spatial += spectres(
                np.arange(nresample) * resample_factor, pix_resampled, lines)

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
                                  prominence=1)

        #print('spec_spatial = ' + str(spec_spatial) + ' of length ' + str(len(spec_spatial)))
        #print('peaks = ' + str(peaks))

        # update the number of spectra if the number of peaks detected is less
        # than the number requested
        self.nspec = min(len(peaks), nspec)

        # Sort the positions by the prominences, and return to the original
        # scale (i.e. with subpixel position)
        spec_init = np.sort(peaks[0][np.argsort(-peaks[1]['prominences'])]
                            [:self.nspec]) / resample_factor
        #print('spec_init = ' + str(spec_init))

        # Create array to populate the spectral locations
        spec_idx = np.zeros((len(spec_init), len(img_split)))

        # Populate the initial values
        spec_idx[:, start_window_idx] = spec_init

        # Pixel positions of the mid point of each data_split (spectral)
        spec_pix = np.arange(len(img_split)) * w_size + w_size / 2.

        # Looping through pixels larger than middle pixel
        for i in range(start_window_idx + 1, nwindow):
            spec_idx[:, i] = (
                spec_idx[:, i - 1] * resample_factor * nscaled[i] / nresample -
                shift_solution[i]) / resample_factor

        # Looping through pixels smaller than middle pixel
        for i in range(start_window_idx - 1, -1, -1):
            spec_idx[:, i] = (spec_idx[:, i + 1] * resample_factor -
                              shift_solution[i + 1]) / (
                                  int(nresample * scale_solution[i + 1]) /
                                  nresample) / resample_factor

        ap = np.zeros((len(spec_idx), nwave))
        ap_sigma = np.zeros(len(spec_idx))

        for i in range(len(spec_idx)):

            # Get the median of the subspectrum and then get the ADU at the
            # centre of the aperture
            ap_val = np.zeros(nwindow)
            for j in range(nwindow):
                idx = int(spec_idx[i][j] + 0.5)
                ap_val[j] = np.nanmedian(img_split[j], axis=1)[idx]

            # Mask out the faintest ap_faint percentile
            mask = (ap_val > np.percentile(ap_val, ap_faint))

            # fit the trace
            ap_p = np.polyfit(spec_pix[mask], spec_idx[i][mask], int(polydeg))
            ap[i] = np.polyval(ap_p, np.arange(nwave))

            # Get the centre of the upsampled spectrum
            ap_centre_idx = ap[i][start_window_idx] * resample_factor

            # Get the indices for the 10 pixels on the left and right of the
            # spectrum, and apply the resampling factor.
            start_idx = int(ap_centre_idx - 10 * resample_factor + 0.5)
            end_idx = start_idx + 20 * resample_factor + 1

            # compute ONE sigma for each trace
            pguess = [
                np.nanmax(spec_spatial[start_idx:end_idx]),
                np.nanpercentile(spec_spatial, 10), ap_centre_idx, 3.
            ]

            popt, pcov = curve_fit(self._gaus,
                                   range(start_idx, end_idx),
                                   spec_spatial[start_idx:end_idx],
                                   p0=pguess)
            ap_sigma[i] = popt[3] / resample_factor

        self.trace = ap
        self.trace_sigma = ap_sigma

        # Plot
        if display:

            # set a side-by-side subplot
            fig = go.Figure()

            # show the image on the left
            fig.add_trace(
                go.Heatmap(z=np.log10(self.img),
                           zmin=self.zmin,
                           zmax=self.zmax,
                           colorscale="Viridis",
                           colorbar=dict(title='log(ADU)')))
            for i in range(len(spec_idx)):
                fig.add_trace(
                    go.Scatter(x=np.arange(nwave),
                               y=ap[i],
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
            fig.update_layout(autosize=True,
                              yaxis_title='SpectralDirection / pixel',
                              xaxis=dict(zeroline=False,
                                         showgrid=False,
                                         title='Spatial Direction / pixel'),
                              bargap=0,
                              hovermode='closest',
                              showlegend=False,
                              height=800)
            if jsonstring:
                return fig.to_json()
            if iframe:
                pio.write_html(fig, 'ap_trace.html')
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

    def ap_trace_quick(self,
                       nspec=1,
                       nwindow=25,
                       recenter=False,
                       prevtrace=(0, ),
                       fittype='spline',
                       order=3,
                       bigbox=8,
                       display=False,
                       renderer='default',
                       jsonstring=False,
                       iframe=False):
        """
        This only works for bright spectra with good wavelength coverage.

        It works by splitting the image in nwindow bins in the wavelength
        direction, then a Gaussian is fitted for each bin to determine the
        spatial position of the trace. Finally, a cubic spline is computed
        for these spatial positions as a function of the spectral position.

        Nothing is returned unless jsonstring of the plotly graph is set to be
        returned. The trace and trace_sigma are stored as properties of the
        TwoDSpec object.

        Parameters
        ----------
        nspec: int
            Number of spectra to be extracted. It does not guarantee returning
            the same number of spectra if fewer can be detected. (Default is 1)
        nwindow: int
            Keyword, number of bins in X direction to chop image into. Use
            fewer bins if ap_trace is having difficulty, such as with faint
            targets (default is 20, minimum is 4)
        recenter: bool
            Set to True to use previous trace, allow small shift in position
            along the spatial direction. Not doing anything if prevtrace is not
            supplied. (Default is False)
        prevtrace: 1-d numpy array
            Provide first guess or refitting the center with different parameters.
        fittype: string
            Set to 'spline' or 'polynomial', using
            scipy.interpolate.UnivariateSpline and numpy.polyfit
        order: string
            Degree of the spline or polynomial. Spline must be <= 5.
            (default is k=3)
        bigbox: float
            The number of sigma away from the main aperture to allow to trace
        silence: boolean
            Set to disable warning/error messages. (Default is False)
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        """

        self.nspec = nspec

        if not self.silence:
            print('Tracing Aperture using nwindow=' + str(nwindow))

        # the valid y-range of the chip (an array of int)
        ydata = np.arange(self.spec_size)
        ztot = np.sum(self.img, axis=1)

        # need at least 3 samples along the trace
        if (nwindow < 3):
            nwindow = 3

        # detect peaks by summing in the spatial direction
        self._identify_spectra(0.01,
                               1,
                               display=False,
                               renderer=renderer,
                               jsonstring=jsonstring,
                               iframe=False)

        if display:
            # set a side-by-side subplot
            fig = go.Figure()

            # show the image on the left
            fig.add_trace(
                go.Heatmap(z=np.log10(self.img),
                           colorscale="Viridis",
                           xaxis='x',
                           yaxis='y',
                           colorbar=dict(title='log(ADU)')))

            # plot the integrated count and the detected peaks on the right
            fig.add_trace(
                go.Scatter(x=np.log10(ztot),
                           y=ydata,
                           line=dict(color='black'),
                           xaxis='x2'))
            fig.add_trace(
                go.Scatter(x=np.log10(self.peak_height),
                           y=self.peak,
                           mode='markers',
                           marker=dict(color='firebrick'),
                           xaxis='x2'))

        my = np.zeros((self.nspec, self.spatial_size))
        y_sigma = np.zeros((self.nspec, self.spatial_size))

        # trace each individual spetrum one by one
        for i in range(self.nspec):

            peak_guess = [
                self.peak_height[i],
                np.nanmedian(ztot), self.peak[i], 2.
            ]

            if (recenter is False) and (len(prevtrace) > 10):
                my[i] = prevtrace
                y_sigma[i] = np.ones(len(prevtrace)) * self.seeing
                self.trace = my
                self.trace_sigma = np.array(y_sigma)
                if display:
                    fig.add_trace(
                        go.Scatter(x=[min(ztot[ztot > 0]),
                                      max(ztot)],
                                   y=[min(self.trace[i]),
                                      max(self.trace[i])],
                                   mode='lines',
                                   xaxis='x1'))
                    fig.add_trace(
                        go.Scatter(x=np.arange(len(self.trace[i])),
                                   y=self.trace[i],
                                   mode='lines',
                                   xaxis='x1'))
                    fig.update_layout(autosize=True,
                                      yaxis_title='Spatial Direction / pixel',
                                      xaxis=dict(
                                          zeroline=False,
                                          domain=[0, 0.5],
                                          showgrid=False,
                                          title='Spectral Direction / pixel'),
                                      xaxis2=dict(zeroline=False,
                                                  domain=[0.5, 1],
                                                  showgrid=True,
                                                  title='Integrated Count'),
                                      bargap=0,
                                      hovermode='closest',
                                      showlegend=False,
                                      height=800)

                    if jsonstring:
                        return fig.to_json()
                    if iframe:
                        pio.write_html(fig, 'ap_trace_quick.html')
                    if renderer == 'default':
                        fig.show()
                    else:
                        fig.show(renderer)

                break

            # use middle of previous trace as starting guess
            elif (recenter is True) and (len(prevtrace) > 10):
                peak_guess[2] = np.nanmedian(prevtrace)

            else:
                # fit a Gaussian to peak
                try:
                    pgaus, pcov = curve_fit(
                        self._gaus,
                        ydata[np.isfinite(ztot)],
                        ztot[np.isfinite(ztot)],
                        p0=peak_guess,
                        bounds=((0., 0., peak_guess[2] - 10, 0.),
                                (np.inf, np.inf, peak_guess[2] + 10, np.inf)))
                    #print(pgaus, pcov)
                except:
                    if not self.silence:
                        ValueError(
                            'Spectrum ' + str(i + 1) + ' of ' +
                            str(self.nspec) +
                            ' is likely to be (1) too faint, (2) in a crowed'
                            ' field, or (3) an extended source. Automated' +
                            ' tracing is sub-optimal. Please (1) reduce nspec,'
                            + ' (2) reduce n_steps, or (3) provide prevtrace.')

                if display:
                    fig.add_trace(
                        go.Scatter(x=np.log10(
                            self._gaus(ydata, pgaus[0], pgaus[1], pgaus[2],
                                       pgaus[3])),
                                   y=ydata,
                                   mode='lines',
                                   xaxis='x2'))

                # only allow data within a box around this peak
                ydata2 = ydata[np.where(
                    (ydata >= pgaus[2] - pgaus[3] * bigbox)
                    & (ydata <= pgaus[2] + pgaus[3] * bigbox))]
                yi = np.arange(self.spec_size)[ydata2]

                # define the X-bin edges
                xbins = np.linspace(0, self.spatial_size, nwindow)
                ybins = np.zeros_like(xbins)
                ybins_sigma = np.zeros_like(xbins)

                # loop through each bin
                for j in range(0, len(xbins) - 1):
                    # fit gaussian w/j each window
                    zi = np.sum(self.img[ydata2,
                                         int(np.floor(xbins[j])
                                             ):int(np.ceil(xbins[j + 1]))],
                                axis=1)
                    # fit gaussian w/j each window
                    if sum(zi) == 0:
                        break
                    else:
                        pguess = [
                            np.nanmax(zi),
                            np.nanmedian(zi), yi[np.nanargmax(zi)], 2.
                        ]
                    try:
                        popt, pcov = curve_fit(self._gaus, yi, zi, p0=pguess)
                    except:
                        if not self.silence:
                            ValueError('Step ' + str(j + 1) + ' of ' +
                                       str(nwindow) + ' of spectrum ' +
                                       str(i + 1) + ' of ' + str(self.nspec) +
                                       ' cannot be fitted.')
                        break

                    # if the peak is lower than background, sigma is too broad or
                    # gaussian fits off chip, then use chip-integrated answer
                    if ((popt[0] < 0) or (popt[3] < 0) or (popt[3] > 10)):
                        ybins[j] = pgaus[2]
                        popt = pgaus
                        if not self.silence:
                            ValueError(
                                'Step ' + str(j + 1) + ' of ' + str(nwindow) +
                                ' of spectrum ' + str(i + 1) + ' of ' +
                                str(self.nspec) +
                                ' has a poor fit. Initial guess is used instead.'
                            )
                    else:
                        ybins[j] = popt[2]
                        ybins_sigma[j] = popt[3]

                # recenter the bin positions, trim the unused bin off in Y
                mxbins = (xbins[:-1] + xbins[1:]) / 2.
                mybins = ybins[:-1]
                mx = np.arange(0, self.spatial_size)

                if (fittype == 'spline'):
                    # run a cubic spline thru the bins
                    interpolated = itp.UnivariateSpline(mxbins,
                                                        mybins,
                                                        ext=0,
                                                        k=order)
                    # interpolate 1 position per column
                    my[i] = interpolated(mx)

                elif (fittype == 'polynomial'):
                    # linear fit
                    npfit = np.polyfit(mxbins, mybins, deg=order)
                    # interpolate 1 position per column
                    my[i] = np.polyval(npfit, mx)

                else:
                    if not self.silence:
                        ValueError(
                            'Unknown fitting type, please choose from ' +
                            '(1) \'spline\'; or (2) \'polynomial\'.')

                # get the uncertainties in the spatial direction along the spectrum
                slope, intercept, r_value, p_value, std_err =\
                        stats.linregress(mxbins, ybins_sigma[:-1])
                y_sigma[i] = np.nanmedian(slope * mx + intercept)

                if display:
                    fig.add_trace(
                        go.Scatter(x=mx, y=my[i], mode='lines', xaxis='x'))

                if not self.silence:
                    if np.sum(ybins_sigma) == 0:
                        print(
                            'Spectrum ' + str(i + 1) + ' of ' +
                            str(self.nspec) +
                            ' is likely to be (1) too faint, (2) in a crowed'
                            ' field, or (3) an extended source. Automated' +
                            ' tracing is sub-optimal. Please disable multi-source'
                            +
                            ' mode and (1) reduce nspec, or (2) reduce n_steps,'
                            +
                            '  or (3) provide prevtrace, or (4) all of above.')

                    ValueError('Spectrum ' + str(i + 1) +
                               ': Trace gaussian width = ' + str(ybins_sigma) +
                               ' pixels')

            # add the minimum pixel value from fmask before returning
            #if len(spatial_mask)>1:
            #    my += min(spatial_mask)

            self.trace = my
            self.trace_sigma = np.array(y_sigma)

            if display:
                fig.update_layout(autosize=True,
                                  yaxis_title='Spatial Direction / pixel',
                                  xaxis=dict(
                                      zeroline=False,
                                      domain=[0, 0.5],
                                      showgrid=False,
                                      title='Spectral Direction / pixel'),
                                  xaxis2=dict(zeroline=False,
                                              domain=[0.5, 1],
                                              showgrid=True,
                                              title='Integrated Count'),
                                  bargap=0,
                                  hovermode='closest',
                                  showlegend=False,
                                  height=800)

                if jsonstring:
                    return fig.to_json()
                if iframe:
                    pio.write_html(fig, 'ap_trace_quick_' + str(j) + '.html')
                if renderer == 'default':
                    fig.show()
                else:
                    fig.show(renderer)

    def add_trace(self, trace, trace_sigma, x_pix=None):
        '''
        Add user-supplied trace. If the trace is of a different size to the
        2D spectral image in the spectral direction, the trace will be
        interpolated and extrapolated.

        Parameters
        ----------
        trace: 1D numpy array of list of 1D numpy array
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: float or list of float or numpy array of float
            Standard deviation of the Gaussian profile of a trace

        '''

        # If only one trace is provided
        if np.shape(np.shape(trace))[0] == 1:
            self.nspec = 1
            if x_pix is None:
                self.trace = [trace]
            else:
                self.trace = [
                    np.interp1d(x_pix, trace)(np.arange(self.spec_size))
                ]

            if isinstance(trace_sigma, float):
                self.trace_sigma = np.array(trace_sigma).reshape(-1)
            else:
                TypeError('The trace_sigma has to be a float. A ' +\
                          str(type(trace_sigma)) + ' is given.')

        # If there are more than one trace
        else:
            self.nspec = np.shape(trace)[0]
            if x_pix is None:
                self.trace = np.array(trace)
            elif len(x_pix) == 1:
                x_pix = np.ones((self.nspec, len(x_pix))) * x_pix
                self.trace = np.zeros((self.nspec, self.spec_size))
                for i, (x, t) in enumerate(zip(x_pix, trace)):
                    self.trace[i] = [
                        np.interp1d(x, t)(np.arange(self.spec_size))
                    ]
            else:
                ValueError(
                    'x_pix should be of the same shape as trace or '
                    'if all traces use the same x_pix, it should be the '
                    'same length as a trace.')

            # If all traces have the same line spread function
            if isinstance(trace_sigma, float):
                self.trace_sigma = np.ones(self.nspec) * trace_sigma
            elif (len(trace_sigma) == self.nspec):
                self.trace_sigma = np.array(trace_sigma)
            else:
                ValueError('The trace_sigma should be a single float or an '
                           'array of a size of the number the of traces.')

    def ap_extract(self,
                   apwidth=7,
                   skysep=3,
                   skywidth=5,
                   skydeg=1,
                   optimal=True,
                   display=False,
                   renderer='default',
                   jsonstring=False,
                   iframe=False):
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
                            function along the entire spectrum.

        Nothing is returned unless jsonstring of the plotly graph is set to be
        returned. The adu, skyadu and aduerr are stored as properties of the
        TwoDSpec object.

        adu: 1-d array
            The summed adu at each column about the trace. Note: is not
            sky subtracted!
        skyadu: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        aduerr: 1-d array
            the uncertainties of the adu values

        Parameters
        ----------
        apwidth: int
            The size of the aperature (fixed value for tophat extraction) or
            the sigma of the Gaussian (for the first iteration of optimal
            extraction).
        skysep: int
            The separation in pixels from the aperture to the sky window.
            (Default is 3)
        skywidth: int
            The width in pixels of the sky windows on either side of the
            aperture. Zero (0) means ignore sky subtraction. (Default is 7)
        skydeg: int
            The polynomial order to fit between the sky windows.
            (Default is 0, i.e. constant flat sky level)
        optimal: boolean
            Set optimal extraction. (Default is True)
        silence: boolean
            Set to disable warning/error messages. (Default is False)
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        """

        len_trace = len(self.trace[0])
        skyadu = np.zeros((self.nspec, len_trace))
        aduerr = np.zeros((self.nspec, len_trace))
        adu = np.zeros((self.nspec, len_trace))

        for j in range(self.nspec):

            median_trace = int(np.median(self.trace[j]))

            for i, pos in enumerate(self.trace[j]):
                itrace = int(round(pos))

                # first do the aperture adu
                widthup = apwidth
                widthdn = apwidth

                # fix width if trace is too close to the edge
                if (itrace + widthup > self.spatial_size):
                    widthup = spatial_size - itrace - 1
                if (itrace - widthdn < 0):
                    widthdn = itrace - 1  # i.e. starting at pixel row 1

                # simply add up the total adu around the trace +/- width
                xslice = self.img[itrace - widthdn:itrace + widthup + 1, i]
                adu_ap = np.sum(xslice)

                if skywidth > 0:
                    # get the indexes of the sky regions
                    y0 = max(itrace - widthdn - skysep - skywidth, 0)
                    y1 = max(itrace - widthdn - skysep, 0)
                    y2 = min(itrace + widthup + skysep + 1, self.spatial_size)
                    y3 = min(itrace + widthup + skysep + skywidth + 1,
                             self.spatial_size)
                    y = np.append(np.arange(y0, y1), np.arange(y2, y3))
                    z = self.img[y, i]

                    if (skydeg > 0):
                        # fit a polynomial to the sky in this column
                        pfit = np.polyfit(y, z, skydeg)
                        # define the aperture in this column
                        ap = np.arange(itrace - apwidth, itrace + apwidth + 1)
                        # evaluate the polynomial across the aperture, and sum
                        skyadu[j][i] = np.sum(np.polyval(pfit, ap))
                    elif (skydeg == 0):
                        skyadu[j][i] = np.sum(
                            np.ones(apwidth * 2 + 1) * np.nanmean(z))
                else:
                    pfit = [0., 0.]

                # if optimal extraction
                if optimal:
                    pix = range(itrace - widthdn, itrace + widthup + 1)
                    # Fit the sky background
                    if (skydeg > 0):
                        sky = np.polyval(pfit, pix)
                    else:
                        sky = np.ones(len(pix)) * np.nanmean(z)
                    # Get the optimal signals
                    adu[j][i], aduerr[j][i] = self._optimal_signal(
                        pix, xslice, sky, self.trace[j][i],
                        self.trace_sigma[j])
                else:
                    #-- finally, compute the error in this pixel
                    sigB = np.std(z)  # stddev in the background data
                    nB = len(y)  # number of bkgd pixels
                    nA = apwidth * 2. + 1  # number of aperture pixels

                    # based on aperture phot err description by F. Masci, Caltech:
                    # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
                    aduerr[j][i] = np.sqrt(
                        np.sum((adu_ap - skyadu[j][i])) / self.gain +
                        (nA + nA**2. / nB) * (sigB**2.))
                    adu[j][i] = adu_ap - skyadu[j][i]

            if display:

                fig = go.Figure()
                img_display = np.log10(
                    self.
                    img[max(0, median_trace - widthdn - skysep - skywidth -
                            1):min(median_trace + widthup + skysep +
                                   skywidth, len(self.img[0])), :])

                # show the image on the top
                fig.add_trace(
                    go.Heatmap(
                        x=np.arange(len_trace),
                        y=np.arange(
                            max(0, median_trace - widthdn - skysep - skywidth -
                                1),
                            min(median_trace + widthup + skysep + skywidth + 1,
                                len(self.img[0]))),
                        z=img_display,
                        colorscale="Viridis",
                        zmin=self.zmin,
                        zmax=self.zmax,
                        xaxis='x',
                        yaxis='y',
                        colorbar=dict(title='log(ADU)')))

                # Middle black box on the image
                fig.add_trace(
                    go.Scatter(
                        x=[0, len_trace, len_trace, 0, 0],
                        y=[
                            median_trace - widthdn - 1,
                            median_trace - widthdn - 1,
                            median_trace - widthdn - 1 + (apwidth * 2 + 1),
                            median_trace - widthdn - 1 + (apwidth * 2 + 1),
                            median_trace - widthdn - 1
                        ],
                        xaxis='x',
                        yaxis='y',
                        mode='lines',
                        line_color='black',
                        showlegend=False))

                # Lower red box on the image
                if (itrace - widthdn >= 0) & (skywidth > 0):
                    fig.add_trace(
                        go.Scatter(
                            x=[0, len_trace, len_trace, 0, 0],
                            y=[
                                max(
                                    0, median_trace - widthdn - skysep -
                                    (y1 - y0) - 1),
                                max(
                                    0, median_trace - widthdn - skysep -
                                    (y1 - y0) - 1),
                                max(
                                    0, median_trace - widthdn - skysep -
                                    (y1 - y0) - 1) + min(skywidth, (y1 - y0)),
                                max(
                                    0, median_trace - widthdn - skysep -
                                    (y1 - y0) - 1) + min(skywidth, (y1 - y0)),
                                max(
                                    0, median_trace - widthdn - skysep -
                                    (y1 - y0) - 1)
                            ],
                            line_color='red',
                            xaxis='x',
                            yaxis='y',
                            mode='lines',
                            showlegend=False))

                # Upper red box on the image
                if (itrace + widthup <= self.spatial_size) & (skywidth > 0):
                    fig.add_trace(
                        go.Scatter(x=[0, len_trace, len_trace, 0, 0],
                                   y=[
                                       min(median_trace + widthup + skysep,
                                           len(self.img[0])),
                                       min(median_trace + widthup + skysep,
                                           len(self.img[0])),
                                       min(median_trace + widthup + skysep,
                                           len(self.img[0])) +
                                       min(skywidth, (y3 - y2)),
                                       min(median_trace + widthup + skysep,
                                           len(self.img[0])) +
                                       min(skywidth, (y3 - y2)),
                                       min(median_trace + widthup + skysep,
                                           len(self.img[0]))
                                   ],
                                   xaxis='x',
                                   yaxis='y',
                                   mode='lines',
                                   line_color='red',
                                   showlegend=False))
                # plot the SNR
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=adu[j] / aduerr[j],
                               xaxis='x2',
                               yaxis='y3',
                               line=dict(color='slategrey'),
                               name='Signal-to-Noise Ratio'))

                # extrated source, sky and uncertainty
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=skyadu[j],
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='firebrick'),
                               name='Sky ADU'))
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=aduerr[j],
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='orange'),
                               name='Uncertainty'))
                fig.add_trace(
                    go.Scatter(x=np.arange(len_trace),
                               y=adu[j],
                               xaxis='x2',
                               yaxis='y2',
                               line=dict(color='royalblue'),
                               name='Target ADU'))

                # Decorative stuff
                fig.update_layout(
                    autosize=True,
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(zeroline=False,
                               domain=[0.5, 1],
                               showgrid=False,
                               title='Spatial Direction / pixel'),
                    yaxis2=dict(
                        range=[
                            min(np.nanmin(np.log10(adu[j])),
                                np.nanmin(np.log10(aduerr[j])),
                                np.nanmin(np.log10(skyadu[j])), 1),
                            max(np.nanmax(np.log10(adu[j])),
                                np.nanmax(np.log10(skyadu[j])))
                        ],
                        zeroline=False,
                        domain=[0, 0.5],
                        showgrid=True,
                        type='log',
                        title='log(ADU / Count)',
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
                    showlegend=True,
                    height=800)
                if jsonstring:
                    return fig.to_json()
                if iframe:
                    pio.write_html(fig, 'ap_extract_' + str(j) + '.html')
                if renderer == 'default':
                    fig.show()
                else:
                    fig.show(renderer)

        self.adu = adu
        self.aduerr = aduerr
        self.skyadu = skyadu


class WavelengthPolyFit():
    def __init__(self, spec, arc, silence=False):
        '''
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
        spec: TwoDSpec object
            TwoDSpec of the science/standard image containing the trace(s).
        arc: 2D numpy array
            The image of the arc image.
        '''

        self.spec = spec

        # If data provided is an numpy array
        if isinstance(arc, np.ndarray):
            self.arc = arc
        # If it is a fits.hdu.image.PrimaryHDU object
        elif isinstance(arc, fits.hdu.image.PrimaryHDU):
            self.arc = arc.data
        # If it is an ImageReduction object
        elif isinstance(arc, ImageReduction):
            self.arc = arc.arc_master
        # If manually calibration is intended
        elif arc == 'manual':
            self.arc = None
            if not silence:
                warnings.warn('Only add_pfit() can be used.')
        else:
            TypeError('Please provide a numpy array, an ' +
                      'astropy.io.fits.hdu.image.PrimaryHDU object or an ' +
                      'ImageReduction object.')

        # the valid y-range of the chip (i.e. spatial direction)
        if (len(self.spec.spatial_mask) > 1):
            if self.spec.saxis is 1:
                self.arc = self.arc[self.spec.spatial_mask]
            else:
                self.arc = self.arc[:, self.spec.spatial_mask]

        # the valid x-range of the chip (i.e. spectral direction)
        if (len(self.spec.spec_mask) > 1):
            if self.spec.saxis is 1:
                self.arc = self.arc[:, self.spec.spec_mask]
            else:
                self.arc = self.arc[self.spec.spec_mask]

        # get the length in the spectral and spatial directions
        if self.spec.saxis is 0:
            self.arc = np.transpose(self.arc)

        if self.spec.flip:
            self.arc = np.flip(self.arc)

    def find_arc_lines(self,
                       percentile=25.,
                       distance=5.,
                       display=False,
                       jsonstring=False,
                       renderer='default',
                       iframe=False):
        '''
        This function applies the trace(s) to the arc image then take median
        average of the stripe before identifying the arc lines (peaks) with
        scipy.signal.find_peaks(), where only the distance and the prominence
        keywords are used. Distance is the minimum separation between peaks,
        the default value is roughly twice the nyquist sampling rate (i.e.
        pixel size is 2.3 times smaller than the object that is being resolved,
        hence, the sepration between two clearly resolved peaks are ~5 pixels
        apart). An crude estimate of the background can exclude random noise
        which look like small peaks.

        Parameters
        ----------
        percentile: float
            The percentile of the flux to be used as the estimate of the
            background sky level to the first order. [ADU]
        distance: float
            Minimum separation between peaks
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        JSON strings if jsonstring is set to True
        '''

        trace_shape = np.shape(self.spec.trace)
        self.nspec = trace_shape[0]

        self.spectrum = np.zeros(trace_shape)
        self.arc_trace = []
        self.peaks = []

        p = np.percentile(self.arc, percentile)

        fig = np.array([None] * self.nspec, dtype='object')
        for j in range(self.nspec):
            trace = int(np.mean(self.spec.trace[j]))
            width = int(np.mean(self.spec.trace_sigma[j]) * 3)

            self.arc_trace.append(
                self.arc[max(0, trace - width -
                             1):min(trace + width, len(self.spec.img[0])), :])

            self.spectrum[j] = np.median(self.arc_trace[j], axis=0)

            peaks, _ = signal.find_peaks(self.spectrum[j],
                                         distance=distance,
                                         prominence=p)

            # Fine ftuning
            self.peaks.append(
                refine_peaks(self.spectrum[j], peaks, window_width=3))

            if display & plotly_imported:
                fig[j] = go.Figure()

                # show the image on the top
                fig[j].add_trace(
                    go.Heatmap(x=np.arange(self.arc.shape[0]),
                               y=np.arange(self.arc.shape[1]),
                               z=np.log10(self.arc),
                               colorscale="Viridis",
                               colorbar=dict(title='log(ADU)')))

                for i in self.peaks[j]:
                    fig[j].add_trace(
                        go.Scatter(x=[i, i],
                                   y=[0, self.arc.shape[0]],
                                   mode='lines',
                                   line=dict(color='firebrick', width=1)))

                fig[j].update_layout(
                    autosize=True,
                    xaxis=dict(zeroline=False,
                               range=[0, self.arc.shape[1]],
                               title='Spectral Direction / pixel'),
                    yaxis=dict(zeroline=False,
                               range=[0, self.arc.shape[0]],
                               title='Spatial Direction / pixel'),
                    hovermode='closest',
                    showlegend=False,
                    height=600)

                if jsonstring:
                    return fig[j].to_json()
                if iframe:
                    pio.write_html(fig[j], 'arc_lines_' + str(j) + '.html')
                if renderer == 'default':
                    fig[j].show()
                else:
                    fig[j].show(renderer)

    def calibrate(self,
                  elements=None,
                  min_wave=3500.,
                  max_wave=8500.,
                  sample_size=5,
                  max_tries=10000,
                  top_n=8,
                  nslopes=5000,
                  range_tolerance=500.,
                  fit_tolerance=10.,
                  polydeg=4,
                  candidate_thresh=10.,
                  ransac_thresh=1.,
                  xbins=250,
                  ybins=250,
                  brute_force=False,
                  fittype='poly',
                  mode='manual',
                  progress=False,
                  pfit=None,
                  display=False,
                  savefig=None):
        '''
        A wrapper function to perform wavelength calibration with RASCAL.

        As of 14 January 2020, it supports He, Ne, Ar, Cu, Kr, Cd, Xe,
        Hg and Th from NIST:

            https://physics.nist.gov/PhysRefData/ASD/lines_form.html

        If there is already a set of good coefficienes, use calibrate_pfit()
        instead.

        Parameters
        ----------
        elements: string or list of string
            String or list of strings of Chemical symbol. Case insensitive.
        min_wave: float
            Minimum wavelength of the bluest arc line, NOT OF THE SPECTRUM.
        max_wave: float
            Maximum wavelength of the reddest arc line, NOT OF THE SPECTRUM.
        sample_size: int
            Number of lines to be fitted in each loop.
        max_tries: int
            Number of trials of polynomial fitting.
        top_n: int
            Top ranked lines to be fitted.
        nslopes: int
            Number of lines to be used in Hough transform.
        range_tolerance: float
            Estimation of the error on the provided spectral range
            e.g. 3000 - 5000 with tolerance 500 will search for
            solutions that may satisfy 2500 - 5500
        fit_tolerance: float
            Maximum RMS allowed
        polydeg: int
            Degree of the polynomial
        candidate_thresh: float
            Threshold for considering a point to be an inlier during candidate
            peak/line selection. Don't make this too small, it should allow
            for the error between a linear and non-linear fit.
        ransac_thresh: float
            The distance criteria to be considered an inlier to a fit. This
            should be close to the size of the expected residuals on the final
            fit.
        xbins: int
            The number of bins in the pixel direction (in Hough space).
        ybins : int
            The number of bins in the wavelength direction (in Hough space).
        brute_force: boolean
            Set to try all possible combinations and choose the best fit as
            the solution. This takes tens of minutes for tens of lines.
        fittype: string
            One of 'poly', 'legendre' or 'chebyshev'.
        mode: string
            Default to 'manual' to read take in user supplied sample_size,
            max_tries, top_n and nslope, which are by default equivalent to
            'normal' mode. Predefined modes are 'fast', 'normal' and 'slow':
            fast:
                sample_size = 3, max_tries = 1000, top_n = 20, nslope = 500
            normal:
                sample_size = 5, max_tries = 5000, top_n = 20, nslope = 1000
            slow:
                sample_size = 5, max_tries = 10000, top_n = 20, nslope = 2000
        progress: boolean
            Set to show the progress using tdqm (if imported).
        pfit: list
            List of the polynomial fit coefficients for the first guess.
        display: boolean
            Set to show diagnostic plot.
        '''

        self.pfit = []
        self.pfit_type = []

        for j in range(self.nspec):
            c = Calibrator(self.peaks[j],
                           min_wavelength=min_wave,
                           max_wavelength=max_wave,
                           num_pixels=len(self.spectrum[j]),
                           plotting_library='plotly')
            c.add_atlas(elements)
            c.set_fit_constraints(num_slopes=nslopes,
                                  range_tolerance=range_tolerance,
                                  fit_tolerance=fit_tolerance,
                                  polydeg=polydeg,
                                  candidate_thresh=candidate_thresh,
                                  ransac_thresh=ransac_thresh,
                                  xbins=xbins,
                                  ybins=ybins,
                                  brute_force=brute_force,
                                  fittype=fittype)

            pfit = c.fit(sample_size=sample_size,
                         max_tries=max_tries,
                         top_n=top_n,
                         n_slope=nslopes,
                         mode=mode,
                         progress=progress,
                         coeff=pfit)

            self.pfit.append(pfit)
            self.pfit_type.append(fittype)

            if display:
                c.plot_fit(np.median(self.arc_trace[j], axis=0),
                           self.pfit[j],
                           plot_atlas=True,
                           log_spectrum=False,
                           tolerance=1.0,
                           output_filename=savefig)

    def calibrate_pfit(self,
                       elements,
                       min_wave=3500.,
                       max_wave=8500.,
                       tolerance=10.,
                       display=False,
                       savefig=None):
        '''
        A wrapper function to fine tune wavelength calibration with RASCAL
        when there is already a set of good coefficienes.

        As of 14 January 2020, it supports He, Ne, Ar, Cu, Kr, Cd, Xe,
        Hg and Th from NIST:

            https://physics.nist.gov/PhysRefData/ASD/lines_form.html

        Parameters
        ----------
        elements: string or list of string
            String or list of strings of Chemical symbol. Case insensitive.
        pfit : list
            List of polynomial fit coefficients
        min_wave: float
            Minimum wavelength of the bluest arc line, NOT OF THE SPECTRUM.
        max_wave: float
            Maximum wavelength of the reddest arc line, NOT OF THE SPECTRUM.
        tolerance : float
            Absolute difference between fit and model.
        '''

        pfit_new = []

        for j in range(self.nspec):
            c = Calibrator(self.peaks[j],
                           min_wavelength=min_wave,
                           max_wavelength=max_wave,
                           num_pixels=len(self.spectrum[j]),
                           plotting_library='plotly')
            c.add_atlas(elements=elements)

            pfit, _, _ = c.match_peaks_to_atlas(self.pfit[j],
                                                tolerance=tolerance)

            pfit_new.append(pfit)

            if display:
                c.plot_fit(np.median(self.arc_trace[j], axis=0),
                           pfit_new[j],
                           plot_atlas=True,
                           log_spectrum=False,
                           tolerance=1.0,
                           output_filename=savefig)

        self.pfit = pfit_new

    def add_pfit(self, pfit, pfit_type='poly'):
        '''
        Add user supplied polynomial coefficient.

        Parameters
        ----------
        pfit: numpy array or list of numpy array
            Coefficients of the polynomial fit.
        pfit_type: str
            One of 'poly', 'legendre' or 'chebyshev'.
        '''
        if isinstance(pfit, list):
            nspec = len(pfit)
        elif isinstance(pfit, np.ndarray):
            if np.shape(np.shape(pfit)) == 1:
                nspec = 1
            else:
                nspec = np.shape(pfit)[0]
        else:
            TypeError('Please provide a numpy array or a list of 1D numpy '
                      'array.')

        self.pfit = pfit
        self.pfit_type = []

        if len(pfit_type) != nspec:
            for i in range(nspec):
                self.pfit_type.append(pfit_type)
            self.pfit_type = np.array(self.pfit_type)
        else:
            self.pfit_type = pfit_type


class StandardFlux:
    def __init__(self, target, group, cutoff=0.4, ftype='flux', silence=False):
        '''
        This class handles flux calibration by comparing the extracted and
        wavelength-calibrated standard observation to the "ground truth"
        from

        https://github.com/iraf-community/iraf/tree/master/noao/lib/onedstds
        https://www.eso.org/sci/observing/tools/standards/spectra.html

        See explanation notes at those links for details.

        The list of targets and groups can be listed with
        >>> from aspired.standard_list import list_all
        >>> list_all()

        The units of the data are
            wavelength: A
            flux:       ergs / cm / cm / s / A
            mag:        mag (AB)

        Parameters
        ----------
        target: string
            Name of the standard star
        group: string
            Name of the group of standard star
        cutoff: float
            The threshold for the word similarity in the range of [0, 1].
        ftype: string
            'flux' or 'mag' (AB magnitude)
        silence: boolean
            Set to suppress all verbose warning.
        '''

        self.target = target
        self.group = group
        self.cutoff = cutoff
        self.ftype = ftype
        self.silence = silence

        self._lookup_standard()

    def _lookup_standard(self):
        '''
        Check if the requested standard and library exist. Return the three
        most similar words if the requested one does not exist. See

            https://docs.python.org/3.7/library/difflib.html
        '''

        # Load the list of targets in the requested group
        try:
            target_list = eval(self.group)
        except:
            raise ValueError('Requested standard star library does not exist.')

        # If the requested target is not in the target list, suggest the three
        # closest match
        if self.target not in target_list:
            best_match = difflib.get_close_matches(self.target,
                                                   target_list,
                                                   cutoff=self.cutoff)
            raise ValueError(
                'Requested standard star is not in the library.', '',
                'The requrested spectrophotometric library contains: ',
                target_list, '', 'Are you looking for these: ', best_match)

    def load_standard(self,
                      display=False,
                      renderer='default',
                      jsonstring=False,
                      iframe=False):
        '''
        Read the standard flux/magnitude file. And return the wavelength and
        flux/mag.

        Returns
        -------
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        JSON strings if jsonstring is set to True
        '''

        flux_multiplier = 1.
        if self.group[:4] == 'iraf':
            target_name = self.target + '.dat'
        else:
            if self.ftype == 'flux':
                target_name = 'f' + self.target + '.dat'
                if self.group != 'xshooter':
                    flux_multiplier = 1e-16
            elif self.ftype == 'mag':
                target_name = 'm' + self.target + '.dat'
            else:
                raise ValueError('The type has to be \'flux\' of \'mag\'.')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(dir_path, 'standards',
                                str(self.group) + 'stan', target_name)

        if self.group[:4] == 'iraf':
            f = np.loadtxt(filepath, skiprows=1)
        else:
            f = np.loadtxt(filepath)

        wave = f[:, 0]
        if (self.group[:4] == 'iraf') & (self.ftype == 'flux'):
            fluxmag = 10.**(-(f[:, 1] / 2.5)) * 3630.780548 / 3.34e4 / wave**2
        else:
            fluxmag = f[:, 1] * flux_multiplier

        self.wave_std = wave
        self.fluxmag_std = fluxmag

        # Note that if the renderer does not generate any image (e.g. JSON)
        # nothing will be displayed
        if display & plotly_imported:
            self.inspect_standard(renderer, jsonstring, iframe)

    def inspect_standard(self,
                         renderer='default',
                         jsonstring=False,
                         iframe=False):
        '''
        Display the standard star plot.

        Parameters
        ----------
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        JSON strings if jsonstring is set to True
        '''
        fig = go.Figure(layout=dict(updatemenus=list([
            dict(
                active=0,
                buttons=list([
                    dict(label='Log Scale',
                         method='update',
                         args=[{
                             'visible': [True, True]
                         }, {
                             'title': 'Log scale',
                             'yaxis': {
                                 'type': 'log'
                             }
                         }]),
                    dict(label='Linear Scale',
                         method='update',
                         args=[{
                             'visible': [True, False]
                         }, {
                             'title': 'Linear scale',
                             'yaxis': {
                                 'type': 'linear'
                             }
                         }])
                ]),
            )
        ]),
                                    title='Log scale'))

        # show the image on the top
        fig.add_trace(
            go.Scatter(x=self.wave_std,
                       y=self.fluxmag_std,
                       line=dict(color='royalblue', width=4)))

        fig.update_layout(
            autosize=True,
            title=self.group + ': ' + self.target + ' ' + self.ftype,
            xaxis_title=r'$\text{Wavelength / A}$',
            yaxis_title=
            r'$\text{Flux / ergs cm}^{-2} \text{s}^{-1} \text{A}^{-1}$',
            hovermode='closest',
            showlegend=False,
            height=800)

        if jsonstring:
            return fig.to_json()
        if iframe:
            pio.write_html(fig, 'standard.html')
        if renderer == 'default':
            fig.show()
        else:
            fig.show(renderer)


class OneDSpec:
    def __init__(self,
                 science,
                 wave_cal,
                 standard=None,
                 wave_cal_std=None,
                 flux_cal=None):
        '''
        This class applies the wavelength calibrations and compute & apply the
        flux calibration to the extracted 1D spectra. The standard TwoDSpec
        object is not required for data reduction, but the flux calibrated
        standard observation will not be available for diagnostic.

        Parameters
        ----------
        science: TwoDSpec object
            The TwoDSpec object with the extracted science target
        wave_cal: WavelengthPolyFit object
            The WavelengthPolyFit object for the science target, flux will
            not be calibrated if this is not provided.
        standard: TwoDSpec object
            The TwoDSpec object with the extracted standard target
        wave_cal_std: WavelengthPolyFit object
            The WavelengthPolyFit object for the standard target, flux will
            not be calibrated if this is not provided.
        flux_cal: StandardFlux object
            The true mag/flux values.
        '''

        try:
            self.adu = science.adu
            self.aduerr = science.aduerr
            self.skyadu = science.skyadu
            self.exptime = science.exptime
            self.nspec = science.nspec
        except:
            raise TypeError('Please provide a valid TwoDSpec.')

        try:
            self._set_wavecal(wave_cal, 'science')
        except:
            raise TypeError('Please provide a WavelengthPolyFit.')

        if standard is not None:
            self._set_standard(standard)
            self.standard_imported = True
        else:
            self.standard_imported = False
            warnings.warn('The TwoDSpec of the standard observation is not '
                          'available. Flux calibration will not be performed.')

        if wave_cal_std is not None:
            self._set_wavecal(wave_cal_std, 'standard')
            self.wav_cal_std_imported = True

        if (wave_cal_std is None) & (standard is not None):
            self._set_wavecal(wave_cal, 'standard')
            self.wav_cal_std_imported = True
            warnings.warn(
                'The WavelengthPolyFit of the standard observation '
                'is not available. The wavelength calibration for the science '
                'frame is applied to the standard.')

        if flux_cal is not None:
            self._set_fluxcal(flux_cal)
            self.flux_imported = True
        else:
            self.flux_imported = False
            warnings.warn('The StandardFlux of the standard star is not '
                          'available. Flux calibration will not be performed.')

    def _set_standard(self, standard):
        '''
        Extract the required information from the TwoDSpec object of the
        standard.

        Parameters
        ----------
        standard: TwoDSpec object
            The TwoDSpec object with the extracted standard target
        '''

        try:
            self.adu_std = standard.adu[0]
            self.aduerr_std = standard.aduerr[0]
            self.skyadu_std = standard.skyadu[0]
            self.exptime_std = standard.exptime
        except:
            raise TypeError('Please provide a valid TwoDSpec.')

    def _set_wavecal(self, wave_cal, stype):
        '''
        Extract the required information from a WavelengthPolyFit object, it
        can be used to apply the polynomial coefficients for science, standard
        or both.

        Parameters
        ----------
        wave_cal: WavelengthPolyFit object
            The WavelengthPolyFit object for the standard target, flux will
            not be calibrated if this is not provided.
        stype: string
            'science', 'standard' or 'all' to indicate type
        '''

        if stype in ['science', 'all']:
            try:
                self.pfit_type = wave_cal.pfit_type
                self.pfit = wave_cal.pfit
                self.polyval = np.array([None] * self.nspec, dtype='object')
                for i in range(self.nspec):
                    if self.pfit_type[i] == 'poly':
                        self.polyval[i] = np.polynomial.polynomial.polyval
                    elif self.pfit_type[i] == 'legendre':
                        self.polyval[i] = np.polynomial.legendre.legval
                    elif self.pfit_type[i] == 'chebyshev':
                        self.polyval[i] = np.polynomial.chebyshev.chebval
                    else:
                        raise ValueError(
                            'fittype must be: (1) poly; (2) legendre; or '
                            '(3) chebyshev')
            except:
                raise TypeError('Please provide a valid WavelengthPolyFit.')
        if stype in ['standard', 'all']:
            try:
                self.pfit_type_std = wave_cal.pfit_type
                self.pfit_std = wave_cal.pfit
                if isinstance(self.pfit_std, list):
                    self.pfit_type_std = self.pfit_type_std[0]
                    self.pfit_std = self.pfit_std[0]
                if self.pfit_type_std == 'poly':
                    self.polyval_std = np.polynomial.polynomial.polyval
                elif self.pfit_type_std == 'legendre':
                    self.polyval_std = np.polynomial.legendre.legval
                elif self.pfit_type_std == 'chebyshev':
                    self.polyval_std = np.polynomial.chebyshev.chebval
                else:
                    raise ValueError(
                        'fittype must be: (1) poly; (2) legendre; or '
                        '(3) chebyshev')
            except:
                raise TypeError('Please provide a valid WavelengthPolyFit.')

        if stype not in ['science', 'standard', 'all']:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all')

    def _set_fluxcal(self, flux_cal):
        '''
        Extract the required information from a StandardFlux object.

        Parameters
        ----------
        flux_cal: StandardFlux object
            The true mag/flux values.
        '''

        try:
            self.group = flux_cal.group
            self.target = flux_cal.target
            self.wave_std_true = flux_cal.wave_std
            self.fluxmag_std_true = flux_cal.fluxmag_std
        except:
            raise TypeError('Please provide a valid StandardFlux.')

    def apply_wavelength_calibration(self, stype):
        '''
        Apply the wavelength calibration.

        Parameters
        ----------
        stype: string
            'science', 'standard' or 'all' to indicate type
        '''

        # Can be multiple spectra in the science frame
        if stype in ['science', 'all']:
            pix = np.arange(len(self.adu[0]))
            self.wave = np.zeros((self.nspec, len(self.adu[0])))
            for i in range(self.nspec):
                self.wave[i] = self.polyval[i](pix, self.pfit[i])
        # Only one spectrum in the standard frame
        if stype in ['standard', 'all']:
            if self.standard_imported:
                pix_std = np.arange(len(self.adu_std))
                self.wave_std = self.polyval_std(pix_std, self.pfit_std)
            else:
                raise AttributeError(
                    'The TwoDSpec of the standard observation is not '
                    'available. Flux calibration will not be performed.')
        if stype not in ['science', 'standard', 'all']:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')

    def compute_sencurve(self,
                         kind=3,
                         smooth=False,
                         slength=5,
                         sorder=3,
                         mask_range=[[6850, 6960], [7150, 7400], [7575, 7700]],
                         display=False,
                         renderer='default',
                         jsonstring=False,
                         iframe=False):
        '''
        The sensitivity curve is computed by dividing the true values by the
        wavelength calibrated standard spectrum, which is resampled with the
        spectres.spectres(). The curve is then interpolated with a cubic spline
        by default and is stored as a scipy interp1d object.

        A Savitzky-Golay filter is available for smoothing before the
        interpolation but it is not used by default.

        6850 - 7000 A,  7150 - 7400 A and 7575 - 7775 A are masked by default.

        Parameters
        ----------
        kind: string or integer [1,2,3,4,5 only]
            interpolation kind
            >>> [‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
                 ‘previous’, ‘next’]
        smooth: boolean
            set to smooth the input spectrum with scipy.signal.savgol_filter
        slength: int
            SG-filter window size
        sorder: int
            SG-filter polynomial order
        mask_range: None or list of list
            Masking out regions not suitable for fitting the sensitivity curve.
                None:         no mask
                list of list: [[min1, max1], [min2, max2],...]
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        JSON strings if jsonstring is set to True.
        '''

        # Get the standard flux/magnitude
        self.slength = slength
        self.sorder = sorder
        self.smooth = smooth

        if spectres_imported:
            # resampling both the observed and the database standard spectra
            # in unit of flux per second
            flux_std = spectres(self.wave_std_true, self.wave_std,
                                self.adu_std / self.exptime_std)
            flux_std_true = self.fluxmag_std_true
        else:
            flux_std = flux_std / self.exptime_std
            flux_std_true = itp.interp1d(self.wave_std_true,
                                         self.fluxmag_std_true)(self.wave_std)
        # Get the sensitivity curve
        sensitivity = flux_std_true / flux_std

        if mask_range is None:
            mask = (np.isfinite(sensitivity) & (sensitivity > 0.))
        else:
            mask = (np.isfinite(sensitivity) & (sensitivity > 0.))
            for m in mask_range:
                mask = mask & ((self.wave_std_true < m[0]) |
                               (self.wave_std_true > m[1]))

        sensitivity = sensitivity[mask]
        wave_std = self.wave_std_true[mask]
        flux_std = flux_std[mask]

        # apply a Savitzky-Golay filter to remove noise and Telluric lines
        if smooth:
            sensitivity = signal.savgol_filter(sensitivity, slength, sorder)

        sencurve = itp.interp1d(wave_std,
                                np.log10(sensitivity),
                                kind=kind,
                                fill_value='extrapolate')

        self.sensitivity = sensitivity
        self.sencurve = sencurve
        self.wave_sen = wave_std
        self.flux_sen = flux_std

        # Diagnostic plot
        if display & plotly_imported:
            self.inspect_sencurve(renderer, jsonstring, iframe)

    def add_sensitivity(self, sensitivity, wave_std, flux_std):
        self.sensitivity = sensitivity
        self.wave_sen = wave_std
        self.flux_sen = flux_std
        self.sencurve = itp.interp1d(wave_std,
                                     np.log10(sensitivity),
                                     kind=kind,
                                     fill_value='extrapolate')

    def add_sencurve():
        pass

    def inspect_sencurve(self,
                         renderer='default',
                         jsonstring=False,
                         iframe=False):
        '''
        Display the computed sensitivity curve.

        Parameters
        ----------
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        JSON strings if jsonstring is set to True.
        '''

        fig = go.Figure(layout=dict(updatemenus=list([
            dict(
                active=0,
                buttons=list([
                    dict(label='Log Scale',
                         method='update',
                         args=[{
                             'visible': [True, True]
                         }, {
                             'title': 'Log scale',
                             'yaxis': {
                                 'type': 'log'
                             }
                         }]),
                    dict(label='Linear Scale',
                         method='update',
                         args=[{
                             'visible': [True, False]
                         }, {
                             'title': 'Linear scale',
                             'yaxis': {
                                 'type': 'linear'
                             }
                         }])
                ]),
            )
        ]),
                                    title='Log scale'))
        # show the image on the top
        fig.add_trace(
            go.Scatter(x=self.wave_sen,
                       y=self.flux_sen,
                       line=dict(color='royalblue', width=4),
                       name='ADU (Observed)'))

        fig.add_trace(
            go.Scatter(x=self.wave_sen,
                       y=self.sensitivity,
                       yaxis='y2',
                       line=dict(color='firebrick', width=4),
                       name='Sensitivity Curve'))

        fig.add_trace(
            go.Scatter(x=self.wave_sen,
                       y=10.**self.sencurve(self.wave_sen),
                       yaxis='y2',
                       line=dict(color='black', width=2),
                       name='Best-fit Sensitivity Curve'))

        if self.smooth:
            fig.update_layout(title='SG(' + str(self.slength) + ', ' +
                              str(self.sorder) + ')-Smoothed ' + self.group +
                              ': ' + self.target,
                              yaxis_title='Smoothed ADU')
        else:
            fig.update_layout(title=self.group + ': ' + self.target,
                              yaxis_title='ADU')

        fig.update_layout(autosize=True,
                          hovermode='closest',
                          showlegend=True,
                          xaxis_title=r'$\text{Wavelength / A}$',
                          yaxis=dict(title='ADU'),
                          yaxis2=dict(title='Sensitivity Curve',
                                      type='log',
                                      anchor="x",
                                      overlaying="y",
                                      side="right"),
                          legend=go.layout.Legend(x=0,
                                                  y=1,
                                                  traceorder="normal",
                                                  font=dict(
                                                      family="sans-serif",
                                                      size=12,
                                                      color="black"),
                                                  bgcolor='rgba(0,0,0,0)'),
                          height=800)
        if jsonstring:
            return fig.to_json()
        if iframe:
            pio.write_html(fig, 'senscurve.html')
        if renderer == 'default':
            fig.show()
        else:
            fig.show(renderer)

    def apply_flux_calibration(self, stype='all'):
        '''
        Apply the computed sensitivity curve.

        Parameters
        ----------
        stype: string
            'science', 'standard' or 'all' to indicate type
        '''

        if stype == 'science' or stype == 'all':
            self.flux = np.zeros((self.nspec, len(self.adu[0])))
            self.fluxerr = np.zeros((self.nspec, len(self.adu[0])))
            self.skyflux = np.zeros((self.nspec, len(self.adu[0])))
            for i in range(self.nspec):
                self.flux[i] = 10.**self.sencurve(self.wave[i]) * self.adu[i]
                self.fluxerr[i] = 10.**self.sencurve(
                    self.wave[i]) * self.aduerr[i]
                self.skyflux[i] = 10.**self.sencurve(
                    self.wave[i]) * self.skyadu[i]
        if stype == 'standard' or stype == 'all':
            self.flux_std = 10.**self.sencurve(self.wave_std) * self.adu_std
            self.fluxerr_std = 10.**self.sencurve(
                self.wave_std) * self.aduerr_std
            self.skyflux_std = 10.**self.sencurve(
                self.wave_std) * self.skyadu_std
        if stype not in ['science', 'standard', 'all']:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')

    def inspect_reduced_spectrum(self,
                                 stype='all',
                                 wave_min=4000.,
                                 wave_max=8000.,
                                 renderer='default',
                                 jsonstring=False,
                                 iframe=False):
        '''
        Display the reduced spectra.

        Parameters
        ----------
        stype: string
            'science', 'standard' or 'all' to indicate type
        wave_min: float
            Minimum wavelength to display
        wave_max: float
            Maximum wavelength to display
        renderer: string
            plotly renderer options.
        jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        JSON strings if jsonstring is set to True.
        '''

        if stype in ['science', 'all']:
            fig_sci = np.array([None] * self.nspec, dtype='object')
            for j in range(self.nspec):

                wave_mask = ((self.wave[j] > wave_min) &
                             (self.wave[j] < wave_max))
                if self.standard_imported:
                    flux_mask = (
                        (self.flux[j] >
                         np.nanpercentile(self.flux[j][wave_mask], 5) / 1.5) &
                        (self.flux[j] <
                         np.nanpercentile(self.flux[j][wave_mask], 95) * 1.5))
                    flux_min = np.log10(np.nanmin(self.flux[j][flux_mask]))
                    flux_max = np.log10(np.nanmax(self.flux[j][flux_mask]))
                else:
                    flux_mask = (
                        (self.adu[j] >
                         np.nanpercentile(self.adu[j][wave_mask], 5) / 1.5) &
                        (self.adu[j] <
                         np.nanpercentile(self.adu[j][wave_mask], 95) * 1.5))
                    flux_min = np.log10(np.nanmin(self.adu[j][flux_mask]))
                    flux_max = np.log10(np.nanmax(self.adu[j][flux_mask]))

                fig_sci[j] = go.Figure(layout=dict(updatemenus=list([
                    dict(
                        active=0,
                        buttons=list([
                            dict(label='Log Scale',
                                 method='update',
                                 args=[{
                                     'visible': [True, True]
                                 }, {
                                     'title': 'Log scale',
                                     'yaxis': {
                                         'type': 'log'
                                     }
                                 }]),
                            dict(label='Linear Scale',
                                 method='update',
                                 args=[{
                                     'visible': [True, False]
                                 }, {
                                     'title': 'Linear scale',
                                     'yaxis': {
                                         'type': 'linear'
                                     }
                                 }])
                        ]),
                    )
                ]),
                                                   title='Log scale'))
                # show the image on the top
                if self.standard_imported:
                    fig_sci[j].add_trace(
                        go.Scatter(x=self.wave[j],
                                   y=self.flux[j],
                                   line=dict(color='royalblue'),
                                   name='Flux'))
                    fig_sci[j].add_trace(
                        go.Scatter(x=self.wave[j],
                                   y=self.fluxerr[j],
                                   line=dict(color='firebrick'),
                                   name='Flux Uncertainty'))
                    fig_sci[j].add_trace(
                        go.Scatter(x=self.wave[j],
                                   y=self.skyflux[j],
                                   line=dict(color='orange'),
                                   name='Sky Flux'))
                else:
                    fig_sci[j].add_trace(
                        go.Scatter(x=self.wave[j],
                                   y=self.adu[j],
                                   line=dict(color='royalblue'),
                                   name='ADU'))
                    fig_sci[j].add_trace(
                        go.Scatter(x=self.wave[j],
                                   y=self.aduerr[j],
                                   line=dict(color='firebrick'),
                                   name='ADU Uncertainty'))
                    fig_sci[j].add_trace(
                        go.Scatter(x=self.wave[j],
                                   y=self.skyadu[j],
                                   line=dict(color='orange'),
                                   name='Sky ADU'))
                fig_sci[j].update_layout(
                    autosize=True,
                    hovermode='closest',
                    showlegend=True,
                    xaxis=dict(title='Wavelength / A',
                               range=[wave_min, wave_max]),
                    yaxis=dict(title='Flux',
                               range=[flux_min, flux_max],
                               type='log'),
                    legend=go.layout.Legend(x=0,
                                            y=1,
                                            traceorder="normal",
                                            font=dict(family="sans-serif",
                                                      size=12,
                                                      color="black"),
                                            bgcolor='rgba(0,0,0,0)'),
                    height=800)

                if jsonstring:
                    return fig_sci[j].to_json()
                if iframe:
                    pio.write_html(fig_sci[j], 'spectrum_' + str(j) + '.html')
                if renderer == 'default':
                    fig_sci[j].show()
                else:
                    fig_sci[j].show(renderer)

        if stype in ['standard', 'all']:

            if not self.standard_imported:
                warnings.warn('Standard observation is not provided.')
            else:
                wave_std_mask = ((self.wave_std > wave_min) &
                                 (self.wave_std < wave_max))
                flux_std_mask = (
                    (self.flux_std >
                     np.nanpercentile(self.flux_std[wave_std_mask], 5) / 1.5) &
                    (self.flux_std <
                     np.nanpercentile(self.flux_std[wave_std_mask], 95) * 1.5))
                flux_std_min = np.log10(np.nanmin(
                    self.flux_std[flux_std_mask]))
                flux_std_max = np.log10(np.nanmax(
                    self.flux_std[flux_std_mask]))

                fig_std = go.Figure(layout=dict(updatemenus=list([
                    dict(
                        active=0,
                        buttons=list([
                            dict(label='Log Scale',
                                 method='update',
                                 args=[{
                                     'visible': [True, True]
                                 }, {
                                     'title': 'Log scale',
                                     'yaxis': {
                                         'type': 'log'
                                     }
                                 }]),
                            dict(label='Linear Scale',
                                 method='update',
                                 args=[{
                                     'visible': [True, False]
                                 }, {
                                     'title': 'Linear scale',
                                     'yaxis': {
                                         'type': 'linear'
                                     }
                                 }])
                        ]),
                    )
                ]),
                                                title='Log scale'))
                # show the image on the top
                fig_std.add_trace(
                    go.Scatter(x=self.wave_std,
                               y=self.flux_std,
                               line=dict(color='royalblue'),
                               name='Flux'))
                fig_std.add_trace(
                    go.Scatter(x=self.wave_std,
                               y=self.fluxerr_std,
                               line=dict(color='orange'),
                               name='Flux Uncertainty'))
                fig_std.add_trace(
                    go.Scatter(x=self.wave_std,
                               y=self.skyflux_std,
                               line=dict(color='firebrick'),
                               name='Sky Flux'))
                fig_std.add_trace(
                    go.Scatter(x=self.wave_std_true,
                               y=self.fluxmag_std_true,
                               line=dict(color='black'),
                               name='Standard'))
                fig_std.update_layout(
                    autosize=True,
                    hovermode='closest',
                    showlegend=True,
                    xaxis=dict(title='Wavelength / A',
                               range=[wave_min, wave_max]),
                    yaxis=dict(title='Flux',
                               range=[flux_std_min, flux_std_max],
                               type='log'),
                    legend=go.layout.Legend(x=0,
                                            y=1,
                                            traceorder="normal",
                                            font=dict(family="sans-serif",
                                                      size=12,
                                                      color="black"),
                                            bgcolor='rgba(0,0,0,0)'),
                    height=800)

                if jsonstring:
                    return fig_std.to_json()
                if iframe:
                    pio.write_html(fig_std, 'spectrum_standard.html')
                if renderer == 'default':
                    fig_std.show()
                else:
                    fig_std.show(renderer)
        if stype not in ['science', 'standard', 'all']:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')
