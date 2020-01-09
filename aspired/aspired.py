import difflib
import os
import sys
from functools import partial
import warnings

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
        ftype: string, optional
            one of csv, tsv and ascii. Default is csv.
        Sxais: int, 0 or 1
            OVERRIDE the SAXIS value in the FITS header, or to provide the
            SAXIS if it does not exist
        saxis_keyword: string
            HDU keyword for the spectral axis direction
        combinetype_light: string, optional
            average of median for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_light: tuple
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
        combinetype_dark: string, optional
            average of median for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_dark: tuple
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
        combinetype_bias: string, optional
            average of median for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_bias: tuple
            perform sigma clipping if set to True. sigma is computed with the
            numpy.ma.std method
        clip_low_bias: float
            lower threshold of the sigma clipping
        clip_high_bias: float
            upper threshold of the sigma clipping
        combinetype_flat: string, optional
            average of median for CCDproc.Combiner.average_combine() and
            CCDproc.Combiner.median_combine(). All the frame types follow
            the same combinetype.
        sigma_clipping_flat: tuple
            perform sigma clipping if set to True. sigma is computed with the
            numpy.ma.std method
        clip_low_flat: float
            lower threshold of the sigma clipping
        clip_high_flat: float
            upper threshold of the sigma clipping
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
        self._check_files()

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
                    warnings.warn('Saxis keyword "' + self.saxis_keyword +
                                  '" is not in the header. Saxis is set to 1.')
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

    def _check_files(self):
        for filepath in self.impath:
            try:
                os.path.isfile(filepath)
            except:
                ValueError('File ' + filepath + ' does not exist.')

    def _bias_subtract(self,
                       combinetype='median',
                       sigma_clipping=True,
                       clip_low=5,
                       clip_high=5):
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
            self.biascombinetype = combinetype
            self.bias_master = bias_combiner.median_combine()
        elif self.combinetype_bias == 'average':
            self.biascombinetype = combinetype
            self.bias_master = bias_combiner.average_combine()
        else:
            self.bias_filename = []
            raise ValueError('ASPIRED: Unknown combinetype.')

        # Bias subtract
        self.light_master = self.light_master.subtract(self.bias_master)

        # Free memory
        del bias_CCDData
        del bias_combiner

    def _dark_subtract(self,
                       combinetype='median',
                       sigma_clipping=True,
                       clip_low=5,
                       clip_high=5):
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
            self.darkcombinetype = combinetype
            self.dark_master = dark_combiner.median_combine()
            self.exptime_dark = np.median(dark_time)
        elif self.combinetype_dark == 'average':
            self.darkcombinetype = combinetype
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
            self.flatcombinetype = combinetype
            self.flat_master = flat_combiner.median_combine()
        elif self.combinetype_flat == 'average':
            self.flatcombinetype = combinetype
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

        # rotate the frame by 90 degrees anti-clockwise if Saxis is 0
        if self.saxis is 0:
            self.light_master = np.rot(self.light_master)

        # Construct a FITS object of the reduced frame
        self.light_master = np.array((self.light_master))

    def savefits(self, filepath='reduced_image.fits', overwrite=False):
        '''
        Save the reduced image to disk.

        Parameters
        ----------
        filepath: String
            Disk location to be written to. Default is at where the Python
            process/subprocess is execuated.
        overwrite: tuple
            Default is False. 

        '''

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
            keyword='LCOMTYPE',
            value=self.combinetype_light,
            comment='Type of image combine of the light frames.')
        self.fits_data.header.set(
            keyword='LSIGCLIP',
            value=self.sigma_clipping_light,
            comment='True if the light frames are sigma clipped.')
        self.fits_data.header.set(
            keyword='LCLIPLOW',
            value=self.clip_low_light,
            comment='Lower threshold of sigma clipping of the light frames.')
        self.fits_data.header.set(
            keyword='LCLIPHIG',
            value=self.clip_high_light,
            comment='Higher threshold of sigma clipping of the light frames.')
        self.fits_data.header.set(
            keyword='LXPOSURE',
            value=self.exptime_light,
            comment='Average exposure time of the light frames.')
        self.fits_data.header.set(
            keyword='LKEYWORD',
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

        # Save file to disk
        self.fits_data.writeto(filepath, overwrite=overwrite)

    def inspect(self, log=True, renderer='default', jsonstring=False):
        '''
        Display the reduced image with a supported plotly renderer or export
        as json strings.

        Parameters
        ----------
        log: tuple
            Log the ADU count per second in the display. Default is True.
        renderer: string
            plotly renderer: jpg, png
        jsonstring: tuple
            set to True to return json string that can be rendered by Plot.ly
            in any support language

        Return
        ------
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
                 img,
                 Saxis=1,
                 spatial_mask=(1, ),
                 spec_mask=(1, ),
                 flip=False,
                 cr=True,
                 cr_sigma=5.,
                 rn=None,
                 gain=None,
                 seeing=None,
                 exptime=None,
                 silence=False):
        '''
        Currently, there is no automated way to decide if a flip is needed.

        The supplied file should contain 2 or 3 columns with the following
        structure:

            column 1: one of bias, dark, flat or light
            column 2: file location
            column 3: HDU number (default to 0 if not given)

        If the 2D spectrum is
        +--------+--------+-------+-------+
        |  blue  |   red  | Saxis |  flip |
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
        img: 2D numpy array (M x N)
            2D spectral image
        Saxis: int, optional
            Spectral direction, 0 for vertical, 1 for horizontal.
            (Default is 1)
        spatial_mask: 1D numpy array (N), optional
            Mask in the spatial direction, can be the indices of the pixels
            to be included (size <N) or a 1D numpy array of True/False (size N)
            (Default is (1,) i.e. keep everything)
        spec_mask: 1D numpy array (M), optional
            Mask in the spectral direction, can be the indices of the pixels
            to be included (size <M) or a 1D numpy array of True/False (size M)
            (Default is (1,) i.e. keep everything)
        flip: tuple, optional
            If the frame has to be left-right flipped, set to True.
            (Deafult is False)
        cr: tuple, optional
            Set to True to apply cosmic ray rejection by sigma clipping with
            astroscrappy if available, otherwise a 2D median filter of size 5
            would be used. (default is True)
        cr_sigma: float, optional
            Cosmic ray sigma clipping limit (Deafult is 5.0)
        rn: float, optional
            Readnoise of the detector, not important if noise estimation is
            not needed.
            (Deafult is None, which will be replaced with 1.0)
        gain: float, optional
            Gain of the detector, not important if noise estimation is
            not needed.
            (Deafult is None, which will be replaced with 1.0)
        seeing: float, optional
            Seeing in unit of arcsec, use as the first guess of the line
            spread function of the spectra.
            (Deafult is None, which will be replaced with 1.0)
        exptime: float, optional
            Esposure time for the observation, not important if absolute flux
            calibration is not needed.
            (Deafult is None, which will be replaced with 1.0)
        silence: tuple, optional
            Set to True to suppress all verbose output.
        '''

        self.Saxis = Saxis
        if self.Saxis is 1:
            self.Waxis = 0
        else:
            self.Waxis = 1
        self.spatial_mask = spatial_mask
        self.spec_mask = spec_mask
        self.flip = flip
        self.cr_sigma = cr_sigma
        if rn is None:
            self.rn = 1.
        else:
            self.rn = rn
        if gain is None:
            self.gain = 1.
        else:
            self.gain = gain
        if seeing is None:
            self.seeing = 1.
        else:
            self.seeing = seeing
        if exptime is None:
            self.exptime = 1.
        else:
            self.exptime = exptime

        
        
        self.silence = silence

        # cosmic ray rejection
        if cr:
            img = detect_cosmics(img,
                                 sigclip=self.cr_sigma,
                                 readnoise=self.rn,
                                 gain=self.gain,
                                 fsmode='convolve',
                                 psffwhm=self.seeing)[1]

        # the valid y-range of the chip (i.e. spatial direction)
        if (len(self.spatial_mask) > 1):
            if self.Saxis is 1:
                img = img[self.spatial_mask]
            else:
                img = img[:, self.spatial_mask]

        # the valid x-range of the chip (i.e. spectral direction)
        if (len(self.spec_mask) > 1):
            if self.Saxis is 1:
                img = img[:, self.spec_mask]
            else:
                img = img[self.spec_mask]

        # get the length in the spectral and spatial directions
        self.spec_size = np.shape(img)[self.Waxis]
        self.spatial_size = np.shape(img)[self.Saxis]
        if self.Saxis is 0:
            self.img = np.transpose(img)
        else:
            self.img = img

        if self.flip:
            self.img = np.flip(self.img)

        # set the 2D histogram z-limits
        self.zmin = np.nanpercentile(np.log10(self.img), 5)
        self.zmax = np.nanpercentile(np.log10(self.img), 95)

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

    def _identify_spectra(self, f_height, display, renderer, jsonstring):
        """
        Identify peaks assuming the spatial and spectral directions are
        aligned with the X and Y direction within a few degrees.

        Parameters
        ----------
        f_height: float
            The minimum intensity as a fraction of maximum height.
        display: tuple
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: tuple
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
        ztot = np.nanmedian(self.img, axis=self.Saxis)

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
            if self.Saxis == 1:
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
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

        self.peak = peaks_y
        self.peak_height = heights_y

    def _optimal_signal(self, pix, xslice, sky, mu, sigma, display, renderer,
                        jsonstring):
        """
        Iterate to get optimal signal, for internal use only

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
        display: tuple
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        jsonstring: tuple
            set to True to return json string that can be rendered by Plotly
            in any support language.

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

        if display:
            fit = self._gaus(pix, max(signal), 0., mu, sigma) + sky
            fig, ax = plt.subplots(ncols=1, figsize=(10, 10))
            ax.plot(pix, xslice)
            ax.plot(pix, fit)
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Count')
            #print(signal, variance)
            #print(np.sum(xslice-sky_const))

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
                 p_bg=5,
                 tol=3,
                 display=False,
                 renderer='default',
                 jsonstring=False):
        '''

                trace: 1-d numpy array (N)
            The spatial positions (Y axis) corresponding to the center of the
            trace for every wavelength (X axis), as returned from ap_trace
        trace_sigma: float, or 1-d array (1 or N)
            Tophat extraction: Float is accepted but will be rounded to an int,
                                which gives the constant aperture size on either
                                side of the trace to extract.
            Optimal extraction: Float or 1-d array of the same size as the trace.
                                If a float is supplied, a fixed standard deviation
                                will be used to construct the gaussian weight
                                function along the entire spectrum.

        Parameters
        ----------
        nspec: int, optional

        nwindow: int, optional

        spec_sep: int, optional

        resample_factor: int, optional

        rescale: tuple, optional

        scaling_min: float, optional

        scaling_max: float, optional

        scaling_step: float, optional

        p_bg: float, optional

        tol: float, optional

        display: tuple, optional
            Set to True to display disgnostic plot.
        renderer: string, optional
            plotly renderer options.
        jsonstring: tuple, optional
            set to True to return json string that can be rendered by Plotly
            in any support language.
        '''

        # Get the shape of the 2D spectrum and define upsampling ratio
        nwave = len(self.img[0])
        nspatial = len(self.img)

        nresample = nspatial * resample_factor

        # window size
        w_size = nwave // nwindow
        img_split = np.array_split(self.img, nwindow, axis=1)

        lines_ref_init = np.nanmedian(img_split[0], axis=1)
        lines_ref_init_resampled = signal.resample(lines_ref_init, nresample)

        # linear scaling limits
        if rescale:
            scaling_range = np.arange(scaling_min, scaling_max, scaling_step)
        else:
            scaling_range = np.ones(1)

        # estimate the 5-th percentile as the sky background level
        lines_ref = lines_ref_init_resampled - np.percentile(
            lines_ref_init_resampled, p_bg)

        shift_solution = np.zeros(nwindow)
        scale_solution = np.ones(nwindow)

        # maximum shift (SEMI-AMPLITUDE) from the neighbour (pixel)
        tol_len = int(tol * resample_factor)

        # Scipy correlate method
        for i in range(nwindow):

            # smooth by taking the median
            lines = np.nanmedian(img_split[i], axis=1)
            lines = signal.resample(lines, nresample)
            lines = lines - np.percentile(lines, p_bg)

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

            # Update (increment) the reference line
            lines_ref = lines

        nscaled = (nresample * scale_solution).astype('int')

        # Find the spectral position in the middle of the gram in the upsampled pixel location location
        peaks = signal.find_peaks(signal.resample(
            np.nanmedian(img_split[nwindow // 2], axis=1), nresample),
                                  distance=spec_sep,
                                  prominence=1)

        # update the number of spectra if the number of peaks detected is less
        # than the number requested
        self.nspec = min(len(peaks), nspec)

        # Sort the positions by the prominences, and return to the original scale (i.e. with subpixel position)
        spec_init = np.sort(peaks[0][np.argsort(-peaks[1]['prominences'])]
                            [:self.nspec]) / resample_factor

        # Create array to populate the spectral locations
        spec = np.zeros((len(spec_init), len(img_split)))
        #spec_val = np.zeros((len(spec_init), len(img_split)))

        # Populate the initial values
        spec[:, nwindow // 2] = spec_init

        # Pixel positions of the mid point of each data_split
        spec_pix = np.arange(len(img_split)) * w_size + w_size / 2.

        # Looping through pixels larger than middle pixel
        for i in range(nwindow // 2 + 1, len(img_split)):
            spec[:, i] = (spec[:, i - 1] * resample_factor * nscaled[i] /
                          nresample - shift_solution[i]) / resample_factor

        # Looping through pixels smaller than middle pixel
        for i in range(nwindow // 2 - 1, -1, -1):
            spec[:, i] = (spec[:, i + 1] * resample_factor +
                          shift_solution[i + 1]) / (
                              int(nresample * scale_solution[i + 1]) /
                              nresample) / resample_factor
            #spec_val[:,i] = signal.resample(np.nanmedian(img_split[i], axis=0), int(nresample*scale_solution[i]))[(spec[:,i] * scale_solution[i]).astype('int')]

        ap = np.zeros((len(spec), nwave))
        ap_sigma = np.zeros((len(spec), nwave))

        for i in range(len(spec)):
            # fit the trace
            ap_p = np.polyfit(spec_pix, spec[i], max(1, nwindow // 10))
            ap[i] = np.polyval(ap_p, np.arange(nwave))

            # stacking up the slices to get a good guess of the LSF
            for j, ap_idx in enumerate(ap[i][spec_pix.astype('int')]):
                ap_slice = np.nanmedian(img_split[j], axis=1)
                # Add 0.5 to allow proper rounding
                start_idx = int(ap_idx - 20 + 0.5)
                end_idx = start_idx + 20 + 20 + 1
                if j == 0:
                    ap_spatial = ap_slice[start_idx:end_idx]
                else:
                    ap_spatial += ap_slice[start_idx:end_idx]

            # compute ONE sigma for each trace
            pguess = [
                np.nanmax(ap_spatial),
                np.nanpercentile(ap_spatial, 10), ap_idx, 3.
            ]

            popt, pcov = curve_fit(self._gaus,
                                   range(start_idx, end_idx),
                                   ap_spatial,
                                   p0=pguess)
            ap_sigma[i] = popt[3]

        if self.nspec is 1:
            self.trace = ap
            self.trace_sigma = ap_sigma
        else:
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
            for i in range(len(spec)):
                fig.add_trace(
                    go.Scatter(x=np.arange(nwave),
                               y=ap[i],
                               line=dict(color='black')))
                fig.add_trace(
                    go.Scatter(x=spec_pix,
                               y=spec[i],
                               mode='markers',
                               marker=dict(color='grey')))
            fig.add_trace(
                go.Scatter(x=np.ones(len(spec)) * spec_pix[nwindow // 2],
                           y=spec[:, nwindow // 2],
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
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

    def ap_trace_quick(self,
                       nspec=1,
                       nsteps=20,
                       recenter=False,
                       prevtrace=(0, ),
                       fittype='spline',
                       order=3,
                       bigbox=8,
                       display=False,
                       renderer='default',
                       jsonstring=False):
        """
        Trace the spectrum aperture in an image. It only works for bright
        spectra with good wavelength coverage.

        It works by chopping image up in bins along the wavelength direction,
        fits a Gaussian for each bin to determine the spatial center of the
        trace. Finally, draws a cubic spline through the bins.

        Parameters
        ----------
        nspec: int, optional
            Number of spectra to be extracted. It does not guarantee returning
            the same number of spectra if fewer can be detected. (Default is 1)
        nsteps: int, optional
            Keyword, number of bins in X direction to chop image into. Use
            fewer bins if ap_trace is having difficulty, such as with faint
            targets (default is 20, minimum is 4)
        recenter: bool, optional
            Set to True to use previous trace, allow small shift in position
            along the spatial direction. Not doing anything if prevtrace is not
            supplied. (Default is False)
        prevtrace: 1-d numpy array, optional
            Provide first guess or refitting the center with different parameters.
        fittype: string, optional
            Set to 'spline' or 'polynomial', using
            scipy.interpolate.UnivariateSpline and numpy.polyfit
        order: string, optional
            Degree of the spline or polynomial. Spline must be <= 5.
            (default is k=3)
        bigbox: float, optional
            The number of sigma away from the main aperture to allow to trace
        silence: tuple, optional
            Set to disable warning/error messages. (Default is False)
        display: tuple, optional
            Set to True to display disgnostic plot.
        renderer: string, optional
            plotly renderer options.
        jsonstring: tuple, optional
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        my: array (N, nspec)
            The spatial (Y) positions of the trace, interpolated over the
            entire wavelength (X) axis
        y_sigma: array (N, nspec)
            The sigma measured at the nsteps.

        """

        self.nspec = nspec

        if not self.silence:
            print('Tracing Aperture using nsteps=' + str(nsteps))

        # the valid y-range of the chip (an array of int)
        ydata = np.arange(self.spec_size)
        ztot = np.sum(self.img, axis=1)

        # need at least 3 samples along the trace
        if (nsteps < 3):
            nsteps = 3

        # detect peaks by summing in the spatial direction
        self._identify_spectra(0.01,
                               1,
                               display=False,
                               renderer=renderer,
                               jsonstring=jsonstring)

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
                self.trace_sigma = y_sigma
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
                xbins = np.linspace(0, self.spatial_size, nsteps)
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
                                       str(nsteps) + ' of spectrum ' +
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
                                'Step ' + str(j + 1) + ' of ' + str(nsteps) +
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
            self.trace_sigma = y_sigma

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
                if renderer == 'default':
                    fig.show()
                else:
                    fig.show(renderer)

    def ap_extract(self,
                   apwidth=7,
                   skysep=3,
                   skywidth=5,
                   skydeg=1,
                   optimal=True,
                   display=False,
                   renderer='default',
                   jsonstring=False):
        """
        Extract the spectra using the traces, support aperture or optimal
        extraction. The sky background is fitted in one dimention only. The
        uncertainty at each pixel is also computed, but it is only meaningful
        if correct gain, read noise and are provided.

        Parameters
        ----------
        apwidth: int, optional
        skysep: int, optional
            The separation in pixels from the aperture to the sky window.
            (Default is 3)
        skywidth: int, optional
            The width in pixels of the sky windows on either side of the
            aperture. (Default is 7)
        skydeg: int, optional
            The polynomial order to fit between the sky windows.
            (Default is 0, i.e. constant flat sky level)
        optimal: tuple, optional
            Set optimal extraction. (Default is True)
        silence: tuple, optional
            Set to disable warning/error messages. (Default is False)
        display: tuple, optional
            Set to True to display disgnostic plot.
        renderer: string, optional
            plotly renderer options.
        jsonstring: tuple, optional
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        onedspec: 1-d array
            The summed adu at each column about the trace. Note: is not
            sky subtracted!
        skyadu: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        aduerr: 1-d array
            the uncertainties of the adu values
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
                        pix,
                        xslice,
                        sky,
                        self.trace[j][i],
                        self.trace_sigma[j][i],
                        display=False,
                        renderer=renderer,
                        jsonstring=jsonstring)
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
                            min(median_trace + widthup + skysep + skywidth,
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
                if (itrace - widthdn >= 0):
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
                if (itrace + widthup <= self.spatial_size):
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
                if renderer == 'default':
                    fig.show()
                else:
                    fig.show(renderer)

        self.adu = adu
        self.aduerr = aduerr
        self.skyadu = skyadu


class WavelengthPolyFit:
    def __init__(self, spec, arc):
        '''
        arc: TwoDSpec object of the arc image
        spec: TwoDSpec object of the science/standard image

        '''

        self.spec = spec
        self.arc = arc

        # the valid y-range of the chip (i.e. spatial direction)
        if (len(self.spec.spatial_mask) > 1):
            if self.spec.Saxis is 1:
                self.arc = self.arc[self.spec.spatial_mask]
            else:
                self.arc = self.arc[:, self.spec.spatial_mask]

        # the valid x-range of the chip (i.e. spectral direction)
        if (len(self.spec.spec_mask) > 1):
            if self.spec.Saxis is 1:
                self.arc = self.arc[:, self.spec.spec_mask]
            else:
                self.arc = self.arc[self.spec.spec_mask]

        # get the length in the spectral and spatial directions
        if self.spec.Saxis is 0:
            self.arc = np.transpose(self.arc)

        if self.spec.flip:
            self.arc = np.flip(self.arc)

    def find_arc_lines(self,
                       percentile=20.,
                       distance=5.,
                       display=False,
                       jsonstring=False,
                       renderer='default'):
        '''
        pixelscale in unit of A/pix

        '''

        p = np.percentile(self.arc, percentile)
        trace = int(np.mean(self.spec.trace))
        width = int(np.mean(self.spec.trace_sigma[0]) * 3)

        self.arc_trace = self.arc[max(0, trace - width -
                                      1):min(trace +
                                             width, len(self.spec.img[0])), :]
        self.spectrum = np.median(self.arc_trace, axis=0)
        peaks, _ = signal.find_peaks(self.spectrum,
                                     distance=distance,
                                     prominence=p)

        self.peaks = refine_peaks(self.spectrum, peaks, window_width=3)

        if display & plotly_imported:
            fig = go.Figure()

            # show the image on the top
            fig.add_trace(
                go.Heatmap(x=np.arange(self.arc.shape[0]),
                           y=np.arange(self.arc.shape[1]),
                           z=np.log10(self.arc),
                           colorscale="Viridis",
                           colorbar=dict(title='log(ADU)')))

            for i in self.peaks:
                fig.add_trace(
                    go.Scatter(x=[i, i],
                               y=[0, self.arc.shape[0]],
                               mode='lines',
                               line=dict(color='firebrick', width=1)))

            fig.update_layout(autosize=True,
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
                return fig.to_json()
            if renderer == 'default':
                fig.show()
            else:
                fig.show(renderer)

    def calibrate(self,
                  elements,
                  sample_size=5,
                  min_wave=3500.,
                  max_wave=8500.,
                  max_tries=5000,
                  display=False,
                  num_slopes=1000,
                  range_tolerance=500,
                  fit_tolerance=20.,
                  polydeg=5,
                  top_n=20,
                  candidate_thresh=15.,
                  ransac_thresh=1,
                  xbins=50,
                  ybins=50,
                  brute_force=False,
                  fittype='poly',
                  mode='manual',
                  progress=True,
                  coeff=None):
        '''
        thresh (A) :: the individual line fitting tolerance to accept as a valid fitting point
        fit_tolerance (A) :: the RMS
        '''

        c = Calibrator(self.peaks,
                       min_wavelength=min_wave,
                       max_wavelength=max_wave,
                       num_pixels=len(self.spectrum))

        c.add_atlas(elements)

        c.set_fit_constraints(num_slopes=num_slopes,
                              range_tolerance=range_tolerance,
                              fit_tolerance=fit_tolerance,
                              polydeg=polydeg,
                              candidate_thresh=candidate_thresh,
                              ransac_thresh=ransac_thresh,
                              xbins=xbins,
                              ybins=ybins,
                              brute_force=brute_force,
                              fittype=fittype)

        p = c.fit(sample_size=sample_size,
                  max_tries=max_tries,
                  top_n=top_n,
                  n_slope=num_slopes,
                  mode=mode,
                  progress=progress,
                  coeff=coeff)

        pfit, _, _ = c.match_peaks_to_atlas(p, tolerance=1)
        pfit, _, _ = c.match_peaks_to_atlas(pfit, tolerance=0.5)

        self.pfit = pfit
        self.pfit_type = 'poly'

        if display:
            c.plot_fit(np.median(self.arc_trace, axis=0),
                       self.pfit,
                       plot_atlas=True,
                       log_spectrum=False,
                       tolerance=0.5)

    def calibrate_pfit(self,
                       elements,
                       pfit,
                       min_wave=3500.,
                       max_wave=8500.,
                       tolerance=10.,
                       display=False):

        c = Calibrator(self.peaks,
                       min_wavelength=min_wave,
                       max_wavelength=max_wave,
                       num_pixels=len(self.spectrum))
        c.add_atlas(elements=elements,
                    min_wavelength=min_wave,
                    max_wavelength=max_wave)
        pfit, _, _ = c.match_peaks_to_atlas(pfit, tolerance=tolerance)
        pfit, _, _ = c.match_peaks_to_atlas(pfit, tolerance=1)
        pfit, _, _ = c.match_peaks_to_atlas(pfit, tolerance=0.5)
        self.pfit = pfit
        self.pfit_type = 'poly'

        if display:
            c.plot_fit(np.median(self.arc_trace, axis=0),
                       self.pfit,
                       plot_atlas=True,
                       log_spectrum=False,
                       tolerance=0.5)


class StandardFlux:
    def __init__(self, target, group, cutoff=0.4, ftype='flux', silence=False):
        self.target = target
        self.group = group
        self.cutoff = cutoff
        self.ftype = ftype
        self.silence = silence
        self._lookup_standard()

    def _lookup_standard(self):
        '''
        Check if the requested standard and library exist.

        '''

        try:
            target_list = eval(self.group)
        except:
            raise ValueError('Requested standard star library does not exist.')

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
                      jsonstring=False):
        '''
        Read the standard flux/magnitude file. And return the wavelength and
        flux/mag in units of

        wavelength: A
        flux:       ergs / cm / cm / s / A
        mag:        mag (AB) 

        Returns
        -------
        display: tuple, optional
            Set to True to display disgnostic plot.
        renderer: string, optional
            plotly renderer options.
        jsonstring: tuple, optional
            set to True to return json string that can be rendered by Plotly
            in any support language.
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
            self.inspect_standard(renderer, jsonstring)

    def inspect_standard(self, renderer='default', jsonstring=False):
        fig = go.Figure()

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
        Parameters
        ----------
        science: TwoDSpec object

        wave_cal: WavelengthPolyFit object

        standard: TwoDSpec object, optional

        wave_cal_std: WavelengthPolyFit object, optional

        flux_cal: StandardFlux object, optional (require wave_cal_std)

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
        Extract the required information from a TwoDSpec object
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
        Extract the required information from a WavelengthPolyFit object
        '''

        if stype == 'science':
            try:
                self.pfit_type = wave_cal.pfit_type
                self.pfit = wave_cal.pfit
                if self.pfit_type == 'poly':
                    self.polyval = np.polyval
                elif self.pfit_type == 'legendre':
                    self.polyval = np.polynomial.legendre.legval
                elif self.pfit_type == 'chebyshev':
                    self.polyval = np.polynomial.chebyshev.chebval
                else:
                    raise ValueError(
                        'fittype must be: (1) poly; (2) legendre; or '
                        '(3) chebyshev')
            except:
                raise TypeError('Please provide a valid WavelengthPolyFit.')
        elif stype == 'standard':
            try:
                self.pfit_type_std = wave_cal.pfit_type
                self.pfit_std = wave_cal.pfit
                if self.pfit_type_std == 'poly':
                    self.polyval_std = np.polyval
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
        elif stype == 'all':
            try:
                self.pfit_type = wave_cal.pfit_type
                self.pfit_type_std = wave_cal.pfit_type
                self.pfit = wave_cal.pfit
                self.pfit_std = wave_cal.pfit
                if self.pfit_type == 'poly':
                    self.polyval = np.polyval
                    self.polyval_std = np.polyval
                elif self.pfit_type == 'legendre':
                    self.polyval = np.polynomial.legendre.legval
                    self.polyval_std = np.polynomial.legendre.legval
                elif self.pfit_type == 'chebyshev':
                    self.polyval = np.polynomial.chebyshev.chebval
                    self.polyval_std = np.polynomial.chebyshev.chebval
                else:
                    raise ValueError(
                        'fittype must be: (1) poly; (2) legendre; or '
                        '(3) chebyshev')
            except:
                raise TypeError('Please provide a valid WavelengthPolyFit.')
        else:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')

    def _set_fluxcal(self, flux_cal):
        '''
        Extract the required information from a StandardFlux object.
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
        Apply the wavelength calibration
        '''

        if stype == 'science':
            pix = np.arange(len(self.adu[0]))
            self.wave = self.polyval(self.pfit, pix)
        elif stype == 'standard':
            if self.standard_imported:
                pix_std = np.arange(len(self.adu_std))
                self.wave_std = self.polyval(self.pfit_std, pix_std)
            else:
                raise AttributeError(
                    'The TwoDSpec of the standard '
                    'observation is not available. Flux calibration will not '
                    'be performed.')
        elif stype == 'all':
            pix = np.arange(len(self.adu[0]))
            pix_std = np.arange(len(self.adu_std))
            self.wave = self.polyval(self.pfit, pix)
            self.wave_std = self.polyval(self.pfit_std, pix_std)
        else:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')

    def compute_sencurve(self,
                         kind=3,
                         smooth=False,
                         slength=5,
                         sorder=3,
                         display=False,
                         renderer='default',
                         jsonstring=False):
        '''
        Get the standard flux or magnitude of the given target and group
        based on the given array of wavelengths. Some standard libraries
        contain the same target with slightly different values.

        Parameters
        ----------
        kind: string or integer [1,2,3,4,5 only]
            interpolation kind
            >>> [‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
                 ‘previous’, ‘next’]
            (default is 'cubic')
        smooth: tuple
            set to smooth the input ADU/flux/mag with scipy.signal.savgol_filter
            (default is True)
        slength: int
            SG-filter window size
        sorder: int
            SG-filter polynomial order
        display: tuple, optional
            Set to True to display disgnostic plot.
        renderer: string, optional
            plotly renderer options.
        jsonstring: tuple, optional
            set to True to return json string that can be rendered by Plotly
            in any support language.

        Returns
        -------
        A scipy interp1d object.

        '''

        # Get the standard flux/magnitude
        self.slength = slength
        self.sorder = sorder
        self.smooth = smooth

        # Compute bin sizes such that the bin is roughly 10 A wide
        #wave_range = self.wave_std[-1] - self.wave_std[0]
        #bin_size = self.wave_std[1] - self.wave_std[0]

        # Find the centres of the first and last bins such that
        # the old and new spectra covers identical wavelength range
        #wave_lhs = self.wave_std[0] - bin_size / 2.
        #wave_rhs = self.wave_std[-1] + bin_size / 2.
        #wave_obs = np.arange(wave_lhs, wave_rhs, bin_size)
        #wave_std = self.wave_std

        if spectres_imported:
            # resampling both the observed and the database standard spectra
            # in unit of flux per second
            flux_std = spectres(self.wave_std_true, self.wave_std,
                                self.adu_std / self.exptime_std)
            flux_std_true = self.fluxmag_std_true
        else:
            flux_std = flux_std / self.exptime_std
            flux_std_true = itp.interp1d(self.wave_std_true,
                                         self.fluxmag_std_true)(self.wave_obs)
        # Get the sensitivity curve
        sensitivity = flux_std_true / flux_std
        mask = (np.isfinite(sensitivity) & (sensitivity > 0.) &
                ((self.wave_std_true < 6850.) | (self.wave_std_true > 7000.)) &
                ((self.wave_std_true < 7150.) | (self.wave_std_true > 7400.)) &
                ((self.wave_std_true < 7575.) | (self.wave_std_true > 7775.)))

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
            self.inspect_sencurve()

    def inspect_sencurve(self, renderer='default', jsonstring=False):
        '''
        Display the computed sensitivity curve.
        '''

        fig = go.Figure()
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
        if renderer == 'default':
            fig.show()
        else:
            fig.show(renderer)

    def apply_flux_calibration(self, stype='all'):
        '''
        Apply the computed sensitivity curve
        '''

        if stype == 'science':
            self.flux = 10.**self.sencurve(self.wave) * self.adu
            self.fluxerr = 10.**self.sencurve(self.wave) * self.aduerr
            self.skyflux = 10.**self.sencurve(self.wave) * self.skyadu
        elif stype == 'standard':
            self.flux_std = 10.**self.sencurve(self.wave_std) * self.adu_std
            self.fluxerr_std = 10.**self.sencurve(
                self.wave_std) * self.aduerr_std
            self.skyflux_std = 10.**self.sencurve(
                self.wave_std) * self.skyadu_std
        elif stype == 'all':
            self.flux = 10.**self.sencurve(self.wave) * self.adu
            self.fluxerr = 10.**self.sencurve(self.wave) * self.aduerr
            self.skyflux = 10.**self.sencurve(self.wave) * self.skyadu
            self.flux_std = 10.**self.sencurve(self.wave_std) * self.adu_std
            self.fluxerr_std = 10.**self.sencurve(
                self.wave_std) * self.aduerr_std
            self.skyflux_std = 10.**self.sencurve(
                self.wave_std) * self.skyadu_std
        else:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')

    def inspect_reduced_spectrum(self,
                                 stype='all',
                                 wave_min=4000.,
                                 wave_max=8000.,
                                 renderer='default',
                                 jsonstring=False):
        '''
        Display the reduced spectra.
        '''

        if stype == 'science':
            for j in range(self.nspec):

                wave_mask = ((self.wave > wave_min) & (self.wave < wave_max))
                flux_mask = (
                    (self.flux[j] >
                     np.nanpercentile(self.flux[j][wave_mask], 5) / 1.5) &
                    (self.flux[j] <
                     np.nanpercentile(self.flux[j][wave_mask], 95) * 1.5))
                flux_min = np.log10(np.nanmin(self.flux[j][flux_mask]))
                flux_max = np.log10(np.nanmax(self.flux[j][flux_mask]))

                fig = go.Figure()
                # show the image on the top
                fig.add_trace(
                    go.Scatter(x=self.wave,
                               y=self.flux[j],
                               line=dict(color='royalblue'),
                               name='Flux'))
                fig.add_trace(
                    go.Scatter(x=self.wave,
                               y=self.fluxerr[j],
                               line=dict(color='firebrick'),
                               name='Flux Uncertainty'))
                fig.add_trace(
                    go.Scatter(x=self.wave,
                               y=self.skyflux[j],
                               line=dict(color='orange'),
                               name='Sky Flux'))
                fig.update_layout(autosize=True,
                                  hovermode='closest',
                                  showlegend=True,
                                  xaxis=dict(title='Wavelength / A',
                                             range=[wave_min, wave_max]),
                                  yaxis=dict(title='Flux',
                                             range=[flux_min, flux_max],
                                             type='log'),
                                  legend=go.layout.Legend(
                                      x=0,
                                      y=1,
                                      traceorder="normal",
                                      font=dict(family="sans-serif",
                                                size=12,
                                                color="black"),
                                      bgcolor='rgba(0,0,0,0)'),
                                  height=800)

                if jsonstring:
                    return fig.to_json()
                if renderer == 'default':
                    fig.show()
                else:
                    fig.show(renderer)

        elif stype == 'standard':

            wave_std_mask = ((self.wave_std > wave_min) &
                             (self.wave_std < wave_max))
            flux_std_mask = (
                (self.flux_std >
                 np.nanpercentile(self.flux_std[wave_std_mask], 5) / 1.5) &
                (self.flux_std <
                 np.nanpercentile(self.flux_std[wave_std_mask], 95) * 1.5))
            flux_std_min = np.log10(np.nanmin(self.flux_std[flux_std_mask]))
            flux_std_max = np.log10(np.nanmax(self.flux_std[flux_std_mask]))

            fig = go.Figure()
            # show the image on the top
            fig.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.flux_std,
                           line=dict(color='royalblue'),
                           name='Flux'))
            fig.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.fluxerr_std,
                           line=dict(color='orange'),
                           name='Flux Uncertainty'))
            fig.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.skyflux_std,
                           line=dict(color='firebrick'),
                           name='Sky Flux'))
            fig.add_trace(
                go.Scatter(x=self.wave_std_true,
                           y=self.fluxmag_std_true,
                           line=dict(color='black'),
                           name='Standard'))
            fig.update_layout(autosize=True,
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
                                                      font=dict(
                                                          family="sans-serif",
                                                          size=12,
                                                          color="black"),
                                                      bgcolor='rgba(0,0,0,0)'),
                              height=800)

            fig.show(renderer)

        elif stype == 'all':

            for j in range(self.nspec):

                wave_mask = ((self.wave > wave_min) & (self.wave < wave_max))
                flux_mask = (
                    (self.flux[j] >
                     np.nanpercentile(self.flux[j][wave_mask], 5) / 1.5) &
                    (self.flux[j] <
                     np.nanpercentile(self.flux[j][wave_mask], 95) * 1.5))
                flux_min = np.log10(np.nanmin(self.flux[j][flux_mask]))
                flux_max = np.log10(np.nanmax(self.flux[j][flux_mask]))

                fig = go.Figure()
                # show the image on the top
                fig.add_trace(
                    go.Scatter(x=self.wave,
                               y=self.flux[j],
                               line=dict(color='royalblue'),
                               name='Flux'))
                fig.add_trace(
                    go.Scatter(x=self.wave,
                               y=self.fluxerr[j],
                               line=dict(color='orange'),
                               name='Flux Uncertainty'))
                fig.add_trace(
                    go.Scatter(x=self.wave,
                               y=self.skyflux[j],
                               line=dict(color='firebrick'),
                               name='Sky Flux'))
                fig.update_layout(autosize=True,
                                  hovermode='closest',
                                  showlegend=True,
                                  xaxis=dict(title='Wavelength / A',
                                             range=[wave_min, wave_max]),
                                  yaxis=dict(title='Flux',
                                             range=[flux_min, flux_max],
                                             type='log'),
                                  legend=go.layout.Legend(
                                      x=0,
                                      y=1,
                                      traceorder="normal",
                                      font=dict(family="sans-serif",
                                                size=12,
                                                color="black"),
                                      bgcolor='rgba(0,0,0,0)'),
                                  height=800)

                if not jsonstring:
                    if renderer == 'default':
                        fig.show()
                    else:
                        fig.show(renderer)

            wave_std_mask = ((self.wave_std > wave_min) &
                             (self.wave_std < wave_max))
            flux_std_mask = (
                (self.flux_std >
                 np.nanpercentile(self.flux_std[wave_std_mask], 5) / 1.5) &
                (self.flux_std <
                 np.nanpercentile(self.flux_std[wave_std_mask], 95) * 1.5))
            flux_std_min = np.log10(np.nanmin(self.flux_std[flux_std_mask]))
            flux_std_max = np.log10(np.nanmax(self.flux_std[flux_std_mask]))

            fig2 = go.Figure()
            # show the image on the top
            fig2.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.flux_std,
                           line=dict(color='royalblue'),
                           name='Flux'))
            fig2.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.fluxerr_std,
                           line=dict(color='orange'),
                           name='Flux Uncertainty'))
            fig2.add_trace(
                go.Scatter(x=self.wave_std,
                           y=self.skyflux_std,
                           line=dict(color='firebrick'),
                           name='Sky Flux'))
            fig2.add_trace(
                go.Scatter(x=self.wave_std_true,
                           y=self.fluxmag_std_true,
                           line=dict(color='black'),
                           name='Standard'))
            fig2.update_layout(autosize=True,
                               hovermode='closest',
                               showlegend=True,
                               xaxis=dict(title='Wavelength / A',
                                          range=[wave_min, wave_max]),
                               yaxis=dict(title='Flux',
                                          range=[flux_std_min, flux_std_max],
                                          type='log'),
                               legend=go.layout.Legend(
                                   x=0,
                                   y=1,
                                   traceorder="normal",
                                   font=dict(family="sans-serif",
                                             size=12,
                                             color="black"),
                                   bgcolor='rgba(0,0,0,0)'),
                               height=800)

            if jsonstring:
                return fig.to_json(), fig2.to_json()
            if renderer == 'default':
                fig2.show()
            else:
                fig2.show(renderer)
        else:
            raise ValueError('Unknown stype, please choose from (1) science; '
                             '(2) standard; or (3) all.')
