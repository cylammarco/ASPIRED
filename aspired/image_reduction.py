import os
import warnings

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import sigma_clip
from ccdproc import Combiner
from plotly import graph_objects as go
from plotly import io as pio


class ImageReduction:
    def __init__(self,
                 filelist,
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
        filelist: string
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
            Set to True to suppress all verbose warnings.
        '''

        if os.path.isabs(filelist):
            self.filelist = filelist
        else:
            self.filelist = os.path.abspath(filelist)

        self.filelist_abspath = self.filelist.rsplit('/', 1)[0]

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

        self.image_fits = None

        self.bias_filename = []
        self.dark_filename = []
        self.flat_filename = []
        self.arc_filename = []
        self.light_filename = []

        # import file with first column as image type and second column as
        # file path

        if isinstance(self.filelist, str):
            self.filelist = np.genfromtxt(self.filelist,
                                          delimiter=self.delimiter,
                                          dtype=str,
                                          autostrip=True).astype('U')

            if np.shape(np.shape(self.filelist))[0] == 2:
                self.imtype = self.filelist[:, 0].astype('object')
                self.impath = self.filelist[:, 1].astype('object')
            elif np.shape(np.shape(self.filelist))[0] == 1:
                self.imtype = self.filelist[0].astype('object')
                self.impath = self.filelist[1].astype('object')
            else:
                raise TypeError(
                    'Please provide a text file with at least 2 columns.')

        elif isinstance(self.filelist, np.ndarray):
            if np.shape(np.shape(self.filelist))[0] == 2:
                self.imtype = self.filelist[:, 0]
                self.impath = self.filelist[:, 1]
            elif np.shape(np.shape(self.filelist))[0] == 1:
                self.imtype = self.filelist[0]
                self.impath = self.filelist[1]
            else:
                raise TypeError(
                    'Please provide a numpy.ndarray with at least 2 columns.')
        else:
            raise TypeError('Please provide a file path to the file list or '
                            'a numpy array with at least 2 columns.')

        for i, im in enumerate(self.impath):
            if not os.path.isabs(im):
                self.impath[i] = os.path.join(self.filelist_abspath, im)
            print(im)
            print(self.impath[i])

        self.imtype = self.imtype.astype('str')

        if np.shape(np.shape(self.filelist))[0] == 2:
            try:
                self.hdunum = self.filelist[:, 2].astype('int')
            except:
                self.hdunum = np.zeros(len(self.impath)).astype('int')
        elif np.shape(np.shape(self.filelist))[0] == 1:
            try:
                self.hdunum = self.filelist[2].astype('int')
            except:
                self.hdunum = 0
        else:
            raise TypeError('Please provide a file path to the file list or '
                            'a numpy array with at least 2 columns.')

        if np.shape(np.shape(self.filelist))[0] == 2:
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

        if np.shape(np.shape(self.filelist))[0] == 1:
            if self.imtype == 'light':
                self.light_list = np.array([self.impath])
                self.bias_list = np.array([])
                self.dark_list = np.array([])
                self.flat_list = np.array([])
                self.arc_list = np.array([])

                self.light_hdunum = np.array([self.hdunum])
                self.bias_hdunum = np.array([])
                self.dark_hdunum = np.array([])
                self.flat_hdunum = np.array([])
                self.arc_hdunum = np.array([])
            else:
                ValueError('You are only providing a single file, it has to '
                           'be a light frame.')

        # If there is no science frames, nothing to process.
        assert (self.light_list.size > 0), 'There is no light frame.'

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
        self.light_header = []

        for i in range(self.light_list.size):
            # Open all the light frames
            light = fits.open(self.light_list[i])[self.light_hdunum[i]]
            light_CCDData.append(CCDData(light.data, unit=u.adu))
            self.light_header.append(light.header)
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

        if len(self.arc_list) > 0:
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

        # Dark subtraction adjusted for exposure time
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
        if self.saxis == 0:
            self.light_master = np.rot90(self.light_master)

        # Construct a FITS object of the reduced frame
        self.light_master = np.array((self.light_master))

    def _create_image_fits(self):
        # Put the reduced data in FITS format with an image header
        # Append header info to the *first* light frame header
        self.image_fits = fits.ImageHDU(self.light_master)

        self.image_fits.header = self.light_header[0]

        # Add the names of all the light frames to header
        if len(self.light_filename) > 0:
            for i in range(len(self.light_filename)):
                self.image_fits.header.set(keyword='light' + str(i + 1),
                                           value=self.light_filename[i],
                                           comment='Light frames')

        # Add the names of all the biad frames to header
        if len(self.bias_filename) > 0:
            for i in range(len(self.bias_filename)):
                self.image_fits.header.set(keyword='bias' + str(i + 1),
                                           value=self.bias_filename[i],
                                           comment='Bias frames')

        # Add the names of all the dark frames to header
        if len(self.dark_filename) > 0:
            for i in range(len(self.dark_filename)):
                self.image_fits.header.set(keyword='dark' + str(i + 1),
                                           value=self.dark_filename[i],
                                           comment='Dark frames')

        # Add the names of all the flat frames to header
        if len(self.flat_filename) > 0:
            for i in range(len(self.flat_filename)):
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

    def save_fits(self,
                  filename='reduced_image',
                  extension='fits',
                  overwrite=False):
        '''
        Save the reduced image to disk.

        Parameters
        ----------
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        extension: String
            File extension without the dot.
        overwrite: boolean
            Default is False.

        '''

        if filename[-5:] == '.fits':
            filename = filename[:-5]
        if filename[-4:] == '.fit':
            filename = filename[:-4]

        self._create_image_fits()
        self.image_fits = fits.PrimaryHDU(self.image_fits)
        # Save file to disk
        self.image_fits.writeto(filename + '.' + extension,
                                overwrite=overwrite)

    def inspect(self,
                log=True,
                display=True,
                renderer='default',
                width=1280,
                height=720,
                return_jsonstring=False,
                save_iframe=False,
                filename=None,
                open_iframe=False):
        '''
        Display the reduced image with a supported plotly renderer or export
        as json strings.

        Parameters
        ----------
        log: boolean
            Log the ADU count per second in the display. Default is True.
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
        JSON strings if return_jsonstring is set to True.

        '''

        if log:

            fig = go.Figure(data=go.Heatmap(z=np.log10(self.light_master),
                                            colorscale="Viridis"))
        else:

            fig = go.Figure(
                data=go.Heatmap(z=self.light_master, colorscale="Viridis"))

        fig.update_layout(yaxis_title='Spatial Direction / pixel',
                          xaxis=dict(zeroline=False,
                                     showgrid=False,
                                     title='Spectral Direction / pixel'),
                          bargap=0,
                          hovermode='closest',
                          showlegend=False,
                          autosize=False,
                          height=height,
                          width=width,)

        if save_iframe:

            if filename is None:

                pio.write_html(fig,
                               'reduced_image.html',
                               auto_open=open_iframe)

            else:

                pio.write_html(fig, filename + '.html', auto_open=open_iframe)

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

        print(self.filelist)
