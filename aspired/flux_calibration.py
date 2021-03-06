import datetime
import difflib
import json
import logging
import os
import pkg_resources

import numpy as np
from plotly import graph_objects as go
from plotly import io as pio
from scipy import signal
from scipy import interpolate as itp
from spectres import spectres
from statsmodels.nonparametric.smoothers_lowess import lowess

from .spectrum1D import Spectrum1D

base_dir = os.path.dirname(__file__)

__all__ = ['StandardLibrary', 'FluxCalibration']


class StandardLibrary:
    def __init__(self,
                 verbose=True,
                 logger_name='StandardLibrary',
                 log_level='WARNING',
                 log_file_folder='default',
                 log_file_name='default'):
        '''
        This class handles flux calibration by comparing the extracted and
        wavelength-calibrated standard observation to the "ground truth"
        from

        https://github.com/iraf-community/iraf/tree/master/noao/lib/onedstandards
        https://www.eso.org/sci/observing/tools/standards/spectra.html

        See explanation notes at those links for details.

        The list of targets and libraries can be listed with

        list_all()

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
            File name of the log, set to None to logging.warning to screen
            only.

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

        self.verbose = verbose

        self._load_standard_dictionary()

    def _load_standard_dictionary(self):
        '''
        Load the dictionaries containing the names of all the standard stars.

        '''

        self.lib_to_uname = json.load(
            open(
                pkg_resources.resource_filename(
                    'aspired', 'standards/lib_to_uname.json')))
        self.uname_to_lib = json.load(
            open(
                pkg_resources.resource_filename(
                    'aspired', 'standards/uname_to_lib.json')))

    def _get_eso_standard(self):
        '''
        Formatter for reading the ESO standards.

        '''

        folder = self.library

        # first letter of the file name
        if self.ftype == 'flux':

            filename = 'f'

        else:

            filename = 'm'

        # the rest of the file name
        filename += self.target

        # the extension
        filename += '.dat'

        filepath = os.path.join(base_dir, 'standards', folder, filename)

        f = np.loadtxt(filepath)

        self.standard_wave_true = f[:, 0]
        self.standard_fluxmag_true = f[:, 1]

        if self.ftype == 'flux':

            if self.library != 'esoxshooter':

                self.standard_fluxmag_true /= 10.0**16.0

    def _get_ing_standard(self):
        '''
        Formatter for reading the ING standards.

        '''

        folder = self.library.split("_")[0]

        # the first part of the file name
        filename = self.target
        extension = self.library.split('_')[-1]

        # last letter (or nothing) of the file name
        if self.ftype == 'flux':

            # .mas only contain magnitude files
            if extension == 'mas':

                filename += 'a'

            if ((filename == 'g24' or filename == 'g157')
                    and (extension == 'fg')):

                filename += 'a'

            if (filename == 'h102') and (extension == 'sto'):

                filename += 'a'

        else:

            filename += 'a'

        # the extension
        filename += '.' + extension

        filepath = os.path.join(base_dir, 'standards', folder, filename)

        f = open(filepath)
        wave = []
        fluxmag = []
        for line in f.readlines():

            if line[0] in ['*', 'S']:

                if line.startswith('SET .Z.UNITS = '):

                    # remove all special characters and white spaces
                    unit = ''.join(e for e in line.split('"')[1].lower()
                                   if e.isalnum())

            else:

                li = line.strip().strip(':').split()
                wave.append(li[0])
                fluxmag.append(li[1])

        f.close()
        self.standard_wave_true = np.array(wave).astype('float')
        self.standard_fluxmag_true = np.array(fluxmag).astype('float')

        if self.ftype == 'flux':

            # Trap the ones without flux files
            if (extension == 'mas' or filename == 'g24a.fg'
                    or filename == 'g157a.fg' or filename == 'h102a.sto'):

                self.standard_fluxmag_true = 10.**(
                    -(self.standard_fluxmag_true / 2.5)
                ) * 3630.780548 / 3.33564095e4 / self.standard_wave_true**2

            # convert milli-Jy into F_lambda
            if unit == 'mjy':

                self.standard_fluxmag_true = (self.standard_fluxmag_true *
                                              1e-3 * 3.33564095e4 *
                                              self.standard_wave_true**2)

            # convert micro-Jy into F_lambda
            if unit == 'microjanskys':

                self.standard_fluxmag_true = (self.standard_fluxmag_true *
                                              1e-6 * 3.33564095e4 *
                                              self.standard_wave_true**2)

    def _get_iraf_standard(self):
        '''
        Formatter for reading the iraf standards.

        '''

        folder = self.library

        # file name and extension
        filename = self.target + '.dat'

        filepath = os.path.join(base_dir, 'standards', folder, filename)

        f = np.loadtxt(filepath, skiprows=1)

        self.standard_wave_true = f[:, 0]
        self.standard_fluxmag_true = f[:, 1]

        # iraf is always in AB magnitude
        if self.ftype == 'flux':

            # Convert from AB mag to flux
            self.standard_fluxmag_true = 10.**(
                -(self.standard_fluxmag_true / 2.5)
            ) * 3630.780548 / 3.33564095e4 / self.standard_wave_true**2

    def lookup_standard_libraries(self, target, cutoff=0.4):
        '''
        Check if the requested standard and library exist. Return the three
        most similar words if the requested one does not exist. See

            https://docs.python.org/3.7/library/difflib.html

        Parameters
        ----------
        target: str
            Name of the standard star
        cutoff: float (Default: 0.4)
            The similarity toleranceold
            [0 (completely different) - 1 (identical)]

        '''

        # Load the list of targets in the requested library
        try:

            libraries = self.uname_to_lib[target]
            return libraries, True

        except Exception as e:

            self.logger.warning(str(e))

            # If the requested target is not in any library, suggest the
            # closest match, Top 5 are returned.
            # difflib uses Gestalt pattern matching.
            target_list = difflib.get_close_matches(
                target, list(self.uname_to_lib.keys()), n=5, cutoff=cutoff)

            if len(target_list) > 0:

                self.logger.warning(
                    'Requested standard star cannot be found, a list of ' +
                    'the closest matching names are returned: {}'.format(
                        target_list))

                return target_list, False

            else:
                error_msg = 'Please check the name of your standard ' +\
                    'star, nothing share a similarity above {}.'.format(
                        cutoff)
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

    def load_standard(self, target, library=None, ftype='flux', cutoff=0.4):
        '''
        Read the standard flux/magnitude file. And return the wavelength and
        flux/mag. The units of the data are always in

        | wavelength: A
        | flux:       ergs / cm / cm / s / A
        | mag:        mag (AB)


        Parameters
        ----------
        target: string
            Name of the standard star
        library: string
            Name of the library of standard star
        ftype: string
            'flux' or 'mag'
        cutoff: float
            The toleranceold for the word similarity in the range of [0, 1].

        '''

        self.target = target
        self.ftype = ftype
        self.cutoff = cutoff

        libraries, success = self.lookup_standard_libraries(self.target)

        if success:

            if library in libraries:

                self.library = library

            else:

                self.library = libraries[0]

                self.logger.warning(
                    'The requested standard star cannot be found in the '
                    'given library,  or the library is not specified. '
                    'ASPIRED is using ' + self.library + '.')

        else:

            # If not, search again with the first one returned from lookup.
            self.target = libraries[0]
            libraries, _ = self.lookup_standard_libraries(self.target)
            self.library = libraries[0]

            self.logger.warning(
                'The requested library does not exist, ' + self.library +
                ' is used because it has the closest matching name.')

        if not self.verbose:

            if library is None:

                # Use the default library order
                self.logger.warning('Standard library is not given, ' +
                                    self.library + ' is used.')

        if self.library.startswith('iraf'):

            self._get_iraf_standard()

        if self.library.startswith('ing'):

            self._get_ing_standard()

        if self.library.startswith('eso'):

            self._get_eso_standard()

    def inspect_standard(self,
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
        Display the standard star plot.

        Parameters
        ----------
        display: bool
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool
            set to True to return json string that can be rendered by Plotly
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
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        '''

        fig = go.Figure(
            layout=dict(autosize=False,
                        height=height,
                        width=width,
                        updatemenus=list([
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
            go.Scatter(x=self.standard_wave_true,
                       y=self.standard_fluxmag_true,
                       line=dict(color='royalblue', width=4)))

        fig.update_layout(
            title=self.library + ': ' + self.target + ' ' + self.ftype,
            xaxis_title=r'$\text{Wavelength / A}$',
            yaxis_title=(r'$\text{Flux / ergs cm}^{-2} \text{s}^{-1}' +
                         '\text{A}^{-1}$'),
            hovermode='closest',
            showlegend=False)

        if filename is None:

            filename = 'standard'

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


class FluxCalibration(StandardLibrary):
    def __init__(self,
                 verbose=True,
                 logger_name='FluxCalibration',
                 log_level='WARNING',
                 log_file_folder='default',
                 log_file_name='default'):
        '''
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
        log_level: str (Default: WARNING)
            Four levels of logging are available, in decreasing order of
            information and increasing order of severity:
            CRITICAL, DEBUG, INFO, WARNING, ERROR
        log_file_folder: None or str (Default: "default")
            Folder in which the file is save, set to default to save to the
            current path.
        log_file_name: None or str (Default: "default")
            File name of the log, set to None to logging.warning to screen
            only.

        '''

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
            '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] '
            '%(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S')

        if log_file_name is None:
            # Only logging.warning log to screen
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
        self.logger.addHandler(handler)

        self.verbose = verbose
        self.logger_name = logger_name
        self.log_level = log_level
        self.log_file_folder = log_file_folder
        self.log_file_name = log_file_name

        # Load the dictionary
        super().__init__(verbose=self.verbose,
                         logger_name=self.logger_name,
                         log_level=self.log_level,
                         log_file_folder=self.log_file_folder,
                         log_file_name=self.log_file_name)
        self.verbose = verbose
        self.spectrum1D = Spectrum1D(spec_id=0,
                                     verbose=self.verbose,
                                     logger_name=self.logger_name,
                                     log_level=self.log_level,
                                     log_file_folder=self.log_file_folder,
                                     log_file_name=self.log_file_name)
        self.target_spec_id = None
        self.standard_wave_true = None
        self.standard_fluxmag_true = None

    def from_spectrum1D(self, spectrum1D, merge=False):
        '''
        This function copies all the info from the spectrum1D, because users
        may supply different level/combination of reduction, everything is
        copied from the spectrum1D even though in most cases only a None
        will be passed.

        By default, this is passing object by reference by default, so it
        directly modifies the spectrum1D supplied. By setting merger to True,
        it copies the data into the Spectrum1D in the FluxCalibration object.

        Parameters
        ----------
        spectrum1D: Spectrum1D object
            The Spectrum1D to be referenced or copied.
        merge: bool (Default: None)
            Set to True to copy everything over to the local Spectrum1D,
            hence FluxCalibration will not be acting on the Spectrum1D
            outside.

        '''

        if merge:
            self.spectrum1D.merge(spectrum1D)
        else:
            self.spectrum1D = spectrum1D

        self.spectrum1D_imported = True

    def remove_spectrum1D(self):

        self.spectrum1D = Spectrum1D(spec_id=0,
                                     verbose=self.verbose,
                                     logger_name=self.logger_name,
                                     log_level=self.log_level,
                                     log_file_folder=self.log_file_folder,
                                     log_file_name=self.log_file_name)
        self.spectrum1D_imported = False

    def load_standard(self, target, library=None, ftype='flux', cutoff=0.4):
        """
        Load the literature values of the standard star.

        """

        super().load_standard(target=target,
                              library=library,
                              ftype=ftype,
                              cutoff=cutoff)
        # the best target and library found can be different from the input
        self.spectrum1D.add_standard_star(library=self.library,
                                          target=self.target)

    def add_standard(self, wavelength, count, count_err=None, count_sky=None):
        '''
        Add spectrum (wavelength, count, count_err & count_sky).

        Parameters
        ----------
        wavelength: 1-d array
            The wavelength at each pixel of the trace.
        count: 1-d array
            The summed count at each column about the trace.
        count_err: 1-d array
            the uncertainties of the count values
        count_sky: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract

        '''

        self.spectrum1D.add_wavelength(wavelength)
        self.spectrum1D.add_count(count, count_err, count_sky)

    def compute_sensitivity(self,
                            k=3,
                            smooth=True,
                            method='interpolate',
                            slength=7,
                            sorder=3,
                            mask_range=[[6850, 6960], [7575, 7700],
                                        [8925, 9050], [9265, 9750]],
                            mask_fit_order=1,
                            mask_fit_size=1,
                            return_function=True,
                            use_lowess=True,
                            lowess_frac=0.1,
                            **kwargs):
        '''
        The sensitivity curve is computed by dividing the true values by the
        wavelength calibrated standard spectrum, which is resampled with the
        spectres.spectres(). The curve is then interpolated with a cubic spline
        by default and is stored as a scipy interp1d object.

        A Savitzky-Golay filter is available for smoothing before the
        interpolation but it is not used by default.

        6850 - 6960, 7575 - 7700, 8925 - 9050 and 9265 - 9750 A are masked by
        default.

        Parameters
        ----------
        k: integer [1,2,3,4,5 only]
            The order of the spline.
        smooth: bool
            Set to smooth the input spectrum with scipy.signal.savgol_filter
        method: str (Default: interpolate)
            This should be either 'interpolate' of 'polynomial'. Note that the
            polynomial is computed from the interpolated function. The
            default is interpolate because it is much more stable at the
            wavelength limits of a spectrum in an automated system.
        slength: int
            SG-filter window size
        sorder: int
            SG-filter polynomial order
        mask_range: None or list of list
            Masking out regions not suitable for fitting the sensitivity curve.
            None for no mask. List of list has the pattern
            [[min1, max1], [min2, max2],...]
        mask_fit_order: int
            Order of polynomial to be fitted over the masked regions
        mask_fit_size: int
            Number of "pixels" to be fitted on each side of the masked regions.
        return_function: bool
            Set to True to return the callable function of the sensitivity
            curve.
        use_lowess: bool (Default: True)
            Use a lowess function to get the continuum.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        '''

        # resampling both the observed and the database standard spectra
        # in unit of flux per second. The higher resolution spectrum is
        # resampled to match the lower resolution one.
        count = getattr(self.spectrum1D, 'count')
        count_err = getattr(self.spectrum1D, 'count_err')
        wave = getattr(self.spectrum1D, 'wave')

        # If the median resolution of the observed spectrum is higher than
        # the literature one
        if np.nanmedian(np.array(np.ediff1d(wave))) < np.nanmedian(
                np.array(np.ediff1d(self.standard_wave_true))):

            standard_flux, standard_flux_err = spectres(
                np.array(self.standard_wave_true).reshape(-1),
                np.array(wave).reshape(-1),
                np.array(count).reshape(-1),
                np.array(count_err).reshape(-1),
                verbose=True)
            standard_flux_true = self.standard_fluxmag_true
            standard_wave_true = self.standard_wave_true

        # If the median resolution of the observed spectrum is lower than
        # the literature one
        else:

            standard_flux = count
            # standard_flux_err = count_err
            standard_flux_true = spectres(
                np.array(wave).reshape(-1),
                np.array(self.standard_wave_true).reshape(-1),
                np.array(self.standard_fluxmag_true).reshape(-1),
                verbose=True)
            standard_wave_true = wave

        # apply a Savitzky-Golay filter to remove noise and Telluric lines
        if smooth:

            standard_flux = signal.savgol_filter(standard_flux, slength,
                                                 sorder)
            # Set the smoothing parameters
            self.spectrum1D.add_smoothing(smooth, slength, sorder)

        if use_lowess:

            standard_flux = lowess(standard_flux,
                                   standard_wave_true,
                                   return_sorted=False,
                                   frac=lowess_frac,
                                   **kwargs)

        # Get the sensitivity curve
        sensitivity = standard_flux_true / standard_flux
        sensitivity_masked = sensitivity.copy()

        if mask_range is not None:

            for m in mask_range:

                # If the mask is partially outside the spectrum, ignore
                if (m[0] < min(standard_wave_true)) or (
                        m[1] > max(standard_wave_true)):

                    continue

                # Get the indices for the two sides of the masking region
                left_end = int(max(np.where(standard_wave_true <= m[0])[0]))
                left_start = int(left_end - mask_fit_size)
                right_start = int(min(np.where(standard_wave_true >= m[1])[0]))
                right_end = int(right_start + mask_fit_size)

                # Get the wavelengths of the two sides
                wave_temp = np.concatenate(
                    (standard_wave_true[left_start:left_end],
                     standard_wave_true[right_start:right_end]))

                # Get the sensitivity of the two sides
                sensitivity_temp = np.concatenate(
                    (sensitivity[left_start:left_end],
                     sensitivity[right_start:right_end]))

                finite_mask = ~np.isnan(sensitivity_temp) & (sensitivity_temp >
                                                             0)

                # Fit the polynomial across the masked region
                coeff = np.polynomial.polynomial.polyfit(
                    wave_temp[finite_mask], sensitivity_temp[finite_mask],
                    mask_fit_order)

                # Replace the snsitivity values with the fitted values
                sensitivity_masked[
                    left_end:right_start] = np.polynomial.polynomial.polyval(
                        standard_wave_true[left_end:right_start], coeff)

        mask = np.isfinite(sensitivity_masked) & (sensitivity_masked > 0)
        sensitivity_masked = sensitivity_masked[mask]
        standard_wave_masked = standard_wave_true[mask]
        standard_flux_masked = standard_flux_true[mask]

        if method == 'interpolate':
            tck = itp.splrep(standard_wave_masked,
                             np.log10(sensitivity_masked),
                             k=k)

            def sensitivity_func(x):
                return itp.splev(x, tck)

        elif method == 'polynomial':

            coeff = np.polynomial.polynomial.polyfit(
                standard_wave_masked, np.log10(sensitivity_masked), deg=7)

            def sensitivity_func(x):
                return np.polynomial.polynomial.polyval(x, coeff)

        else:

            error_msg = '{} is not implemented.'.format(method)
            self.logger.critical(error_msg)
            raise NotImplementedError(error_msg)

        self.spectrum1D.add_sensitivity(sensitivity_masked)
        self.spectrum1D.add_literature_standard(standard_wave_masked,
                                                standard_flux_masked)

        # Add to each Spectrum1D object
        self.spectrum1D.add_sensitivity_func(sensitivity_func)

        if return_function:

            return sensitivity_func

    def add_sensitivity_func(self, sensitivity_func):
        '''
        parameters
        ----------
        sensitivity_func: callable function
            Interpolated sensivity curve object.

        '''

        # Add to both science and standard spectrum_list
        self.spectrum1D.add_sensitivity_func(sensitivity_func=sensitivity_func)
        self.spectrum1D.add_literature_standard(self.standard_wave_true,
                                                self.standard_fluxmag_true)

    def get_sensitivity_func(self):
        pass

    def save_sensitivity_func(self):
        pass

    def inspect_sensitivity(self,
                            display=True,
                            renderer='default',
                            width=1280,
                            height=720,
                            return_jsonstring=False,
                            save_fig=False,
                            fig_type='iframe+png',
                            filename=None,
                            open_iframe=False,
                            browser='default'):
        '''
        Display the computed sensitivity curve.

        Parameters
        ----------
        display: bool
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool
            set to True to return json string that can be rendered by Plotly
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

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        '''

        wave_literature = getattr(self.spectrum1D, 'wave_literature')
        flux_literature = getattr(self.spectrum1D, 'flux_literature')
        sensitivity = getattr(self.spectrum1D, 'sensitivity')
        sensitivity_func = getattr(self.spectrum1D, 'sensitivity_func')

        smooth = getattr(self.spectrum1D, 'smooth')
        library = getattr(self.spectrum1D, 'library')
        target = getattr(self.spectrum1D, 'target')

        fig = go.Figure(
            layout=dict(autosize=False,
                        height=height,
                        width=width,
                        updatemenus=list([
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
            go.Scatter(x=wave_literature,
                       y=flux_literature,
                       line=dict(color='royalblue', width=4),
                       name='Count / s (Observed)'))

        fig.add_trace(
            go.Scatter(x=wave_literature,
                       y=sensitivity,
                       yaxis='y2',
                       line=dict(color='firebrick', width=4),
                       name='Sensitivity Curve'))

        fig.add_trace(
            go.Scatter(x=wave_literature,
                       y=10.**sensitivity_func(wave_literature),
                       yaxis='y2',
                       line=dict(color='black', width=2),
                       name='Best-fit Sensitivity Curve'))

        if smooth:

            slength = getattr(self.spectrum1D, 'slength')
            sorder = getattr(self.spectrum1D, 'sorder')
            fig.update_layout(title='SG(' + str(slength) + ', ' + str(sorder) +
                              ')-Smoothed ' + library + ': ' + target,
                              yaxis_title='Smoothed Count / s')

        else:

            fig.update_layout(title=library + ': ' + target,
                              yaxis_title='Count / s')

        fig.update_layout(hovermode='closest',
                          showlegend=True,
                          xaxis_title=r'$\text{Wavelength / A}$',
                          yaxis=dict(title='Count / s'),
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
                                                  bgcolor='rgba(0,0,0,0)'))

        if filename is None:

            filename = 'senscurve'

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

    def apply_flux_calibration(self,
                               target_spectrum1D,
                               inspect=False,
                               wave_min=4000.,
                               wave_max=8000.,
                               display=False,
                               renderer='default',
                               width=1280,
                               height=720,
                               return_jsonstring=False,
                               save_fig=False,
                               fig_type='iframe+png',
                               filename=None,
                               open_iframe=False):
        '''
        Apply the computed sensitivity curve. And resample the spectra to
        match the highest resolution (the smallest wavelength bin) part of the
        spectrum.

        Note: This function directly modify the *target_spectrum1D*.

        Note 2: the wave_min and wave_max are for DISPLAY purpose only.

        Parameters
        ----------
        target_spectrum1D: Spectrum1D object
            The spectrum to be flux calibrated.
        sensitivity: Spectrum1D object or callable function
            To use for flux calibration of the target.
        inspect: bool
            Set to True to create/display/save figure
        wave_min: float
            Minimum wavelength to display
        wave_max: float
            Maximum wavelength to display
        display: bool
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
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

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        '''

        self.target_spec_id = getattr(target_spectrum1D, 'spec_id')

        wave = getattr(target_spectrum1D, 'wave')
        wave_resampled = getattr(target_spectrum1D, 'wave_resampled')
        if wave_resampled is None:
            wave_resampled = wave
        count = getattr(target_spectrum1D, 'count')
        count_err = getattr(target_spectrum1D, 'count_err')
        count_sky = getattr(target_spectrum1D, 'count_sky')

        # apply the flux calibration
        sensitivity_func = getattr(self.spectrum1D, 'sensitivity_func')
        sensitivity = 10.**sensitivity_func(wave)

        flux = sensitivity * count

        flux_resampled = spectres(np.array(wave_resampled).reshape(-1),
                                  np.array(wave).reshape(-1),
                                  np.array(flux).reshape(-1),
                                  verbose=True)

        if count_err is None:

            flux_err_resampled = np.zeros_like(flux_resampled)

        else:

            flux_err = sensitivity * count_err
            flux_err_resampled = spectres(np.array(wave_resampled).reshape(-1),
                                          np.array(wave).reshape(-1),
                                          np.array(flux_err).reshape(-1),
                                          verbose=True)

        if count_sky is None:

            flux_sky_resampled = np.zeros_like(flux_resampled)

        else:

            flux_sky = sensitivity * count_sky
            flux_sky_resampled = spectres(np.array(wave_resampled).reshape(-1),
                                          np.array(wave).reshape(-1),
                                          np.array(flux_sky).reshape(-1),
                                          verbose=True)

        # Only computed for diagnostic
        sensitivity_resampled = spectres(np.array(wave_resampled).reshape(-1),
                                         np.array(wave).reshape(-1),
                                         np.array(sensitivity).reshape(-1),
                                         verbose=True)

        flux_resampled[np.isnan(flux_resampled)] = 0.
        flux_err_resampled[np.isnan(flux_err_resampled)] = 0.
        flux_sky_resampled[np.isnan(flux_sky_resampled)] = 0.

        target_spectrum1D.add_flux(flux, flux_err, flux_sky)
        target_spectrum1D.add_flux_resampled(flux_resampled,
                                             flux_err_resampled,
                                             flux_sky_resampled)
        target_spectrum1D.add_sensitivity(sensitivity)
        target_spectrum1D.add_sensitivity_resampled(sensitivity_resampled)

        # Add the rest of the flux calibration parameters
        target_spectrum1D.merge(self.spectrum1D)

        if inspect:

            wave_mask = ((np.array(wave_resampled).reshape(-1) > wave_min)
                         & (np.array(wave_resampled).reshape(-1) < wave_max))

            flux_low = np.nanpercentile(
                np.array(flux_resampled).reshape(-1)[wave_mask], 5) / 1.5
            flux_high = np.nanpercentile(
                np.array(flux_resampled).reshape(-1)[wave_mask], 95) * 1.5
            flux_mask = ((np.array(flux_resampled).reshape(-1) > flux_low)
                         & (np.array(flux_resampled).reshape(-1) < flux_high))
            flux_min = np.log10(
                np.nanmin(np.array(flux_resampled).reshape(-1)[flux_mask]))
            flux_max = np.log10(
                np.nanmax(np.array(flux_resampled).reshape(-1)[flux_mask]))

            fig = go.Figure(
                layout=dict(autosize=False,
                            height=height,
                            width=width,
                            updatemenus=list([
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
                go.Scatter(x=wave_resampled,
                           y=flux_resampled,
                           line=dict(color='royalblue'),
                           name='Flux'))

            if flux_err is not None:

                fig.add_trace(
                    go.Scatter(x=wave_resampled,
                               y=flux_err_resampled,
                               line=dict(color='firebrick'),
                               name='Flux Uncertainty'))

            if flux_sky is not None:

                fig.add_trace(
                    go.Scatter(x=wave_resampled,
                               y=flux_sky_resampled,
                               line=dict(color='orange'),
                               name='Sky Flux'))

            fig.update_layout(hovermode='closest',
                              showlegend=True,
                              xaxis=dict(title='Wavelength / A',
                                         range=[wave_min, wave_max]),
                              yaxis=dict(title='Flux',
                                         range=[flux_min, flux_max],
                                         type='log'),
                              legend=go.layout.Legend(x=0,
                                                      y=1,
                                                      traceorder="normal",
                                                      font=dict(
                                                          family="sans-serif",
                                                          size=12,
                                                          color="black"),
                                                      bgcolor='rgba(0,0,0,0)'))

            if filename is None:

                filename = "spectrum_" + str(self.target_spec_id)

            else:

                filename = os.path.splitext(filename)[0] + "_" + str(
                    self.target_spec_id)

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

    def create_fits(self,
                    output='count+count_resampled+flux+flux_resampled',
                    empty_primary_hdu=True,
                    recreate=False):
        '''
        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

            trace: 2 HDUs
                Trace, and trace width (pixel)
            count: 4 HDUs
                Count, uncertainty, sky, optimal flag, and weight (pixel)
            arc_spec: 3 HDUs
                1D arc spectrum, arc line pixels, and arc line effective pixels
            wavecal: 1 HDU
                Polynomial coefficients for wavelength calibration
            wavelength: 1 HDU
                Wavelength of each pixel
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank (Default: True)
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.

        '''

        # If flux is calibrated
        self.spectrum1D.create_fits(output=output,
                                    empty_primary_hdu=empty_primary_hdu,
                                    recreate=recreate)

    def save_fits(
            self,
            output='count_resampled+sensitivity_resampled+flux_resampled',
            filename='fluxcal',
            empty_primary_hdu=True,
            overwrite=False,
            recreate=False):
        '''
        Save the reduced data to disk, with a choice of any combination of
        the data that are already present in the Spectrum1D, see below the
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
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            wavelength: 1 HDU
                Wavelength of each pixel
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank (Default: True)
        overwrite: bool
            Default is False.
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.

        '''

        # Fix the names and extensions
        if self.target_spec_id is not None:
            filename = os.path.splitext(filename)[0] + "_" + str(
                self.target_spec_id)
        else:
            filename = os.path.splitext(filename)[0]

        self.spectrum1D.save_fits(output=output,
                                  filename=filename,
                                  overwrite=overwrite,
                                  recreate=recreate,
                                  empty_primary_hdu=empty_primary_hdu)

    def save_csv(self,
                 output='sensitivity_resampled+flux_resampled',
                 filename='fluxcal',
                 overwrite=False,
                 recreate=False):
        '''
        Save the reduced data to disk, with a choice of any combination of
        the data that are already present in the Spectrum1D, see below the
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
                Count, uncertainty, sky, and weight (pixel)
            count_resampled: 3 HDUs
                Resampled Count, uncertainty, and sky (wavelength)
            wavelength: 1 HDU
                Wavelength of each pixel
            flux: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (pixel)
            flux_resampled: 4 HDUs
                Flux, uncertainty, sky, and sensitivity (wavelength)
        filename: String
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        overwrite: bool
            Default is False.
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.

        '''

        # Fix the names and extensions
        if self.target_spec_id is not None:
            filename = os.path.splitext(filename)[0] + "_" + str(
                self.target_spec_id)
        else:
            filename = os.path.splitext(filename)[0]

        self.spectrum1D.save_csv(output=output,
                                 filename=filename,
                                 overwrite=overwrite,
                                 recreate=recreate)

    def get_spectrum1D(self):

        return self.spectrum1D
