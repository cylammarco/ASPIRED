import difflib
import json
import os
import pkg_resources
import warnings

import numpy as np
from plotly import graph_objects as go
from plotly import io as pio
from scipy import signal
from scipy import interpolate as itp
from spectres import spectres

from .spectrum1D import Spectrum1D

base_dir = os.path.dirname(__file__)


class StandardLibrary:
    def __init__(self, silence=False):
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
        silence: boolean
            Set to True to suppress all verbose warnings.

        '''

        self.silence = silence

        self._load_standard_dictionary()

    def _load_standard_dictionary(self):
        '''
        Load the dictionaries

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

        self.wave_standard_true = f[:, 0]
        self.fluxmag_standard_true = f[:, 1]

        if self.ftype == 'flux':

            if self.library != 'esoxshooter':

                self.fluxmag_standard_true *= 1e-16

    def _get_ing_standard(self):

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
        self.wave_standard_true = np.array(wave).astype('float')
        self.fluxmag_standard_true = np.array(fluxmag).astype('float')

        if self.ftype == 'flux':

            # Trap the ones without flux files
            if (extension == 'mas' or filename == 'g24a.fg'
                    or filename == 'g157a.fg' or filename == 'h102a.sto'):

                self.fluxmag_standard_true = 10.**(
                    -(self.fluxmag_standard_true / 2.5)
                ) * 3630.780548 / 3.33564095e4 / self.wave_standard_true**2

            # convert milli-Jy into F_lambda
            if unit == 'mjy':

                self.fluxmag_standard_true = (self.fluxmag_standard_true *
                                              1e-3 * 3.33564095e4 *
                                              self.wave_standard_true**2)

            # convert micro-Jy into F_lambda
            if unit == 'microjanskys':

                self.fluxmag_standard_true = (self.fluxmag_standard_true *
                                              1e-6 * 3.33564095e4 *
                                              self.wave_standard_true**2)

    def _get_iraf_standard(self):
        # iraf is always in AB magnitude

        folder = self.library

        # file name and extension
        filename = self.target + '.dat'

        filepath = os.path.join(base_dir, 'standards', folder, filename)

        f = np.loadtxt(filepath, skiprows=1)

        self.wave_standard_true = f[:, 0]
        self.fluxmag_standard_true = f[:, 1]

        if self.ftype == 'flux':

            # Convert from AB mag to flux
            self.fluxmag_standard_true = 10.**(
                -(self.fluxmag_standard_true / 2.5)
            ) * 3630.780548 / 3.33564095e4 / self.wave_standard_true**2

    def lookup_standard_libraries(self, target, cutoff=0.4):
        '''
        Check if the requested standard and library exist. Return the three
        most similar words if the requested one does not exist. See

            https://docs.python.org/3.7/library/difflib.html

        Parameters
        ----------
        target: str
            Name of the standard star
        cutoff: float (default: 0.4)
            The similarity toleranceold
            [0 (completely different) - 1 (identical)]

        '''

        # Load the list of targets in the requested library
        try:

            libraries = self.uname_to_lib[target]
            return libraries, True

        except Exception as e:

            warnings.warn(str(e))

            # If the requested target is not in any library, suggest the
            # closest match, Top 5 are returned.
            # difflib uses Gestalt pattern matching.
            target_list = difflib.get_close_matches(
                target, list(self.uname_to_lib.keys()), n=5, cutoff=cutoff)

            if len(target_list) > 0:

                if not self.silence:

                    warnings.warn(
                        'Requested standard star cannot be found, a list of ' +
                        'the closest matching names are returned:' +
                        str(target_list))

                return target_list, False

            else:

                raise ValueError(
                    'Please check the name of your standard star, nothing '
                    'share a similarity above ' + str(cutoff) + '.')

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

                if not self.silence:

                    warnings.warn(
                        'The requested standard star cannot be found in the '
                        'given library,  or the library is not specified. '
                        'ASPIRED is using ' + self.library + '.')

        else:

            # If not, search again with the first one returned from lookup.
            self.target = libraries[0]
            libraries, _ = self.lookup_standard_libraries(self.target)
            self.library = libraries[0]

            if not self.silence:

                print('The requested library does not exist, ' + self.library +
                      ' is used because it has the closest matching name.')

        if not self.silence:

            if library is None:

                # Use the default library order
                if not self.silence:

                    print('Standard library is not given, ' + self.library +
                          ' is used.')

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
                         save_iframe=False,
                         filename=None,
                         open_iframe=False):
        '''
        Display the standard star plot.

        Parameters
        ----------
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
            go.Scatter(x=self.wave_standard_true,
                       y=self.fluxmag_standard_true,
                       line=dict(color='royalblue', width=4)))

        fig.update_layout(
            title=self.library + ': ' + self.target + ' ' + self.ftype,
            xaxis_title=r'$\text{Wavelength / A}$',
            yaxis_title=(r'$\text{Flux / ergs cm}^{-2} \text{s}^{-1}' +
                         '\text{A}^{-1}$'),
            hovermode='closest',
            showlegend=False)

        if save_iframe:

            if filename is None:

                pio.write_html(fig, 'standard.html', auto_open=open_iframe)

            else:

                pio.write_html(fig, filename + '.html', auto_open=open_iframe)

        if display:

            if renderer == 'default':

                fig.show()

            else:

                fig.show(renderer)

        if return_jsonstring:

            return fig.to_json()


class FluxCalibration(StandardLibrary):
    def __init__(self, silence=False):
        '''
        Initialise a FluxCalibration object.

        Parameters
        ----------
        silence: boolean
            Set to True to suppress all verbose warnings.

        '''

        # Load the dictionary
        super().__init__()
        self.silence = silence
        self.spectrum1D = Spectrum1D()
        self.target_spec_id = None

    def from_spectrum1D(self, spectrum1D):
        self.spectrum1D.merge(spectrum1D)
        self.spectrum1D_imported = True

    def remove_spectrum1D(self):
        self.spectrum1D = Spectrum1D()
        self.spectrum1D_imported = False

    """
    def add_spec(self,
                 count,
                 spec_id=None,
                 count_err=None,
                 count_sky=None,
                 stype='science+standard'):
        '''
        Add spectrum (count, count_err & count_sky) one at a time.

        Parameters
        ----------
        count: 1-d array
            The summed count at each column about the trace.
        spec_id: int
            The ID corresponding to the spectrum1D object
        count_err: 1-d array
            the uncertainties of the count values
        count_sky: 1-d array
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'standard' in stype_split:

            if len(self.spectrum_list_standard.keys()) == 0:

                self.spectrum_list_standard[0] = Spectrum1D(0)

            if spec_id in list(self.spectrum_list_standard.keys()):

                if self.spectrum_list_standard[spec_id].count is not None:
                    warnings.warn(
                        'The given spec_id is in use already, the given '
                        'trace, trace_sigma and pixel_list will overwrite the '
                        'current data.')

            if spec_id is None:

                # If spectrum_list is not empty
                if not self.spectrum_list_standard:

                    spec_id = max(self.spectrum_list_standard.keys()) + 1

                else:

                    spec_id = 0

            self.spectrum_list_standard[spec_id].add_count(
                count, count_err, count_sky)

        elif 'science' in stype_split:

            if len(self.spectrum_list_science.keys()) == 0:

                self.spectrum_list_science[0] = Spectrum1D(0)

            if spec_id in list(self.spectrum_list_science.keys()):

                if self.spectrum_list_science[spec_id].count is not None:
                    warnings.warn(
                        'The given spec_id is in use already, the given '
                        'trace, trace_sigma and pixel_list will overwrite the '
                        'current data.')

            if spec_id is None:

                # If spectrum_list is not empty
                if not self.spectrum_list_science:

                    spec_id = max(self.spectrum_list_science.keys()) + 1

                else:

                    spec_id = 0

            self.spectrum_list_science[spec_id].add_count(
                count, count_err, count_sky)

        else:

            if stype not in ['science', 'standard']:

                raise ValueError('Unknown stype, please choose from '
                                 '(1) science; and/or (2) standard.')

    def add_wavelength(self,
                       wave=None,
                       wave_resampled=None,
                       spec_id=None,
                       stype='science+standard'):
        '''
        Add wavelength (must be same length as the spectrum), one at a time.

        Parameters
        ----------
        wave : numeric value, list or numpy 1D array (N) (default: None)
            The wavelength of each pixels of the spectrum.
        wave_resampled:
            The resampled wavelength of each pixels of the spectrum.
        spec_id: int
            The ID corresponding to the spectrum1D object
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if (wave is None) & (wave_resampled is None):
            raise ValueError('wave and wave_resampled cannot be None at '
                             'at the same time.')

        elif wave_resampled is None:
            wave_resampled = wave

        elif wave is None:
            wave = wave_resampled

        stype_split = stype.split('+')

        if 'standard' in stype_split:

            if len(self.spectrum_list_standard.keys()) == 0:
                self.spectrum_list_standard[0] = Spectrum1D(0)

            spec = self.spectrum_list_standard[0]

            spec.add_wavelength(wave)
            spec.add_wavelength_resampled(
                wave_bin=np.nanmedian(np.array(np.ediff1d(wave_resampled))),
                wave_start=wave_resampled[0],
                wave_end=wave_resampled[-1],
                wave_resampled=wave_resampled,
            )

        elif 'science' in stype_split:

            if spec_id is not None:
                if spec_id not in list(self.spectrum_list_science.keys()):
                    warnings.warn(
                        'The spec_id provided is not in the '
                        'spectrum_list_science, new Spectrum1D with the ID '
                        'is created.')
                    self.spectrum_list_science[spec_id] = Spectrum1D(spec_id)
            else:
                if not self.spectrum_list_science:
                    self.spectrum_list_science[0] = Spectrum1D(0)
                spec_id = list(self.spectrum_list_science.keys())

            if isinstance(spec_id, int):
                spec_id = [spec_id]

            for i, s in enumerate(spec_id):

                spec = self.spectrum_list_science[s]

                if (len(np.shape(np.array(wave))) == 2):

                    spec.add_wavelength(wave[i])

                else:

                    spec.add_wavelength(wave)

                if (len(np.shape(np.array(wave))) == 2):

                    spec.add_wavelength_resampled(
                        wave_bin=np.nanmedian(
                            np.array(np.ediff1d(wave_resampled[i]))),
                        wave_start=wave_resampled[i][0],
                        wave_end=wave_resampled[i][-1],
                        wave_resampled=wave_resampled[i],
                    )

                else:

                    spec.add_wavelength_resampled(
                        wave_bin=np.nanmedian(
                            np.array(np.ediff1d(wave_resampled))),
                        wave_start=wave_resampled[0],
                        wave_end=wave_resampled[-1],
                        wave_resampled=wave_resampled,
                    )

    def from_twodspec(self,
                      twodspec,
                      pixel_list=None,
                      stype='science+standard'):
        '''
        To add a TwoDSpec object or numpy array to provide the traces, line
        spread function of the traces, optionally the pixel values
        correcponding to the traces.

        Science type twodspec can contain more than 1 spectrum.

        Parameters
        ----------
        twodspec: TwoDSpec object
            TwoDSpec of the science/standard image containin the trace(s)
        pixel_list: list or numpy array
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(len(spec.trace[0])), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'standard' in stype_split:

            # Loop through the twodspec.spectrum_list to update the
            # spectrum_list_standard
            if not self.spectrum_list_standard:

                self.spectrum_list_standard[0] = Spectrum1D(spec_id=0)

            for key, value in twodspec.spectrum_list[0].__dict__.items():

                setattr(self.spectrum_list_standard[0], key, value)

        if 'science' in stype_split:

            # Loop through the spec_id in twodspec
            for i in twodspec.spectrum_list.keys():

                # Loop through the twodspec.spectrum_list to update the
                # spectrum_list_standard
                if i not in self.spectrum_list_science:

                    self.spectrum_list_science[i] = Spectrum1D(spec_id=i)

                for key, value in twodspec.spectrum_list[i].__dict__.items():

                    setattr(self.spectrum_list_science[i], key, value)

    def add_wavecal(self, wavecal, stype='science+standard'):
        '''
        Copy the spectrum_list from the WavelengthCalibration object to here.

        Parameters
        ----------
        wavecal: WavelengthPolyFit object
            The WavelengthPolyFit object for the science target, flux will
            not be calibrated if this is not provided.
        stype: string
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'standard' in stype_split:

            # Loop through the wavecal.spectrum_list to update the
            # spectrum_list_standard
            for key, value in wavecal.spectrum_list[0].__dict__.items():

                if not self.spectrum_list_standard[0]:

                    self.spectrum_list_standard[0] = Spectrum1D(spec_id=0)

                setattr(self.spectrum_list_standard[0], key, value)

        if 'science' in stype_split:

            # Loop through the spec_id in wavecal
            for i in wavecal.spectrum_list.keys():

                # Loop through the wavecal.spectrum_list to update the
                # spectrum_list_science
                for key, value in wavecal.spectrum_list[i].__dict__.items():

                    if not self.spectrum_list_science[i]:

                        self.spectrum_list_science[i] = Spectrum1D(spec_id=i)

                    setattr(self.spectrum_list_science[i], key, value)
    """

    def compute_sensitivity(self,
                            k=3,
                            smooth=True,
                            method='interpolate',
                            slength=5,
                            sorder=3,
                            mask_range=[[6850, 6960], [7575, 7700],
                                        [8925, 9050], [9265, 9750]],
                            mask_fit_order=1,
                            mask_fit_size=1,
                            return_function=True):
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
        smooth: boolean
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
                np.array(np.ediff1d(self.wave_standard_true))):

            flux_standard, flux_err_standard = spectres(
                np.array(self.wave_standard_true).reshape(-1),
                np.array(wave).reshape(-1),
                np.array(count).reshape(-1),
                np.array(count_err).reshape(-1),
                verbose=False)
            flux_standard_true = self.fluxmag_standard_true
            wave_standard_true = self.wave_standard_true

        # If the median resolution of the observed spectrum is lower than
        # the literature one
        else:

            flux_standard = count
            # flux_err_standard = count_err
            flux_standard_true = spectres(
                np.array(wave).reshape(-1),
                np.array(self.wave_standard_true).reshape(-1),
                np.array(self.fluxmag_standard_true).reshape(-1),
                verbose=False)
            wave_standard_true = wave

        # Get the sensitivity curve
        sensitivity = flux_standard_true / flux_standard
        sensitivity_masked = sensitivity.copy()

        if mask_range is not None:

            for m in mask_range:

                # If the mask is partially outside the spectrum, ignore
                if (m[0] < min(wave_standard_true)) or (
                        m[1] > max(wave_standard_true)):

                    continue

                # Get the indices for the two sides of the masking region
                left_end = int(max(np.where(wave_standard_true <= m[0])[0]))
                left_start = int(left_end - mask_fit_size)
                right_start = int(min(np.where(wave_standard_true >= m[1])[0]))
                right_end = int(right_start + mask_fit_size)

                # Get the wavelengths of the two sides
                wave_temp = np.concatenate(
                    (wave_standard_true[left_start:left_end],
                     wave_standard_true[right_start:right_end]))

                # Get the sensitivity of the two sides
                sensitivity_temp = np.concatenate(
                    (sensitivity[left_start:left_end],
                     sensitivity[right_start:right_end]))

                # Fit the polynomial across the masked region
                coeff = np.polynomial.polynomial.polyfit(
                    wave_temp, sensitivity_temp, mask_fit_order)

                # Replace the snsitivity values with the fitted values
                sensitivity_masked[
                    left_end:right_start] = np.polynomial.polynomial.polyval(
                        wave_standard_true[left_end:right_start], coeff)

        mask = np.isfinite(sensitivity_masked)
        sensitivity_masked = sensitivity_masked[mask]
        wave_standard_masked = wave_standard_true[mask]
        flux_standard_masked = flux_standard_true[mask]

        # apply a Savitzky-Golay filter to remove noise and Telluric lines
        if smooth:

            sensitivity_masked = signal.savgol_filter(sensitivity_masked,
                                                      slength, sorder)
            # Set the smoothing parameters
            self.spectrum1D.add_smoothing(smooth, slength, sorder)

        if method == 'interpolate':
            tck = itp.splrep(wave_standard_masked,
                             np.log10(sensitivity_masked),
                             k=k)

            def sensitivity_func(x):
                return itp.splev(x, tck)

        elif method == 'polynomial':

            coeff = np.polynomial.polynomial.polyfit(
                wave_standard_masked, np.log10(sensitivity_masked), deg=7)

            def sensitivity_func(x):
                return np.polynomial.polynomial.polyval(x, coeff)

        else:

            raise NotImplementedError('{} is not implemented.'.format(method))

        self.spectrum1D.add_sensitivity(sensitivity_masked)
        self.spectrum1D.add_literature_standard(wave_standard_masked,
                                                flux_standard_masked)

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
                            save_iframe=False,
                            filename=None,
                            open_iframe=False):
        '''
        Display the computed sensitivity curve.

        Parameters
        ----------
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

        if save_iframe:

            if filename is None:

                pio.write_html(fig, 'senscurve.html', auto_open=open_iframe)

            else:

                pio.write_html(fig, filename + '.html', auto_open=open_iframe)

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
                              display=True,
                              renderer='default',
                              width=1280,
                              height=720,
                              filename=None,
                              save_png=False,
                              save_jpg=False,
                              save_svg=False,
                              save_pdf=False,
                              return_jsonstring=False,
                              save_iframe=False,
                              open_iframe=False):
        '''
        Apply the computed sensitivity curve. And resample the spectra to
        match the highest resolution (the smallest wavelength bin) part of the
        spectrum.

        Note: This function directly modify the *target_spectrum1D*.

        Parameters
        ----------
        target_spectrum1D: Spectrum1D object
            The spectrum to be flux calibrated.
        sensitivity: Spectrum1D object or callable function
            To use for flux calibration of the target.
        inspect: boolean
            Set to True to create/display/save figure
        wave_min: float
            Minimum wavelength to display
        wave_max: float
            Maximum wavelength to display
        display: boolean
            Set to True to display disgnostic plot.
        renderer: string
            plotly renderer options.
        width: int/float
            Number of pixels in the horizontal direction of the outputs
        height: int/float
            Number of pixels in the vertical direction of the outputs
        filename: str
            Filename for the output, all of them will share the same name but
            will have different extension.
        save_png: boolean
            Save an png image of the Plotly plot
        save_jpg: boolean
            Save an png image of the Plotly plot
        save_svg: boolean
            Save an png image of the Plotly plot
        save_pdf: boolean
            Save a pdf of the Plotly plot
        return_jsonstring: boolean
            set to True to return json string that can be rendered by Plotly
            in any support language.
        save_iframe: boolean
            Save as an save_iframe, can work concurrently with other renderer
            apart from exporting return_jsonstring.
        open_iframe: boolean
            Open the save_iframe in the default browser if set to True.

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
                                  verbose=False)

        if count_err is None:

            flux_err_resampled = np.zeros_like(flux_resampled)

        else:

            flux_err = sensitivity * count_err
            flux_err_resampled = spectres(np.array(wave_resampled).reshape(-1),
                                          np.array(wave).reshape(-1),
                                          np.array(flux_err).reshape(-1),
                                          verbose=False)

        if count_sky is None:

            flux_sky_resampled = np.zeros_like(flux_resampled)

        else:

            flux_sky = sensitivity * count_sky
            flux_sky_resampled = spectres(np.array(wave_resampled).reshape(-1),
                                          np.array(wave).reshape(-1),
                                          np.array(flux_sky).reshape(-1),
                                          verbose=False)

        # Only computed for diagnostic
        sensitivity_resampled = spectres(np.array(wave_resampled).reshape(-1),
                                         np.array(wave).reshape(-1),
                                         np.array(sensitivity).reshape(-1),
                                         verbose=False)

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

            fig_sci = go.Figure(
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
            fig_sci.add_trace(
                go.Scatter(x=wave_resampled,
                           y=flux_resampled,
                           line=dict(color='royalblue'),
                           name='Flux'))

            if flux_err is not None:

                fig_sci.add_trace(
                    go.Scatter(x=wave_resampled,
                               y=flux_err_resampled,
                               line=dict(color='firebrick'),
                               name='Flux Uncertainty'))

            if flux_sky is not None:

                fig_sci.add_trace(
                    go.Scatter(x=wave_resampled,
                               y=flux_sky_resampled,
                               line=dict(color='orange'),
                               name='Sky Flux'))

            fig_sci.update_layout(hovermode='closest',
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
                                      bgcolor='rgba(0,0,0,0)'))

            if filename is None:

                filename_output = "spectrum_" + str(self.target_spec_id)

            else:

                filename_output = os.path.splitext(filename)[0] + "_" + str(
                    self.target_spec_id)

            if save_iframe:

                pio.write_html(fig_sci,
                               filename_output + '.html',
                               auto_open=open_iframe)

            if display:

                if renderer == 'default':

                    fig_sci.show()

                else:

                    fig_sci.show(renderer)

            if save_jpg:

                fig_sci.write_image(filename_output + '.jpg', format='jpg')

            if save_png:

                fig_sci.write_image(filename_output + '.png', format='png')

            if save_svg:

                fig_sci.write_image(filename_output + '.svg', format='svg')

            if save_pdf:

                fig_sci.write_image(filename_output + '.pdf', format='pdf')

            if return_jsonstring:

                return fig_sci.to_json()

    def create_fits(self,
                    output='count+count_resampled+flux+flux_resampled',
                    empty_primary_hdu=True):
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
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        overwrite: boolean
            Default is False.

        '''

        # If flux is calibrated
        self.spectrum1D.create_fits(output=output,
                                    empty_primary_hdu=empty_primary_hdu)

    def save_fits(
            self,
            output='count_resampled+sensitivity_resampled+flux_resampled',
            filename='fluxcal',
            empty_primary_hdu=True,
            overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

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
        empty_primary_hdu: boolean (default: True)
            Set to True to leave the Primary HDU blank (default: True)
        overwrite: boolean
            Default is False.

        '''

        # Fix the names and extensions
        if self.target_spec_id is not None:
            filename = os.path.splitext(filename)[0] + "_" + str(
                self.target_spec_id)
        else:
            filename = os.path.splitext(filename)[0]

        # Create the FITS here to go through all the checks, the
        # save_fits() below does not re-create the FITS. A warning will
        # be given, but it can be ignored.
        self.create_fits(output=output, empty_primary_hdu=empty_primary_hdu)

        self.spectrum1D.save_fits(output=output,
                                  filename=filename,
                                  overwrite=overwrite,
                                  empty_primary_hdu=empty_primary_hdu)

    def save_csv(self,
                 output='count_resampled+sensitivity_resampled+flux_resampled',
                 filename='fluxcal',
                 overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        output: String
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

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
        overwrite: boolean
            Default is False.

        '''

        # Fix the names and extensions
        if self.target_spec_id is not None:
            filename = os.path.splitext(filename)[0] + "_" + str(
                self.target_spec_id)
        else:
            filename = os.path.splitext(filename)[0]

        # Create the FITS here to go through all the checks, the
        # save_fits() below does not re-create the FITS. A warning will be
        # given, but it can be ignored.
        self.create_fits(output=output, empty_primary_hdu=False)

        self.spectrum1D.save_csv(output=output,
                                 filename=filename,
                                 overwrite=overwrite)

    def get_spectrum1D(self):

        return self.spectrum1D
