import datetime
import logging
import os
import pkg_resources

import numpy as np
from plotly import graph_objects as go
from plotly import io as pio
from spectres import spectres
from scipy.interpolate import interp1d

from .wavelength_calibration import WavelengthCalibration
from .flux_calibration import FluxCalibration
from .spectrum1D import Spectrum1D

__all__ = ['OneDSpec']


class OneDSpec():
    def __init__(self,
                 verbose=True,
                 logger_name='OneDSpec',
                 log_level='WARNING',
                 log_file_folder='default',
                 log_file_name='default'):
        '''
        This class applies the wavelength calibrations and compute & apply the
        flux calibration to the extracted 1D spectra. The standard TwoDSpec
        object is not required for data reduction, but the flux calibrated
        standard observation will not be available for diagnostic.

        Parameters
        ----------
        verbose: bool (Default: True)
            Set to False to suppress all verbose warnings, except for
            critical failure.
        logger_name: str (Default: 'OneDSpec')
            This will set the name of the logger, if the name is used already,
            it will reference to the existing logger. This will be the
            first part of the default log file name unless log_file_name is
            provided.
        log_level: str (Default: 'WARNING')
            Four levels of logging are available, in decreasing order of
            information and increasing order of severity: (1) DEBUG, (2) INFO,
            (3) WARNING, (4) ERROR and (5) CRITICAL. WARNING means that
            there is suboptimal operations in some parts of that step. ERROR
            means that the requested operation cannot be performed, but the
            software can handle it by either using the default setting or
            skipping the operation. CRITICAL means that the requested
            operation cannot be resolved without human interaction, this is
            most usually coming from missing data.
        log_file_folder: None or str (Default: 'default')
            Folder in which the file is save, set to default to save to the
            current path.
        log_file_name: None or str (Default: 'default')
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

        self.verbose = verbose
        self.logger_name = logger_name
        self.log_level = log_level
        self.log_file_folder = log_file_folder
        self.log_file_name = log_file_name

        # Initialise empty calibration objects
        self.science_wavecal = {}
        self.standard_wavecal = WavelengthCalibration(
            verbose=self.verbose,
            logger_name=self.logger_name,
            log_level=self.log_level,
            log_file_folder=self.log_file_folder,
            log_file_name=self.log_file_name)
        self.fluxcal = FluxCalibration(verbose=self.verbose,
                                       logger_name=self.logger_name,
                                       log_level=self.log_level,
                                       log_file_folder=self.log_file_folder,
                                       log_file_name=self.log_file_name)

        # Create empty dictionary
        self.science_spectrum_list = {}
        self.standard_spectrum_list = {
            0:
            Spectrum1D(spec_id=0,
                       verbose=self.verbose,
                       logger_name=self.logger_name,
                       log_level=self.log_level,
                       log_file_folder=self.log_file_folder,
                       log_file_name=self.log_file_name)
        }

        self.add_science_spectrum1D(0)

        # Link them up
        self.science_wavecal[0].from_spectrum1D(self.science_spectrum_list[0])
        self.standard_wavecal.from_spectrum1D(self.standard_spectrum_list[0])
        self.fluxcal.from_spectrum1D(self.standard_spectrum_list[0])

        # Tracking data availability
        self.science_data_available = False
        self.standard_data_available = False

        self.science_arc_available = False
        self.standard_arc_available = False

        self.science_trace_available = False
        self.standard_trace_available = False

        self.science_arc_spec_available = False
        self.standard_arc_spec_available = False

        self.science_arc_lines_available = False
        self.standard_arc_lines_available = False

        self.science_atlas_available = False
        self.standard_atlas_available = False

        self.science_hough_pairs_available = False
        self.standard_hough_pairs_available = False

        self.science_wavecal_polynomial_available = False
        self.standard_wavecal_polynomial_available = False

        self.science_wavelength_calibrated = False
        self.standard_wavelength_calibrated = False

        self.science_wavelength_resampled = False
        self.standard_wavelength_resampled = False

        self.atmospheric_extinction_correction_available = False
        self.atmospheric_extinction_corrected = False

        self.sensitivity_curve_available = False

        self.science_flux_calibrated = False
        self.standard_flux_calibrated = False

        self.science_flux_resampled = False
        self.standard_flux_calibrated = False

    def add_science_spectrum1D(self, spec_id):
        self.science_wavecal.update({
            spec_id:
            WavelengthCalibration(verbose=self.verbose,
                                  logger_name=self.logger_name,
                                  log_level=self.log_level,
                                  log_file_folder=self.log_file_folder,
                                  log_file_name=self.log_file_name)
        })

        self.science_spectrum_list.update({
            spec_id:
            Spectrum1D(spec_id=spec_id,
                       verbose=self.verbose,
                       logger_name=self.logger_name,
                       log_level=self.log_level,
                       log_file_folder=self.log_file_folder,
                       log_file_name=self.log_file_name)
        })

        self.science_wavecal[spec_id].from_spectrum1D(
            self.science_spectrum_list[spec_id])

    def add_fluxcalibration(self, fluxcal):
        '''
        Provide the pre-calibrated FluxCalibration object.

        Parameters
        ----------
        fluxcal: FluxCalibration object
            The true mag/flux values.

        '''

        if type(fluxcal) == FluxCalibration:

            self.fluxcal = fluxcal

        else:

            err_msg = 'Please provide a valid FluxCalibration object'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

    def add_wavelengthcalibration(self,
                                  wavecal,
                                  spec_id=None,
                                  stype='science+standard'):
        '''
        Provide the pre-calibrated WavelengthCalibration object.

        Parameters
        ----------
        wavecal: list of WavelengthCalibration object
            The WavelengthPolyFit object for the science target, flux will
            not be calibrated if this is not provided.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if type(wavecal) == WavelengthCalibration:

            wavecal = [wavecal]

        elif type(wavecal) == list:

            pass

        else:

            err_msg = 'Please provide a WavelengthCalibration object or ' +\
                'a list of them.'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split('+')

        for s in stype_split:

            if s == 'science':

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(wavecal) == len(spec_id):

                    wavecal = {
                        spec_id[i]: wavecal[i]
                        for i in range(len(spec_id))
                    }

                elif len(wavecal) == 1:

                    wavecal = {
                        spec_id[i]: wavecal[0]
                        for i in range(len(spec_id))
                    }

                else:

                    error_msg = 'wavecal must be the same length of shape ' +\
                        'as spec_id.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

                for i in spec_id:

                    if type(wavecal[i]) == WavelengthCalibration:

                        self.science_wavecal[i] = wavecal[i]

                    else:

                        err_msg = 'Please provide a valid ' +\
                            'WavelengthCalibration object.'
                        self.logger.critical(err_msg)
                        raise TypeError(err_msg)

            if s == 'standard':

                if type(wavecal[0]) == WavelengthCalibration:

                    self.standard_wavecal = wavecal[0]

                else:

                    err_msg = 'Please provide a valid ' +\
                        'WavelengthCalibration object'
                    self.logger.critical(err_msg)
                    raise TypeError(err_msg)

    def add_wavelength(self, wave, spec_id=None, stype='science+standard'):
        '''
        Three combinations of wave and spec_id shapes are accepted.

        +-----------+-----------------+
        | Parameter |       Size      |
        +-----------+-----+-----+-----+
        | wave      |  1  |  1  |  N  |
        +-----------+-----+-----+-----+
        | spec_id   |  1  |  N  |  N  |
        +-----------+-----+-----+-----+

        Parameters
        ----------
        wave : numeric value, list or numpy 1D array (N) (Default: None)
            The wavelength of each pixels of the spectrum.
        spec_id: int
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if type(wave) == np.ndarray:

            wave = [wave]

        elif type(wave) == list:

            pass

        else:

            err_msg = 'Please provide a numpy array or a list of them.'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_data_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(wave) == len(spec_id):

                    wave = {spec_id[i]: wave[i] for i in range(len(spec_id))}

                elif len(wave) == 1:

                    wave = {spec_id[i]: wave[0] for i in range(len(spec_id))}

                else:

                    error_msg = 'wave must be the same length of shape ' +\
                        'as spec_id.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

                for i in spec_id:

                    if len(wave[i]) == len(
                            self.science_spectrum_list[i].count):

                        self.science_spectrum_list[i].add_wavelength(
                            wave=wave[i])

                    else:

                        err_msg = 'The wavelength provided has a different ' +\
                            'size to that of the extracted science spectrum.'
                        self.logger.critical(err_msg)
                        raise RuntimeError(err_msg)

                self.science_wavelength_calibrated = True

            else:

                err_msg = 'science data is not available, wavelength ' +\
                    'cannot be added.'
                self.logger.critical(err_msg)
                raise RuntimeError(err_msg)

        if 'standard' in stype_split:

            if self.standard_data_available:

                if len(wave[0]) == len(self.standard_spectrum_list[0].count):

                    self.standard_spectrum_list[0].add_wavelength(wave=wave[0])

                else:

                    err_msg = 'The wavelength provided is of a different ' +\
                        'size to that of the extracted standard spectrum.'
                    self.logger.critical(err_msg)
                    raise RuntimeError(err_msg)

                self.standard_wavelength_calibrated = True

            else:

                err_msg = 'standard data is not available, wavelength ' +\
                    'cannot be added.'
                self.logger.critical(err_msg)
                raise RuntimeError(err_msg)

    def add_wavelength_resampled(self,
                                 wave_resampled,
                                 spec_id=None,
                                 stype='science+standard'):
        '''
        Three combinations of wave and spec_id shapes are accepted.

        +-----------+-----------------+
        | Parameter |       Size      |
        +-----------+-----+-----+-----+
        | wave      |  1  |  1  |  N  |
        +-----------+-----+-----+-----+
        | spec_id   |  1  |  N  |  N  |
        +-----------+-----+-----+-----+

        Parameters
        ----------
        wave_resampled:
            The wavelength of the resampled spectrum.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if type(wave_resampled) == np.ndarray:

            wave_resampled = [wave_resampled]

        elif type(wave_resampled) == list:

            pass

        else:

            err_msg = 'Please provide a numpy array or a list of them.'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_data_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(wave_resampled) == len(spec_id):

                    wave_resampled = {
                        spec_id[i]: wave_resampled[i]
                        for i in range(len(spec_id))
                    }

                elif len(wave_resampled) == 1:

                    wave_resampled = {
                        spec_id[i]: wave_resampled[0]
                        for i in range(len(spec_id))
                    }

                else:

                    error_msg = 'wave must be the same length of shape ' +\
                        'as spec_id.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

                for i in spec_id:

                    if len(wave_resampled[i]) == len(
                            self.science_spectrum_list[i].count):

                        self.science_spectrum_list[i].add_wavelength_resampled(
                            wave_resampled=wave_resampled[i])

                    else:

                        err_msg = 'The wavelength provided has a different ' +\
                            'size to that of the extracted science spectrum.'
                        self.logger.critical(err_msg)
                        raise RuntimeError(err_msg)

                self.science_wavelength_resampled = True

            else:

                err_msg = 'science data is not available, ' +\
                    'wavelength_resampled cannot be added.'
                self.logger.critical(err_msg)
                raise RuntimeError(err_msg)

        if 'standard' in stype_split:

            if self.standard_data_available:

                if len(wave_resampled[0]) == len(
                        self.standard_spectrum_list[0].count):

                    self.standard_spectrum_list[0].add_wavelength_resampled(
                        wave_resampled=wave_resampled[0])

                else:

                    err_msg = 'The wavelength provided is of a different ' +\
                        'size to that of the extracted standard spectrum.'
                    self.logger.critical(err_msg)
                    raise RuntimeError(err_msg)

                self.standard_wavelength_resampled_calibrated = True

            else:

                err_msg = 'standard data is not available, ' +\
                    'wavelength_resampled cannot be added.'
                self.logger.critical(err_msg)
                raise RuntimeError(err_msg)

    def add_spec(self,
                 count,
                 spec_id=None,
                 count_err=None,
                 count_sky=None,
                 stype='science+standard'):
        '''
        Parameters
        ----------
        count: 1-d array
            The summed count at each column about the trace.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        count_err: 1-d array (Default: None)
            the uncertainties of the count values
        count_sky: 1-d array (Default: None)
            The integrated sky values along each column, suitable for
            subtracting from the output of ap_extract
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if type(count) == np.ndarray:

            count = [count]

        elif type(count) == list:

            pass

        else:

            err_msg = 'Please provide a numpy array or a list of them.'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        if count_err is not None:
            if type(count_err) == np.ndarray:

                count_err = [count_err]

            elif type(count_err) == list:

                pass

            else:

                err_msg = 'Please provide a numpy array or a list of them.'
                self.logger.critical(err_msg)
                raise TypeError(err_msg)

        else:

            count_err = [None]

        if count_sky is not None:

            if type(count_sky) == np.ndarray:

                count_sky = [count_sky]

            elif type(count_sky) == list:

                pass

            else:

                err_msg = 'Please provide a numpy array or a list of them.'
                self.logger.critical(err_msg)
                raise TypeError(err_msg)

        else:

            count_sky = [None]

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    for i in spec_id:

                        if i not in list(self.science_spectrum_list.keys()):

                            self.add_science_spectrum1D(i)

                            self.logger.warning(
                                'The given spec_id, {}, does not exist. A new '
                                'spectrum1D is created. Please check you are '
                                'providing the correct spec_id.'.format(
                                    spec_id))

                        else:

                            pass

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            # Check the sizes of the count and spec_id and convert count
            # into a dictionary
            if len(count) == len(spec_id):

                count = {spec_id[i]: count[i] for i in range(len(spec_id))}

            elif len(count) == 1:

                count = {spec_id[i]: count[0] for i in range(len(spec_id))}

            else:

                error_msg = 'count must be the same length of shape ' +\
                    'as spec_id, or of size 1.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            # Check the sizes of the count_sky and spec_id and convert
            # count_sky into a dictionary
            if count_sky is [None]:

                count_sky = {spec_id[i]: None for i in range(len(spec_id))}

            elif len(count_sky) == len(spec_id):

                count_sky = {
                    spec_id[i]: count_sky[i]
                    for i in range(len(spec_id))
                }

            elif len(count_sky) == 1:

                count_sky = {
                    spec_id[i]: count_sky[0]
                    for i in range(len(spec_id))
                }

            else:

                error_msg = 'count_sky must be the same length of shape ' +\
                    'as spec_id, or of size 1.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            # Check the sizes of the count_err and spec_id and convert
            # count_err into a dictionary
            if count_err is [None]:

                count_err = {spec_id[i]: None for i in range(len(spec_id))}

            elif len(count_err) == len(spec_id):

                count_err = {
                    spec_id[i]: count_err[i]
                    for i in range(len(spec_id))
                }

            elif len(count_err) == 1:

                count_err = {
                    spec_id[i]: count_err[0]
                    for i in range(len(spec_id))
                }

            else:

                error_msg = 'count_err must be the same length of shape ' +\
                    'as spec_id, or of size 1.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            for i in spec_id:

                self.science_spectrum_list[i].add_count(count=count[i],
                                                        count_err=count_err[i],
                                                        count_sky=count_sky[i])

            self.science_data_available = True

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].add_count(count=count[0],
                                                     count_err=count_err[0],
                                                     count_sky=count_sky[0])

            self.standard_data_available = True

    def add_arc_spec(self, arc_spec, spec_id=None, stype='science+standard'):
        '''
        Parameters
        ----------
        arc_spec: 1-d array
            The count of the summed 1D arc spec
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if type(arc_spec) == np.ndarray:

            arc_spec = [arc_spec]

        elif type(arc_spec) == list:

            pass

        else:

            err_msg = 'Please provide a numpy array or a list of them.'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_data_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        for i in spec_id:

                            if i not in list(
                                    self.science_spectrum_list.keys()):

                                self.add_science_spectrum1D(i)

                                self.logger.warning(
                                    'The given spec_id, {}, does not '
                                    'exist. A new spectrum1D is created. '
                                    'Please check you are providing the '
                                    'correct spec_id.'.format(spec_id))

                            else:

                                pass

                    else:

                        pass

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(arc_spec) == len(spec_id):

                    arc_spec = {
                        spec_id[i]: arc_spec[i]
                        for i in range(len(spec_id))
                    }

                elif len(arc_spec) == 1:

                    arc_spec = {
                        spec_id[i]: arc_spec[0]
                        for i in range(len(spec_id))
                    }

                else:

                    error_msg = 'arc_spec must be the same length of shape ' +\
                        'as spec_id.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

                for i in spec_id:

                    self.science_spectrum_list[i].add_arc_spec(
                        arc_spec=arc_spec[i])

                self.science_arc_spec_available = True

        if 'standard' in stype_split:

            if self.standard_data_available:

                self.standard_spectrum_list[0].add_arc_spec(
                    arc_spec=arc_spec[0])

                self.standard_arc_spec_available = True

    def add_arc_lines(self, peaks, spec_id=None, stype='science+standard'):
        '''
        Parameters
        ----------
        peaks: list of list or list of arrays
            The pixel locations of the arc lines. Multiple traces of the arc
            can be provided as list of list or list of arrays.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if type(peaks) == np.ndarray:

            peaks = [peaks]

        elif type(peaks) == list:

            pass

        else:

            err_msg = 'Please provide a numpy array or a list of them.'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    for i in spec_id:

                        if i not in list(self.science_spectrum_list.keys()):

                            self.add_science_spectrum1D(i)

                            self.logger.warning(
                                'The given spec_id, {}, does not '
                                'exist. A new spectrum1D is created. '
                                'Please check you are providing the '
                                'correct spec_id.'.format(spec_id))

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            # Check the sizes of the wave and spec_id and convert wave
            # into a dictionary
            if len(peaks) == len(spec_id):

                peaks = {spec_id[i]: peaks[i] for i in range(len(spec_id))}

            elif len(peaks) == 1:

                peaks = {i: peaks[0] for i in spec_id}

            else:

                error_msg = 'peaks must be the same length of shape ' +\
                    'as spec_id.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            for i in spec_id:

                self.science_spectrum_list[i].add_peaks(peaks=peaks[i])

            self.science_arc_lines_available = True

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].add_peaks(peaks=peaks[0])

            self.standard_arc_lines_available = True

    def add_trace(self,
                  trace,
                  trace_sigma,
                  spec_id=None,
                  pixel_list=None,
                  stype='science+standard'):
        '''
        Parameters
        ----------
        trace: list or numpy.ndarray (N)
            The spatial pixel value (can be sub-pixel) of the trace at each
            spectral position.
        trace_sigma: list or numpy.ndarray (N)
            Standard deviation of the Gaussian profile of a trace
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        pixel_list: list or numpy.narray
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(len(spec.trace[0])), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if type(trace) == np.ndarray:

            trace = [trace]

        elif type(trace) == list:

            pass

        else:

            err_msg = 'Please provide a numpy array or a list of them.'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        if type(trace_sigma) == np.ndarray:

            trace_sigma = [trace_sigma]

        elif type(trace_sigma) == list:

            pass

        else:

            err_msg = 'Please provide a numpy array or a list of them.'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    for i in spec_id:

                        if i not in list(self.science_spectrum_list.keys()):

                            self.add_science_spectrum1D(i)

                            self.logger.warning(
                                'The given spec_id, {}, does not '
                                'exist. A new spectrum1D is created. '
                                'Please check you are providing the '
                                'correct spec_id.'.format(spec_id))

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            # Check the sizes of the wave and spec_id and convert wave
            # into a dictionary
            if len(trace) == len(spec_id):

                trace = {spec_id[i]: trace[i] for i in range(len(spec_id))}

            elif len(trace) == 1:

                trace = {spec_id[i]: trace[0] for i in range(len(spec_id))}

            else:

                error_msg = 'trace must be the same length of shape ' +\
                    'as spec_id.'
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

            # Check the sizes of the wave and spec_id and convert wave
            # into a dictionary
            if len(trace_sigma) == len(spec_id):

                trace_sigma = {
                    spec_id[i]: trace_sigma[i]
                    for i in range(len(spec_id))
                }

            elif len(trace_sigma) == 1:

                trace_sigma = {
                    spec_id[i]: trace_sigma[0]
                    for i in range(len(spec_id))
                }

            else:

                error_msg = 'wave must be the same length of shape ' +\
                    'as spec_id.'
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

            for i in spec_id:

                self.science_spectrum_list[i].add_trace(
                    trace=trace[i],
                    trace_sigma=trace_sigma[i],
                    pixel_list=pixel_list)

            self.science_trace_available = True

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].add_trace(
                trace=trace[0],
                trace_sigma=trace_sigma[0],
                pixel_list=pixel_list)

            self.standard_trace_available = True

    def add_fit_coeff(self,
                      fit_coeff,
                      fit_type='poly',
                      spec_id=None,
                      stype='science+standard'):
        '''
        Parameters
        ----------
        fit_coeff: list or numpy array, or a list of them
            Polynomial fit coefficients.
        fit_type: str or list of str
            Strings starting with 'poly', 'leg' or 'cheb' for polynomial,
            legendre and chebyshev fits. Case insensitive.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if type(fit_coeff) == np.ndarray:

            fit_coeff = [fit_coeff]

        elif all(isinstance(i, list) for i in fit_coeff):

            pass

        elif isinstance(fit_coeff, list):

            if isinstance(fit_coeff[0], (list, np.ndarray)):

                pass

            elif isinstance(fit_coeff[0], (int, float)):

                fit_coeff = [fit_coeff]

            else:

                pass

        elif all(isinstance(i, np.ndarray) for i in fit_coeff):

            pass

        else:

            err_msg = 'Please provide a numpy array or a list of them.'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        if type(fit_type) == str:

            fit_type = [fit_type]

        elif all(isinstance(i, (str, list, np.ndarray)) for i in fit_type):

            for i, ft in enumerate(fit_type):

                if isinstance(ft, (list, np.ndarray)):

                    fit_type[i] = ft[0]

        else:

            err_msg = 'Please provide a numpy array or a list of them.'
            self.logger.critical(err_msg)
            raise TypeError(err_msg)

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_data_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(fit_coeff) == len(spec_id):

                    fit_coeff = {
                        spec_id[i]: fit_coeff[i]
                        for i in range(len(spec_id))
                    }

                elif len(fit_coeff) == 1:

                    fit_coeff = {
                        spec_id[i]: fit_coeff[0]
                        for i in range(len(spec_id))
                    }

                else:

                    error_msg = 'fit_coeff must be the same length of ' +\
                        'shape as spec_id.'
                    self.logger.critical(error_msg)
                    raise RuntimeError(error_msg)

                # Check the sizes of the wave and spec_id and convert wave
                # into a dictionary
                if len(fit_type) == len(spec_id):

                    fit_type = {
                        spec_id[i]: fit_type[i]
                        for i in range(len(spec_id))
                    }

                elif len(fit_type) == 1:

                    fit_type = {
                        spec_id[i]: fit_type[0]
                        for i in range(len(spec_id))
                    }

                else:

                    error_msg = 'wave must be the same length of shape ' +\
                        'as spec_id.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

                for i in spec_id:

                    self.science_spectrum_list[i].add_fit_coeff(
                        fit_coeff=fit_coeff[i])
                    self.science_spectrum_list[i].add_fit_type(
                        fit_type=fit_type[i])

            self.science_wavecal_polynomial_available = True

        if 'standard' in stype_split:

            if self.standard_data_available:

                self.standard_spectrum_list[0].add_fit_coeff(
                    fit_coeff=fit_coeff[0])
                self.standard_spectrum_list[0].add_fit_type(
                    fit_type=fit_type[0])

            self.standard_wavecal_polynomial_available = True

    def from_twodspec(self, twodspec, spec_id=None, stype='science+standard'):
        '''
        To add a TwoDSpec object or numpy array to provide the traces, line
        spread function of the traces, optionally the pixel values
        correcponding to the traces. If the arc is provided, the saxis and flip
        properties of the TwoDSpec will be applied to the arc, and then
        the spec_mask and the spatial_mask from the TwoDSpec object will be
        applied.

        Parameters
        ----------
        twodspec: TwoDSpec object
            TwoDSpec of the science image containin the trace(s) and
            trace_sigma(s).
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(twodspec.spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(twodspec.spectrum_list.keys())

            # reference the spectrum1D to the WavelengthCalibration
            for i in spec_id:

                self.add_science_spectrum1D(i)
                self.science_wavecal[i] = WavelengthCalibration(
                    verbose=self.verbose,
                    logger_name=self.logger_name,
                    log_level=self.log_level,
                    log_file_folder=self.log_file_folder,
                    log_file_name=self.log_file_name)

                # By reference
                self.science_wavecal[i].from_spectrum1D(
                    twodspec.spectrum_list[i])
                self.science_spectrum_list[i] =\
                    self.science_wavecal[i].spectrum1D

            self.science_data_available = True
            self.science_arc_available = True
            self.science_arc_spec_available = True

        if 'standard' in stype_split:

            # By reference
            self.standard_wavecal = WavelengthCalibration(
                verbose=self.verbose,
                logger_name=self.logger_name,
                log_level=self.log_level,
                log_file_folder=self.log_file_folder,
                log_file_name=self.log_file_name)
            self.standard_wavecal.from_spectrum1D(twodspec.spectrum_list[0])
            self.fluxcal.from_spectrum1D(twodspec.spectrum_list[0])
            self.standard_spectrum_list[0] = self.standard_wavecal.spectrum1D

            self.standard_data_available = True
            self.standard_arc_available = True
            self.standard_arc_spec_available = True

    def find_arc_lines(self,
                       spec_id=None,
                       background=None,
                       percentile=2.,
                       prominence=5.,
                       top_n_peaks=None,
                       distance=5.,
                       refine=True,
                       refine_window_width=5,
                       display=False,
                       width=1280,
                       height=720,
                       return_jsonstring=False,
                       renderer='default',
                       save_fig=False,
                       fig_type='iframe+png',
                       filename=None,
                       open_iframe=False,
                       stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        background: int or None (Default: None)
            User-supplied estimated background level
        percentile: float (Default: 2.)
            The percentile of the flux to be used as the estimate of the
            background sky level to the first order. Only used if background
            is None. [Count]
        prominence: float (Default: 5.)
            The minimum prominence to be considered as a peak (normalised)
        distance: float (Default: 5.)
            Minimum separation between peaks
        refine: bool (Default: True)
            Set to true to fit a gaussian to get the peak at sub-pixel
            precision
        refine_window_width: int or float (Default: 5)
            The number of pixels (on each side of the existing peaks) to be
            fitted with gaussian profiles over.
        display: bool (Default: False)
            Set to True to display disgnostic plot.
        renderer: str (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs
        return_jsonstring: bool (Default: False)
            set to True to return  JSON-string that can be rendered by Plotly
            in any support language.
        renderer: str (Default: 'default')
            plotly renderer options.
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        Returns
        -------
        JSON strings if return_jsonstring is set to True

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_arc_spec_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    print(list(self.science_spectrum_list.keys()))
                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].find_arc_lines(
                        background=background,
                        percentile=percentile,
                        prominence=prominence,
                        top_n_peaks=top_n_peaks,
                        distance=distance,
                        refine=refine,
                        refine_window_width=refine_window_width,
                        display=display,
                        renderer=renderer,
                        width=width,
                        height=height,
                        return_jsonstring=return_jsonstring,
                        save_fig=save_fig,
                        fig_type=fig_type,
                        filename=filename,
                        open_iframe=open_iframe)

                self.science_arc_lines_available = True

            else:

                self.logger.warning('Science arc spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_arc_spec_available:

                self.standard_wavecal.find_arc_lines(
                    background=background,
                    percentile=percentile,
                    prominence=prominence,
                    top_n_peaks=top_n_peaks,
                    distance=distance,
                    refine=refine,
                    refine_window_width=refine_window_width,
                    display=display,
                    renderer=renderer,
                    width=width,
                    height=height,
                    return_jsonstring=return_jsonstring,
                    save_fig=save_fig,
                    fig_type=fig_type,
                    filename=filename,
                    open_iframe=open_iframe)

                self.standard_arc_lines_available = True

            else:

                self.logger.warning(
                    'Standard arc spectrum/a are not imported.')

    def initialise_calibrator(self,
                              spec_id=None,
                              peaks=None,
                              arc_spec=None,
                              stype='science+standard'):
        '''
        If the peaks were found with find_arc_lines(), peaks and spectrum can
        be None.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        peaks: list, numpy.ndarray or None (Default: None)
            The pixel values of the peaks (start from zero)
        spectrum: list, numpy.ndarray or None (Default: None)
            The spectral intensity as a function of pixel.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_arc_lines_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        for i in spec_id:

                            if i not in list(
                                    self.science_spectrum_list.keys()):

                                self.add_science_spectrum1D(i)

                                self.logger.warning(
                                    'The given spec_id, {}, does not '
                                    'exist. A new spectrum1D is created. '
                                    'Please check you are providing the '
                                    'correct spec_id.'.format(spec_id))

                            else:

                                pass

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].from_spectrum1D(
                        self.science_spectrum_list[i])
                    self.science_wavecal[i].initialise_calibrator(
                        peaks=peaks, arc_spec=arc_spec)
                    self.science_wavecal[i].set_calibrator_properties()
                    self.science_wavecal[i].set_hough_properties()
                    self.science_wavecal[i].set_ransac_properties()

            else:

                self.logger.warning('Science arc lines are not available.')

        if 'standard' in stype_split:

            if self.standard_arc_lines_available:

                self.standard_wavecal.from_spectrum1D(
                    self.standard_spectrum_list[0])
                self.standard_wavecal.initialise_calibrator(peaks=peaks,
                                                            arc_spec=arc_spec)
                self.standard_wavecal.set_calibrator_properties()
                self.standard_wavecal.set_hough_properties()
                self.standard_wavecal.set_ransac_properties()

            else:

                self.logger.warning('Standard arc lines are not available.')

    def set_calibrator_properties(self,
                                  spec_id=None,
                                  num_pix=None,
                                  pixel_list=None,
                                  plotting_library='plotly',
                                  logger_name='Calibrator',
                                  log_level='info',
                                  stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        num_pix: int (Default: None)
            The number of pixels in the dispersion direction
        pixel_list: list or numpy array (Default: None)
            The pixel position of the trace in the dispersion direction.
            This should be provided if you wish to override the default
            range(num_pix), for example, in the case of accounting
            for chip gaps (10 pixels) in a 3-CCD setting, you should provide
            [0,1,2,...90, 100,101,...190, 200,201,...290]
        plotting_library : str (Default: 'plotly')
            Choose between matplotlib and plotly.
        logger_name: str (Default: 'Calibrator')
            This will set the name of the logger, if the name is used already,
            it will reference to the existing logger. This will be the
            first part of the default log file name unless log_file_name is
            provided.
        log_level : str (Default: 'info')
            Choose {critical, error, warning, info, debug, notset}.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_arc_lines_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].set_calibrator_properties(
                        num_pix=num_pix,
                        pixel_list=pixel_list,
                        plotting_library=plotting_library,
                        logger_name=logger_name,
                        log_level=log_level)

            else:

                self.logger.warning('Science arc lines are not available.')

        if 'standard' in stype_split:

            if self.standard_arc_lines_available:

                self.standard_wavecal.set_calibrator_properties(
                    num_pix=num_pix,
                    pixel_list=pixel_list,
                    plotting_library=plotting_library,
                    logger_name=logger_name,
                    log_level=log_level)

            else:

                self.logger.warning('Standard arc lines are not available.')

    def set_hough_properties(self,
                             spec_id=None,
                             num_slopes=5000,
                             xbins=200,
                             ybins=200,
                             min_wavelength=3500,
                             max_wavelength=8500,
                             range_tolerance=500,
                             linearity_tolerance=100,
                             stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        num_slopes: int (Default: 5000)
            Number of slopes to consider during Hough transform
        xbins: int (Default: 200)
            Number of bins for Hough accumulation
        ybins: int (Default: 200)
            Number of bins for Hough accumulation
        min_wavelength: float (Default: 3500)
            Minimum wavelength of the spectrum.
        max_wavelength: float (Default: 8500)
            Maximum wavelength of the spectrum.
        range_tolerance: float (Default: 500)
            Estimation of the error on the provided spectral range
            e.g. 3000-5000 with tolerance 500 will search for
            solutions that may satisfy 2500-5500
        linearity_tolerance: float (Default: 100)
            A toleranceold (Ansgtroms) which defines some padding around the
            range tolerance to allow for non-linearity. This should be the
            maximum expected excursion from linearity.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_arc_lines_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].set_hough_properties(
                        num_slopes=num_slopes,
                        xbins=xbins,
                        ybins=ybins,
                        min_wavelength=min_wavelength,
                        max_wavelength=max_wavelength,
                        range_tolerance=range_tolerance,
                        linearity_tolerance=linearity_tolerance)

            else:

                self.logger.warning('Science arc lines are not available.')

        if 'standard' in stype_split:

            if self.standard_arc_lines_available:

                self.standard_wavecal.set_hough_properties(
                    num_slopes=num_slopes,
                    xbins=xbins,
                    ybins=ybins,
                    min_wavelength=min_wavelength,
                    max_wavelength=max_wavelength,
                    range_tolerance=range_tolerance,
                    linearity_tolerance=linearity_tolerance)

            else:

                self.logger.warning('Standard arc lines are not available.')

    def set_ransac_properties(self,
                              spec_id=None,
                              sample_size=5,
                              top_n_candidate=5,
                              linear=True,
                              filter_close=False,
                              ransac_tolerance=5,
                              candidate_weighted=True,
                              hough_weight=1.0,
                              stype='science+standard'):
        '''
        Configure the Calibrator. This may require some manual twiddling before
        the calibrator can work efficiently. However, in theory, a large
        max_tries in fit() should provide a good solution in the expense of
        performance (minutes instead of seconds).

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        sample_size: int (Default: 5)
            Number of pixel-wavelength hough pairs to be used for each arc line
            being picked.
        top_n_candidate: int (Default: 5)
            Top ranked lines to be fitted.
        linear: bool (Default: True)
            True to use the hough transformed gradient, otherwise, use the
            known polynomial.
        filter_close: bool (Default: False)
            Remove the pairs that are out of bounds in the hough space.
        ransac_tolerance: float (Default: 1)
            The distance criteria  (Angstroms) to be considered an inlier to a
            fit. This should be close to the size of the expected residuals on
            the final fit (e.g. 1A is typical)
        candidate_weighted: bool (Default: True)
            Set to True to down-weight pairs that are far from the fit.
        hough_weight: float or None (Default: 1.0)
            Set to use the hough space to weigh the fit. The theoretical
            optimal weighting is unclear. The larger the value, the heavily it
            relies on the overdensity in the hough space for a good fit.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_arc_lines_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].set_ransac_properties(
                        sample_size=sample_size,
                        top_n_candidate=top_n_candidate,
                        linear=linear,
                        filter_close=filter_close,
                        ransac_tolerance=ransac_tolerance,
                        candidate_weighted=candidate_weighted,
                        hough_weight=hough_weight)

            else:

                self.logger.warning('Science arc lines are not available.')

        if 'standard' in stype_split:

            if self.standard_arc_lines_available:

                self.standard_wavecal.set_ransac_properties(
                    sample_size=sample_size,
                    top_n_candidate=top_n_candidate,
                    linear=linear,
                    filter_close=filter_close,
                    ransac_tolerance=ransac_tolerance,
                    candidate_weighted=candidate_weighted,
                    hough_weight=hough_weight)

            else:

                self.logger.warning('Standard arc lines are not available.')

    def set_known_pairs(self,
                        pix=None,
                        wave=None,
                        spec_id=None,
                        stype='science+standard'):
        '''
        Parameters
        ----------
        pix : numeric value, list or numpy 1D array (N) (Default: None)
            Any pixel value, can be outside the detector chip and
            serve purely as anchor points.
        wave : numeric value, list or numpy 1D array (N) (Default: None)
            The matching wavelength for each of the pix.
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_arc_lines_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].set_known_pairs(pix=pix, wave=wave)

            else:

                self.logger.warning('Science arc lines are not available.')

        if 'standard' in stype_split:

            if self.standard_arc_lines_available:

                self.standard_wavecal.set_known_pairs(pix=pix, wave=wave)

            else:

                self.logger.warning('Standard arc lines are not available.')

    def add_user_atlas(self,
                       elements,
                       wavelengths,
                       spec_id=None,
                       intensities=None,
                       candidate_tolerance=10.,
                       constrain_poly=False,
                       vacuum=False,
                       pressure=101325,
                       temperature=273.15,
                       relative_humidity=0.,
                       stype='science+standard'):
        '''
        Append the user supplied arc lines to the calibrator.

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements : list
            Element (required). Preferably a standard (i.e. periodic table)
            name for convenience with built-in atlases
        wavelengths : list or None (Default: None)
            Wavelength to add (Angstrom)
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        intensities : list or None (Default: None)
            Relative line intensities
        candidate_tolerance: float (Default: 10.)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: bool (Default: False)
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: bool (Default: False)
            Set to true to convert the input wavelength to air-wavelengths
            based on the given pressure, temperature and humidity.
        pressure: float (Default: 101325)
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float (Default: 273.15)
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float (Default: 0.)
            In percentage.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        if pressure is None:
            pressure = 101325.0
            self.logger.warning('Pressure is not provided, set to 1 unit of '
                                'standard atmosphere.')

        if temperature is None:
            temperature = 273.15
            self.logger.warning(
                'Temperature is not provided, set to 0 degrees '
                'Celsius.')

        if relative_humidity is None:
            relative_humidity = 0.
            self.logger.warning(
                'Relative humidity is not provided, set to 0%.')

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_arc_lines_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].add_user_atlas(
                        elements=elements,
                        wavelengths=wavelengths,
                        intensities=intensities,
                        candidate_tolerance=candidate_tolerance,
                        constrain_poly=constrain_poly,
                        vacuum=vacuum,
                        pressure=pressure,
                        temperature=temperature,
                        relative_humidity=relative_humidity)

                self.science_atlas_available = True

            else:

                self.logger.warning('Science arc lines are not available.')

        if 'standard' in stype_split:

            if self.standard_data_available:

                self.standard_wavecal.add_user_atlas(
                    elements=elements,
                    wavelengths=wavelengths,
                    intensities=intensities,
                    candidate_tolerance=candidate_tolerance,
                    constrain_poly=constrain_poly,
                    vacuum=vacuum,
                    pressure=pressure,
                    temperature=temperature,
                    relative_humidity=relative_humidity)

                self.standard_atlas_available = True

            else:

                self.logger.warning('Standard arc lines are not available.')

    def add_atlas(self,
                  elements,
                  spec_id=None,
                  min_atlas_wavelength=1000.,
                  max_atlas_wavelength=30000.,
                  min_intensity=10.,
                  min_distance=10.,
                  candidate_tolerance=10.,
                  constrain_poly=False,
                  vacuum=False,
                  pressure=101325.,
                  temperature=273.15,
                  relative_humidity=0,
                  stype='science+standard'):
        '''
        Parameters
        ----------
        elements: str or list of strings
            Chemical symbol, case insensitive
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        min_atlas_wavelength: float (Default: None)
            Minimum wavelength of the arc lines.
        max_atlas_wavelength: float (Default: None)
            Maximum wavelength of the arc lines.
        min_intensity: float (Default: 10.)
            Minimum intensity of the arc lines. Refer to NIST for the
            intensity.
        min_distance: float (Default: 10.)
            Minimum separation between neighbouring arc lines.
        candidate_tolerance: float (Default: 10.)
            toleranceold  (Angstroms) for considering a point to be an inlier
            during candidate peak/line selection. This should be reasonable
            small as we want to search for candidate points which are
            *locally* linear.
        constrain_poly: bool (Default: False)
            Apply a polygonal constraint on possible peak/atlas pairs
        vacuum: bool (Default: False)
            Set to true to convert the input wavelength to air-wavelengths
            based on the given pressure, temperature and humidity.
        pressure: float (Default: 101325)
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float (Default: 273.15)
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float (Default: 0)
            In percentage.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_wavecal[i].add_atlas(
                    elements=elements,
                    min_atlas_wavelength=min_atlas_wavelength,
                    max_atlas_wavelength=max_atlas_wavelength,
                    min_intensity=min_intensity,
                    min_distance=min_distance,
                    candidate_tolerance=candidate_tolerance,
                    constrain_poly=constrain_poly,
                    vacuum=vacuum,
                    pressure=pressure,
                    temperature=temperature,
                    relative_humidity=relative_humidity)

            self.science_atlas_available = True

        if 'standard' in stype_split:

            self.standard_wavecal.add_atlas(
                elements=elements,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=min_intensity,
                min_distance=min_distance,
                candidate_tolerance=candidate_tolerance,
                constrain_poly=constrain_poly,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity)

            self.standard_atlas_available = True

    def remove_atlas_lines_range(self,
                                 wavelength,
                                 tolerance=10.,
                                 spec_id=None,
                                 stype='science+standard'):
        '''
        Remove arc lines within a certain wavelength range.

        Parameters
        ----------
        wavelength: float
            Wavelength to remove (Angstrom)
        tolerance: float
            Tolerance around this wavelength where atlas lines will be removed
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_atlas_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].remove_atlas_lines_range(
                        wavelength, tolerance)

            else:

                self.logger.warning('Science atlas is not available.')

        if 'standard' in stype_split:

            if self.standard_atlas_available:

                self.standard_wavecal.remove_atlas_lines_range(
                    wavelength, tolerance)

            else:

                self.logger.warning('Standard atlas is not available.')

    def clear_atlas(self, spec_id=None, stype='science+standard'):
        '''
        Remove all the atlas lines from the calibrator.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_atlas_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].clear_atlas()

                self.science_atlas_available = False

            else:

                self.logger.warning('Science atlas is not available.')

        if 'standard' in stype_split:

            if self.standard_atlas_available:

                self.standard_wavecal.clear_atlas()

                self.standard_atlas_available = False

            else:

                self.logger.warning('Standard atlas is not available.')

    def list_atlas(self, spec_id=None, stype='science+standard'):
        '''
        Remove all the atlas lines from the calibrator.

        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_atlas_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].list_atlas()

            else:

                self.logger.warning('Science atlas is not available.')

        if 'standard' in stype_split:

            if self.standard_atlas_available:

                self.standard_wavecal.list_atlas()

            else:

                self.logger.warning('Standard atlas is not available.')

    def do_hough_transform(self,
                           spec_id=None,
                           brute_force=True,
                           stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        brute_force: bool (Default: Ture)
            Set to true to compute the gradient and intercept between
            every two data points
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_atlas_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].do_hough_transform(
                        brute_force=brute_force)

                self.science_hough_pairs_available = True

            else:

                self.logger.warning('Science atlas is not available.')

        if 'standard' in stype_split:

            if self.standard_atlas_available:

                self.standard_wavecal.do_hough_transform(
                    brute_force=brute_force)

                self.standard_hough_pairs_available = True

            else:

                self.logger.warning('Standard atlas is not available.')

    def fit(self,
            spec_id=None,
            max_tries=5000,
            fit_deg=4,
            fit_coeff=None,
            fit_tolerance=10.,
            fit_type='poly',
            brute_force=False,
            progress=True,
            display=False,
            save_fig=False,
            filename=None,
            stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        max_tries: int (Default: 5000)
            Number of trials of polynomial fitting.
        fit_deg: int (Default: 4)
            The degree of the polynomial to be fitted.
        fit_coeff: list (Default: None)
            Set the baseline of the least square fit. If no fits outform this
            set of polynomial coefficients, this will be used as the best fit.
        fit_tolerance: float (Default: 10)
            Sets a tolerance on whether a fit found by RANSAC is considered
            acceptable
        fit_type: str (Default: 'poly')
            One of 'poly', 'legendre' or 'chebyshev'
        brute_force: bool (Default: False)
            Set to True to try all possible combination in the given parameter
            space
        progress: bool (Default: True)
            Set to show the progress using tdqm (if imported).
        display: bool (Default: False)
            Set to show diagnostic plot.
        save_fig: bool (Default: False)
            Set to save figure.
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_hough_pairs_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.science_wavecal[i].fit(max_tries=max_tries,
                                                fit_deg=fit_deg,
                                                fit_coeff=fit_coeff,
                                                fit_tolerance=fit_tolerance,
                                                fit_type=fit_type,
                                                brute_force=brute_force,
                                                progress=progress,
                                                display=display,
                                                save_fig=save_fig,
                                                filename=filename)

                self.science_wavecal_polynomial_available = True

            else:

                self.logger.warning('Science hough pairs are not available.')

        if 'standard' in stype_split:

            if self.standard_hough_pairs_available:

                self.standard_wavecal.fit(max_tries=max_tries,
                                          fit_deg=fit_deg,
                                          fit_coeff=fit_coeff,
                                          fit_tolerance=fit_tolerance,
                                          fit_type=fit_type,
                                          brute_force=brute_force,
                                          progress=progress,
                                          display=display,
                                          save_fig=save_fig,
                                          filename=filename)

                self.standard_wavecal_polynomial_available = True

            else:

                self.logger.warning('Standard spectrum/a are not imported.')

    def refine_fit(self,
                   spec_id=None,
                   fit_coeff=None,
                   n_delta=None,
                   refine=True,
                   tolerance=10.,
                   method='Nelder-Mead',
                   convergence=1e-6,
                   robust_refit=True,
                   fit_deg=None,
                   display=False,
                   save_fig=False,
                   filename=None,
                   stype='science+standard'):
        '''
        ** EXPERIMENTAL, as of 17 Jan 2021 **
        Refine the fitted solution with a minimisation method as provided by
        scipy.optimize.minimize().

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        fit_coeff: list or None (Default: None)
            List of polynomial fit coefficients.
        n_delta: int (Default: None)
            The number of the highest polynomial order to be adjusted
        refine: bool (Default: True)
            Set to True to refine solution.
        tolerance : float (Default: 10.)
            Absolute difference between fit and model in the unit of nm.
        method: str (Default: 'Nelder-Mead')
            scipy.optimize.minimize method.
        convergence: float (Default: 1e-6)
            scipy.optimize.minimize tol.
        robust_refit: bool (Default: True)
            Set to True to fit all the detected peaks with the given polynomial
            solution.
        fit_deg: int (Default: length of the input coefficients - 1)
            Order of polynomial fit with all the detected peaks.
        display: bool (Default: False)
            Set to show diagnostic plot.
        save_fig: bool (Default: False)
            Set to save figure.
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_wavecal_polynomial_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)
                else:

                    # if spec_id is None, calibrators are initialised to all
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    if fit_coeff is None:

                        fit_coeff = self.science_wavecal[
                            i].spectrum1D.calibrator.fit_coeff

                    self.science_wavecal[i].refine_fit(
                        fit_coeff=fit_coeff,
                        n_delta=n_delta,
                        refine=refine,
                        tolerance=tolerance,
                        method=method,
                        convergence=convergence,
                        robust_refit=robust_refit,
                        fit_deg=fit_deg,
                        display=display,
                        save_fig=save_fig,
                        filename=filename)

            else:

                self.logger.warning('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_wavecal_polynomial_available:

                if fit_coeff is None:

                    fit_coeff = self.standard_wavecal[
                        0].spectrum1D.calibrator.fit_coeff

                self.standard_wavecal.refine_fit(fit_coeff=fit_coeff,
                                                 n_delta=n_delta,
                                                 refine=refine,
                                                 tolerance=tolerance,
                                                 method=method,
                                                 convergence=convergence,
                                                 robust_refit=robust_refit,
                                                 fit_deg=fit_deg,
                                                 display=display,
                                                 save_fig=save_fig,
                                                 filename=filename)

            else:

                self.logger.warning('Standard spectrum/a are not imported.')

    def apply_wavelength_calibration(self,
                                     spec_id=None,
                                     wave_start=None,
                                     wave_end=None,
                                     wave_bin=None,
                                     stype='science+standard'):
        '''
        Apply the wavelength calibration. Because the polynomial fit can run
        away at the two ends, the default minimum and maximum are limited to
        1,000 and 12,000 A, respectively. This can be overridden by providing
        user's choice of wave_start and wave_end.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        wave_start: float or None (Default: None)
            Provide the minimum wavelength for resampling.
        wave_end: float or None (Default: None)
            Provide the maximum wavelength for resampling
        wave_bin: float or None (Default: None)
            Provide the resampling bin size
        stype: str or None (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if 'science' in stype_split:

            if self.science_wavecal_polynomial_available:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, contraints are applied to all
                    #  calibrators
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    spec = self.science_spectrum_list[i]

                    # Adjust for pixel shift due to chip gaps
                    wave = self.science_wavecal[i].polyval[spec.fit_type](
                        np.array(spec.pixel_list), spec.fit_coeff).reshape(-1)

                    # compute the new equally-spaced wavelength array
                    if wave_bin is None:

                        wave_bin = np.nanmedian(np.ediff1d(wave))

                    if wave_start is None:

                        wave_start = wave[0]

                    if wave_end is None:

                        wave_end = wave[-1]

                    wave_resampled = np.arange(wave_start, wave_end, wave_bin)

                    # apply the flux calibration and resample
                    count_resampled = spectres(
                        np.array(wave_resampled).reshape(-1),
                        np.array(wave).reshape(-1),
                        np.array(spec.count).reshape(-1),
                        verbose=True)

                    if spec.count_err is not None:

                        count_err_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.count_err).reshape(-1),
                            verbose=True)

                    if spec.count_sky is not None:

                        count_sky_resampled = spectres(
                            np.array(wave_resampled).reshape(-1),
                            np.array(wave).reshape(-1),
                            np.array(spec.count_sky).reshape(-1),
                            verbose=True)

                    spec.add_wavelength(wave)
                    spec.add_wavelength_resampled(wave_resampled)
                    spec.add_count_resampled(count_resampled,
                                             count_err_resampled,
                                             count_sky_resampled)

                self.science_wavelength_calibrated = True
                self.science_wavelength_resampled = True

            else:

                self.logger.warning('Science spectrum/a are not imported.')

        if 'standard' in stype_split:

            if self.standard_wavecal_polynomial_available:

                spec = self.standard_spectrum_list[0]

                # Adjust for pixel shift due to chip gaps
                wave = self.standard_wavecal.polyval[spec.fit_type](np.array(
                    spec.pixel_list), spec.fit_coeff).reshape(-1)

                # compute the new equally-spaced wavelength array
                if wave_bin is None:

                    wave_bin = np.nanmedian(np.ediff1d(wave))

                if wave_start is None:

                    wave_start = wave[0]

                if wave_end is None:

                    wave_end = wave[-1]

                wave_resampled = np.arange(wave_start, wave_end, wave_bin)

                # apply the flux calibration and resample
                count_resampled = spectres(
                    np.array(wave_resampled).reshape(-1),
                    np.array(wave).reshape(-1),
                    np.array(spec.count).reshape(-1),
                    verbose=True)

                if spec.count_err is not None:

                    count_err_resampled = spectres(
                        np.array(wave_resampled).reshape(-1),
                        np.array(wave).reshape(-1),
                        np.array(spec.count_err).reshape(-1),
                        verbose=True)

                if spec.count_sky is not None:

                    count_sky_resampled = spectres(
                        np.array(wave_resampled).reshape(-1),
                        np.array(wave).reshape(-1),
                        np.array(spec.count_sky).reshape(-1),
                        verbose=True)

                spec.add_wavelength(wave)
                spec.add_wavelength_resampled(wave_resampled)
                spec.add_count_resampled(count_resampled, count_err_resampled,
                                         count_sky_resampled)

                self.standard_wavelength_calibrated = True
                self.standard_wavelength_resampled = True

            else:

                self.logger.warning('Standard spectrum is not imported.')

            self.standard_wavelength_calibrated = True

    def set_atmospheric_extinction(self, location='orm', extinction_func=None):
        '''
        ** EXPERIMENTAL, as of 17 Jan 2021 **

        The ORM atmospheric extinction correction chart is taken from
        http://www.ing.iac.es/astronomy/observing/manuals/ps/tech_notes/tn031.pdf

        Parameters
        ----------
        location: str (Default: orm)
            Location of the observatory, currently contains:
            (1) ORM - Roque de los Muchachos Observatory (ORM)
        extinction_func: callable function (Default: None)
            Input wavelength in Angstrom, output magnitude of extinction per
            airmass. It will override the 'location'.

        '''

        if (extinction_func is not None) and (callable(extinction_func)):

            self.extinction_func = extinction_func
            self.logger.info(
                'Manual extinction correction function is loaded.')

        else:

            filename = pkg_resources.resource_filename(
                'aspired', 'extinction/{}_atm_extinct.txt'.format(location))
            extinction_table = np.loadtxt(filename, delimiter=',')
            self.extinction_func = interp1d(extinction_table[:, 0],
                                            extinction_table[:, 1],
                                            kind='cubic',
                                            fill_value='extrapolate')
            self.logger.info(
                '{} extinction correction function is loaded.'.format(
                    location))

        self.atmospheric_extinction_correction_available = True

    def lookup_standard_libraries(self, target, cutoff=0.4):
        '''
        Parameters
        ----------
        target: str
            Name of the standard star
        cutoff: float (Default: 0.4)
            The similarity tolerance [0=completely different, 1=identical]

        '''

        self.fluxcal.lookup_standard_libraries(target=target, cutoff=cutoff)

    def load_standard(self, target, library=None, ftype='flux', cutoff=0.4):
        '''
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

        self.fluxcal.load_standard(target=target,
                                   library=library,
                                   ftype=ftype,
                                   cutoff=cutoff)

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
        Parameters
        ----------
        display: bool (Default: True)
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
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.

        Returns
        -------
        JSON strings if return_jsonstring is set to True.

        '''

        self.fluxcal.inspect_standard(renderer=renderer,
                                      return_jsonstring=return_jsonstring,
                                      display=display,
                                      height=height,
                                      width=width,
                                      save_fig=save_fig,
                                      fig_type=fig_type,
                                      filename=filename,
                                      open_iframe=open_iframe)

        if return_jsonstring:

            return return_jsonstring

    def compute_sensitivity(self,
                            k=3,
                            smooth=False,
                            method='interpolate',
                            slength=5,
                            sorder=3,
                            mask_range=[[6850, 6960], [7575, 7700],
                                        [8925, 9050], [9265, 9750]],
                            mask_fit_order=1,
                            mask_fit_size=1,
                            return_function=False,
                            use_lowess=True,
                            **kwargs):
        '''
        Parameters
        ----------
        k: integer [1,2,3,4,5 only]
            The order of the spline.
        smooth: bool (Default: False)
            set to smooth the input spectrum with scipy.signal.savgol_filter
        weighted: bool (Default: True)
            Set to weight the interpolation/polynomial fit by the
            inverse uncertainty.
        method: str (Default: interpolate)
            This should be either 'interpolate' of 'polynomial'. Note that the
            polynomial is computed from the interpolated function. The
            default is interpolate because it is much more stable at the
            wavelength limits of a spectrum in an automated system.
        slength: int (Default: 5)
            SG-filter window size
        sorder: int (Default: 3)
            SG-filter polynomial order
        mask_range: None or list of list
            (Default: 6850-6960, 7575-7700, 8925-9050, 9265-9750)
            Masking out regions not suitable for fitting the sensitivity curve.
            None for no mask. List of list has the pattern
            [[min1, max1], [min2, max2],...]
        mask_fit_order: int (Default: 1)
            Order of polynomial to be fitted over the masked regions
        mask_fit_size: int (Default: 1)
            Number of "pixels" to be fitted on each side of the masked regions.

        '''

        if self.standard_wavelength_calibrated:

            self.fluxcal.compute_sensitivity(k=k,
                                             smooth=smooth,
                                             method=method,
                                             slength=slength,
                                             sorder=sorder,
                                             mask_range=mask_range,
                                             mask_fit_order=mask_fit_order,
                                             mask_fit_size=mask_fit_size,
                                             return_function=return_function,
                                             use_lowess=use_lowess,
                                             **kwargs)
            self.sensitivity_curve_available = True

        else:

            error_msg = "Standard star is not wavelength calibrated, " +\
                "sensitivity curve cannot be computed."
            self.logger.critical(error_msg)
            raise RuntimeError(error_msg)

    def save_sensitivity_func(self, filename='sensitivity_func.npy'):
        '''
        Parameters
        ----------
        filename: str
            Filename for the output interpolated sensivity curve.

        '''

        self.fluxcal.save_sensitivity_func(filename=filename)

    def add_sensitivity_func(self, sensitivity_func):
        '''
        Parameters
        ----------
        sensitivity_func: str
            Interpolated sensivity curve object.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        self.fluxcal.add_sensitivity_func(sensitivity_func=sensitivity_func)
        self.sensitivity_curve_available = True

    def inspect_sensitivity(self,
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
        Parameters
        ----------
        display: bool (Default: True)
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
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.

        '''

        if self.sensitivity_curve_available:

            self.fluxcal.inspect_sensitivity(
                renderer=renderer,
                width=width,
                height=height,
                return_jsonstring=return_jsonstring,
                display=display,
                save_fig=save_fig,
                fig_type=fig_type,
                filename=filename,
                open_iframe=open_iframe)

    def apply_flux_calibration(self, spec_id=None, stype='science+standard'):
        '''
        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        '''

        stype_split = stype.split('+')

        if self.sensitivity_curve_available:

            if 'science' in stype_split:

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, contraints are applied to all
                    #  calibrators
                    spec_id = list(self.science_spectrum_list.keys())

                for i in spec_id:

                    self.fluxcal.apply_flux_calibration(
                        target_spectrum1D=self.science_spectrum_list[i])
                    self.science_spectrum_list[i].add_standard_header(
                        self.standard_spectrum_list[0].spectrum_header)

                self.science_flux_calibrated = True

            if 'standard' in stype_split:

                self.fluxcal.apply_flux_calibration(
                    target_spectrum1D=self.standard_spectrum_list[0])
                self.standard_spectrum_list[0].add_standard_header(
                    self.standard_spectrum_list[0].spectrum_header)

                self.standard_flux_calibrated = True

    def apply_atmospheric_extinction_correction(self,
                                                science_airmass=None,
                                                standard_airmass=None,
                                                spec_id=None):
        '''
        ** EXPERIMENTAL, as of 17 Jan 2021 **
        This is the first step in allowing atmospheric extinction correction
        of the spectra. Currently it only works if both the science and
        standard spectra are present and both airmass values are provided.
        Towards completion, this function should allow atmospheric
        extinction correction on any meaningful combination of
        (1) science and/or standard spectrum/a, and (2) airmass of either
        or both science and standard observations.

        Parameters
        ----------
        science_airmass: float, str or None (Default: None)
            - If None, it will look for the airmass in the header, if the
              keyword AIRMASS is not found, correction will not be performed.
            - A string input will be used as the header keyword of the airmass,
              if the keyword or header is not found, correction will not be
              performed.
            - A floatpoint value will override the other two and directly be
              use as the airmass
        standard_airmass: float, str or None (Default: None)
            The same as science_airmass.
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object

        '''

        if isinstance(spec_id, int):

            spec_id = [spec_id]

        if spec_id is not None:

            if not set(spec_id).issubset(
                    list(self.science_spectrum_list.keys())):

                error_msg = 'The given spec_id does not exist.'
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        else:

            # if spec_id is None, contraints are applied to all
            #  calibrators
            spec_id = list(self.science_spectrum_list.keys())

        if not self.atmospheric_extinction_correction_available:

            self.logger.error(
                "Atmospheric extinction correction is not configured, "
                "sensitivity curve will be generated without extinction "
                "correction.")

        else:

            if self.science_flux_calibrated & self.standard_flux_calibrated:

                standard_spec = self.standard_spectrum_list[0]

                if standard_airmass is not None:

                    if isinstance(standard_airmass, (int, float)):

                        standard_am = standard_airmass
                        self.logger.info(
                            'Airmass is set to be {}.'.format(standard_am))

                    if isinstance(standard_airmass, str):

                        try:

                            standard_am = standard_spec.spectrum_header[
                                standard_airmass]

                        except Exception as e:

                            self.logger.warning(str(e))

                            standard_am = 1.0
                            self.logger.warning(
                                'Keyword for airmass: {} cannot be found '
                                'in header.'.format(standard_airmass))
                            self.logger.warning('Airmass is set to be 1.0')

                else:

                    try:

                        standard_am = standard_spec.spectrum_header['AIRMASS']

                    except Exception as e:

                        self.logger.warning(str(e))

                        standard_am = 1.0
                        self.logger.warning(
                            'Keyword for airmass: AIRMASS cannot be found '
                            'in header.')
                        self.logger.warning('Airmass is set to be 1.0')

                if spec_id is not None:

                    if not set(spec_id).issubset(
                            list(self.science_spectrum_list.keys())):

                        error_msg = 'The given spec_id does not exist.'
                        self.logger.critical(error_msg)
                        raise ValueError(error_msg)

                else:

                    # if spec_id is None, contraints are applied to all
                    #  calibrators
                    spec_id = list(self.science_spectrum_list.keys())

                if isinstance(spec_id, int):

                    spec_id = [spec_id]

                for i in spec_id:

                    science_spec = self.science_spectrum_list[i]

                    if science_airmass is not None:

                        if isinstance(science_airmass, (int, float)):

                            science_am = science_airmass

                        if isinstance(science_airmass, str):

                            try:

                                science_am = science_spec.spectrum_header[
                                    science_airmass]

                            except Exception as e:

                                self.logger.warning(str(e))
                                science_am = 1.0

                    else:

                        if science_airmass is None:

                            try:

                                science_am =\
                                    science_spec.spectrum_header['AIRMASS']

                            except Exception as e:

                                self.logger.warning(str(e))
                                science_am = 1.0

                    if science_am is None:

                        science_am = 1.0

                    self.logger.info(
                        'Standard airmass is {}.'.format(standard_am))
                    self.logger.info(
                        'Science airmass is {}.'.format(science_am))

                    # Get the atmospheric extinction correction factor
                    science_flux_extinction_factor = 10.**(
                        -(self.extinction_func(science_spec.wave) * science_am)
                        / 2.5)
                    standard_flux_extinction_factor = 10.**(
                        -(self.extinction_func(science_spec.wave) *
                          standard_am) / 2.5)
                    science_spec.flux /= (science_flux_extinction_factor /
                                          standard_flux_extinction_factor)

                    science_flux_resampled_extinction_factor = 10.**(
                        -(self.extinction_func(science_spec.wave_resampled) *
                          science_am) / 2.5)
                    standard_flux_resampled_extinction_factor = 10.**(
                        -(self.extinction_func(science_spec.wave_resampled) *
                          standard_am) / 2.5)
                    science_spec.flux_resampled /= (
                        science_flux_resampled_extinction_factor /
                        standard_flux_resampled_extinction_factor)

                self.atmospheric_extinction_corrected = True
                self.logger.info('Atmospheric extinction is corrected.')

            else:

                self.logger.error('Flux calibration was not performed, the '
                                  'spectrum cannot be extinction corrected. '
                                  'Process continues with the uncorrected '
                                  'spectrum.')

    def inspect_reduced_spectrum(self,
                                 spec_id=None,
                                 stype='science+standard',
                                 wave_min=4000.,
                                 wave_max=8000.,
                                 display=True,
                                 renderer='default',
                                 width=1280,
                                 height=720,
                                 save_fig=False,
                                 fig_type='iframe+png',
                                 filename=None,
                                 open_iframe=False,
                                 return_jsonstring=False):
        '''
        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        wave_min: float (Default: 4000.)
            Minimum wavelength to display
        wave_max: float (Default: 8000.)
            Maximum wavelength to display
        display: bool (Default: True)
            Set to True to display disgnostic plot.
        renderer: str (Default: 'default')
            plotly renderer options.
        width: int/float (Default: 1280)
            Number of pixels in the horizontal direction of the outputs
        height: int/float (Default: 720)
            Number of pixels in the vertical direction of the outputs
        save_fig: bool (default: False)
            Save an image if set to True. Plotly uses the pio.write_html()
            or pio.write_image(). The support format types should be provided
            in fig_type.
        fig_type: string (default: 'iframe+png')
            Image type to be saved, choose from:
            jpg, png, svg, pdf and iframe. Delimiter is '+'.
        filename: str or None (Default: None)
            Filename for the output, all of them will share the same name but
            will have different extension.
        open_iframe: bool (Default: False)
            Open the iframe in the default browser if set to True.
        return_jsonstring: bool (Default: False)
            set to True to return JSON-string that can be rendered by Plotly
            in any support language.

        '''

        stype_split = stype.split('+')
        to_return = []

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, contraints are applied to all calibrators
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                spec = self.science_spectrum_list[i]

                if self.science_wavelength_calibrated:

                    wave = spec.wave

                    if self.science_flux_calibrated:
                        fluxcount = spec.flux
                        fluxcount_sky = spec.flux_sky
                        fluxcount_err = spec.flux_err
                        fluxcount_name = 'Flux'
                        fluxcount_sky_name = 'Sky Flux'
                        fluxcount_err_name = 'Flux Uncertainty'
                    else:
                        fluxcount = spec.count
                        fluxcount_sky = spec.count_sky
                        fluxcount_err = spec.count_err
                        fluxcount_name = 'Count / (e- / s)'
                        fluxcount_sky_name = 'Sky Count / (e- / s)'
                        fluxcount_err_name = 'Count Uncertainty / (e- / s)'

                if self.science_wavelength_resampled:

                    wave = spec.wave_resampled

                    if self.science_flux_calibrated:
                        fluxcount = spec.flux_resampled
                        fluxcount_sky = spec.flux_sky_resampled
                        fluxcount_err = spec.flux_err_resampled
                        fluxcount_name = 'Flux'
                        fluxcount_sky_name = 'Sky Flux'
                        fluxcount_err_name = 'Flux Uncertainty'
                    else:
                        fluxcount = spec.count_resampled
                        fluxcount_sky = spec.count_sky_resampled
                        fluxcount_err = spec.count_err_resampled
                        fluxcount_name = 'Count / (e- / s)'
                        fluxcount_sky_name = 'Sky Count / (e- / s)'
                        fluxcount_err_name = 'Count Uncertainty / (e- / s)'

                wave_mask = ((np.array(wave).reshape(-1) > wave_min)
                             & (np.array(wave).reshape(-1) < wave_max))

                flux_low = np.nanpercentile(
                    np.array(fluxcount).reshape(-1)[wave_mask], 5) / 1.5
                flux_high = np.nanpercentile(
                    np.array(fluxcount).reshape(-1)[wave_mask], 95) * 1.5
                flux_mask = ((np.array(fluxcount).reshape(-1) > flux_low)
                             & (np.array(fluxcount).reshape(-1) < flux_high))

                if np.sum(flux_mask) > 0:

                    flux_min = np.log10(
                        np.nanmin(np.array(fluxcount).reshape(-1)[flux_mask]))
                    flux_max = np.log10(
                        np.nanmax(np.array(fluxcount).reshape(-1)[flux_mask]))

                else:

                    flux_min = np.log10(
                        np.nanmin(np.array(fluxcount).reshape(-1)))
                    flux_max = np.log10(
                        np.nanmax(np.array(fluxcount).reshape(-1)))

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
                    go.Scatter(x=wave,
                               y=fluxcount,
                               line=dict(color='royalblue'),
                               name=fluxcount_name))

                if fluxcount_err is not None:

                    fig_sci.add_trace(
                        go.Scatter(x=wave,
                                   y=fluxcount_err,
                                   line=dict(color='firebrick'),
                                   name=fluxcount_err_name))

                if fluxcount_sky is not None:

                    fig_sci.add_trace(
                        go.Scatter(x=wave,
                                   y=fluxcount_sky,
                                   line=dict(color='orange'),
                                   name=fluxcount_sky_name))

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

                    filename = "spectrum"

                if save_fig:

                    fig_type_split = fig_type.split('+')

                    for t in fig_type_split:

                        if t == 'iframe':

                            pio.write_html(fig_sci,
                                           filename + '_' + str(i) + '.' + t,
                                           auto_open=open_iframe)

                        elif t in ['jpg', 'png', 'svg', 'pdf']:

                            pio.write_image(fig_sci,
                                            filename + '_' + str(i) + '.' + t)

                if display:

                    if renderer == 'default':

                        fig_sci.show()

                    else:

                        fig_sci.show(renderer)

                if return_jsonstring:

                    to_return.append(fig_sci.to_json())

        if 'standard' in stype_split:

            spec = self.standard_spectrum_list[0]

            if self.standard_wavelength_calibrated:

                standard_wave = spec.wave

                if self.standard_flux_calibrated:
                    standard_fluxcount = spec.flux
                    standard_fluxcount_sky = spec.flux_sky
                    standard_fluxcount_err = spec.flux_err
                    standard_fluxcount_name = 'Flux'
                    standard_fluxcount_sky_name = 'Sky Flux'
                    standard_fluxcount_err_name = 'Flux Uncertainty'
                else:
                    standard_fluxcount = spec.count
                    standard_fluxcount_sky = spec.count_sky
                    standard_fluxcount_err = spec.count_err
                    standard_fluxcount_name = 'Count / (e- / s)'
                    standard_fluxcount_sky_name = 'Sky Count / (e- / s)'
                    standard_fluxcount_err_name =\
                        'Count Uncertainty / (e- / s)'

            if self.standard_wavelength_resampled:

                standard_wave = spec.wave_resampled

                if self.standard_flux_calibrated:
                    standard_fluxcount = spec.flux_resampled
                    standard_fluxcount_sky = spec.flux_sky_resampled
                    standard_fluxcount_err = spec.flux_err_resampled
                    standard_fluxcount_name = 'Flux'
                    standard_fluxcount_sky_name = 'Sky Flux'
                    standard_fluxcount_err_name = 'Flux Uncertainty'
                else:
                    standard_fluxcount = spec.count_resampled
                    standard_fluxcount_sky = spec.count_sky_resampled
                    standard_fluxcount_err = spec.count_err_resampled
                    standard_fluxcount_name = 'Count / (e- / s)'
                    standard_fluxcount_sky_name = 'Sky Count / (e- / s)'
                    standard_fluxcount_err_name =\
                        'Count Uncertainty / (e- / s)'

            standard_wave_mask = (
                (np.array(standard_wave).reshape(-1) > wave_min) &
                (np.array(standard_wave).reshape(-1) < wave_max))
            standard_flux_mask = (
                (np.array(standard_fluxcount).reshape(-1) > np.nanpercentile(
                    np.array(standard_fluxcount).reshape(-1)
                    [standard_wave_mask], 5) / 1.5) &
                (np.array(standard_fluxcount).reshape(-1) < np.nanpercentile(
                    np.array(standard_fluxcount).reshape(-1)
                    [standard_wave_mask], 95) * 1.5))

            if np.nansum(standard_flux_mask) > 0:

                standard_flux_min = np.log10(
                    np.nanmin(
                        np.array(standard_fluxcount).reshape(-1)
                        [standard_flux_mask]))
                standard_flux_max = np.log10(
                    np.nanmax(
                        np.array(standard_fluxcount).reshape(-1)
                        [standard_flux_mask]))

            else:

                standard_flux_min = np.log10(
                    np.nanmin(np.array(standard_fluxcount).reshape(-1)))
                standard_flux_max = np.log10(
                    np.nanmax(np.array(standard_fluxcount).reshape(-1)))

            fig_standard = go.Figure(layout=dict(updatemenus=list([
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
                                                 autosize=False,
                                                 height=height,
                                                 width=width,
                                                 title='Log scale'))

            # show the image on the top
            fig_standard.add_trace(
                go.Scatter(x=standard_wave,
                           y=standard_fluxcount,
                           line=dict(color='royalblue'),
                           name=standard_fluxcount_name))

            if standard_fluxcount_err is not None:

                fig_standard.add_trace(
                    go.Scatter(x=standard_wave,
                               y=standard_fluxcount_err,
                               line=dict(color='firebrick'),
                               name=standard_fluxcount_err_name))

            if standard_fluxcount_sky is not None:

                fig_standard.add_trace(
                    go.Scatter(x=standard_wave,
                               y=standard_fluxcount_sky,
                               line=dict(color='orange'),
                               name=standard_fluxcount_sky_name))

            if self.fluxcal.standard_fluxmag_true is not None:

                fig_standard.add_trace(
                    go.Scatter(x=self.fluxcal.standard_wave_true,
                               y=self.fluxcal.standard_fluxmag_true,
                               line=dict(color='black'),
                               name='Standard'))

            fig_standard.update_layout(
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='Wavelength / A', range=[wave_min, wave_max]),
                yaxis=dict(title='Flux',
                           range=[standard_flux_min, standard_flux_max],
                           type='log'),
                legend=go.layout.Legend(x=0,
                                        y=1,
                                        traceorder="normal",
                                        font=dict(family="sans-serif",
                                                  size=12,
                                                  color="black"),
                                        bgcolor='rgba(0,0,0,0)'))

            if filename is None:

                filename = "spectrum_standard"

            if save_fig:

                fig_type_split = fig_type.split('+')

                for t in fig_type_split:

                    if t == 'iframe':

                        pio.write_html(fig_standard,
                                       filename + '.' + t,
                                       auto_open=open_iframe)

                    elif t in ['jpg', 'png', 'svg', 'pdf']:

                        pio.write_image(fig_standard, filename + '.' + t)

            if display:

                if renderer == 'default':

                    fig_standard.show(height=height, width=width)

                else:

                    fig_standard.show(renderer, height=height, width=width)

            if return_jsonstring:

                to_return.append(fig_standard.to_json())

        if return_jsonstring:

            return to_return

        if ('science' not in stype_split) and ('standard' not in stype_split):

            error_msg = 'Unknown stype, please choose from (1) science; ' +\
                'and/or (2) standard. use + as delimiter.'
            self.logger.critical(error_msg)
            raise TypeError(error_msg)

    def create_fits(self,
                    output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                    spec_id=None,
                    stype='science+standard',
                    recreate=True,
                    empty_primary_hdu=True):
        '''
        Create a HDU list, with a choice of any combination of the
        data, see below the 'output' parameters for details.

        Parameters
        ----------
        output: String
            (Default: 'arc_spec+wavecal+wavelength+flux+flux_resampled')
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 4 HDUs
                    Count, uncertainty, sky, optimal flag, and weight (pixel)
                arc_spec: 3 HDUs
                    1D arc spectrum, arc line pixels, and arc line effective
                    pixels
                wavecal: 1 HDU
                    Polynomial coefficients for wavelength calibration
                wavelength: 1 HDU
                    Wavelength of each pixel
                count_resampled: 3 HDUs
                    Resampled Count, uncertainty, and sky (wavelength)
                sensitivity: 1 HDU
                    Sensitivity (pixel)
                flux: 4 HDUs
                    Flux, uncertainty, sky, and sensitivity (pixel)
                sensitivity_resampled: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 4 HDUs
                    Flux, uncertainty, sky, and sensitivity (wavelength)

        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        recreate: bool (Default: True)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank

        '''

        # Split the string into strings
        stype_split = stype.split('+')
        output_split = output.split('+')

        for i in output_split:

            if i not in [
                    'trace', 'count', 'weight_map', 'arc_spec', 'wavecal',
                    'wavelength', 'count_resampled', 'sensitivity', 'flux',
                    'sensitivity_resampled', 'flux_resampled'
            ]:

                error_msg = '{} is not a valid output.'.format(i)
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        if ('science' not in stype_split) and ('standard' not in stype_split):

            error_msg = 'Unknown stype, please choose from (1) science; ' +\
                'and/or (2) standard. use + as delimiter.'
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, contraints are applied to all
                #  calibrators
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].create_fits(
                    output=output,
                    recreate=recreate,
                    empty_primary_hdu=empty_primary_hdu)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].create_fits(
                output=output,
                recreate=recreate,
                empty_primary_hdu=empty_primary_hdu)

    def modify_trace_header(self,
                            idx,
                            method,
                            *args,
                            spec_id=None,
                            stype='science+standard'):
        """
        Wrapper function to modify the trace header.

        Parameters
        ----------
        idx: int
            The HDU number of the trace FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_trace_header(
                    idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_trace_header(
                idx, method, *args)

    def modify_count_header(self,
                            idx,
                            method,
                            *args,
                            spec_id=None,
                            stype='science+standard'):
        """
        Wrapper function to modify the count header.

        Parameters
        ----------
        idx: int
            The HDU number of the trace FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_count_header(
                    idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_count_header(
                idx, method, *args)

    def modify_weight_map_header(self,
                                 idx,
                                 method,
                                 *args,
                                 spec_id=None,
                                 stype='science+standard'):
        """
        Wrapper function to modify the weight map header.

        Parameters
        ----------
        idx: int
            The HDU number of the weight map FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_weight_map_header(
                    idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_weight_map_header(
                idx, method, *args)

    def modify_count_resampled_header(self,
                                      idx,
                                      method,
                                      *args,
                                      spec_id=None,
                                      stype='science+standard'):
        """
        Wrapper function to modify the count resampled header.

        Parameters
        ----------
        idx: int
            The HDU number of the count resampled FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_count_resampled_header(
                    idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_count_resampled_header(
                idx, method, *args)

    def modify_arc_spec_header(self,
                               idx,
                               method,
                               *args,
                               spec_id=None,
                               stype='science+standard'):
        """
        Wrapper function to modify the arc spectrum header.

        Parameters
        ----------
        idx: int
            The HDU number of the arc spectrum FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_arc_spec_header(
                    idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_arc_spec_header(
                idx, method, *args)

    def modify_wavecal_header(self,
                              idx,
                              method,
                              *args,
                              spec_id=None,
                              stype='science+standard'):
        """
        Wrapper function to modify the wavecal header.

        Parameters
        ----------
        idx: int
            The HDU number of the wavecal FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_wavecal_header(
                    idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_wavecal_header(
                idx, method, *args)

    def modify_wavelength_header(self,
                                 idx,
                                 method,
                                 *args,
                                 spec_id=None,
                                 stype='science+standard'):
        """
        Wrapper function to modify the wavelength header.

        Parameters
        ----------
        idx: int
            The HDU number of the wavelength FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_wavelength_header(
                    idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_wavelength_header(
                idx, method, *args)

    def modify_sensitivity_header(self,
                                  idx,
                                  method,
                                  *args,
                                  spec_id=None,
                                  stype='science+standard'):
        """
        Wrapper function to modify the sensitivity header.

        Parameters
        ----------
        idx: int
            The HDU number of the sensitivity FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_sensitivity_header(
                    idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_sensitivity_header(
                idx, method, *args)

    def modify_flux_header(self,
                           idx,
                           method,
                           *args,
                           spec_id=None,
                           stype='science+standard'):
        """
        Wrapper function to modify the flux header.

        Parameters
        ----------
        idx: int
            The HDU number of the flux FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_flux_header(
                    idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_flux_header(
                idx, method, *args)

    def modify_sensitivity_resampled_header(self,
                                            idx,
                                            method,
                                            *args,
                                            spec_id=None,
                                            stype='science+standard'):
        """
        Wrapper function to modify the sensitivity resampled header.

        Parameters
        ----------
        idx: int
            The HDU number of the sensitivity resampled FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[
                    i].modify_sensitivity_resampled_header(idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_sensitivity_resampled_header(
                idx, method, *args)

    def modify_flux_resampled_header(self,
                                     idx,
                                     method,
                                     *args,
                                     spec_id=None,
                                     stype='science+standard'):
        """
        Wrapper function to modify the flux resampled header.

        Parameters
        ----------
        idx: int
            The HDU number of the flux resampled FITS
        method: str
            The operation to modify the header with
        *args:
            Extra arguments for the method
        spec_id: int (Default: None)
            The ID corresponding to the spectrum1D object
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter

        """

        # Split the string into strings
        stype_split = stype.split('+')

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, calibrators are initialised to all
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                self.science_spectrum_list[i].modify_flux_resampled_header(
                    idx, method, *args)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].modify_flux_resampled_header(
                idx, method, *args)

    def save_fits(self,
                  spec_id=None,
                  output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                  filename='reduced',
                  stype='science+standard',
                  recreate=False,
                  empty_primary_hdu=True,
                  overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        output: String
            (Default: 'arc_spec+wavecal+wavelength+flux+flux_resampled')
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 4 HDUs
                    Count, uncertainty, sky, optimal flag, and weight (pixel)
                arc_spec: 3 HDUs
                    1D arc spectrum, arc line pixels, and arc line effective
                    pixels
                wavecal: 1 HDU
                    Polynomial coefficients for wavelength calibration
                wavelength: 1 HDU
                    Wavelength of each pixel
                count_resampled: 3 HDUs
                    Resampled Count, uncertainty, and sky (wavelength)
                sensitivity: 1 HDU
                    Sensitivity (pixel)
                flux: 3 HDUs
                    Flux, uncertainty, and sky (pixel)
                sensitivity_resampled: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 3 HDUs
                    Flux, uncertainty, and sky (wavelength)

        filename: String (Default: 'reduced')
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.
        empty_primary_hdu: bool (Default: True)
            Set to True to leave the Primary HDU blank
        overwrite: bool (Default: False)
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        stype_split = stype.split('+')
        output_split = output.split('+')

        for i in output_split:

            if i not in [
                    'trace', 'count', 'weight_map', 'arc_spec', 'wavecal',
                    'wavelength', 'count_resampled', 'sensitivity', 'flux',
                    'sensitivity_resampled', 'flux_resampled'
            ]:

                error_msg = '{} is not a valid output.'.format(i)
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        if ('science' not in stype_split) and ('standard' not in stype_split):

            error_msg = 'Unknown stype, please choose from (1) science; ' +\
                'and/or (2) standard. use + as delimiter.'
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    error_msg = 'The given spec_id does not exist.'
                    self.logger.critical(error_msg)
                    raise ValueError(error_msg)

            else:

                # if spec_id is None, contraints are applied to all
                #  calibrators
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                filename_i = filename + '_science_' + str(i)

                self.science_spectrum_list[i].save_fits(
                    output=output,
                    filename=filename_i,
                    overwrite=overwrite,
                    recreate=recreate,
                    empty_primary_hdu=empty_primary_hdu)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].save_fits(
                output=output,
                filename=filename + '_standard',
                overwrite=overwrite,
                recreate=recreate,
                empty_primary_hdu=empty_primary_hdu)

    def save_csv(self,
                 spec_id=None,
                 output='arc_spec+wavecal+wavelength+flux+flux_resampled',
                 filename='reduced',
                 stype='science+standard',
                 recreate=False,
                 overwrite=False):
        '''
        Save the reduced data to disk, with a choice of any combination of the
        5 sets of data, see below the 'output' parameters for details.

        Parameters
        ----------
        spec_id: int or None (Default: None)
            The ID corresponding to the spectrum1D object
        output: String
            (Default: 'arc_spec+wavecal+wavelength+flux+flux_resampled')
            Type of data to be saved, the order is fixed (in the order of
            the following description), but the options are flexible. The
            input strings are delimited by "+",

                trace: 2 HDUs
                    Trace, and trace width (pixel)
                count: 4 HDUs
                    Count, uncertainty, sky, optimal flag, and weight (pixel)
                arc_spec: 3 HDUs
                    1D arc spectrum, arc line pixels, and arc line effective
                    pixels
                wavecal: 1 HDU
                    Polynomial coefficients for wavelength calibration
                wavelength: 1 HDU
                    Wavelength of each pixel
                count_resampled: 3 HDUs
                    Resampled Count, uncertainty, and sky (wavelength)
                sensitivity: 1 HDU
                    Sensitivity (pixel)
                flux: 3 HDUs
                    Flux, uncertainty, sky, and sensitivity (pixel)
                sensitivity_resampled: 1 HDU
                    Sensitivity (wavelength)
                flux_resampled: 3 HDUs
                    Flux, uncertainty, sky, and sensitivity (wavelength)

        filename: String (Default: 'reduced')
            Disk location to be written to. Default is at where the
            process/subprocess is execuated.
        stype: str (Default: 'science+standard')
            'science' and/or 'standard' to indicate type, use '+' as delimiter
        recreate: bool (Default: False)
            Set to True to overwrite the FITS data and header.
        overwrite: bool (Default: False)
            Default is False.

        '''

        # Fix the names and extensions
        filename = os.path.splitext(filename)[0]

        # Split the string into strings
        stype_split = stype.split('+')
        output_split = output.split('+')

        for i in output_split:

            if i not in [
                    'trace', 'count', 'weight_map', 'arc_spec', 'wavecal',
                    'wavelength', 'count_resampled', 'sensitivity', 'flux',
                    'sensitivity', 'flux_resampled'
            ]:

                error_msg = '{} is not a valid output.'.format(i)
                self.logger.critical(error_msg)
                raise ValueError(error_msg)

        if ('science' not in stype_split) and ('standard' not in stype_split):

            error_msg = 'Unknown stype, please choose from (1) science; ' +\
                'and/or (2) standard. use + as delimiter.'
            self.logger.critical(error_msg)
            raise ValueError(error_msg)

        if 'science' in stype_split:

            if isinstance(spec_id, int):

                spec_id = [spec_id]

            if spec_id is not None:

                if not set(spec_id).issubset(
                        list(self.science_spectrum_list.keys())):

                    raise ValueError('The given spec_id does not exist.')

            else:

                # if spec_id is None, contraints are applied to all
                #  calibrators
                spec_id = list(self.science_spectrum_list.keys())

            for i in spec_id:

                filename_i = filename + '_science_' + str(i)

                self.science_spectrum_list[i].save_csv(output=output,
                                                       filename=filename_i,
                                                       recreate=recreate,
                                                       overwrite=overwrite)

        if 'standard' in stype_split:

            self.standard_spectrum_list[0].save_csv(output=output,
                                                    filename=filename +
                                                    '_standard',
                                                    recreate=recreate,
                                                    overwrite=overwrite)
