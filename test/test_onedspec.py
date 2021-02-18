import logging
import os
import numpy as np
import pytest
from aspired import image_reduction
from aspired import spectral_reduction
from aspired.wavelengthcalibration import WavelengthCalibration
from aspired.fluxcalibration import FluxCalibration

base_dir = os.path.dirname(__file__)
abs_dir = os.path.abspath(os.path.join(base_dir, '..'))


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def test_logger():
    onedspec_debug = spectral_reduction.OneDSpec(
        log_level='DEBUG',
        logger_name='onedspec_debug',
        log_file_name='onedspec_debug.log',
        log_file_folder='test/test_output/')
    onedspec_info = spectral_reduction.OneDSpec(
        log_level='INFO',
        logger_name='onedspec_info',
        log_file_name='onedspec_info.log',
        log_file_folder='test/test_output/')
    onedspec_warning = spectral_reduction.OneDSpec(
        log_level='WARNING',
        logger_name='onedspec_warning',
        log_file_name='onedspec_warning.log',
        log_file_folder='test/test_output/')
    onedspec_error = spectral_reduction.OneDSpec(
        log_level='ERROR',
        logger_name='onedspec_error',
        log_file_name='onedspec_error.log',
        log_file_folder='test/test_output/')
    onedspec_critical = spectral_reduction.OneDSpec(
        log_level='CRITICAL',
        logger_name='onedspec_critical',
        log_file_name='onedspec_critical.log',
        log_file_folder='test/test_output/')

    onedspec_debug.logger.debug('debug: debug mode')
    onedspec_debug.logger.info('debug: info mode')
    onedspec_debug.logger.warning('debug: warning mode')
    onedspec_debug.logger.error('debug: error mode')
    onedspec_debug.logger.critical('debug: critical mode')

    onedspec_info.logger.debug('info: debug mode')
    onedspec_info.logger.info('info: info mode')
    onedspec_info.logger.warning('info: warning mode')
    onedspec_info.logger.error('info: error mode')
    onedspec_info.logger.critical('info: critical mode')

    onedspec_warning.logger.debug('warning: debug mode')
    onedspec_warning.logger.info('warning: info mode')
    onedspec_warning.logger.warning('warning: warning mode')
    onedspec_warning.logger.error('warning: error mode')
    onedspec_warning.logger.critical('warning: critical mode')

    onedspec_error.logger.debug('error: debug mode')
    onedspec_error.logger.info('error: info mode')
    onedspec_error.logger.warning('error: warning mode')
    onedspec_error.logger.error('error: error mode')
    onedspec_error.logger.critical('error: critical mode')

    onedspec_critical.logger.debug('critical: debug mode')
    onedspec_critical.logger.info('critical: info mode')
    onedspec_critical.logger.warning('critical: warning mode')
    onedspec_critical.logger.error('critical: error mode')
    onedspec_critical.logger.critical('critical: critical mode')

    debug_debug_length = file_len('test/test_output/onedspec_debug.log')
    debug_info_length = file_len('test/test_output/onedspec_info.log')
    debug_warning_length = file_len('test/test_output/onedspec_warning.log')
    debug_error_length = file_len('test/test_output/onedspec_error.log')
    debug_critical_length = file_len('test/test_output/onedspec_critical.log')

    assert debug_debug_length == 5, 'Expecting 5 lines in the log file, {} is logged.'.format(
        debug_debug_length)
    assert debug_info_length == 4, 'Expecting 4 lines in the log file, {} is logged.'.format(
        debug_info_length)
    assert debug_warning_length == 3, 'Expecting 3 lines in the log file, {} is logged.'.format(
        debug_warning_length)
    assert debug_error_length == 2, 'Expecting 2 lines in the log file, {} is logged.'.format(
        debug_error_length)
    assert debug_critical_length == 1, 'Expecting 1 lines in the log file, {} is logged.'.format(
        debug_critical_length)


os.remove('test/test_output/onedspec_debug.log')
os.remove('test/test_output/onedspec_info.log')
os.remove('test/test_output/onedspec_warning.log')
os.remove('test/test_output/onedspec_error.log')
os.remove('test/test_output/onedspec_critical.log')


def test_add_fluxcalibration():
    # Create a dummy FluxCalibration
    dummy_fluxcal = FluxCalibration(log_file_name=None)
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_fluxcalibration(dummy_fluxcal)


@pytest.mark.xfail(raises=TypeError)
def test_add_fluxcalibration_fail():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_fluxcalibration(None)


def test_add_wavelengthcalibration():
    # Create a dummy WavelengthCalibration
    dummy_wavecal = WavelengthCalibration(log_file_name=None)
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration(dummy_wavecal)
    onedspec.add_wavelengthcalibration(dummy_wavecal, spec_id=0)
    onedspec.add_wavelengthcalibration([dummy_wavecal])
    onedspec.add_wavelengthcalibration([dummy_wavecal], spec_id=0)


@pytest.mark.xfail(raises=TypeError)
def test_add_wavelengthcalibration_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration(None)


@pytest.mark.xfail(raises=ValueError)
def test_add_wavelengthcalibration_fail_spec_id():
    # Create a dummy WavelengthCalibration
    dummy_wavecal = WavelengthCalibration(log_file_name=None)
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration(dummy_wavecal, spec_id=1)


# science
def test_add_wavelength_science():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_wavelength(np.arange(100), stype='science')
    onedspec.add_wavelength([np.arange(100)], stype='science')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_science_fail_no_science_data():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelength(np.arange(100), stype='science')
    onedspec.add_wavelength([np.arange(100)], stype='science')


@pytest.mark.xfail(raises=TypeError)
def test_add_wavelength_science_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_wavelength(None, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_add_wavelength_science_fail_spec_id():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_wavelength(np.arange(100), spec_id=1, stype='science')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_science_fail_wavelength_size():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_wavelength(np.arange(10), stype='science')


# standard
def test_add_wavelength_standard():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='standard')
    onedspec.add_wavelength(np.arange(100), stype='standard')
    onedspec.add_wavelength([np.arange(100)], stype='standard')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_standard_fail_no_standard_data():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelength(np.arange(100), stype='standard')
    onedspec.add_wavelength([np.arange(100)], stype='standard')


@pytest.mark.xfail(raises=TypeError)
def test_add_wavelength_standard_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='standard')
    onedspec.add_wavelength(None, stype='standard')


# Note that standard does not care about spec_id, there can only be one
def test_add_wavelength_standard_spec_id():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='standard')
    onedspec.add_wavelength(np.arange(100), spec_id=1, stype='standard')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_standard_fail_wavelength_size():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='standard')
    onedspec.add_wavelength(np.arange(10), stype='standard')
