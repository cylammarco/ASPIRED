import os
import numpy as np
import pytest
from aspired import image_reduction
from aspired import spectral_reduction
from aspired.wavelength_calibration import WavelengthCalibration
from aspired.flux_calibration import FluxCalibration

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

    assert debug_debug_length == 5, 'Expecting 5 lines in the log file, ' +\
        '{} is logged.'.format(debug_debug_length)
    assert debug_info_length == 4, 'Expecting 4 lines in the log file, ' +\
        '{} is logged.'.format(debug_info_length)
    assert debug_warning_length == 3, 'Expecting 3 lines in the log file, ' +\
        '{} is logged.'.format(debug_warning_length)
    assert debug_error_length == 2, 'Expecting 2 lines in the log file, ' +\
        '{} is logged.'.format(debug_error_length)
    assert debug_critical_length == 1, 'Expecting 1 lines in the log file, ' +\
        '{} is logged.'.format(debug_critical_length)


try:
    os.remove('test/test_output/onedspec_debug.log')
except Exception as e:
    print(e)

try:
    os.remove('test/test_output/onedspec_info.log')
except Exception as e:
    print(e)

try:
    os.remove('test/test_output/onedspec_warning.log')
except Exception as e:
    print(e)

try:
    os.remove('test/test_output/onedspec_error.log')
except Exception as e:
    print(e)

try:
    os.remove('test/test_output/onedspec_critical.log')
except Exception as e:
    print(e)


def test_add_fluxcalibration():
    # Create a dummy FluxCalibration
    dummy_fluxcal = FluxCalibration(log_file_name=None)
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_fluxcalibration(dummy_fluxcal)


@pytest.mark.xfail(raises=TypeError)
def test_add_fluxcalibration_fail():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_fluxcalibration(None)


# science add_wavelengthcalibration
def test_add_wavelengthcalibration_science():
    # Create a dummy WavelengthCalibration
    dummy_wavecal = WavelengthCalibration(log_file_name=None)
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration(dummy_wavecal)
    onedspec.add_wavelengthcalibration(dummy_wavecal,
                                       spec_id=0,
                                       stype='science')
    onedspec.add_wavelengthcalibration([dummy_wavecal])
    onedspec.add_wavelengthcalibration([dummy_wavecal],
                                       spec_id=[0],
                                       stype='science')


# science add_wavelengthcalibration to two traces
def test_add_wavelengthcalibration_science_two_spec():
    # Create a dummy WavelengthCalibration
    dummy_wavecal = WavelengthCalibration(log_file_name=None)
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_science_spectrum1D(1)
    onedspec.add_wavelengthcalibration(dummy_wavecal,
                                       spec_id=[0, 1],
                                       stype='science')


@pytest.mark.xfail(raises=ValueError)
# science add_wavelengthcalibration to two traces
def test_add_wavelengthcalibration_science_expect_fail():
    # Create a dummy WavelengthCalibration
    dummy_wavecal = WavelengthCalibration(log_file_name=None)
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration(dummy_wavecal,
                                       spec_id=[0, 1],
                                       stype='science')


# standard add_wavelengthcalibration
def test_add_wavelengthcalibration_standard():
    # Create a dummy WavelengthCalibration
    dummy_wavecal = WavelengthCalibration(log_file_name=None)
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration(dummy_wavecal)
    onedspec.add_wavelengthcalibration(dummy_wavecal, stype='standard')
    onedspec.add_wavelengthcalibration([dummy_wavecal])
    onedspec.add_wavelengthcalibration([dummy_wavecal], stype='standard')


# science
@pytest.mark.xfail(raises=TypeError)
def test_add_wavelengthcalibration_science_fail_type_None():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration(None, stype='science')


@pytest.mark.xfail(raises=TypeError)
def test_add_wavelengthcalibration_science_fail_type_list_of_None():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration([None], stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_add_wavelengthcalibration_science_fail_spec_id():
    # Create a dummy WavelengthCalibration
    dummy_wavecal = WavelengthCalibration(log_file_name=None)
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration(dummy_wavecal,
                                       spec_id=1,
                                       stype='science')


# standard
@pytest.mark.xfail(raises=TypeError)
def test_add_wavelengthcalibration_standard_fail_type_None():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration(None, stype='standard')


@pytest.mark.xfail(raises=TypeError)
def test_add_wavelengthcalibration_standard_fail_type_list_of_None():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration([None], stype='standard')


# spec_id is ignored in standard, so this passes
def test_add_wavelengthcalibration_standard_fail_spec_id():
    # Create a dummy WavelengthCalibration
    dummy_wavecal = WavelengthCalibration(log_file_name=None)
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelengthcalibration(dummy_wavecal,
                                       spec_id=1,
                                       stype='standard')


# science add_spec
def test_add_spec_science():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(count=np.arange(100),
                      count_err=np.arange(100),
                      count_sky=np.arange(100),
                      spec_id=0,
                      stype='science')
    onedspec.add_spec(count=[np.arange(200)],
                      count_err=[np.arange(200)],
                      count_sky=[np.arange(200)],
                      spec_id=1,
                      stype='science')


@pytest.mark.xfail(raises=TypeError)
def test_add_spec_science_fail_count_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(count=0.,
                      count_err=np.arange(100),
                      count_sky=np.arange(200),
                      spec_id=0,
                      stype='science')


@pytest.mark.xfail(raises=TypeError)
def test_add_spec_science_fail_count_sky_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(count=np.arange(100),
                      count_err=np.arange(100),
                      count_sky=0.,
                      spec_id=0,
                      stype='science')


@pytest.mark.xfail(raises=TypeError)
def test_add_spec_science_fail_count_err_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(count=np.arange(100),
                      count_err=0.,
                      count_sky=np.arange(200),
                      spec_id=0,
                      stype='science')


@pytest.mark.xfail(raises=AssertionError)
def test_add_spec_science_fail_count_sky_length_mismatch():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(count=np.arange(100),
                      count_err=np.arange(100),
                      count_sky=np.arange(200),
                      spec_id=0,
                      stype='science')


@pytest.mark.xfail(raises=AssertionError)
def test_add_spec_science_fail_count_err_length_mismatch():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(count=np.arange(100),
                      count_err=np.arange(200),
                      count_sky=np.arange(100),
                      spec_id=0,
                      stype='science')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_spec_science_fail_count_shape_mismatch():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(count=[np.arange(100)],
                      count_err=[np.arange(100),
                                 np.arange(100)],
                      count_sky=[np.arange(100),
                                 np.arange(100)],
                      spec_id=[0, 1, 2],
                      stype='science')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_spec_science_fail_count_sky_shape_mismatch():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(count=[np.arange(100), np.arange(100)],
                      count_err=[np.arange(100)],
                      count_sky=[np.arange(100),
                                 np.arange(100)],
                      spec_id=[0, 1, 2],
                      stype='science')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_spec_science_fail_count_err_shape_mismatch():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(count=[np.arange(100), np.arange(100)],
                      count_err=[np.arange(100),
                                 np.arange(100)],
                      count_sky=[np.arange(100)],
                      spec_id=[0, 1, 2],
                      stype='science')


# science add_wavelength
def test_add_wavelength_science():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), spec_id=0, stype='science')
    onedspec.add_spec(np.arange(200), spec_id=1, stype='science')
    onedspec.add_wavelength(np.arange(100), spec_id=0, stype='science')
    onedspec.add_wavelength(np.arange(200), spec_id=1, stype='science')
    onedspec.add_wavelength([np.arange(100)], spec_id=0, stype='science')
    onedspec.add_wavelength([np.arange(200)], spec_id=1, stype='science')


# science add_wavelengthcalibration to two traces
def test_add_wavelength_science_two_spec():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), spec_id=0, stype='science')
    onedspec.add_spec(np.arange(100), spec_id=1, stype='science')
    onedspec.add_wavelength(np.arange(100), spec_id=[0, 1], stype='science')


@pytest.mark.xfail(raises=ValueError)
# science add_wavelengthcalibration to two traces
def test_add_wavelength_science_expect_fail():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), spec_id=0, stype='science')
    onedspec.add_wavelength(np.arange(100), spec_id=[0, 1], stype='science')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_science_fail_no_science_data():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelength(np.arange(100), stype='science')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_science_fail_no_science_data_2():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
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


# standard add_wavelength
def test_add_wavelength_standard():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), spec_id=0, stype='standard')
    onedspec.add_wavelength(np.arange(100), spec_id=0, stype='standard')
    onedspec.add_wavelength([np.arange(100)], spec_id=0, stype='standard')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_standard_fail_no_standard_data():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelength(np.arange(100), stype='standard')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_standard_fail_no_standard_data_2():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
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


# science add_wavelength_resampled
def test_add_wavelength_resampled_science():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), spec_id=0, stype='science')
    onedspec.add_spec(np.arange(200), spec_id=1, stype='science')
    onedspec.add_spec(np.arange(100), spec_id=10, stype='science')
    onedspec.add_wavelength_resampled(np.arange(100),
                                      spec_id=0,
                                      stype='science')
    onedspec.add_wavelength_resampled(np.arange(200),
                                      spec_id=1,
                                      stype='science')
    onedspec.add_wavelength_resampled([np.arange(100)],
                                      spec_id=0,
                                      stype='science')
    onedspec.add_wavelength_resampled([np.arange(200)],
                                      spec_id=1,
                                      stype='science')
    onedspec.add_wavelength_resampled(
        [np.arange(100), np.arange(200),
         np.arange(100)],
        spec_id=[0, 1, 10],
        stype='science')


# science add_wavelengthcalibration to two traces
def test_add_wavelength_resampled_science_two_spec():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), spec_id=0, stype='science')
    onedspec.add_spec(np.arange(100), spec_id=1, stype='science')
    onedspec.add_wavelength_resampled(np.arange(100),
                                      spec_id=[0, 1],
                                      stype='science')


@pytest.mark.xfail(raises=ValueError)
# science add_wavelengthcalibration to two traces
def test_add_wavelength_resampled_science_two_spec_expect_fail():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), spec_id=0, stype='science')
    onedspec.add_wavelength_resampled(np.arange(100),
                                      spec_id=[0, 1],
                                      stype='science')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_resampled_science_fail_no_science_data():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelength_resampled(np.arange(100), stype='science')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_resampled_science_fail_no_science_data_2():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelength_resampled([np.arange(100)], stype='science')


@pytest.mark.xfail(raises=TypeError)
def test_add_wavelength_resampled_science_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_wavelength_resampled(None, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_add_wavelength_resampled_science_fail_spec_id():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_wavelength_resampled(np.arange(100),
                                      spec_id=1,
                                      stype='science')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_resampled_science_fail_wavelength_size():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_wavelength_resampled(np.arange(10), stype='science')


# standard add_wavelength_resampled
def test_add_wavelength_resampled_standard():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), spec_id=0, stype='standard')
    onedspec.add_wavelength_resampled(np.arange(100),
                                      spec_id=0,
                                      stype='standard')
    onedspec.add_wavelength_resampled([np.arange(100)],
                                      spec_id=0,
                                      stype='standard')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_resampled_standard_fail_no_standard_data():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelength_resampled(np.arange(100), stype='standard')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_resampled_standard_fail_no_standard_data_2():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_wavelength_resampled([np.arange(100)], stype='standard')


@pytest.mark.xfail(raises=TypeError)
def test_add_wavelength_resampled_standard_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='standard')
    onedspec.add_wavelength_resampled(None, stype='standard')


# Note that standard does not care about spec_id, there can only be one,
# so this test passes
def test_add_wavelength_resampled_standard_spec_id():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='standard')
    onedspec.add_wavelength_resampled(np.arange(100),
                                      spec_id=1,
                                      stype='standard')


@pytest.mark.xfail(raises=RuntimeError)
def test_add_wavelength_resampled_standard_fail_wavelength_size():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='standard')
    onedspec.add_wavelength_resampled(np.arange(10), stype='standard')


# science arc_spec
def test_add_arc_spec_science():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_arc_spec(np.arange(100), stype='science')
    onedspec.add_arc_spec(np.arange(100), spec_id=0, stype='science')
    onedspec.add_arc_spec([np.arange(100)], spec_id=0, stype='science')
    onedspec.add_arc_spec(np.arange(200), spec_id=1, stype='science')
    onedspec.add_arc_spec(np.arange(200), spec_id=[1], stype='science')
    onedspec.add_arc_spec([np.arange(100), np.arange(200)],
                          spec_id=[0, 2],
                          stype='science')


# science find_arc_lines
def test_find_arc_lines_science():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_arc_spec(np.arange(100), stype='science')
    onedspec.add_arc_spec([np.arange(100)], spec_id=0, stype='science')
    onedspec.find_arc_lines(spec_id=0, stype='science')
    onedspec.find_arc_lines(spec_id=[0], stype='science')


# science find_arc_lines fail spec_id
@pytest.mark.xfail(raises=ValueError)
def test_find_arc_lines_science_fail_spec_id():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_arc_spec(np.arange(100), spec_id=0, stype='science')
    onedspec.find_arc_lines(spec_id=7, stype='science')


# science add_arc_lines
def test_add_arc_lines_science():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_spec(np.arange(100), spec_id=0, stype='science')
    onedspec.add_spec(np.arange(200), spec_id=1, stype='science')
    onedspec.add_spec(np.arange(100), spec_id=10, stype='science')
    onedspec.add_arc_lines(np.arange(7), spec_id=0, stype='science')
    onedspec.add_arc_lines(np.arange(5), spec_id=1, stype='science')
    onedspec.add_arc_lines([np.arange(7)], spec_id=0, stype='science')
    onedspec.add_arc_lines([np.arange(15)], spec_id=1, stype='science')
    onedspec.add_arc_lines(
        [np.arange(7), np.arange(15),
         np.arange(7)],
        spec_id=[0, 1, 10],
        stype='science')


# mismatched spec lengths
def test_add_arc_lines_science_fail_spec_id():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_arc_lines(np.arange(5), spec_id=0, stype='science')
    onedspec.add_arc_lines(np.arange(5), spec_id=7, stype='science')


@pytest.mark.xfail(raises=TypeError)
def test_add_arc_lines_science_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='science')
    onedspec.add_arc_lines(None, stype='science')


# standard add_arc_lines
def test_add_arc_lines_standard():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='standard')
    onedspec.add_spec(np.arange(100), spec_id=0, stype='standard')
    onedspec.add_arc_lines(np.arange(5), spec_id=0, stype='standard')
    onedspec.add_arc_lines([np.arange(15)], spec_id=0, stype='standard')
    onedspec.add_arc_lines([np.arange(15)], spec_id=[0], stype='standard')


@pytest.mark.xfail(raises=TypeError)
def test_add_arc_lines_standard_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='standard')
    onedspec.add_arc_lines(None, stype='standard')


# science add_trace
def test_add_trace_science():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_trace(np.arange(100), np.arange(100), stype='science')
    onedspec.add_trace(np.arange(100),
                       np.arange(100),
                       spec_id=0,
                       stype='science')
    onedspec.add_trace(np.arange(200),
                       np.arange(200),
                       spec_id=1,
                       stype='science')
    onedspec.add_trace(np.arange(200),
                       np.arange(200),
                       spec_id=[0, 1],
                       stype='science')
    onedspec.add_trace([np.arange(100), np.arange(200)],
                       [np.arange(100), np.arange(200)],
                       spec_id=[0, 2],
                       stype='science')


@pytest.mark.xfail(raises=TypeError)
def test_add_trace_science_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_trace(np.polyfit, np.ndarray, stype='science')


# standard add_trace
def test_add_trace_standard():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_trace(np.arange(100), np.arange(100), stype='standard')
    onedspec.add_trace([np.arange(100)], [np.arange(100)], stype='standard')


@pytest.mark.xfail(raises=TypeError)
def test_add_trace_standard_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_trace(np.polyfit, np.ndarray, stype='standard')


# science add_fit_coeff
def test_add_fit_coeff_science():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), spec_id=0, stype='science')
    onedspec.add_spec(np.arange(200), spec_id=1, stype='science')
    onedspec.add_spec(np.arange(100), spec_id=10, stype='science')
    onedspec.add_fit_coeff(np.arange(100), stype='science')
    onedspec.add_fit_coeff([0, 1, 2, 3, 4, 5], stype='science')
    onedspec.add_fit_coeff(np.arange(100), fit_type='leg', stype='science')
    onedspec.add_fit_coeff(np.arange(100), fit_type='cheb', stype='science')
    onedspec.add_fit_coeff(np.arange(100), spec_id=0, stype='science')
    onedspec.add_fit_coeff(np.arange(200), spec_id=1, stype='science')
    onedspec.add_fit_coeff([np.arange(100)], spec_id=0, stype='science')
    onedspec.add_fit_coeff([np.arange(200)], spec_id=1, stype='science')
    onedspec.add_fit_coeff(
        [np.arange(100), np.arange(200),
         np.arange(100)],
        fit_type=['poly', 'poly', 'poly'],
        spec_id=[0, 1, 10],
        stype='science')
    onedspec.add_fit_coeff(
        [[np.arange(100)], [np.arange(200)], [np.arange(100)]],
        fit_type=[['poly'], ['poly'], ['poly']],
        spec_id=[0, 1, 10],
        stype='science')


@pytest.mark.xfail(raises=TypeError)
def test_add_fit_coeff_science_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(10), stype='science')
    onedspec.add_wavelength_resampled(None, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_add_fit_coeff_science_fail_spec_id():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(10), stype='science')
    onedspec.add_fit_coeff(np.arange(10), spec_id=1, stype='science')


# standard add_fit_coeff
def test_add_fit_coeff_standard():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(100), stype='standard')
    onedspec.add_fit_coeff(np.arange(100), spec_id=0, stype='standard')
    onedspec.add_fit_coeff([0, 1, 2, 3, 4, 5], stype='standard')
    onedspec.add_fit_coeff(np.arange(100), fit_type='leg', stype='standard')
    onedspec.add_fit_coeff(np.arange(100), fit_type='cheb', stype='standard')
    onedspec.add_fit_coeff(np.arange(100), spec_id=0, stype='standard')
    onedspec.add_fit_coeff([np.arange(100)], spec_id=0, stype='standard')
    onedspec.add_fit_coeff([np.arange(100)], spec_id=[0], stype='standard')


@pytest.mark.xfail(raises=TypeError)
def test_add_fit_coeff_standard_fail_type():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(10), stype='standard')
    onedspec.add_fit_coeff(None, stype='standard')


# Note that standard does not care about spec_id, there can only be one
# so this test passes
def test_add_fit_coeff_standard_not_fail_spec_id():
    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_spec(np.arange(10), stype='standard')
    onedspec.add_fit_coeff(np.arange(10), spec_id=1, stype='standard')


# Note that this is testing the "relay" to the WavelengthCalibrator, but
# not testing the calibrator itself.
def test_calibrator_science():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.initialise_calibrator(spec_id=0, stype='science')
    onedspec.initialise_calibrator(spec_id=[1], stype='science')
    onedspec.initialise_calibrator(spec_id=[11, 75], stype='science')

    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_calibrator_properties(spec_id=0, stype='science')
    onedspec.set_calibrator_properties(spec_id=[1],
                                       num_pix=100,
                                       stype='science')
    onedspec.set_calibrator_properties(spec_id=[11, 75],
                                       pixel_list=np.arange(100),
                                       log_level='debug',
                                       stype='science')

    onedspec.set_hough_properties(num_slopes=1000, stype='science')
    onedspec.set_hough_properties(spec_id=0, num_slopes=1000, stype='science')
    onedspec.set_hough_properties(spec_id=[1],
                                  num_slopes=1000,
                                  stype='science')
    onedspec.set_hough_properties(spec_id=[11, 75],
                                  num_slopes=1000,
                                  stype='science')

    onedspec.set_ransac_properties(filter_close=True, stype='science')
    onedspec.set_ransac_properties(spec_id=0,
                                   filter_close=True,
                                   stype='science')
    onedspec.set_ransac_properties(spec_id=[1],
                                   filter_close=True,
                                   stype='science')
    onedspec.set_ransac_properties(spec_id=[11, 75],
                                   filter_close=True,
                                   stype='science')

    onedspec.add_user_atlas(elements=['HeXe'] * 10,
                            wavelengths=np.arange(10),
                            stype='science')
    onedspec.add_user_atlas(spec_id=0,
                            elements=['HeXe'] * 10,
                            wavelengths=np.arange(10),
                            stype='science')
    onedspec.add_user_atlas(spec_id=[1],
                            elements=['HeXe'] * 10,
                            wavelengths=np.arange(10),
                            stype='science')
    onedspec.add_user_atlas(spec_id=[11, 75],
                            elements=['HeXe'] * 10,
                            wavelengths=np.arange(10),
                            stype='science')

    assert len(
        onedspec.science_wavecal[0].spectrum1D.calibrator.atlas_elements) == 20
    onedspec.remove_atlas_lines_range(wavelength=5., tolerance=1.5, spec_id=0)
    assert len(
        onedspec.science_wavecal[0].spectrum1D.calibrator.atlas_elements) == 10
    assert len(
        onedspec.science_wavecal[1].spectrum1D.calibrator.atlas_elements) == 20
    onedspec.remove_atlas_lines_range(wavelength=5., tolerance=1.5)
    assert len(
        onedspec.science_wavecal[1].spectrum1D.calibrator.atlas_elements) == 10

    onedspec.clear_atlas()
    assert onedspec.science_wavecal[
        0].spectrum1D.calibrator.atlas_elements == []

    onedspec.add_atlas(elements=['Ar'] * 10, stype='science')
    onedspec.add_atlas(spec_id=0, elements=['Ar'] * 10, stype='science')
    onedspec.add_atlas(spec_id=[1], elements=['Ar'] * 10, stype='science')
    onedspec.add_atlas(spec_id=[11, 75], elements=['Ar'] * 10, stype='science')

    onedspec.do_hough_transform(stype='science')
    onedspec.do_hough_transform(spec_id=0, stype='science')
    onedspec.do_hough_transform(spec_id=[1], stype='science')
    onedspec.do_hough_transform(spec_id=[11, 75], stype='science')

    onedspec.set_known_pairs(pix=100, wave=4500., stype='science')
    onedspec.set_known_pairs(pix=[100], wave=[4500.], stype='science')
    onedspec.set_known_pairs(pix=[100, 200],
                             wave=[4500., 5500.],
                             stype='science')


# Fail at the RASCAL initilisation
@pytest.mark.xfail(raises=TypeError)
def test_calibrator_science2():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.initialise_calibrator(spec_id=[1, 5], stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_calibrator_properties_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(spec_id=7, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_hough_properties_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_hough_properties(spec_id=7, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_ransac_properties_science():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_hough_properties(stype='science')
    onedspec.set_ransac_properties(spec_id=7, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_known_pairs_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_hough_properties(stype='science')
    onedspec.set_ransac_properties(stype='science')
    onedspec.set_known_pairs(spec_id=7, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_add_user_atlas_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_hough_properties(stype='science')
    onedspec.set_ransac_properties(stype='science')
    onedspec.add_user_atlas(elements=['bla'],
                            wavelengths=[1234.],
                            spec_id=7,
                            stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_add_atlas_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_hough_properties(stype='science')
    onedspec.set_ransac_properties(stype='science')
    onedspec.add_atlas(elements=['Xe'], spec_id=7, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_remove_atlas_lines_range_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_hough_properties(stype='science')
    onedspec.set_ransac_properties(stype='science')
    onedspec.add_atlas(elements=['Xe'], stype='science')
    onedspec.remove_atlas_lines_range(wavelength=6000.,
                                      spec_id=7,
                                      stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_clear_atlas_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_hough_properties(stype='science')
    onedspec.set_ransac_properties(stype='science')
    onedspec.add_atlas(elements=['Xe'], stype='science')
    onedspec.clear_atlas(spec_id=0, stype='science')
    onedspec.add_atlas(elements=['Xe'], stype='science')
    onedspec.list_atlas(stype='science')
    onedspec.list_atlas(spec_id=0, stype='science')
    onedspec.clear_atlas(spec_id=7, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_list_atlas_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.add_atlas(elements=['Xe'], stype='science')
    onedspec.list_atlas(spec_id=[0, 1], stype='science')
    onedspec.list_atlas(spec_id=7, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_hough_transform_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_hough_properties(stype='science')
    onedspec.set_ransac_properties(stype='science')
    onedspec.add_atlas(elements=['Xe'], stype='science')
    onedspec.do_hough_transform(spec_id=0, stype='science')
    onedspec.do_hough_transform(spec_id=[1, 11], stype='science')
    onedspec.do_hough_transform(spec_id=7, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_fit_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_hough_properties(stype='science')
    onedspec.set_ransac_properties(stype='science')
    onedspec.add_atlas(elements=['Xe'], stype='science')
    onedspec.do_hough_transform(stype='science')
    onedspec.fit(spec_id=7, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_refine_fit_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_arc_lines(np.arange(5),
                           spec_id=[0, 1, 11, 75],
                           stype='science')
    onedspec.add_arc_spec(np.arange(100),
                          spec_id=[0, 1, 11, 75],
                          stype='science')

    onedspec.initialise_calibrator(stype='science')
    onedspec.set_calibrator_properties(stype='science')
    onedspec.set_hough_properties(stype='science')
    onedspec.set_ransac_properties(stype='science')
    onedspec.add_atlas(elements=['Xe'], stype='science')
    onedspec.do_hough_transform(stype='science')
    onedspec.science_wavecal_polynomial_available = True
    onedspec.refine_fit(spec_id=7, stype='science')


@pytest.mark.xfail(raises=ValueError)
def test_calibrator_science_fail_ap_extract_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.add_fit_coeff([1, 2, 3, 4, 5])
    onedspec.apply_wavelength_calibration(spec_id=7)


img = image_reduction.ImageReduction(
    filelist='test/test_data/sprat_LHS6328.list', log_file_name=None)
img.reduce()
twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
twodspec.add_data(img)
twodspec.ap_trace()
twodspec.ap_extract(model='lowess')


def test_from_twodspec():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.from_twodspec(twodspec, spec_id=0)


@pytest.mark.xfail()
def test_from_twodspec_fail_spec_id():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.from_twodspec(twodspec, spec_id=10)


def test_extinction_function():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.set_atmospheric_extinction()
    onedspec.set_atmospheric_extinction(
        extinction_func=np.polyfit([0, 1], [0, 1], 1))


def test_standard_library_lookup():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.lookup_standard_libraries(target='cd32d9927')
    onedspec.lookup_standard_libraries(target='agk_81d266_005')
    onedspec.lookup_standard_libraries(target='bd28')
    onedspec.lookup_standard_libraries(target='hr3454')
    onedspec.load_standard(library='esowdstan', target='agk_81d266_005')
    onedspec.inspect_standard(display=False)


def test_sensitivity():

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)

    onedspec.add_trace(np.ones(70) * 37,
                       np.ones(70),
                       spec_id=0,
                       stype='science+standard')
    onedspec.add_spec(np.arange(1, 71), spec_id=0, stype='science+standard')
    onedspec.add_arc_spec(np.arange(1, 71),
                          spec_id=0,
                          stype='science+standard')
    onedspec.add_fit_coeff(np.array((4000., 1, 0.2, 0.0071)),
                           spec_id=0,
                           stype='science+standard')
    onedspec.apply_wavelength_calibration(wave_bin=100.,
                                          wave_start=4000.,
                                          wave_end=9000.)

    onedspec.load_standard(target='cd32d9927')
    onedspec.inspect_standard(
        display=False,
        return_jsonstring=True,
        save_fig=True,
        filename='test/test_output/test_onedspec_inspect_standard')

    coeff = np.polynomial.polynomial.polyfit(np.arange(1, 1001),
                                             np.random.random(1000) * 20, 2)

    onedspec.add_sensitivity_func(
        lambda x: np.polynomial.polynomial.polyval(x, coeff))

    # Not implemented yet
    # onedspec.save_sensitivity_func('test/test_output/' +\
    # 'test_onedspec_sensitivity_func')
    onedspec.inspect_sensitivity(
        display=False,
        return_jsonstring=True,
        save_fig=True,
        filename='test/test_output/test_onedspec_inspect_sensitivity')


onedspec = spectral_reduction.OneDSpec(log_file_name=None)

onedspec.add_trace(np.ones(70) * 37,
                   np.ones(70),
                   spec_id=0,
                   stype='science+standard')
onedspec.add_spec(np.arange(1, 71), spec_id=0, stype='science+standard')
onedspec.add_arc_spec(np.arange(1, 71), spec_id=0, stype='science+standard')
onedspec.add_fit_coeff(np.array((4000., 1, 0.2, 0.0071)),
                       spec_id=0,
                       stype='science+standard')
onedspec.apply_wavelength_calibration(wave_bin=100.,
                                      wave_start=4000.,
                                      wave_end=9000.)

onedspec.load_standard(target='cd32d9927')
coeff = np.polynomial.polynomial.polyfit(np.arange(1, 1001),
                                         np.random.random(1000) * 20, 2)

onedspec.add_sensitivity_func(
    lambda x: np.polynomial.polynomial.polyval(x, coeff))
# Not implemented yet
# onedspec.save_sensitivity_func('test/test_output/' +\
# 'test_onedspec_sensitivity_func')
# onedspec.compute_sensitivity()
onedspec.set_atmospheric_extinction()
onedspec.apply_flux_calibration()

onedspec.apply_atmospheric_extinction_correction(science_airmass=1.2,
                                                 standard_airmass=1.5)

onedspec.create_fits(output='trace+count+wavelength+count_resampled+'
                     'sensitivity+flux+sensitivity_resampled+'
                     'flux_resampled',
                     empty_primary_hdu=False)


def test_miscellaneous():

    onedspec.apply_flux_calibration()
    onedspec.apply_flux_calibration(spec_id=0)
    onedspec.apply_flux_calibration(spec_id=[0])

    onedspec.apply_atmospheric_extinction_correction(science_airmass=1.2)
    onedspec.apply_atmospheric_extinction_correction(science_airmass=1.2,
                                                     standard_airmass=1.5)
    onedspec.apply_atmospheric_extinction_correction(spec_id=0,
                                                     science_airmass=1.2,
                                                     standard_airmass=1.5)
    onedspec.apply_atmospheric_extinction_correction(spec_id=[0],
                                                     science_airmass=1.2,
                                                     standard_airmass=1.5)

    onedspec.apply_flux_calibration()

    onedspec.set_atmospheric_extinction(extinction_func=np.poly1d([1, 2, 3]))

    onedspec.apply_atmospheric_extinction_correction(science_airmass=1.2)
    onedspec.apply_atmospheric_extinction_correction(science_airmass=1.2,
                                                     standard_airmass=1.5)
    onedspec.apply_atmospheric_extinction_correction(spec_id=0,
                                                     science_airmass=1.2,
                                                     standard_airmass=1.5)
    onedspec.apply_atmospheric_extinction_correction(spec_id=[0],
                                                     science_airmass=1.2,
                                                     standard_airmass=1.5)

    onedspec.apply_flux_calibration()
    onedspec.set_atmospheric_extinction()

    onedspec.apply_atmospheric_extinction_correction(science_airmass=1.2)
    onedspec.apply_atmospheric_extinction_correction(science_airmass=1.2,
                                                     standard_airmass=1.5)
    onedspec.apply_atmospheric_extinction_correction(spec_id=0,
                                                     science_airmass=1.2,
                                                     standard_airmass=1.5)
    onedspec.apply_atmospheric_extinction_correction(spec_id=[0],
                                                     science_airmass=1.2,
                                                     standard_airmass=1.5)

    onedspec.inspect_reduced_spectrum(
        display=False,
        filename='test/test_output/test_onedspec_inspect_reduced_spectrum')
    onedspec.inspect_reduced_spectrum(
        display=False,
        spec_id=0,
        renderer='default',
        return_jsonstring=True,
        save_fig=True,
        fig_type='iframe+png',
        filename='test/test_output/test_onedspec_inspect_reduced_spectrum')
    onedspec.inspect_sensitivity(
        display=False,
        renderer='default',
        return_jsonstring=True,
        save_fig=True,
        fig_type='iframe+png',
        filename='test/test_output/test_onedspec_inspect_reduced_spectrum')

    onedspec.create_fits(output='trace+count+weight_map+arc_spec+wavecal+'
                         'wavelength+count_resampled+sensitivity+flux+'
                         'sensitivity_resampled+flux_resampled',
                         empty_primary_hdu=False)
    onedspec.create_fits(output='trace+count+weight_map+arc_spec+wavecal+'
                         'wavelength+count_resampled+sensitivity+flux+'
                         'sensitivity_resampled+flux_resampled',
                         recreate=True)
    onedspec.create_fits(output='trace+count+weight_map+arc_spec+wavecal+'
                         'wavelength+count_resampled+sensitivity+flux+'
                         'sensitivity_resampled+flux_resampled',
                         spec_id=0,
                         empty_primary_hdu=False,
                         recreate=True)

    onedspec.modify_trace_header(0, 'set', 'COMMENT', 'Hello Trace!')
    onedspec.modify_count_header(0, 'set', 'COMMENT', 'Hello Count!')
    onedspec.modify_weight_map_header('set', 'COMMENT', 'Hello Weight!')
    onedspec.modify_arc_spec_header(0, 'set', 'COMMENT', 'Hello Arc Spec!')
    onedspec.modify_wavecal_header('set', 'COMMENT', 'Hello Wavecal!')
    onedspec.modify_wavelength_header('set', 'COMMENT', 'Hello Wavelength!')
    onedspec.modify_count_resampled_header(0, 'set', 'COMMENT',
                                           'Hello Count Resampled!')
    onedspec.modify_sensitivity_header('set', 'COMMENT', 'Hello Sensitivity!')
    onedspec.modify_flux_header(0, 'set', 'COMMENT', 'Hello Flux!')
    onedspec.modify_sensitivity_resampled_header(
        'set', 'COMMENT', 'Hello Sensitivity Resampled!')
    onedspec.modify_flux_resampled_header(0, 'set', 'COMMENT',
                                          'Hello Flux Resampled!')

    onedspec.modify_trace_header(0,
                                 'set',
                                 'COMMENT',
                                 'Hello Trace!',
                                 spec_id=0)
    onedspec.modify_count_header(0,
                                 'set',
                                 'COMMENT',
                                 'Hello Count!',
                                 spec_id=0)
    onedspec.modify_weight_map_header('set',
                                      'COMMENT',
                                      'Hello Weight!',
                                      spec_id=0)
    onedspec.modify_arc_spec_header(0,
                                    'set',
                                    'COMMENT',
                                    'Hello Arc Spec!',
                                    spec_id=0)
    onedspec.modify_wavecal_header('set',
                                   'COMMENT',
                                   'Hello Wavecal!',
                                   spec_id=0)
    onedspec.modify_wavelength_header('set',
                                      'COMMENT',
                                      'Hello Wavelength!',
                                      spec_id=0)
    onedspec.modify_count_resampled_header(0,
                                           'set',
                                           'COMMENT',
                                           'Hello Count Resampled!',
                                           spec_id=0)
    onedspec.modify_sensitivity_header('set',
                                       'COMMENT',
                                       'Hello Sensitivity!',
                                       spec_id=0)
    onedspec.modify_flux_header(0, 'set', 'COMMENT', 'Hello Flux!', spec_id=0)
    onedspec.modify_sensitivity_resampled_header(
        'set', 'COMMENT', 'Hello Sensitivity Resampled!', spec_id=0)
    onedspec.modify_flux_resampled_header(0,
                                          'set',
                                          'COMMENT',
                                          'Hello Flux Resampled!',
                                          spec_id=0)


@pytest.mark.xfail(raises=ValueError)
def test_modify_trace_header_fail_spec_id():
    onedspec.modify_trace_header(0,
                                 'set',
                                 'COMMENT',
                                 'Hello Trace!',
                                 spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_modify_count_header_fail_spec_id():
    onedspec.modify_count_header(0,
                                 'set',
                                 'COMMENT',
                                 'Hello Count!',
                                 spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_modify_weight_map_header_fail_spec_id():
    onedspec.modify_weight_map_header(0,
                                      'set',
                                      'COMMENT',
                                      'Hello Weight!',
                                      spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_modify_arc_spec_header_fail_spec_id():
    onedspec.modify_arc_spec_header(0,
                                    'set',
                                    'COMMENT',
                                    'Hello Arc Spec!',
                                    spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_modify_wavecal_header_fail_spec_id():
    onedspec.modify_wavecal_header('set',
                                   'COMMENT',
                                   'Hello Wavecal!',
                                   spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_modify_wavelength_header_fail_spec_id():
    onedspec.modify_wavelength_header('set',
                                      'COMMENT',
                                      'Hello Wavelength!',
                                      spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_modify_count_resampled_header_fail_spec_id():
    onedspec.modify_count_resampled_header(0,
                                           'set',
                                           'COMMENT',
                                           'Hello Count Resampled!',
                                           spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_modify_sensitivity_header_fail_spec_id():
    onedspec.modify_sensitivity_header(0,
                                       'set',
                                       'COMMENT',
                                       'Hello Sensitivity!',
                                       spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_modify_flux_header_fail_spec_id():
    onedspec.modify_flux_header(0, 'set', 'COMMENT', 'Hello Flux!', spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_modify_sensitivity_resampled_header_fail_spec_id():
    onedspec.modify_sensitivity_resampled_header(
        0, 'set', 'COMMENT', 'Hello Sensitivity Resampled!', spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_modify_flux_resampled_header_fail_spec_id():
    onedspec.modify_flux_resampled_header(0,
                                          'set',
                                          'COMMENT',
                                          'Hello Flux Resampled!',
                                          spec_id=10)


@pytest.mark.xfail(raises=ValueError)
def test_save_fits_fail_output_type():
    onedspec.save_fits(output='wave')


@pytest.mark.xfail(raises=ValueError)
def test_save_fits_fail_stype():
    onedspec.save_fits(stype='sci')


@pytest.mark.xfail(raises=ValueError)
def test_save_fits_fail_spec_id():
    onedspec.save_csv(spec_id=100)


@pytest.mark.xfail(raises=ValueError)
def test_save_csv_fail_output_type():
    onedspec.save_csv(output='wave')


@pytest.mark.xfail(raises=ValueError)
def test_save_csv_fail_stype():
    onedspec.save_csv(stype='sci')


@pytest.mark.xfail(raises=ValueError)
def test_save_csv_fail_spec_id():
    onedspec.save_csv(spec_id=100)
