import numpy as np
import os
import pytest
from aspired import image_reduction
from astropy.io import fits

base_dir = os.path.dirname(__file__)
abs_dir = os.path.abspath(os.path.join(base_dir, '..'))


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def test_logger():
    imred_debug = image_reduction.ImageReduction(
        log_level='DEBUG',
        logger_name='imred_debug',
        log_file_name='imred_debug.log',
        log_file_folder='test/test_output/')
    imred_info = image_reduction.ImageReduction(
        log_level='INFO',
        logger_name='imred_info',
        log_file_name='imred_info.log',
        log_file_folder='test/test_output/')
    imred_warning = image_reduction.ImageReduction(
        log_level='WARNING',
        logger_name='imred_warning',
        log_file_name='imred_warning.log',
        log_file_folder='test/test_output/')
    imred_error = image_reduction.ImageReduction(
        log_level='ERROR',
        logger_name='imred_error',
        log_file_name='imred_error.log',
        log_file_folder='test/test_output/')
    imred_critical = image_reduction.ImageReduction(
        log_level='CRITICAL',
        logger_name='imred_critical',
        log_file_name='imred_critical.log',
        log_file_folder='test/test_output/')

    imred_debug.logger.debug('debug: debug mode')
    imred_debug.logger.info('debug: info mode')
    imred_debug.logger.warning('debug: warning mode')
    imred_debug.logger.error('debug: error mode')
    imred_debug.logger.critical('debug: critical mode')

    imred_info.logger.debug('info: debug mode')
    imred_info.logger.info('info: info mode')
    imred_info.logger.warning('info: warning mode')
    imred_info.logger.error('info: error mode')
    imred_info.logger.critical('info: critical mode')

    imred_warning.logger.debug('warning: debug mode')
    imred_warning.logger.info('warning: info mode')
    imred_warning.logger.warning('warning: warning mode')
    imred_warning.logger.error('warning: error mode')
    imred_warning.logger.critical('warning: critical mode')

    imred_error.logger.debug('error: debug mode')
    imred_error.logger.info('error: info mode')
    imred_error.logger.warning('error: warning mode')
    imred_error.logger.error('error: error mode')
    imred_error.logger.critical('error: critical mode')

    imred_critical.logger.debug('critical: debug mode')
    imred_critical.logger.info('critical: info mode')
    imred_critical.logger.warning('critical: warning mode')
    imred_critical.logger.error('critical: error mode')
    imred_critical.logger.critical('critical: critical mode')

    debug_debug_length = file_len('test/test_output/imred_debug.log')
    debug_info_length = file_len('test/test_output/imred_info.log')
    debug_warning_length = file_len('test/test_output/imred_warning.log')
    debug_error_length = file_len('test/test_output/imred_error.log')
    debug_critical_length = file_len('test/test_output/imred_critical.log')

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
    os.remove('test/test_output/imred_debug.log')
except Exception as e:
    print(e)

try:
    os.remove('test/test_output/imred_info.log')
except Exception as e:
    print(e)

try:
    os.remove('test/test_output/imred_warning.log')
except Exception as e:
    print(e)

try:
    os.remove('test/test_output/imred_error.log')
except Exception as e:
    print(e)

try:
    os.remove('test/test_output/imred_critical.log')
except Exception as e:
    print(e)


def test_absolute_path():
    current_absolute_path = os.path.abspath(
        'test/test_data/sprat_LHS6328.list')
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist=current_absolute_path)
    img.reduce()


def test_space_separated_input():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist='test/test_data/sprat_LHS6328.txt',
                     ftype='ascii')
    img.reduce()


def test_space_separated_input_2():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist='test/test_data/sprat_LHS6328.txt',
        delimiter=' ',
    )
    img.reduce()


def test_tsv_input_2():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist='test/test_data/sprat_LHS6328.tsv',
        ftype='tsv',
        delimiter='\t',
    )
    img.reduce()


def test_input_with_hdu():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist='test/test_data/sprat_LHS6328_with_hdu.list', )
    img.reduce()


def test_input_with_extra_bracket():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist='test/test_data/sprat_LHS6328_fake_extra_bracket.list')
    img.reduce()


def test_input_with_data_cube():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist='test/test_data/sprat_LHS6328_fake_data_cube.list')
    img.reduce()


def test_input_with_data_cube_extra_bracket():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist=
        'test/test_data/sprat_LHS6328_fake_data_cube_extra_bracket.list')
    img.reduce()


@pytest.mark.xfail(raises=RuntimeError)
def test_input_with_one_dimensional_data():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist='test/test_data/sprat_LHS6328_one_dimensional_data.list')


@pytest.mark.xfail(raises=RuntimeError)
def test_input_with_wrong_light_combine_type():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist='test/test_data/sprat_LHS6328.list',
                     combinetype_light='FAIL')


@pytest.mark.xfail(raises=RuntimeError)
def test_input_with_wrong_dark_combine_type():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist='test/test_data/sprat_LHS6328.list',
        combinetype_dark='FAIL',
    )
    img.reduce()


# The bad combinetype does not affect the reduction because
# there is not any flat frame
def test_input_with_wrong_flat_combine_type():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist='test/test_data/sprat_LHS6328.list',
        combinetype_flat='FAIL',
    )
    img.reduce()


def test_input_with_wrong_bias_combine_type():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist='test/test_data/sprat_LHS6328.list',
        combinetype_bias='FAIL',
    )
    img.reduce()


@pytest.mark.xfail(raises=RuntimeError)
def test_input_with_only_one_column():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist='test/test_data/sprat_LHS6328_expect_fail.list', )


def test_cosmicray_cleaning_x_then_y():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist='test/test_data/sprat_LHS6328.list',
                     cosmicray=True,
                     psfmodel='gaussxy',
                     combinetype_light='average')
    img.reduce()


def test_cosmicray_cleaning_y_then_x():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist='test/test_data/sprat_LHS6328.list',
                     cosmicray=True,
                     psfmodel='gaussyx',
                     combinetype_dark='average')
    img.reduce()


def test_cosmicray_cleaning():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist='test/test_data/sprat_LHS6328.list',
                     cosmicray=True,
                     psfmodel='gaussx',
                     combinetype_bias='average')
    img.reduce()


def test_input_with_multiple_frames_to_combine_and_reduction():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(
        filelist='test/test_data/sprat_LHS6328_repeated_data.list',
        combinetype_flat='average')
    img.reduce()


def test_input_with_one_line_and_reduction():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist='test/test_data/sprat_LHS6328_one_line.list')
    img.reduce()


def test_input_with_numpy_array_and_reduction():
    filelist = np.loadtxt('test/test_data/sprat_LHS6328.list',
                          delimiter=',',
                          dtype='object')
    for i, filepath in enumerate(filelist[:, 1]):
        filelist[:, 1][i] = os.path.join('test/test_data/', filepath.strip())
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist=filelist)
    img.reduce()


def test_reduction_and_save():
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist='test/test_data/sprat_LHS6328.list')
    img.reduce()
    img.save_fits('test/test_output/reduced_image', overwrite=True)
    img.inspect(display=False,
                filename='test/test_output/reduced_image',
                save_fig=True,
                fig_type='iframe+png+svg+pdf+jpg')
    img.list_files()


def test_input_with_numpy_array_and_clean_bad_pixels():
    filelist = np.loadtxt('test/test_data/sprat_LHS6328.list',
                          delimiter=',',
                          dtype='object')
    for i, filepath in enumerate(filelist[:, 1]):
        filelist[:, 1][i] = os.path.join('test/test_data/', filepath.strip())
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist=filelist)
    img.reduce()
    img.heal_bad_pixels()


def test_input_with_numpy_array_and_set_every_pixel_bad():
    filelist = np.loadtxt('test/test_data/sprat_LHS6328.list',
                          delimiter=',',
                          dtype='object')
    for i, filepath in enumerate(filelist[:, 1]):
        filelist[:, 1][i] = os.path.join('test/test_data/', filepath.strip())
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist=filelist)
    img.reduce()
    img.heal_bad_pixels(np.zeros_like(img.light_reduced))


@pytest.mark.xfail(raises=RuntimeError)
def test_input_with_numpy_array_and_clean_bad_pixels_expect_fail():
    filelist = np.loadtxt('test/test_data/sprat_LHS6328.list',
                          delimiter=',',
                          dtype='object')
    for i, filepath in enumerate(filelist[:, 1]):
        filelist[:, 1][i] = os.path.join('test/test_data/', filepath.strip())
    img = image_reduction.ImageReduction(log_file_name=None)
    img.add_filelist(filelist=filelist)
    img.reduce()
    img.create_bad_mask()
    img.heal_bad_pixels(bad_mask=np.ones(100))


def test_input_with_fits_data_object():
    filelist = np.loadtxt('test/test_data/sprat_LHS6328.list',
                          delimiter=',',
                          dtype='object')
    for i, filepath in enumerate(filelist[:, 1]):
        filelist[:, 1][i] = os.path.join('test/test_data/', filepath.strip())
    img = image_reduction.ImageReduction(log_file_name=None)
    imtype = filelist[:, 0].astype('object')
    impath = filelist[:, 1].astype('object')

    dark_list = impath[imtype == 'dark']
    arc_list = impath[imtype == 'arc']
    light_list = impath[imtype == 'light']

    for i in range(light_list.size):

        light = fits.open(light_list[i])[0]
        img.add_light(light.data, light.header, light.header['EXPTIME'])

    for i in range(arc_list.size):

        arc = fits.open(arc_list[i])[0]
        img.add_arc(arc.data, arc.header)

    for i in range(dark_list.size):

        dark = fits.open(dark_list[i])[0]
        img.add_dark(dark.data, dark.header, dark.header['EXPTIME'])

    img.reduce()
