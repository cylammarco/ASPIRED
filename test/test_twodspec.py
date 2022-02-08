# -*- coding: utf-8 -*-
import copy
import os
from unittest.mock import patch

from astropy.io import fits
import numpy as np
import pytest

from aspired import image_reduction
from aspired import spectral_reduction
from aspired import util

HERE = os.path.dirname(os.path.realpath(__file__))

img = image_reduction.ImageReduction(log_file_name=None)
img.add_filelist(
    filelist=os.path.join(HERE, "test_data", "sprat_LHS6328.list")
)
img.load_data()
img.reduce()

img_with_fits = copy.copy(img)
img_with_fits._create_image_fits()

img_in_hdulist = fits.HDUList(img_with_fits.image_fits)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def test_logger():
    twodspec_debug = spectral_reduction.TwoDSpec(
        log_level="DEBUG",
        logger_name="twodspec_debug",
        log_file_name="twodspec_debug.log",
        log_file_folder=os.path.join(HERE, "test_output"),
    )
    twodspec_info = spectral_reduction.TwoDSpec(
        log_level="INFO",
        logger_name="twodspec_info",
        log_file_name="twodspec_info.log",
        log_file_folder=os.path.join(HERE, "test_output"),
    )
    twodspec_warning = spectral_reduction.TwoDSpec(
        log_level="WARNING",
        logger_name="twodspec_warning",
        log_file_name="twodspec_warning.log",
        log_file_folder=os.path.join(HERE, "test_output"),
    )
    twodspec_error = spectral_reduction.TwoDSpec(
        log_level="ERROR",
        logger_name="twodspec_error",
        log_file_name="twodspec_error.log",
        log_file_folder=os.path.join(HERE, "test_output"),
    )
    twodspec_critical = spectral_reduction.TwoDSpec(
        log_level="CRITICAL",
        logger_name="twodspec_critical",
        log_file_name="twodspec_critical.log",
        log_file_folder=os.path.join(HERE, "test_output"),
    )

    twodspec_debug.logger.debug("debug: debug mode")
    twodspec_debug.logger.info("debug: info mode")
    twodspec_debug.logger.warning("debug: warning mode")
    twodspec_debug.logger.error("debug: error mode")
    twodspec_debug.logger.critical("debug: critical mode")

    twodspec_info.logger.debug("info: debug mode")
    twodspec_info.logger.info("info: info mode")
    twodspec_info.logger.warning("info: warning mode")
    twodspec_info.logger.error("info: error mode")
    twodspec_info.logger.critical("info: critical mode")

    twodspec_warning.logger.debug("warning: debug mode")
    twodspec_warning.logger.info("warning: info mode")
    twodspec_warning.logger.warning("warning: warning mode")
    twodspec_warning.logger.error("warning: error mode")
    twodspec_warning.logger.critical("warning: critical mode")

    twodspec_error.logger.debug("error: debug mode")
    twodspec_error.logger.info("error: info mode")
    twodspec_error.logger.warning("error: warning mode")
    twodspec_error.logger.error("error: error mode")
    twodspec_error.logger.critical("error: critical mode")

    twodspec_critical.logger.debug("critical: debug mode")
    twodspec_critical.logger.info("critical: info mode")
    twodspec_critical.logger.warning("critical: warning mode")
    twodspec_critical.logger.error("critical: error mode")
    twodspec_critical.logger.critical("critical: critical mode")

    debug_debug_length = file_len(
        os.path.join(HERE, "test_output", "twodspec_debug.log")
    )
    debug_info_length = file_len(
        os.path.join(HERE, "test_output", "twodspec_info.log")
    )
    debug_warning_length = file_len(
        os.path.join(HERE, "test_output", "twodspec_warning.log")
    )
    debug_error_length = file_len(
        os.path.join(HERE, "test_output", "twodspec_error.log")
    )
    debug_critical_length = file_len(
        os.path.join(HERE, "test_output", "twodspec_critical.log")
    )

    assert (
        debug_debug_length == 10
    ), "Expecting 10 lines in the log file, " + "{} is logged.".format(
        debug_debug_length
    )
    assert (
        debug_info_length == 9
    ), "Expecting 9 lines in the log file, " + "{} is logged.".format(
        debug_info_length
    )
    assert (
        debug_warning_length == 8
    ), "Expecting 8 lines in the log file, " + "{} is logged.".format(
        debug_warning_length
    )
    assert (
        debug_error_length == 2
    ), "Expecting 2 lines in the log file, " + "{} is logged.".format(
        debug_error_length
    )
    assert (
        debug_critical_length == 1
    ), "Expecting 1 lines in the log file, " + "{} is logged.".format(
        debug_critical_length
    )


try:
    os.remove("test/test_output/twodspec_debug.log")
except Exception as e:
    print(e)

try:
    os.remove("test/test_output/twodspec_info.log")
except Exception as e:
    print(e)

try:
    os.remove("test/test_output/twodspec_warning.log")
except Exception as e:
    print(e)

try:
    os.remove("test/test_output/twodspec_error.log")
except Exception as e:
    print(e)

try:
    os.remove("test/test_output/twodspec_critical.log")
except Exception as e:
    print(e)


def test_add_data_image_reduction():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(img)
    assert (twodspec.img == img.image_fits.data).all()
    assert twodspec.header == img.image_fits.header


def test_add_data_numpy_array():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(
        img_with_fits.image_fits.data, img_with_fits.image_fits.header
    )
    assert (twodspec.img == img_with_fits.image_fits.data).all()
    assert twodspec.header == img_with_fits.image_fits.header


def test_add_data_primaryhdu():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(img_with_fits.image_fits)
    assert (twodspec.img == img_with_fits.image_fits.data).all()
    assert twodspec.header == img_with_fits.image_fits.header


def test_add_data_hdulist():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(img_in_hdulist)
    assert (twodspec.img == img_with_fits.image_fits.data).all()
    assert twodspec.header == img_with_fits.image_fits.header


def test_add_data_path():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(
        os.path.join(HERE, "test_data", "v_e_20180810_12_1_0_0.fits.gz")
    )
    data_array = fits.open(
        os.path.join(HERE, "test_data", "v_e_20180810_12_1_0_0.fits.gz")
    )[0]
    assert (twodspec.img == data_array.data).all()
    assert twodspec.header == data_array.header


def test_add_no_data():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(None)


@pytest.mark.xfail()
def test_add_random_data_expect_fail():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(np.polyfit)


def test_set_all_properties():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(img)
    twodspec.set_properties(
        saxis=1,
        variance=10.0,
        spatial_mask=(1,),
        spec_mask=(1,),
        flip=True,
        cosmicray=True,
        sigclip=3.0,
        readnoise=10.0,
        gain=2.6,
        seeing=1.7,
        exptime=300.0,
        airmass=1.5,
        verbose=True,
    )
    assert twodspec.saxis == 1
    assert twodspec.waxis == 0
    assert (twodspec.variance == 10.0).all()
    assert twodspec.spatial_mask == (1,)
    assert twodspec.spec_mask == (1,)
    assert twodspec.flip
    assert twodspec.cosmicray
    assert twodspec.readnoise == 10.0
    assert twodspec.gain == 2.6
    assert twodspec.seeing == 1.7
    assert twodspec.exptime == 300.0
    assert twodspec.airmass == 1.5
    assert twodspec.verbose
    # Assert the cosmic ray cleaning is a different copy of the data
    assert twodspec.img is not img.image_fits.data

    # Changing the saxis, and everything else should still be the same
    twodspec.set_properties(
        saxis=0,
        gain=None,
        seeing=None,
        exptime=None,
        airmass=None,
        variance=None,
    )
    assert twodspec.saxis == 0
    assert twodspec.waxis == 1
    assert twodspec.spatial_mask == (1,)
    assert twodspec.spec_mask == (1,)
    assert twodspec.flip
    assert twodspec.cosmicray
    assert twodspec.readnoise == 10.0
    # The gain is now read from the header, so it's 2.45
    assert twodspec.gain == 2.45
    # The seeing is now read from the header, so it's 0.712134
    assert twodspec.seeing == 0.712134
    # The *total* exposure time is now read from the header, so it's 240.
    assert twodspec.exptime == 240.0
    # The airmass is now read from the header, so it's 1.250338
    assert twodspec.airmass == 1.250338
    assert twodspec.verbose

    # Resetting all values to the header values
    twodspec.set_properties(saxis=0, cosmicray=False, variance=None)
    # The readnoise is now set
    twodspec.set_readnoise(20.0)
    # The gain is now set
    twodspec.set_gain(np.pi)
    # The seeing is now set
    twodspec.set_seeing(123.4)
    # The seeing is now set
    twodspec.set_exptime(0.1234)
    # The airmass is now set
    twodspec.set_airmass(1.2345)
    # Asset all the changes
    assert twodspec.readnoise == 20.0
    assert twodspec.gain == np.pi
    assert twodspec.seeing == 123.4
    assert twodspec.exptime == 0.1234
    assert twodspec.airmass == 1.2345

    # Now without resetting, setting all the values based on the given
    # FITS header keyword.
    twodspec.set_readnoise("CRDNOISE")
    # The gain is now read from the header, so it's 2.45
    twodspec.set_gain("GAIN")
    # The seeing is now read from the header, so it's 0.712134
    twodspec.set_seeing("ESTSEE")
    # The exptime is now read from the header, so it's 120.
    twodspec.set_exptime("EXPTIME")
    # The airmass is now read from the header, so it's 1.250338
    twodspec.set_airmass("AIRMASS")
    # Asset all the changes
    assert twodspec.readnoise == 0.0
    assert twodspec.gain == 2.45
    assert twodspec.seeing == 0.712134
    assert twodspec.exptime == 120.0
    assert twodspec.airmass == 1.250338

    twodspec.add_data(img_with_fits)
    # Now set all of them to zeros
    twodspec.set_readnoise(0)
    twodspec.set_gain(0)
    twodspec.set_seeing(0)
    twodspec.set_exptime(0)
    twodspec.set_airmass(0)

    # Now add the header keywords without updating
    twodspec.set_readnoise_keyword("CRDNOISE", append=True, update=False)
    twodspec.set_gain_keyword("GAIN", append=True, update=False)
    twodspec.set_seeing_keyword("ESTSEE", append=True, update=False)
    twodspec.set_exptime_keyword("EXPTIME", append=True, update=False)
    twodspec.set_airmass_keyword("AIRMASS", append=True, update=False)

    # Asset nothing is changed
    assert twodspec.readnoise == 0
    assert twodspec.gain == 0
    assert twodspec.seeing == 0
    assert twodspec.exptime == 0
    assert twodspec.airmass == 0

    # Now update
    twodspec.set_readnoise_keyword("CRDNOISE")
    twodspec.set_gain_keyword("GAIN")
    twodspec.set_seeing_keyword("ESTSEE")
    twodspec.set_exptime_keyword("EXPTIME")
    twodspec.set_airmass_keyword("AIRMASS")

    # Asset all the changes
    assert twodspec.readnoise == 0.0
    assert twodspec.gain == 2.45
    assert twodspec.seeing == 0.712134
    assert twodspec.exptime == 120.0
    assert twodspec.airmass == 1.250338

    # Again, supplying with list instead
    twodspec.set_readnoise_keyword(["CRDNOISE"])
    twodspec.set_gain_keyword(["GAIN"])
    twodspec.set_seeing_keyword(["ESTSEE"])
    twodspec.set_exptime_keyword(["EXPTIME"])
    twodspec.set_airmass_keyword(["AIRMASS"])

    # Asset
    assert twodspec.readnoise == 0.0
    assert twodspec.gain == 2.45
    assert twodspec.seeing == 0.712134
    assert twodspec.exptime == 120.0
    assert twodspec.airmass == 1.250338

    # Again, supplying with numpy.ndarray instead
    twodspec.set_readnoise_keyword(np.array(["CRDNOISE"]))
    twodspec.set_gain_keyword(np.array(["GAIN"]))
    twodspec.set_seeing_keyword(np.array(["ESTSEE"]))
    twodspec.set_exptime_keyword(np.array(["EXPTIME"]))
    twodspec.set_airmass_keyword(np.array(["AIRMASS"]))

    # Asset
    assert twodspec.readnoise == 0.0
    assert twodspec.gain == 2.45
    assert twodspec.seeing == 0.712134
    assert twodspec.exptime == 120.0
    assert twodspec.airmass == 1.250338

    # Again, supplying None
    twodspec.set_readnoise_keyword(None)
    twodspec.set_gain_keyword(None)
    twodspec.set_seeing_keyword(None)
    twodspec.set_exptime_keyword(None)
    twodspec.set_airmass_keyword(None)

    # Asset
    assert twodspec.readnoise == 0
    assert twodspec.gain == 1
    assert twodspec.seeing == 1
    assert twodspec.exptime == 1
    assert twodspec.airmass == 1


image_fits = fits.open(
    os.path.join(HERE, "test_data", "v_e_20180810_12_1_0_0.fits.gz")
)[0]
arc_fits = fits.open(
    os.path.join(HERE, "test_data", "v_a_20180810_13_1_0_1.fits.gz")
)[0]


def test_add_arc_image_reduction():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(img)
    assert (twodspec.arc == arc_fits.data).all()
    assert twodspec.arc_header == arc_fits.header


def test_add_arc_numpy_array():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_arc(arc_fits.data, arc_fits.header)
    assert (twodspec.arc == arc_fits.data).all()
    assert twodspec.arc_header == arc_fits.header


def test_add_arc_numpy_array_and_header_in_list():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_arc(arc_fits.data, [arc_fits.header])
    assert (twodspec.arc == arc_fits.data).all()
    assert twodspec.arc_header == arc_fits.header


def test_add_arc_primaryhdu():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_arc(arc_fits)
    assert (twodspec.arc == arc_fits.data).all()
    assert twodspec.arc_header == arc_fits.header


def test_add_arc_hdulist():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_arc(fits.HDUList(arc_fits))
    assert (twodspec.arc == arc_fits.data).all()
    assert twodspec.arc_header == arc_fits.header


def test_add_arc_path():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_arc(
        os.path.join(HERE, "test_data", "v_a_20180810_13_1_0_1.fits.gz")
    )
    assert (twodspec.arc == arc_fits.data).all()
    assert twodspec.arc_header == arc_fits.header


def test_add_arc_path_with_hdu_number():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_arc(
        os.path.join(HERE, "test_data", "v_a_20180810_13_1_0_1.fits.gz[0]")
    )
    assert (twodspec.arc == arc_fits.data).all()
    assert twodspec.arc_header == arc_fits.header


@pytest.mark.xfail()
def test_add_arc_path_wrong_hdu_number_expect_fail():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_arc(
        os.path.join(HERE, "test_data", "v_a_20180810_13_1_0_1.fits.gz[10]")
    )


# Create some bad data
some_bad_data = copy.copy(image_fits.data)
len_x, len_y = image_fits.data.shape
random_x = np.random.choice(np.arange(len_x), size=10, replace=False)
random_y = np.random.choice(np.arange(len_y), size=10, replace=False)
for i, j in zip(random_x, random_y):
    some_bad_data[i, j] += 1e10

# Add a bad pixel on the spectrum
some_bad_data[130, 500] += 1e10

cmask, _ = util.create_cutoff_mask(image_fits.data, cutoff=1000)
bmask, _ = util.create_bad_pixel_mask(image_fits.data)
bad_mask = bmask & bmask


def test_add_bad_pixel_mask_numpy_array():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_bad_mask(bad_mask)


def test_add_bad_pixel_mask_hdu():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_bad_mask(fits.ImageHDU(bad_mask.astype("int")))


def test_add_bad_pixel_mask_hdu_list():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_bad_mask(fits.HDUList(fits.ImageHDU(bad_mask.astype("int"))))


@pytest.mark.xfail()
def test_add_bad_pixel_mask_expect_fail():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_bad_mask(np.polyval)


# gauss ap_extract
@patch("plotly.graph_objects.Figure.show")
def test_gauss_ap_extract(mock_show):
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(img)
    twodspec.ap_trace()
    twodspec.ap_extract(model="gauss")
    twodspec.inspect_extracted_spectrum(
        display=False,
        save_fig=True,
        return_jsonstring=True,
        filename=os.path.join(HERE, "test_output", "extracted_spectrum"),
    )
    twodspec.inspect_residual(
        display=True,
        save_fig=True,
        return_jsonstring=True,
        filename=os.path.join(HERE, "test_output", "residual_image"),
    )


# lowess ap_extract
@patch("plotly.graph_objects.Figure.show")
def test_lowess_ap_extract(mock_show):
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(img)
    twodspec.ap_trace()
    twodspec.ap_extract(model="lowess")
    twodspec.inspect_extracted_spectrum(
        display=True,
        save_fig=True,
        return_jsonstring=True,
        filename="test/test_output/extracted_spectrum",
    )
    twodspec.inspect_residual(
        display=False,
        save_fig=True,
        return_jsonstring=True,
        filename=os.path.join(HERE, "test_output", "residual_image"),
    )


# compare gauss and lowess ap_extract
def test_gauss_vs_lowess_ap_extract():
    twodspec_gauss = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec_gauss.add_data(img)
    twodspec_gauss.ap_trace()
    twodspec_lowess = copy.copy(twodspec_gauss)
    twodspec_gauss.ap_extract(model="gauss")
    twodspec_lowess.ap_extract(model="lowess")
    count_g = np.nansum(twodspec_gauss.spectrum_list[0].count)
    count_l = np.nansum(twodspec_lowess.spectrum_list[0].count)
    assert (count_l > count_g * 0.95) & (count_l < count_g * 1.05)


missing_lower_half_spectrum_image_fits = fits.open(
    "test/test_data/v_e_20180810_12_1_0_0.fits.gz"
)[0]
missing_lower_half_spectrum_image_fits.data = (
    missing_lower_half_spectrum_image_fits.data[126:200]
)


# ap_extract at detector edge
def test_gauss_ap_extract_lower_detector_edge():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(missing_lower_half_spectrum_image_fits)
    twodspec.ap_trace()
    twodspec.ap_extract(model="gauss")


# ap_extract at detector edge
def test_lowess_ap_extract_lower_detector_edge():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(missing_lower_half_spectrum_image_fits)
    twodspec.ap_trace()
    twodspec.ap_extract(model="lowess")


missing_upper_half_spectrum_image_fits = fits.open(
    "test/test_data/v_e_20180810_12_1_0_0.fits.gz"
)[0]
missing_upper_half_spectrum_image_fits.data = (
    missing_upper_half_spectrum_image_fits.data[100:131]
)


# ap_extract at detector edge
def test_gauss_ap_extract_upper_detector_edge():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(missing_upper_half_spectrum_image_fits)
    twodspec.ap_trace()
    twodspec.ap_extract(model="gauss")


# ap_extract at detector edge
def test_lowess_ap_extract_upper_detector_edge():
    twodspec = spectral_reduction.TwoDSpec(log_file_name=None)
    twodspec.add_data(missing_upper_half_spectrum_image_fits)
    twodspec.ap_trace()
    twodspec.ap_extract(model="lowess")
