# -*- coding: utf-8 -*-
import os

import numpy as np
from aspired import image_reduction, spectral_reduction
from astropy.io import fits

HERE = os.path.dirname(os.path.realpath(__file__))


def gaussian(pixels, central_pixel):
    """
    Parameters
    ----------
    log_age: array
        the age to return the SFH.
    peak_age: float
        the time of the maximum star formation.
    Returns
    -------
    The relative SFH at the given log_age location.
    """
    stdv = 1.0
    variance = stdv**2.0
    g = (
        np.exp(-((pixels - central_pixel) ** 2.0) / 2 / variance)
        / np.sqrt(2 * np.pi)
        / stdv
    )
    return g


# background noise
bg_level = 5.0

# Prepare dummy data
# total signal should be [2 * (2 + 5 + 10) + 20] - [1 * 7] = 47
dummy_data = np.ones((100, 1000)) * bg_level

dummy_data[47] += 2.0
dummy_data[48] += 5.0
dummy_data[49] += 10.0
dummy_data[50] += 50.0
dummy_data[51] += 10.0
dummy_data[52] += 5.0
dummy_data[53] += 2.0
dummy_data = np.random.normal(dummy_data)


# Prepare dummy gaussian data
dummy_gaussian_data = (
    np.ones((100, 1000)).T * gaussian(np.arange(100), 50)
).T * 10000.0
dummy_gaussian_data = (
    np.random.normal(dummy_gaussian_data, scale=bg_level) + bg_level
)

# Prepare faint dummy gaussian data
dummy_gaussian_data_faint = (
    np.ones((100, 1000)).T * gaussian(np.arange(100), 50)
).T * 100.0
dummy_gaussian_data_faint = (
    np.random.normal(dummy_gaussian_data_faint, scale=bg_level) + bg_level
)


def test_spectral_extraction():
    # masking
    spec_mask = np.arange(10, 900)
    spatial_mask = np.arange(15, 85)

    # initialise the two spectral_reduction.TwoDSpec()
    dummy_twodspec = spectral_reduction.TwoDSpec(
        dummy_data,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        log_file_name=None,
        log_level="DEBUG",
        saxis=1,
        flip=False,
        cosmicray_sigma=5.0,
        readnoise=0.1,
        gain=1.0,
        seeing=1.0,
        exptime=1.0,
    )

    # Trace the spectrum, note that the first 15 rows were trimmed from the
    # spatial_mask
    dummy_twodspec.ap_trace(
        rescale=True,
        ap_faint=0,
        save_fig=True,
        fig_type="iframe+png",
        filename=os.path.join(HERE, "test_output", "test_extraction_aptrace"),
        return_jsonstring=True,
    )
    trace = np.round(np.mean(dummy_twodspec.spectrum_list[0].trace))
    assert np.isclose(trace, 35, atol=1.0), (
        "Trace is at row "
        + str(trace)
        + ", but it is expected to be at row 35."
    )

    # Optimal extracting spectrum by summing over the aperture along the trace
    dummy_twodspec.ap_extract(
        apwidth=5,
        optimal=False,
        filename=os.path.join(
            HERE, "test_output", "test_extraction_apextract"
        ),
        save_fig=True,
        fig_type="iframe+png",
        return_jsonstring=True,
    )

    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    assert np.isclose(count, 84, atol=1.0), (
        "Extracted count is " + str(count) + " but it should be 84."
    )

    dummy_twodspec.inspect_extracted_spectrum(
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_extraction_extracted_spectrum"
        ),
        save_fig=True,
        fig_type="iframe+png",
        return_jsonstring=True,
    )


def test_gaussian_spectral_extraction():
    # masking
    spec_mask = np.arange(10, 900)
    spatial_mask = np.arange(15, 85)

    # initialise the two spectral_reduction.TwoDSpec()
    dummy_twodspec = spectral_reduction.TwoDSpec(
        dummy_gaussian_data,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        log_file_name=None,
        log_level="CRITICAL",
        saxis=1,
        flip=False,
        cosmicray_sigma=5.0,
        readnoise=0.1,
        gain=1.0,
        seeing=1.0,
        exptime=1.0,
    )

    # Trace the spectrum, note that the first 15 rows were trimmed from the
    # spatial_mask
    dummy_twodspec.ap_trace(
        rescale=True,
        fit_deg=0,
    )
    trace = np.round(np.mean(dummy_twodspec.spectrum_list[0].trace))
    assert np.isclose(trace, 35, atol=1.0), (
        "Trace is at row "
        + str(trace)
        + ", but it is expected to be at row 35."
    )

    # Direct extraction by summing over the aperture along the trace
    dummy_twodspec.ap_extract(
        apwidth=5,
        optimal=False,
    )
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    count_err = np.mean(dummy_twodspec.spectrum_list[0].count_err)
    snr_tophat = count / count_err
    assert np.isclose(count, 10000.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~10000."
    )

    # Optimal extraction (Horne86 gauss)
    dummy_twodspec.ap_extract(apwidth=5, optimal=True, model="gauss")
    dummy_twodspec.inspect_line_spread_function(
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_extraction_line_spread_function_gauss"
        ),
        save_fig=True,
        fig_type="iframe+png",
        return_jsonstring=True,
    )
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    count_err = np.mean(dummy_twodspec.spectrum_list[0].count_err)
    snr_horne = count / count_err
    assert np.isclose(count, 10000.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~10000."
    )

    # Optimal extraction (Horne86 lowess)
    dummy_twodspec.ap_extract(
        apwidth=5,
        optimal=True,
        model="lowess",
        lowess_frac=0.05,
    )
    dummy_twodspec.inspect_line_spread_function(
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_extraction_line_spread_function_lowess"
        ),
        save_fig=True,
        fig_type="iframe+png",
        return_jsonstring=True,
    )
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    print(count)
    count_err = np.mean(dummy_twodspec.spectrum_list[0].count_err)
    snr_horne = count / count_err
    assert np.isclose(count, 10000.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~10000."
    )

    # Optimal extraction (Marsh89)
    dummy_twodspec.ap_extract(apwidth=5, optimal=True, algorithm="marsh89")
    dummy_twodspec.inspect_line_spread_function(
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_extraction_line_spread_function_marsh"
        ),
        save_fig=True,
        fig_type="iframe+png",
        return_jsonstring=True,
    )
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    count_err = np.mean(dummy_twodspec.spectrum_list[0].count_err)
    snr_marsh = count / count_err
    assert np.isclose(count, 10000.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~10000."
    )


def test_gaussian_spectral_extraction_top_hat_low_signal():
    # masking
    spec_mask = np.arange(10, 900)
    spatial_mask = np.arange(15, 85)

    # initialise the two spectral_reduction.TwoDSpec()
    dummy_twodspec = spectral_reduction.TwoDSpec(
        dummy_gaussian_data_faint,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        log_file_name=None,
        log_level="CRITICAL",
        saxis=1,
        flip=False,
        cosmicray_sigma=5.0,
        readnoise=0.1,
        gain=1.0,
        seeing=1.0,
        exptime=1.0,
    )

    # Trace the spectrum, note that the first 15 rows were trimmed from the
    # spatial_mask
    dummy_twodspec.ap_trace(
        rescale=True,
        fit_deg=0,
    )
    trace = np.round(np.mean(dummy_twodspec.spectrum_list[0].trace))
    assert np.isclose(trace, 35, atol=1.0), (
        "Trace is at row "
        + str(trace)
        + ", but it is expected to be at row 35."
    )

    # Direct extraction by summing over the aperture along the trace
    dummy_twodspec.ap_extract(
        apwidth=5,
        optimal=False,
    )
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    count_err = np.mean(dummy_twodspec.spectrum_list[0].count_err)
    snr_tophat = count / count_err
    assert np.isclose(count, 100.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~100."
    )


def test_gaussian_spectral_extraction_horne86_gaussian_low_signal():
    # masking
    spec_mask = np.arange(10, 900)
    spatial_mask = np.arange(15, 85)

    # initialise the two spectral_reduction.TwoDSpec()
    dummy_twodspec = spectral_reduction.TwoDSpec(
        dummy_gaussian_data_faint,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        log_file_name=None,
        log_level="CRITICAL",
        saxis=1,
        flip=False,
        cosmicray_sigma=5.0,
        readnoise=0.1,
        gain=1.0,
        seeing=1.0,
        exptime=1.0,
    )

    # Trace the spectrum, note that the first 15 rows were trimmed from the
    # spatial_mask
    dummy_twodspec.ap_trace(
        rescale=True,
        fit_deg=0,
    )

    # Optimal extraction (Horne86 gauss)
    dummy_twodspec.ap_extract(
        apwidth=5, optimal=True, algorithm="horne86", model="gauss"
    )
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    count_err = np.mean(dummy_twodspec.spectrum_list[0].count_err)
    snr_horne = count / count_err
    assert np.isclose(count, 100.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~100."
    )


def test_gaussian_spectral_extraction_horne86_lowess_low_signal():
    # masking
    spec_mask = np.arange(10, 900)
    spatial_mask = np.arange(15, 85)

    # initialise the two spectral_reduction.TwoDSpec()
    dummy_twodspec = spectral_reduction.TwoDSpec(
        dummy_gaussian_data_faint,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        log_file_name=None,
        log_level="CRITICAL",
        saxis=1,
        flip=False,
        cosmicray_sigma=5.0,
        readnoise=0.1,
        gain=1.0,
        seeing=1.0,
        exptime=1.0,
    )

    # Trace the spectrum, note that the first 15 rows were trimmed from the
    # spatial_mask
    dummy_twodspec.ap_trace(
        rescale=True,
        fit_deg=0,
    )

    # Optimal extraction (Horne86 lowess)
    dummy_twodspec.ap_extract(
        apwidth=5,
        optimal=True,
        model="lowess",
        lowess_frac=0.05,
    )
    count = np.median(dummy_twodspec.spectrum_list[0].count)
    count_err = np.median(dummy_twodspec.spectrum_list[0].count_err)
    snr_horne = count / count_err
    assert np.isclose(count, 100.0, rtol=0.01, atol=count_err * 2.0), (
        "Extracted count is " + str(count) + " but it should be ~100."
    )


def test_gaussian_spectral_extraction_marsh89_low_signal():
    # masking
    spec_mask = np.arange(10, 900)
    spatial_mask = np.arange(15, 85)

    # initialise the two spectral_reduction.TwoDSpec()
    dummy_twodspec = spectral_reduction.TwoDSpec(
        dummy_gaussian_data_faint,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        log_file_name=None,
        log_level="CRITICAL",
        saxis=1,
        flip=False,
        cosmicray_sigma=5.0,
        readnoise=0.1,
        gain=1.0,
        seeing=1.0,
        exptime=1.0,
    )

    # Trace the spectrum, note that the first 15 rows were trimmed from the
    # spatial_mask
    dummy_twodspec.ap_trace(
        rescale=True,
        fit_deg=0,
    )

    # Optimal extraction (Marsh89)
    dummy_twodspec.ap_extract(apwidth=5, optimal=True, algorithm="marsh89")
    count = np.median(dummy_twodspec.spectrum_list[0].count)
    count_err = np.median(dummy_twodspec.spectrum_list[0].count_err)
    snr_marsh = count / count_err
    assert np.isclose(count, 100.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~100."
    )


def test_user_supplied_trace():
    spatial_mask = np.arange(20, 200)
    spec_mask = np.arange(100, 1024)

    # Loading a single pre-saved spectral trace.
    lhs6328_extracted = fits.open(
        os.path.join(HERE, "test_data", "test_full_run_science_0.fits")
    )
    lhs6328_trace = lhs6328_extracted["trace"].data
    lhs6328_trace_sigma = lhs6328_extracted["trace_sigma"].data

    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        os.path.join(HERE, "test_data", "v_e_20180810_12_1_0_0.fits.gz"),
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        readnoise=2.34,
        log_file_name=None,
    )

    lhs6328_twodspec.add_trace(
        trace=lhs6328_trace, trace_sigma=lhs6328_trace_sigma
    )

    lhs6328_twodspec.ap_extract(
        apwidth=15, optimal=True, skywidth=10, skydeg=1, display=False
    )

    lhs6328_twodspec.save_fits(
        output="count",
        filename=os.path.join(
            HERE, "test_output", "user_supplied_trace_for_extraction"
        ),
        overwrite=True,
    )

    lhs6328_twodspec.remove_trace()


spatial_mask = np.arange(30, 200)
spec_mask = np.arange(50, 1024)

# Science frame
lhs6328_frame = image_reduction.ImageReduction(
    log_level="INFO", log_file_name=None
)
lhs6328_frame.add_filelist(
    os.path.join(HERE, "test_data", "sprat_LHS6328.list")
)
lhs6328_frame.load_data()
lhs6328_frame.reduce()

lhs6328_twodspec = spectral_reduction.TwoDSpec(
    lhs6328_frame,
    spatial_mask=spatial_mask,
    spec_mask=spec_mask,
    cosmicray=True,
    readnoise=5.7,
    log_level="DEBUG",
    log_file_name=None,
)

lhs6328_twodspec.ap_trace(nspec=1, display=False)


def test_tophat_extraction():
    lhs6328_twodspec.ap_extract(optimal=False, display=False)
    lhs6328_twodspec.ap_extract(optimal=False, display=False, spec_id=0)
    lhs6328_twodspec.ap_extract(optimal=False, display=False, spec_id=[0])
    lhs6328_twodspec.ap_extract(
        optimal=False,
        apwidth=[4, 5],
        skysep=[8, 10],
        skywidth=[3, 4],
        display=False,
    )


def test_extraction_wrong_size_extraction_description():
    lhs6328_twodspec.ap_extract(
        optimal=False, apwidth=[1, 2, 3], skysep=[4, 5, 6], skywidth=[7, 8, 9]
    )


def test_horne_extraction():
    lhs6328_twodspec.ap_extract(
        optimal=True, algorithm="horne86", display=False
    )


def test_marsh_extraction_fast():
    lhs6328_twodspec.ap_extract(
        optimal=True, algorithm="marsh89", qmode="fast-nearest", display=False
    )


def test_marsh_extraction_fast_linear():
    lhs6328_twodspec.ap_extract(
        optimal=True, algorithm="marsh89", qmode="fast-linear", display=False
    )


def test_marsh_extraction_slow():
    lhs6328_twodspec.ap_extract(
        optimal=True, algorithm="marsh89", qmode="slow-nearest", display=False
    )


def test_marsh_extraction_slow_linear():
    lhs6328_twodspec.ap_extract(
        optimal=True, algorithm="marsh89", qmode="slow-linear", display=False
    )


def test_marsh_extraction_fast_apwidth():
    lhs6328_twodspec.ap_extract(
        optimal=True,
        apwidth=5,
        algorithm="marsh89",
        qmode="fast-nearest",
        display=False,
    )


def test_marsh_extraction_fast_apwidth_str():
    lhs6328_twodspec.ap_extract(
        optimal=True,
        apwidth="blabla",
        algorithm="marsh89",
        qmode="fast-nearest",
        display=False,
    )


# Exposure time and gain should not have an effect on the extraction itself
def test_gaussian_spectral_extraction_10000s_exptime_2x_gain():
    # masking
    spec_mask = np.arange(10, 900)
    spatial_mask = np.arange(15, 85)

    # initialise the two spectral_reduction.TwoDSpec()
    dummy_twodspec = spectral_reduction.TwoDSpec(
        dummy_gaussian_data,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        log_file_name=None,
        log_level="CRITICAL",
        saxis=1,
        flip=False,
        cosmicray_sigma=5.0,
        readnoise=0.1,
        gain=2.0,
        seeing=1.0,
        exptime=10000.0,
    )

    # Trace the spectrum, note that the first 15 rows were trimmed from the
    # spatial_mask
    dummy_twodspec.ap_trace(
        rescale=True,
        fit_deg=0,
    )
    trace = np.round(np.mean(dummy_twodspec.spectrum_list[0].trace))
    assert np.isclose(trace, 35, atol=1.0), (
        "Trace is at row "
        + str(trace)
        + ", but it is expected to be at row 35."
    )

    # Direct extraction by summing over the aperture along the trace
    dummy_twodspec.ap_extract(
        apwidth=5,
        optimal=False,
    )
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    count_err = np.mean(dummy_twodspec.spectrum_list[0].count_err)
    snr_tophat = count / count_err
    assert np.isclose(count, 10000.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~10000."
    )

    # Optimal extraction (Horne86 gauss)
    dummy_twodspec.ap_extract(apwidth=5, optimal=True, model="gauss")
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    count_err = np.mean(dummy_twodspec.spectrum_list[0].count_err)
    snr_horne = count / count_err
    assert np.isclose(count, 10000.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~10000."
    )

    # Optimal extraction (Horne86 lowess)
    dummy_twodspec.ap_extract(
        apwidth=5,
        optimal=True,
        model="lowess",
        lowess_frac=0.05,
    )
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    count_err = np.mean(dummy_twodspec.spectrum_list[0].count_err)
    snr_horne = count / count_err
    assert np.isclose(count, 10000.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~10000."
    )

    # Optimal extraction (Marsh89)
    dummy_twodspec.ap_extract(apwidth=5, optimal=True, algorithm="marsh89")
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    count_err = np.mean(dummy_twodspec.spectrum_list[0].count_err)
    snr_marsh = count / count_err
    assert np.isclose(count, 10000.0, rtol=0.01, atol=count_err), (
        "Extracted count is " + str(count) + " but it should be ~10000."
    )
