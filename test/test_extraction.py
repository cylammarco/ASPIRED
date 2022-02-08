# -*- coding: utf-8 -*-
import os

from astropy.io import fits
import numpy as np

from aspired import spectral_reduction
from aspired import image_reduction

HERE = os.path.dirname(os.path.realpath(__file__))


def test_spectral_extraction():
    # Prepare dummy data
    # total signal should be [2 * (2 + 5 + 10) + 20] - [1 * 7] = 47
    dummy_data = np.ones((100, 1000))
    dummy_noise = np.random.random((100, 1000)) * 0.1 - 0.05

    dummy_data[47] += 2.0
    dummy_data[48] += 5.0
    dummy_data[49] += 10.0
    dummy_data[50] += 20.0
    dummy_data[51] += 10.0
    dummy_data[52] += 5.0
    dummy_data[53] += 2.0
    dummy_data += dummy_noise

    # masking
    spec_mask = np.arange(10, 90)
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
        filename=os.path.join(HERE, "test_output", "test_full_run_aptrace"),
        return_jsonstring=True,
    )
    trace = np.round(np.mean(dummy_twodspec.spectrum_list[0].trace))
    assert trace == 35, (
        "Trace is at row "
        + str(trace)
        + ", but it is expected to be at row 35."
    )

    # Optimal extracting spectrum by summing over the aperture along the trace
    dummy_twodspec.ap_extract(
        apwidth=5,
        optimal=False,
        filename=os.path.join(HERE, "test_output", "test_full_run_apextract"),
        save_fig=True,
        fig_type="iframe+png",
        return_jsonstring=True,
    )

    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    assert np.round(count).astype("int") == 54, (
        "Extracted count is " + str(count) + " but it should be 54."
    )

    dummy_twodspec.inspect_extracted_spectrum(
        display=False,
        filename=os.path.join(
            HERE, "test_output", "test_full_run_extracted_spectrum"
        ),
        save_fig=True,
        fig_type="iframe+png",
        return_jsonstring=True,
    )


def test_user_supplied_trace():

    spatial_mask = np.arange(50, 200)
    spec_mask = np.arange(50, 1024)

    # Loading a single pre-saved spectral trace.
    lhs6328_extracted = fits.open(
        os.path.join(HERE, "test_data", "test_full_run_science_0.fits")
    )
    lhs6328_trace = lhs6328_extracted[1].data
    lhs6328_trace_sigma = lhs6328_extracted[2].data

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
