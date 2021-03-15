import copy
import os
import numpy as np
from aspired import image_reduction
from aspired import spectral_reduction

base_dir = os.path.dirname(__file__)
abs_dir = os.path.abspath(os.path.join(base_dir, '..'))

# Line list
atlas = [
    4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
    4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
    6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32, 6976.18,
    7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65, 7887.40,
    7967.34, 8057.258
]
element = ['Xe'] * len(atlas)

spatial_mask = np.arange(35, 200)
spec_mask = np.arange(50, 1024)

# Science frame
lhs6328_frame = image_reduction.ImageReduction(
    'test/test_data/sprat_LHS6328.list',
    log_level='DEBUG',
    log_file_folder='test/test_output/')
lhs6328_frame.reduce()


def test_forced_extraction_tophat():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        log_level='DEBUG',
        log_file_folder='test/test_output/')

    lhs6328_twodspec.ap_trace(nspec=1, display=False)

    # Optimal extraction to get the LSF for force extraction below
    lhs6328_twodspec.ap_extract(apwidth=15,
                                skywidth=10,
                                skydeg=1,
                                optimal=False,
                                display=False,
                                save_iframe=False)

    # Store the extracted count
    count = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=False,
        forced=True,
        variances=lhs6328_twodspec.spectrum_list[0].var,
        display=False,
        save_iframe=False)

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(count) >= np.nansum(count_forced) * 0.9999) & (
        np.nansum(count) <= np.nansum(count_forced) * 1.0001)


def test_forced_extraction_horne86_gauss():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        log_level='DEBUG',
        log_file_folder='test/test_output/')

    lhs6328_twodspec.ap_trace(nspec=1, display=False)

    # Optimal extraction to get the LSF for force extraction below
    lhs6328_twodspec.ap_extract(apwidth=15,
                                skywidth=10,
                                skydeg=1,
                                optimal=True,
                                display=False,
                                save_iframe=False)

    # Store the extracted count
    count = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        forced=True,
        variances=lhs6328_twodspec.spectrum_list[0].var,
        display=False,
        save_iframe=False)

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(count) >= np.nansum(count_forced) * 0.9999) & (
        np.nansum(count) <= np.nansum(count_forced) * 1.0001)


def test_forced_extraction_horne86_lowess():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        log_level='DEBUG',
        log_file_folder='test/test_output/')

    lhs6328_twodspec.ap_trace(nspec=1, display=False)

    # Optimal extraction to get the LSF for force extraction below
    lhs6328_twodspec.ap_extract(apwidth=15,
                                skywidth=10,
                                skydeg=1,
                                optimal=True,
                                model='lowess',
                                display=False,
                                save_iframe=False)

    # Store the extracted count
    count = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        model='lowess',
        forced=True,
        variances=lhs6328_twodspec.spectrum_list[0].var,
        display=False,
        save_iframe=False)

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(count) >= np.nansum(count_forced) * 0.9999) & (
        np.nansum(count) <= np.nansum(count_forced) * 1.0001)


def test_forced_extraction_marsh89():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        log_level='DEBUG',
        log_file_folder='test/test_output/')

    lhs6328_twodspec.ap_trace(nspec=1, display=False)

    # Optimal extraction to get the LSF for force extraction below
    lhs6328_twodspec.ap_extract(apwidth=15,
                                skywidth=10,
                                skydeg=1,
                                optimal=True,
                                algorithm='marsh89',
                                display=False,
                                save_iframe=False)

    # Store the extracted count
    count = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        algorithm='marsh89',
        forced=True,
        variances=lhs6328_twodspec.spectrum_list[0].var,
        display=False,
        save_iframe=False)

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(count) >= np.nansum(count_forced) * 0.9999) & (
        np.nansum(count) <= np.nansum(count_forced) * 1.0001)


def test_forced_extraction_horne86_lowess_int_var():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        log_level='DEBUG',
        log_file_folder='test/test_output/')

    lhs6328_twodspec.ap_trace(nspec=1, display=False)

    # Optimal extraction to get the LSF for force extraction below
    lhs6328_twodspec.ap_extract(apwidth=15,
                                skywidth=10,
                                skydeg=1,
                                optimal=True,
                                model='lowess',
                                display=False,
                                save_iframe=False)

    # Store the extracted count
    count = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    # Force extraction
    lhs6328_twodspec.ap_extract(apwidth=15,
                                skywidth=10,
                                skydeg=1,
                                optimal=True,
                                model='lowess',
                                forced=True,
                                variances=np.nanmedian(
                                    lhs6328_twodspec.spectrum_list[0].var),
                                display=False,
                                save_iframe=False)

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(count) >= np.nansum(count_forced) * 0.99) & (
        np.nansum(count) <= np.nansum(count_forced) * 1.01)


def test_forced_extraction_horne86_lowess_str_var():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        log_level='DEBUG',
        log_file_folder='test/test_output/')

    lhs6328_twodspec.ap_trace(nspec=1, display=False)

    # Optimal extraction to get the LSF for force extraction below
    lhs6328_twodspec.ap_extract(apwidth=15,
                                skywidth=10,
                                skydeg=1,
                                optimal=True,
                                model='lowess',
                                display=False,
                                save_iframe=False)

    # Store the extracted count
    count = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    # Force extraction
    lhs6328_twodspec.ap_extract(apwidth=15,
                                skywidth=10,
                                skydeg=1,
                                optimal=True,
                                model='lowess',
                                forced=True,
                                variances='blabla',
                                display=False,
                                save_iframe=False)

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(count) >= np.nansum(count_forced) * 0.95) & (
        np.nansum(count) <= np.nansum(count_forced) * 1.05)
