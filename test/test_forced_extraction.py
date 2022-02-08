# -*- coding: utf-8 -*-
import copy
import os

import numpy as np

from aspired import image_reduction
from aspired import spectral_reduction

HERE = os.path.dirname(os.path.realpath(__file__))

spatial_mask = np.arange(35, 200)
spec_mask = np.arange(50, 1024)

# Science frame
lhs6328_frame = image_reduction.ImageReduction(
    log_level="INFO", log_file_folder=os.path.join(HERE, "test_output")
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
    gain=2.45,
    log_level="INFO",
    log_file_folder=os.path.join(HERE, "test_output"),
)

lhs6328_twodspec.ap_trace(nspec=1, display=False)

# Tophat extraction to get the LSF for force extraction below
lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=False,
    display=False,
    save_fig=True,
    filename=os.path.join(HERE, "test_output", "test_force_extraxtion"),
    fig_type="iframe+png",
)

trace = copy.copy(lhs6328_twodspec.spectrum_list[0].trace)
trace_sigma = copy.copy(lhs6328_twodspec.spectrum_list[0].trace_sigma)

# Store the extracted count
tophat_count = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

# Optimal extraction to get the LSF for force extraction below
lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=True,
    display=False,
    save_fig=True,
    filename=os.path.join(HERE, "test_output", "test_force_extraxtion2"),
    fig_type="iframe+png",
)

# Store the extracted count
horne86_count = copy.copy(lhs6328_twodspec.spectrum_list[0].count)
horne86_var = copy.copy(lhs6328_twodspec.spectrum_list[0].var)

# Optimal extraction to get the LSF for force extraction below
lhs6328_twodspec.ap_extract(
    apwidth=15,
    skywidth=10,
    skydeg=1,
    optimal=True,
    algorithm="marsh89",
    display=False,
    save_fig=True,
    filename=os.path.join(HERE, "test_output", "test_force_extraxtion3"),
    fig_type="iframe+png",
)

# Store the extracted count
marsh89_count = copy.copy(lhs6328_twodspec.spectrum_list[0].count)
marsh89_var = copy.copy(lhs6328_twodspec.spectrum_list[0].var)


def test_forced_extraction_tophat():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        gain=2.45,
        log_level="INFO",
        log_file_folder=os.path.join(HERE, "test_output"),
    )

    lhs6328_twodspec.add_trace(trace, trace_sigma)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=False,
        forced=True,
        display=False,
        save_fig=True,
        filename=os.path.join(HERE, "test_output", "test_force_extraxtion4"),
        fig_type="iframe+png",
    )

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(tophat_count) >= np.nansum(count_forced) * 0.999) & (
        np.nansum(tophat_count) <= np.nansum(count_forced) * 1.001
    )


def test_forced_extraction_horne86_gauss():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        gain=2.45,
        log_level="INFO",
        log_file_folder=os.path.join(HERE, "test_output"),
    )

    lhs6328_twodspec.add_trace(trace, trace_sigma)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        forced=True,
        variances=horne86_var,
        display=False,
    )

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(horne86_count) >= np.nansum(count_forced) * 0.999) & (
        np.nansum(horne86_count) <= np.nansum(count_forced) * 1.001
    )


def test_forced_extraction_horne86_lowess():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        gain=2.45,
        log_level="INFO",
        log_file_folder=os.path.join(HERE, "test_output"),
    )

    lhs6328_twodspec.add_trace(trace, trace_sigma)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        model="lowess",
        forced=True,
        variances=horne86_var,
        display=False,
    )

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(horne86_count) >= np.nansum(count_forced) * 0.999) & (
        np.nansum(horne86_count) <= np.nansum(count_forced) * 1.001
    )


def test_forced_extraction_marsh89():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        gain=2.45,
        log_level="INFO",
        log_file_folder=os.path.join(HERE, "test_output"),
    )

    lhs6328_twodspec.add_trace(trace, trace_sigma)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        algorithm="marsh89",
        forced=True,
        variances=np.transpose(marsh89_var),
        display=False,
        save_fig=True,
        filename=os.path.join(HERE, "test_output", "test_force_extraxtion5"),
        fig_type="iframe+png",
    )

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(marsh89_count) >= np.nansum(count_forced) * 0.999) & (
        np.nansum(marsh89_count) <= np.nansum(count_forced) * 1.001
    )


def test_forced_extraction_horne86_lowess_int_var():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        gain=2.45,
        log_level="INFO",
        log_file_folder=os.path.join(HERE, "test_output"),
    )

    lhs6328_twodspec.add_trace(trace, trace_sigma)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        model="lowess",
        forced=True,
        variances=np.nanmedian(horne86_var),
        display=False,
        save_fig=True,
        filename=os.path.join(HERE, "test_output", "test_force_extraxtion6"),
        fig_type="iframe+png",
    )

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(horne86_count) >= np.nansum(count_forced) * 0.99) & (
        np.nansum(horne86_count) <= np.nansum(count_forced) * 1.01
    )


def test_forced_extraction_horne86_lowess_str_var():
    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        gain=2.45,
        log_level="INFO",
        log_file_folder=os.path.join(HERE, "test_output"),
    )

    lhs6328_twodspec.add_trace(trace, trace_sigma)

    # Force extraction
    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        model="lowess",
        forced=True,
        variances="blabla",
        display=False,
        save_fig=True,
        filename=os.path.join(HERE, "test_output", "test_force_extraxtion7"),
        fig_type="iframe+png",
    )

    # Store the forced extracted count
    count_forced = copy.copy(lhs6328_twodspec.spectrum_list[0].count)

    assert (np.nansum(horne86_count) >= np.nansum(count_forced) * 0.95) & (
        np.nansum(horne86_count) <= np.nansum(count_forced) * 1.05
    )
