# -*- coding: utf-8 -*-
import copy
import os
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits

from aspired.wavelength_calibration import WavelengthCalibration
from aspired.spectrum1D import Spectrum1D
from aspired import spectral_reduction

HERE = os.path.dirname(os.path.realpath(__file__))

# Line list
atlas = [
    4193.5,
    4385.77,
    4500.98,
    4524.68,
    4582.75,
    4624.28,
    4671.23,
    4697.02,
    4734.15,
    4807.02,
    4921.48,
    5028.28,
    5618.88,
    5823.89,
    5893.29,
    5934.17,
    6182.42,
    6318.06,
    6472.841,
    6595.56,
    6668.92,
    6728.01,
    6827.32,
    6976.18,
    7119.60,
    7257.9,
    7393.8,
    7584.68,
    7642.02,
    7740.31,
    7802.65,
    7887.40,
    7967.34,
    8057.258,
]
element = ["Xe"] * len(atlas)

arc_spec = np.loadtxt(
    os.path.join(HERE, "test_data", "test_full_run_science_0_arc_spec.csv"),
    delimiter=",",
    skiprows=1,
)

wavecal = WavelengthCalibration(log_file_name=None)
wavecal.add_arc_spec(arc_spec)

# Find the peaks of the arc
wavecal.find_arc_lines()

arc_lines = wavecal.spectrum1D.peaks

np.random.seed(0)


def test_wavecal():

    lhs6328_spectrum1D = Spectrum1D(log_file_name=None)
    wavecal = WavelengthCalibration(log_file_name=None)

    # Science arc_spec
    lhs6328_spectrum1D.add_arc_spec(arc_spec)
    wavecal.from_spectrum1D(lhs6328_spectrum1D)

    # Find the peaks of the arc
    wavecal.find_arc_lines(
        save_fig=True,
        fig_type="iframe+png",
        filename=os.path.join(
            HERE, "test_output", "test_wavecal_find_arc_lines"
        ),
        display=False,
        return_jsonstring=True,
    )

    # Configure the wavelength calibrator
    wavecal.initialise_calibrator()
    wavecal.set_hough_properties(
        num_slopes=1000,
        xbins=200,
        ybins=200,
        min_wavelength=3500,
        max_wavelength=8500,
    )
    wavecal.set_ransac_properties(filter_close=True)

    wavecal.add_user_atlas(elements=element, wavelengths=atlas)

    # Remove all lines between 3500 and 4000
    wavecal.remove_atlas_lines_range(wavelength=3750, tolerance=250)
    wavecal.do_hough_transform()

    # Solve for the pixel-to-wavelength solution
    wavecal.fit(max_tries=500, display=False)

    # Getting the calibrator
    wavecal.get_calibrator()

    # Save a FITS file
    wavecal.save_fits(
        output="wavecal",
        filename=os.path.join(HERE, "test_output", "test_wavecal"),
        overwrite=True,
    )

    # Save a CSV file
    wavecal.save_csv(
        output="wavecal",
        filename=os.path.join(HERE, "test_output", "test_wavecal"),
        overwrite=True,
    )

    # Getting the calibrator
    wavecal.get_spectrum1D()

    wavecal.list_atlas()
    wavecal.clear_atlas()
    wavecal.list_atlas()


def test_setting_a_known_pair():

    lhs6328_spectrum1D = Spectrum1D(log_file_name=None)
    wavecal = WavelengthCalibration(log_file_name=None)
    lhs6328_spectrum1D.add_arc_spec(arc_spec)
    wavecal.from_spectrum1D(lhs6328_spectrum1D)
    # Find the peaks of the arc
    wavecal.find_arc_lines(
        save_fig=True,
        fig_type="iframe+png",
        filename=os.path.join(
            HERE, "test_output", "test_wavecal_find_arc_lines"
        ),
        display=False,
        return_jsonstring=True,
    )
    wavecal.initialise_calibrator()
    wavecal.set_known_pairs(123, 456)
    assert wavecal.spectrum1D.calibrator.pix_known == 123
    assert wavecal.spectrum1D.calibrator.wave_known == 456


@patch("plotly.graph_objects.Figure.show")
def test_setting_known_pairs(mock_show):

    lhs6328_spectrum1D = Spectrum1D(log_file_name=None)
    wavecal = WavelengthCalibration(log_file_name=None)
    lhs6328_spectrum1D.add_arc_spec(arc_spec)
    wavecal.from_spectrum1D(lhs6328_spectrum1D)
    # Find the peaks of the arc
    wavecal.find_arc_lines(
        save_fig=True,
        fig_type="iframe+png",
        filename=os.path.join(
            HERE, "test_output", "test_wavecal_find_arc_lines"
        ),
        display=True,
        return_jsonstring=True,
    )
    wavecal.initialise_calibrator()
    wavecal.set_known_pairs([123, 234], [456, 567])
    assert len(wavecal.spectrum1D.calibrator.pix_known) == 2
    assert len(wavecal.spectrum1D.calibrator.wave_known) == 2


@pytest.mark.xfail()
def test_setting_a_none_to_known_pairs_expect_fail():

    lhs6328_spectrum1D = Spectrum1D(log_file_name=None)
    wavecal = WavelengthCalibration(log_file_name=None)
    lhs6328_spectrum1D.add_arc_spec(arc_spec)
    wavecal.from_spectrum1D(lhs6328_spectrum1D)
    # Find the peaks of the arc
    wavecal.find_arc_lines(
        save_fig=True,
        fig_type="iframe+png",
        filename=os.path.join(
            HERE, "test_output", "test_wavecal_find_arc_lines"
        ),
        display=False,
        return_jsonstring=True,
    )
    wavecal.initialise_calibrator()
    wavecal.set_known_pairs([1.0], [None])


@pytest.mark.xfail()
def test_setting_nones_to_known_pairs_expect_fail():

    lhs6328_spectrum1D = Spectrum1D(log_file_name=None)
    wavecal = WavelengthCalibration(log_file_name=None)
    lhs6328_spectrum1D.add_arc_spec(arc_spec)
    wavecal.from_spectrum1D(lhs6328_spectrum1D)
    # Find the peaks of the arc
    wavecal.find_arc_lines(
        save_fig=True,
        fig_type="iframe+png",
        filename=os.path.join(
            HERE, "test_output", "test_wavecal_find_arc_lines"
        ),
        display=False,
        return_jsonstring=True,
    )
    wavecal.initialise_calibrator()
    wavecal.set_known_pairs([None], [None])


def test_user_supplied_arc_spec():

    wavecal = WavelengthCalibration(log_file_name=None)

    # Science arc_spec
    wavecal.add_arc_spec(arc_spec)

    # Find the peaks of the arc
    wavecal.find_arc_lines()

    # Configure the wavelength calibrator
    wavecal.initialise_calibrator()
    wavecal.set_hough_properties(
        num_slopes=200,
        xbins=40,
        ybins=40,
        min_wavelength=3500,
        max_wavelength=8500,
    )
    wavecal.set_ransac_properties(filter_close=True)

    wavecal.add_user_atlas(elements=element, wavelengths=atlas)
    wavecal.do_hough_transform()

    # Solve for the pixel-to-wavelength solution
    wavecal.fit(max_tries=500, display=False)

    # Save a FITS file
    wavecal.save_fits(
        output="wavecal",
        filename=os.path.join(
            HERE, "test_output", "test_wavecal_user_supplied_arc_spec"
        ),
        overwrite=True,
    )

    wavecal.remove_arc_lines()


def test_user_supplied_arc_spec_2():

    wavecal = WavelengthCalibration(log_file_name=None)

    # Find the peaks of the arc
    wavecal.find_arc_lines(arc_spec=arc_spec)

    # Configure the wavelength calibrator
    wavecal.initialise_calibrator()
    wavecal.set_hough_properties(
        num_slopes=200,
        xbins=40,
        ybins=40,
        min_wavelength=3500,
        max_wavelength=8500,
    )
    wavecal.set_ransac_properties(filter_close=True)

    wavecal.add_user_atlas(elements=element, wavelengths=atlas)
    wavecal.do_hough_transform()

    # Solve for the pixel-to-wavelength solution
    wavecal.fit(max_tries=500, display=False)

    # Save a FITS file
    wavecal.save_fits(
        output="wavecal",
        filename=os.path.join(
            HERE, "test_output", "test_wavecal_user_supplied_arc_spec"
        ),
        overwrite=True,
    )

    wavecal.remove_arc_lines()


def test_user_supplied_arc_spec_arc_lines_from_at_initilisation():

    wavecal = WavelengthCalibration(log_file_name=None)

    # Configure the wavelength calibrator
    wavecal.initialise_calibrator(peaks=arc_lines, arc_spec=arc_spec)
    wavecal.set_hough_properties(
        num_slopes=200,
        xbins=40,
        ybins=40,
        min_wavelength=3500,
        max_wavelength=8500,
    )
    wavecal.set_ransac_properties(filter_close=True)

    wavecal.add_user_atlas(elements=element, wavelengths=atlas)
    wavecal.do_hough_transform()

    # Solve for the pixel-to-wavelength solution
    wavecal.fit(max_tries=500, display=False)

    # Save a FITS file
    wavecal.save_fits(
        output="wavecal",
        filename=os.path.join(
            HERE, "test_output", "test_wavecal_user_supplied_arc_spec"
        ),
        overwrite=True,
    )


def test_overwritten_copy_of_spectrum1Ds_are_different():

    lhs6328_spectrum1D = Spectrum1D(log_file_name=None)
    wavecal_1 = WavelengthCalibration(log_file_name=None)
    wavecal_1.from_spectrum1D(lhs6328_spectrum1D)
    memory_1 = id(wavecal_1.spectrum1D)
    wavecal_1.from_spectrum1D(copy.copy(lhs6328_spectrum1D), overwrite=True)
    memory_2 = id(wavecal_1.spectrum1D)

    assert memory_1 != memory_2


@patch("plotly.graph_objects.Figure.show")
def test_user_supplied_arc_lines(mock_show):

    wavecal = WavelengthCalibration(log_file_name=None)

    # Find the peaks of the arc
    wavecal.add_arc_lines(arc_lines)

    # Configure the wavelength calibrator
    wavecal.initialise_calibrator()
    wavecal.set_hough_properties(
        num_slopes=200,
        xbins=40,
        ybins=40,
        min_wavelength=3500,
        max_wavelength=8500,
    )
    wavecal.set_ransac_properties(filter_close=True)

    wavecal.add_user_atlas(elements=element, wavelengths=atlas)
    wavecal.do_hough_transform()

    # Solve for the pixel-to-wavelength solution
    wavecal.fit(max_tries=500, display=True)

    # Save a FITS file
    wavecal.save_fits(
        output="wavecal",
        filename=os.path.join(
            HERE, "test_output", "test_wavecal_user_supplied_arc_spec"
        ),
        overwrite=True,
    )

    wavecal.remove_arc_lines()


def test_user_supplied_poly_coeff_twodspec():
    # Load the image
    lhs6328_fits = fits.open(
        os.path.join(HERE, "test_data", "v_e_20180810_12_1_0_0.fits.gz")
    )[0]
    spatial_mask = np.arange(50, 200)
    spec_mask = np.arange(50, 1024)

    #
    # Loading two pre-saved spectral traces from a single FITS file.
    #
    lhs6328 = spectral_reduction.TwoDSpec(
        lhs6328_fits,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=2.34,
        log_file_name=None,
    )

    # Trace the spectra
    lhs6328.ap_trace(nspec=2, display=False)

    # Extract the spectra
    lhs6328.ap_extract(apwidth=10, optimal=True, skywidth=10, display=False)

    # Calibrate the 1D spectra
    lhs6328_onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    lhs6328_onedspec.from_twodspec(lhs6328)

    fit_coeff = np.array(
        [
            3.09833375e03,
            5.98842823e00,
            -2.83963934e-03,
            2.84842392e-06,
            -1.03725267e-09,
        ]
    )
    fit_type = "poly"

    # Note that there are two science traces, so two polyfit coefficients
    # have to be supplied by in a list
    lhs6328_onedspec.add_fit_coeff(fit_coeff, fit_type)
    lhs6328_onedspec.apply_wavelength_calibration()

    # Inspect reduced spectrum
    lhs6328_onedspec.inspect_reduced_spectrum(display=False)

    # Save as a FITS file
    lhs6328_onedspec.save_fits(
        output="wavecal+count",
        filename=os.path.join(
            HERE,
            "test_output",
            "user_supplied_wavelength_polyfit_coefficients",
        ),
        stype="science",
        overwrite=True,
    )


def test_user_supplied_poly_coeff_and_add_arc_twodspec():
    # Load the image
    lhs6328_fits = fits.open(
        os.path.join(HERE, "test_data", "v_e_20180810_12_1_0_0.fits.gz")
    )[0]
    spatial_mask = np.arange(50, 200)
    spec_mask = np.arange(50, 1024)

    #
    # Loading two pre-saved spectral traces from a single FITS file.
    #
    lhs6328 = spectral_reduction.TwoDSpec(
        lhs6328_fits,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=2.34,
        log_file_name=None,
    )

    # Trace the spectra
    lhs6328.ap_trace(nspec=2, display=False)

    # Extract the spectra
    lhs6328.ap_extract(apwidth=10, optimal=True, skywidth=10, display=False)

    # Supply arc manually
    lhs6328.add_arc(
        os.path.join(HERE, "test_data", "v_a_20180810_13_1_0_1.fits.gz")
    )
    lhs6328.apply_mask_to_arc()

    # Calibrate the 1D spectra
    lhs6328_onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    lhs6328_onedspec.from_twodspec(lhs6328)

    fit_coeff = np.array(
        [
            3.09833375e03,
            5.98842823e00,
            -2.83963934e-03,
            2.84842392e-06,
            -1.03725267e-09,
        ]
    )
    fit_type = "poly"

    # Note that there are two science traces, so two polyfit coefficients
    # have to be supplied by in a list
    lhs6328_onedspec.add_fit_coeff(fit_coeff, fit_type)
    lhs6328_onedspec.apply_wavelength_calibration()

    # Inspect reduced spectrum
    lhs6328_onedspec.inspect_reduced_spectrum(display=False)

    # Save as a FITS file
    lhs6328_onedspec.save_fits(
        output="wavecal+count",
        filename=os.path.join(
            HERE,
            "test_output",
            "user_supplied_wavelength_polyfit_coefficients",
        ),
        stype="science",
        overwrite=True,
    )


@patch("plotly.graph_objects.Figure.show")
def test_user_supplied_wavelength_twodspec(mock_show):
    # Load the image
    lhs6328_fits = fits.open(
        os.path.join(HERE, "test_data", "v_e_20180810_12_1_0_0.fits.gz")
    )[0]
    spatial_mask = np.arange(50, 200)
    spec_mask = np.arange(50, 1024)

    #
    # Loading two pre-saved spectral traces from a single FITS file.
    #
    lhs6328 = spectral_reduction.TwoDSpec(
        lhs6328_fits,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=2.34,
        log_file_name=None,
    )

    # Trace the spectra
    lhs6328.ap_trace(nspec=2, display=True)

    # Extract the spectra
    lhs6328.ap_extract(apwidth=10, optimal=True, skywidth=10, display=True)

    # Calibrate the 1D spectra
    lhs6328_onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    lhs6328_onedspec.from_twodspec(lhs6328)

    wavelength = np.genfromtxt(
        os.path.join(
            HERE, "test_data", "test_full_run_standard_wavelength.csv"
        )
    )
    # Manually supply wavelengths
    lhs6328_onedspec.add_wavelength([wavelength, wavelength])

    # Inspect reduced spectrum
    lhs6328_onedspec.inspect_reduced_spectrum(display=False)

    # Save as a FITS file
    lhs6328_onedspec.save_fits(
        output="count",
        filename=os.path.join(HERE, "test_output", "user_supplied_wavelength"),
        stype="science",
        overwrite=True,
    )


peaks = np.sort(np.random.random(31) * 1000.0)
# Removed the closely spaced peaks
distance_mask = np.isclose(peaks[:-1], peaks[1:], atol=5.0)
distance_mask = np.insert(distance_mask, 0, False)
peaks = peaks[~distance_mask]

# Line list
wavelengths_linear = (
    3000.0 + 5.0 * peaks + (np.random.random(len(peaks)) - 0.5) * 2.0
)
wavelengths_quadratic = (
    3000.0
    + 4 * peaks
    + 1.0e-3 * peaks**2.0
    + (np.random.random(len(peaks)) - 0.5) * 2.0
)

elements_linear = ["Linear"] * len(wavelengths_linear)
elements_quadratic = ["Quadratic"] * len(wavelengths_quadratic)


def test_linear_fit():

    wavecal = WavelengthCalibration(log_file_name=None)
    wavecal.initialise_calibrator(peaks)

    wavecal.set_calibrator_properties(num_pix=1000)
    wavecal.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=200,
        ybins=200,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    wavecal.add_user_atlas(
        elements=elements_linear, wavelengths=wavelengths_linear
    )
    wavecal.set_ransac_properties(minimum_matches=20)
    wavecal.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.fit(max_tries=500, fit_deg=1)
    # Refine solution
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.robust_refit(best_p, refine=False, robust_refit=True)

    assert np.abs(best_p[1] - 5.0) / 5.0 < 0.001
    assert np.abs(best_p[0] - 3000.0) / 3000.0 < 0.001
    assert peak_utilisation > 0.8
    assert atlas_utilisation > 0.0


def test_manual_refit():

    # Initialise the calibrator
    wavecal = WavelengthCalibration(log_file_name=None)
    wavecal.initialise_calibrator(peaks)

    wavecal.set_calibrator_properties(num_pix=1000)
    wavecal.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=200,
        ybins=200,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    wavecal.add_user_atlas(
        elements=elements_linear, wavelengths=wavelengths_linear
    )
    wavecal.set_ransac_properties(minimum_matches=25)
    wavecal.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.fit(max_tries=500, fit_deg=1)

    # Refine solution
    (
        best_p_robust,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.robust_refit(best_p, refine=False, robust_refit=True)

    (
        best_p_manual,
        matched_peaks,
        matched_atlas,
        rms,
        residuals,
    ) = wavecal.manual_refit(matched_peaks, matched_atlas)

    assert np.abs(best_p_manual[0] - best_p[0]) < 10.0
    assert np.abs(best_p_manual[1] - best_p[1]) < 0.1


def test_manual_refit_remove_points():

    # Initialise the calibrator
    wavecal = WavelengthCalibration(log_file_name=None)
    wavecal.initialise_calibrator(peaks)

    wavecal.set_calibrator_properties(num_pix=1000)
    wavecal.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=200,
        ybins=200,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    wavecal.add_user_atlas(
        elements=elements_linear, wavelengths=wavelengths_linear
    )
    wavecal.set_ransac_properties(minimum_matches=25)
    wavecal.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        atched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.fit(max_tries=500, fit_deg=1)

    # Refine solution
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.robust_refit(best_p, refine=False, robust_refit=True)

    wavecal.remove_pix_wave_pair(5)

    (
        best_p_manual,
        matched_peaks,
        matched_atlas,
        rms,
        residuals,
    ) = wavecal.manual_refit(matched_peaks, matched_atlas)

    assert np.allclose(best_p_manual, best_p)


def test_manual_refit_add_points():

    # Initialise the calibrator
    wavecal = WavelengthCalibration(log_file_name=None)
    wavecal.initialise_calibrator(peaks)

    wavecal.set_calibrator_properties(num_pix=1000)
    wavecal.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=200,
        ybins=200,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    wavecal.add_user_atlas(
        elements=elements_linear, wavelengths=wavelengths_linear
    )
    wavecal.set_ransac_properties(minimum_matches=25)
    wavecal.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        atched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.fit(max_tries=500, fit_deg=1)

    # Refine solution
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.robust_refit(best_p, refine=False, robust_refit=True)

    wavecal.add_pix_wave_pair(
        2000.0, 3000.0 + 4 * 2000.0 + 1.0e-3 * 2000.0**2.0
    )
    (
        best_p_manual,
        matched_peaks,
        matched_atlas,
        rms,
        residuals,
    ) = wavecal.manual_refit(matched_peaks, matched_atlas)

    assert np.allclose(best_p_manual, best_p)


def test_quadratic_fit():

    # Initialise the calibrator
    wavecal = WavelengthCalibration(log_file_name=None)
    wavecal.initialise_calibrator(peaks)

    wavecal.set_calibrator_properties(num_pix=1000)
    wavecal.set_hough_properties(
        num_slopes=1000,
        range_tolerance=500.0,
        xbins=100,
        ybins=100,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    wavecal.add_user_atlas(
        elements=elements_quadratic, wavelengths=wavelengths_quadratic
    )
    wavecal.set_ransac_properties(minimum_matches=20)
    wavecal.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.fit(
        max_tries=2000, fit_tolerance=5.0, candidate_tolerance=2.0, fit_deg=2
    )
    # Refine solution
    (
        best_p_robust,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.robust_refit(best_p, refine=False, robust_refit=True)


def test_quadratic_fit_legendre():

    # Initialise the calibrator
    wavecal = WavelengthCalibration(log_file_name=None)
    wavecal.initialise_calibrator(peaks)

    wavecal.set_calibrator_properties(num_pix=1000)
    wavecal.set_hough_properties(
        num_slopes=500,
        range_tolerance=200.0,
        xbins=100,
        ybins=100,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    wavecal.add_user_atlas(
        elements=elements_quadratic, wavelengths=wavelengths_quadratic
    )
    wavecal.set_ransac_properties(sample_size=10, minimum_matches=20)
    wavecal.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.fit(
        max_tries=2000,
        fit_tolerance=5.0,
        candidate_tolerance=2.0,
        fit_deg=2,
        fit_type="legendre",
    )


def test_quadratic_fit_chebyshev():

    # Initialise the calibrator
    wavecal = WavelengthCalibration(log_file_name=None)
    wavecal.initialise_calibrator(peaks)

    wavecal.set_calibrator_properties(num_pix=1000)
    wavecal.set_hough_properties(
        num_slopes=500,
        range_tolerance=200.0,
        xbins=100,
        ybins=100,
        min_wavelength=3000.0,
        max_wavelength=8000.0,
    )
    wavecal.add_user_atlas(
        elements=elements_quadratic, wavelengths=wavelengths_quadratic
    )
    wavecal.set_ransac_properties(sample_size=10, minimum_matches=20)
    wavecal.do_hough_transform(brute_force=False)

    # Run the wavelength calibration
    (
        best_p,
        matched_peaks,
        matched_atlas,
        rms,
        residual,
        peak_utilisation,
        atlas_utilisation,
    ) = wavecal.fit(
        max_tries=2000,
        fit_tolerance=5.0,
        candidate_tolerance=2.0,
        fit_deg=2,
        fit_type="chebyshev",
    )
