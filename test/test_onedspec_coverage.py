# -*- coding: utf-8 -*-
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from aspired import spectral_reduction
from aspired.wavelength_calibration import WavelengthCalibration
from astropy.io import fits

HERE = os.path.dirname(os.path.realpath(__file__))
COEFF = np.array((4000.0, 1, 0.2, 0.0071))


def make(stype="science+standard", calibrate=True, n=70):
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    onedspec.add_trace(np.ones(n) * 37, np.ones(n), spec_id=0, stype=stype)
    onedspec.add_spec(np.arange(1, n + 1), spec_id=0, stype=stype)
    onedspec.add_arc_spec(np.arange(1, n + 1), spec_id=0, stype=stype)
    onedspec.add_fit_coeff(COEFF, spec_id=0, stype=stype)
    if calibrate:
        onedspec.apply_wavelength_calibration(stype=stype)
    return onedspec


def make_flux():
    onedspec = make()
    onedspec.load_standard(target="cd32d9927")
    coeff = np.polynomial.polynomial.polyfit(
        np.arange(1, 1001), np.random.random(1000) * 20, 2
    )
    onedspec.add_sensitivity_func(
        lambda x: np.polynomial.polynomial.polyval(x, coeff)
    )
    onedspec.set_atmospheric_extinction()
    onedspec.apply_flux_calibration()
    return onedspec


# add_spec / add_arc_spec / add_fit_coeff argument handling
def test_add_spec_variants():
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    onedspec.add_trace([np.ones(10)] * 2, [np.ones(10)] * 2, spec_id=[0, 1])
    onedspec.add_spec(
        [np.arange(10.0)] * 2,
        count_err=[np.ones(10)],
        count_sky=[np.ones(10)],
        spec_id=[0, 1],
    )
    onedspec.add_spec(
        [np.arange(10.0)] * 2,
        count_err=[np.ones(10)] * 2,
        count_sky=[np.ones(10)] * 2,
        spec_id=[0, 1],
    )


def test_add_spec_bad_sizes():
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    onedspec.add_trace([np.ones(10)] * 3, [np.ones(10)] * 3, spec_id=[0, 1, 2])
    with pytest.raises(RuntimeError):
        onedspec.add_spec(
            [np.arange(10.0)] * 3,
            count_sky=[np.ones(10)] * 2,
            spec_id=[0, 1, 2],
        )
    with pytest.raises(RuntimeError):
        onedspec.add_spec(
            [np.arange(10.0)] * 3,
            count_err=[np.ones(10)] * 2,
            spec_id=[0, 1, 2],
        )


def test_add_arc_spec_variants():
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    onedspec.add_arc_spec([np.arange(10.0)] * 2, spec_id=[0, 1])
    onedspec.add_arc_spec([np.arange(10.0)], spec_id=[0, 1])
    with pytest.raises(ValueError):
        onedspec.add_arc_spec([np.arange(10.0)] * 3, spec_id=[0, 1])
    with pytest.raises(TypeError):
        onedspec.add_arc_spec("abc")


def test_add_fit_coeff_variants():
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    onedspec.add_arc_spec([np.arange(10.0)] * 2, spec_id=[0, 1])
    onedspec.add_fit_coeff(
        [COEFF, COEFF], fit_type=["poly", "poly"], spec_id=[0, 1]
    )
    onedspec.add_fit_coeff([COEFF], fit_type=["poly"], spec_id=[0, 1])
    onedspec.add_fit_coeff(
        [COEFF, COEFF], fit_type=[["poly"], ["poly"]], spec_id=[0, 1]
    )
    with pytest.raises(RuntimeError):
        onedspec.add_fit_coeff([COEFF] * 3, spec_id=[0, 1])
    with pytest.raises(ValueError):
        onedspec.add_fit_coeff(
            [COEFF, COEFF], fit_type=["poly"] * 3, spec_id=[0, 1]
        )
    with pytest.raises(TypeError):
        onedspec.add_fit_coeff([COEFF], fit_type=[1, 2], spec_id=[0])
    with pytest.raises(TypeError):
        onedspec.add_fit_coeff("abc")


# from_fits
def _twodspec_hdul(with_headers=True):
    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.append(fits.ImageHDU(np.ones(70) * 37, name="trace"))
    hdul.append(fits.ImageHDU(np.ones(70), name="trace_sigma"))
    h = fits.ImageHDU(np.arange(1.0, 71.0), name="count")
    if with_headers:
        h.header["GAIN"] = 1.0
        h.header["RNOISE"] = 2.0
        h.header["XPOSURE"] = 100.0
        h.header["SEEING"] = 1.0
        h.header["AIRMASS"] = 1.2
    hdul.append(h)
    hdul.append(fits.ImageHDU(np.ones(70), name="count_err"))
    hdul.append(fits.ImageHDU(np.ones(70), name="count_sky"))
    hdul.append(fits.ImageHDU(np.ones((2, 70)), name="weight_map"))
    hdul.append(fits.ImageHDU(np.arange(70.0), name="arc_spec"))
    return hdul


@pytest.mark.parametrize("with_headers", [True, False])
@pytest.mark.parametrize("stype", ["science", "standard", "science+standard"])
def test_from_fits(with_headers, stype):
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    onedspec.add_science_spectrum_oned(0)
    onedspec.from_fits(_twodspec_hdul(with_headers), spec_id=0, stype=stype)


def test_from_fits_from_path_and_missing_hdus(tmp_path):
    path = str(tmp_path / "a.fits")
    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.writeto(path)
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    onedspec.from_fits(path, spec_id=0, stype="science+standard")
    with pytest.raises(TypeError):
        onedspec.from_fits(["not_hdul"], spec_id=1, stype="science")


# add_variance
@pytest.mark.parametrize(
    "name, value",
    [
        ("add_variance", np.ones(70)),
        ("add_gain", 1.0),
        ("add_readnoise", 2.0),
        ("add_exptime", 100.0),
        ("add_seeing", 1.0),
        ("add_airmass", 1.2),
    ],
)
def test_add_scalar_metadata(name, value):
    onedspec = make(calibrate=False)
    getattr(onedspec, name)(value, stype="science", spec_id=0)
    getattr(onedspec, name)(value, stype="standard", spec_id=0)
    getattr(onedspec, name)(value, stype="unknown", spec_id=0)


# set_calibrator_logger
def test_set_calibrator_logger():
    onedspec = make(calibrate=False)
    onedspec.initialise_calibrator(peaks=np.arange(10.0) * 10 + 5)
    for wavecal in (onedspec.science_wavecal[0], onedspec.standard_wavecal):
        wavecal.spectrum_oned.calibrator.set_logger = MagicMock()
    onedspec.set_calibrator_logger("Cal", "INFO")
    onedspec.set_calibrator_logger("Cal", "INFO", stype="science")
    onedspec.set_calibrator_logger("Cal", "INFO", stype="standard")


def test_set_calibrator_logger_no_calibrator():
    onedspec = make(calibrate=False)
    onedspec.set_calibrator_logger("Cal", "INFO")


# atlas
def test_atlas_functions():
    onedspec = make(calibrate=False)
    onedspec.initialise_calibrator(peaks=np.arange(10.0) * 10 + 5)
    onedspec.set_calibrator_properties(num_pix=70)
    onedspec.add_atlas(
        elements=["Hg", "Ar"],
        min_atlas_wavelength=3500.0,
        max_atlas_wavelength=8000.0,
    )
    onedspec.list_atlas()
    onedspec.remove_atlas_lines_range(5000.0)
    onedspec.clear_atlas()
    onedspec.list_atlas(stype="standard")
    onedspec.remove_atlas_lines_range(5000.0, stype="standard")
    onedspec.clear_atlas(stype="standard")


def test_atlas_functions_no_calibrator():
    onedspec = make(calibrate=False)
    onedspec.add_atlas(elements=["Hg"])
    onedspec.list_atlas()
    onedspec.remove_atlas_lines_range(5000.0)
    onedspec.clear_atlas()


# sensitivity / continuum / telluric
def test_get_sensitivity_default_and_uncalibrated():
    onedspec = make()
    onedspec.load_standard(target="cd32d9927")
    onedspec.get_sensitivity()
    onedspec = make(calibrate=False)
    onedspec.standard_spectrum_list[0].add_count_continuum(np.ones(70))
    with pytest.raises(RuntimeError):
        onedspec.get_sensitivity()


def test_get_count_continuum_variants():
    onedspec = make()
    onedspec.get_count_continuum()
    onedspec.get_count_continuum(method="lowess", frac=0.5)
    onedspec.get_count_continuum(stype="science")
    onedspec.get_count_continuum(stype="standard")


def test_telluric_misc():
    onedspec = make_flux()
    onedspec.add_telluric_function(
        [np.arange(10000), np.ones(10000)], stype="science"
    )
    onedspec.get_telluric_correction()
    onedspec.get_telluric_strength(auto_apply=False)


def test_telluric_no_profile():
    onedspec = make_flux()
    onedspec.get_telluric_correction()
    with pytest.raises(ValueError):
        onedspec.get_telluric_strength()


def test_telluric_science_only_profile():
    onedspec = make_flux()
    onedspec.add_telluric_function(lambda x: np.ones_like(x), stype="science")
    onedspec.get_telluric_correction()


def test_telluric_uncalibrated_data():
    onedspec = make(calibrate=False)
    onedspec.add_telluric_function(lambda x: np.ones_like(x))
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    onedspec.add_telluric_function(lambda x: np.ones_like(x))


# header modifiers
def test_modify_arc_and_resampled_headers():
    onedspec = make_flux()
    for spec in (
        onedspec.science_spectrum_list[0],
        onedspec.standard_spectrum_list[0],
    ):
        spec.arc_lines_hdulist = fits.HDUList([fits.ImageHDU(np.ones(3))])
        spec.wavelength_resampled_hdulist = fits.HDUList(
            [fits.ImageHDU(np.ones(3))]
        )
    for kw in ({}, {"spec_id": 0}, {"stype": "standard"}):
        onedspec.modify_arc_lines_header(0, "set", "COMMENT", "a", **kw)
        onedspec.modify_wavelength_resampled_header(
            "set", "COMMENT", "b", **kw
        )


def test_modify_headers_no_data():
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    onedspec.modify_arc_lines_header(0, "set", "COMMENT", "a")
    onedspec.modify_wavelength_resampled_header("set", "COMMENT", "b")


HEADER_METHODS = [
    "modify_trace_header",
    "modify_count_header",
    "modify_arc_spec_header",
    "modify_count_resampled_header",
    "modify_flux_header",
    "modify_flux_atm_ext_corrected_header",
    "modify_flux_telluric_corrected_header",
    "modify_flux_atm_ext_telluric_corrected_header",
    "modify_flux_resampled_header",
    "modify_flux_resampled_atm_ext_corrected_header",
    "modify_flux_resampled_telluric_corrected_header",
    "modify_flux_resampled_atm_ext_telluric_corrected_header",
]
HEADER_METHODS_NO_IDX = [
    "modify_weight_map_header",
    "modify_wavecal_header",
    "modify_wavelength_header",
    "modify_sensitivity_header",
    "modify_atm_ext_header",
    "modify_telluric_profile_header",
    "modify_sensitivity_resampled_header",
    "modify_atm_ext_resampled_header",
    "modify_telluric_profile_resampled_header",
]


@pytest.mark.parametrize("name", HEADER_METHODS)
def test_modify_header_data_unavailable_idx(name):
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    getattr(onedspec, name)(0, "set", "COMMENT", "a")


@pytest.mark.parametrize("name", HEADER_METHODS_NO_IDX)
def test_modify_header_data_unavailable(name):
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    getattr(onedspec, name)("set", "COMMENT", "a")


# Everything that needs a calibrator / calibrated data, on an empty object
def test_unavailable_branches():
    o = spectral_reduction.OneDSpec(log_file_name=None, log_level="ERROR")
    o.find_arc_lines()
    o.inspect_arc_lines()
    o.set_calibrator_properties()
    o.set_hough_properties()
    o.set_ransac_properties()
    o.set_known_pairs(pix=[1.0], wave=[5000.0])
    o.add_user_atlas(elements=["a"], wavelengths=[5000.0])
    o.add_atlas(elements=["Hg"], pressure=None)
    o.do_hough_transform()
    o.plot_search_space()
    o.fit(max_tries=1)
    o.robust_refit(fit_coeff=[1.0, 1.0])
    o.get_pix_wave_pairs()
    o.add_pix_wave_pair(1.0, 5000.0)
    o.remove_pix_wave_pair(0)
    o.manual_refit()
    o.get_calibrator()
    o.apply_wavelength_calibration()
    o.inspect_sensitivity()
    o.apply_flux_calibration()
    o.resample()
    o.create_fits(output="*")
    o.add_wavelengthcalibration  # attribute exists


def test_pressure_default_and_bad_sizes():
    onedspec = make(calibrate=False)
    onedspec.initialise_calibrator(peaks=np.arange(10.0) * 10 + 5)
    onedspec.add_user_atlas(
        elements=["a"],
        wavelengths=[5000.0],
        pressure=None,
        temperature=None,
        relative_humidity=None,
    )
    with pytest.raises(ValueError):
        onedspec.create_fits(output="bad")
    with pytest.raises(ValueError):
        onedspec.create_fits(stype="bad")


# continuum
@pytest.mark.parametrize("name", ["get_count_continuum", "get_flux_continuum"])
def test_continuum_fit_method_and_unavailable(name):
    onedspec = make_flux()
    getattr(onedspec, name)(method="fit")
    getattr(onedspec, name)(method="fit", median_window=5)
    getattr(onedspec, name)(method="lowess", frac=0.3)
    bare = spectral_reduction.OneDSpec(log_file_name=None, log_level="ERROR")
    getattr(bare, name)()


# telluric
def _telluric_onedspec():
    onedspec = make_flux()
    onedspec.get_flux_continuum()
    onedspec.get_telluric_profile()
    return onedspec


def test_telluric_profile_options():
    onedspec = make_flux()
    func = onedspec.get_telluric_profile(return_function=True)
    assert callable(func)
    onedspec = make_flux()
    onedspec.get_flux_continuum()
    onedspec.get_telluric_profile(use_continuum=True)
    onedspec = make_flux()
    onedspec.standard_spectrum_list[0].flux_continuum = None
    onedspec.get_telluric_profile()


@pytest.mark.parametrize("auto_apply", [False, True])
def test_telluric_strength_and_apply(auto_apply):
    onedspec = _telluric_onedspec()
    onedspec.get_telluric_strength(auto_apply=auto_apply)
    onedspec.get_telluric_strength(auto_apply=auto_apply, spec_id=0)
    onedspec.apply_telluric_correction()
    onedspec.apply_telluric_correction(factor=0.5, stype="science")
    onedspec.apply_telluric_correction(factor=0.5, stype="standard")


def test_telluric_strength_extra_args_and_standard_profile():
    onedspec = _telluric_onedspec()
    onedspec.science_telluric_function_available = False
    onedspec.science_spectrum_list[0].telluric_func = None
    onedspec.get_telluric_strength()


def test_telluric_strength_without_science_function():
    onedspec = _telluric_onedspec()
    onedspec.science_spectrum_list[0].telluric_func = None
    onedspec.science_spectrum_list[0].telluric_profile = None
    onedspec.get_telluric_strength()


def test_apply_telluric_no_function():
    onedspec = make_flux()
    with pytest.raises(ValueError):
        onedspec.apply_telluric_correction()


def test_apply_telluric_missing_func_and_strength():
    onedspec = _telluric_onedspec()
    onedspec.science_telluric_strength_available = False
    onedspec.science_spectrum_list[0].telluric_func = None
    onedspec.apply_telluric_correction(stype="science")
    bare = spectral_reduction.OneDSpec(log_file_name=None, log_level="ERROR")
    bare.science_telluric_function_available = True
    bare.apply_telluric_correction()


@patch("plotly.graph_objects.Figure.show")
def test_inspect_telluric_correction(mock_show):
    onedspec = _telluric_onedspec()
    onedspec.get_telluric_strength()
    onedspec.apply_telluric_correction()
    for kw in (
        {},
        {"spec_id": 0, "save_fig": True, "fig_type": "png+jpg+svg+pdf"},
    ):
        onedspec.inspect_telluric_correction(
            return_jsonstring=True,
            filename=os.path.join(HERE, "test_output", "cov_telluric"),
            **kw,
        )


# atmospheric extinction
@pytest.mark.parametrize(
    "sci, std",
    [
        (None, None),
        ("AIRMASS", "AIRMASS"),
        ("MISSING", "MISSING"),
        (1.2, 1.5),
        (1, 1),
    ],
)
def test_atmospheric_extinction_airmass_variants(sci, std):
    onedspec = make_flux()
    onedspec.atmospheric_extinction_correction_available = False
    onedspec.apply_atmospheric_extinction_correction(
        science_airmass=sci, standard_airmass=std
    )


def test_atmospheric_extinction_airmass_from_header():
    onedspec = make_flux()
    for spec in (
        onedspec.science_spectrum_list[0],
        onedspec.standard_spectrum_list[0],
    ):
        spec.spectrum_header = fits.Header({"AIRMASS": 1.3, "AM": 1.1})
    onedspec.apply_atmospheric_extinction_correction()
    onedspec.apply_atmospheric_extinction_correction("AM", "AM")


def _fully_reduced(telluric=True, atm_ext=True):
    onedspec = _telluric_onedspec()
    if atm_ext:
        onedspec.apply_atmospheric_extinction_correction(
            science_airmass=1.2, standard_airmass=1.5
        )
    if telluric:
        onedspec.get_telluric_strength()
        onedspec.apply_telluric_correction()
    return onedspec


@patch("plotly.graph_objects.Figure.show")
@pytest.mark.parametrize(
    "telluric, atm_ext", [(True, True), (True, False), (False, True)]
)
def test_inspect_reduced_spectrum_variants(mock_show, telluric, atm_ext):
    onedspec = _fully_reduced(telluric=telluric, atm_ext=atm_ext)
    for kw in (
        {},
        {"atm_ext_corrected": False},
        {"telluric_corrected": False},
        {"atm_ext_corrected": False, "telluric_corrected": False},
        {"stype": "science", "spec_id": 0},
        {"stype": "standard"},
    ):
        onedspec.inspect_reduced_spectrum(
            wave_min=3000.0,
            wave_max=9000.0,
            display=False,
            return_jsonstring=True,
            **kw,
        )
    onedspec.inspect_reduced_spectrum(
        display=False,
        save_fig=True,
        fig_type="png+jpg+svg+pdf+html",
        filename=os.path.join(HERE, "test_output", "cov_reduced"),
    )


@patch("plotly.graph_objects.Figure.show")
def test_inspect_reduced_spectrum_uncalibrated(mock_show):
    onedspec = make(calibrate=False)
    onedspec.inspect_reduced_spectrum(
        display=False, return_jsonstring=True, stype="science"
    )


# save
@pytest.mark.parametrize("name", ["save_fits", "save_csv"])
def test_save_variants(tmp_path, name):
    onedspec = make_flux()
    onedspec.add_trace(
        np.ones(70) * 37, np.ones(70), spec_id=1, stype="science"
    )
    onedspec.add_spec(np.arange(1, 71), spec_id=1, stype="science")
    fn = str(tmp_path / "out")
    getattr(onedspec, name)(output="trace+count", filename=fn, spec_id=[0])
    getattr(onedspec, name)(
        output="trace+count", filename=fn, spec_id=[0], overwrite=True
    )
    getattr(onedspec, name)(
        output="trace", filename=fn, stype="standard", overwrite=True
    )
    getattr(onedspec, name)(
        output="trace", filename=fn, spec_id=[0, 1], overwrite=True
    )
    bare = spectral_reduction.OneDSpec(log_file_name=None, log_level="ERROR")
    getattr(bare, name)(filename=fn)


def test_resample_uncalibrated():
    onedspec = make(calibrate=False)
    onedspec.resample()
    onedspec.resample(stype="science")
    onedspec.resample(stype="standard")


def test_add_wavelengthcalibration_errors():
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    wavecal = WavelengthCalibration(log_file_name=None)
    onedspec.add_science_spectrum_oned(1)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        onedspec.add_wavelengthcalibration(
            [wavecal, wavecal, wavecal], spec_id=[0, 1], stype="science"
        )


def test_add_arc_lines_and_trace_errors():
    onedspec = spectral_reduction.OneDSpec(
        log_file_name=None, log_level="ERROR"
    )
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        onedspec.add_arc_lines([np.ones(3)] * 3, spec_id=[0, 1])
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        onedspec.add_trace("a", "b", spec_id=0)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        onedspec.add_trace([np.ones(3)] * 3, [np.ones(3)] * 2, spec_id=[0, 1])
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        onedspec.add_trace([np.ones(3)] * 3, [np.ones(3)] * 3, spec_id=[0, 1])
