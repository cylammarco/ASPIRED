# -*- coding: utf-8 -*-
import numpy as np
from aspired import spectrum1D


def test_spectrum1D():
    spec = spectrum1D.Spectrum1D(log_file_name=None)

    spec.add_count(count=np.arange(10))
    assert (spec.count == np.arange(10)).all()
    assert (spec.count_err == np.zeros(10)).all()
    assert (spec.count_sky == np.zeros(10)).all()
    spec.remove_count()
    assert spec.count is None
    assert spec.count_err is None
    assert spec.count_sky is None

    spec.add_variances(np.ones((100, 200)))
    assert (spec.var == np.ones((100, 200))).all()
    spec.remove_variances()
    assert spec.var is None

    spec.add_arc_spec(np.ones(100))
    assert (spec.arc_spec == np.ones(100)).all()
    spec.remove_arc_spec()
    assert spec.arc_spec is None

    spec.add_peaks(np.ones(100))
    assert (spec.peaks == np.ones(100)).all()
    spec.remove_peaks()
    assert spec.peaks is None

    spec.add_peaks_wave(np.ones(100))
    assert (spec.peaks_wave == np.ones(100)).all()
    spec.remove_peaks_wave()
    assert spec.peaks_wave is None

    spec.add_calibrator("lalala")
    assert spec.calibrator == "lalala"
    spec.remove_calibrator()
    assert spec.calibrator is None

    spec.add_atlas_wavelength_range(
        min_atlas_wavelength=1000.0, max_atlas_wavelength=10000.0
    )
    assert spec.min_atlas_wavelength == 1000.0
    assert spec.max_atlas_wavelength == 10000.0
    spec.remove_atlas_wavelength_range()
    assert spec.min_atlas_wavelength is None
    assert spec.max_atlas_wavelength is None

    spec.add_min_atlas_intensity(123456.0)
    assert spec.min_atlas_intensity == 123456.0
    spec.remove_min_atlas_intensity()
    assert spec.min_atlas_intensity is None

    spec.add_min_atlas_distance(0.123)
    assert spec.min_atlas_distance == 0.123
    spec.remove_min_atlas_distance()
    assert spec.min_atlas_distance is None

    spec.add_gain(2.71)
    assert spec.gain == 2.71
    spec.remove_gain()
    assert spec.gain is None

    spec.add_readnoise(10.234)
    assert spec.readnoise == 10.234
    spec.remove_readnoise()
    assert spec.readnoise is None

    spec.add_exptime(3600.0)
    assert spec.exptime == 3600.0
    spec.remove_exptime()
    assert spec.exptime is None

    spec.add_airmass(1.57)
    assert spec.airmass == 1.57
    spec.remove_airmass()
    assert spec.airmass is None

    spec.add_seeing(2.39)
    assert spec.seeing == 2.39
    spec.remove_seeing()
    assert spec.seeing is None

    spec.add_weather_condition(
        pressure=123456.0, temperature=279.3, relative_humidity=15.1
    )
    assert spec.pressure == 123456.0
    assert spec.temperature == 279.3
    assert spec.relative_humidity == 15.1
    spec.remove_weather_condition()
    assert spec.pressure is None
    assert spec.temperature is None
    assert spec.relative_humidity is None

    spec.add_fit_type("leg")
    assert spec.fit_type == "leg"
    spec.remove_fit_type()
    assert spec.fit_type is None

    spec.add_fit_coeff(np.arange(10))
    assert (spec.fit_coeff == np.arange(10)).all()
    spec.remove_fit_coeff()
    assert spec.fit_coeff is None

    spec.add_calibrator_properties(
        num_pix=1024,
        pixel_list=np.arange(1024),
        plotting_library="plotly",
        log_level="info",
    )
    assert spec.num_pix == 1024
    assert (spec.pixel_list == np.arange(1024)).all()
    assert spec.plotting_library == "plotly"
    assert spec.log_level == "info"
    spec.remove_calibrator_properties()
    assert spec.num_pix is None
    assert spec.pixel_list is None
    assert spec.plotting_library is None
    assert spec.log_level is None

    spec.add_hough_properties(
        num_slopes=1000,
        xbins=120,
        ybins=250,
        min_wavelength=3000.0,
        max_wavelength=7800.0,
        range_tolerance=369.0,
        linearity_tolerance=135.0,
    )
    assert spec.num_slopes == 1000
    assert spec.xbins == 120
    assert spec.ybins == 250
    assert spec.min_wavelength == 3000.0
    assert spec.max_wavelength == 7800.0
    assert spec.range_tolerance == 369.0
    assert spec.linearity_tolerance == 135.0
    spec.remove_hough_properties()
    assert spec.num_slopes is None
    assert spec.xbins is None
    assert spec.ybins is None
    assert spec.min_wavelength is None
    assert spec.max_wavelength is None
    assert spec.range_tolerance is None
    assert spec.linearity_tolerance is None

    spec.add_ransac_properties(
        sample_size=999,
        top_n_candidate=7,
        linear=True,
        filter_close=True,
        ransac_tolerance=5.0,
        candidate_weighted=True,
        hough_weight=1.3,
        minimum_matches=5,
        minimum_peak_utilisation=80.0,
        minimum_fit_error=0.1,
    )
    assert spec.sample_size == 999
    assert spec.top_n_candidate == 7
    assert spec.linear
    assert spec.filter_close
    assert spec.ransac_tolerance == 5.0
    assert spec.candidate_weighted
    assert spec.hough_weight == 1.3
    assert spec.minimum_matches == 5
    assert spec.minimum_peak_utilisation == 80.0
    assert spec.minimum_fit_error == 0.1
    spec.remove_ransac_properties()
    assert spec.sample_size is None
    assert spec.top_n_candidate is None
    assert spec.linear is None
    assert spec.filter_close is None
    assert spec.ransac_tolerance is None
    assert spec.candidate_weighted is None
    assert spec.hough_weight is None
    assert spec.minimum_matches is None
    assert spec.minimum_peak_utilisation is None
    assert spec.minimum_fit_error is None

    spec.add_fit_output_final(
        fit_coeff=[1, 2, 5, 7, 10],
        matched_peaks=[0, 1, 2, 3],
        matched_atlas=[10, 11, 12, 13],
        rms=0.123456,
        residual=0.56789,
        peak_utilisation=87.67894,
        atlas_utilisation=51.7643,
    )
    assert spec.fit_coeff == [1, 2, 5, 7, 10]
    assert spec.matched_peaks == [0, 1, 2, 3]
    assert spec.matched_atlas == [10, 11, 12, 13]
    assert spec.rms == 0.123456
    assert spec.residual == 0.56789
    assert spec.peak_utilisation == 87.67894
    assert spec.atlas_utilisation == 51.7643
    spec.remove_fit_output_final()
    assert spec.fit_coeff is None
    assert spec.matched_peaks is None
    assert spec.matched_atlas is None
    assert spec.rms is None
    assert spec.residual is None
    assert spec.peak_utilisation is None
    assert spec.atlas_utilisation is None

    spec.add_fit_output_rascal(
        fit_coeff=[1, 2, 5, 7, 10],
        matched_peaks=[0, 1, 2, 3],
        matched_atlas=[10, 11, 12, 13],
        rms=0.123456,
        residual=0.56789,
        peak_utilisation=87.67894,
        atlas_utilisation=51.7643,
    )
    assert spec.fit_coeff_rascal == [1, 2, 5, 7, 10]
    assert spec.matched_peaks == [0, 1, 2, 3]
    assert spec.matched_atlas == [10, 11, 12, 13]
    assert spec.rms_rascal == 0.123456
    assert spec.residual_rascal == 0.56789
    assert spec.peak_utilisation_rascal == 87.67894
    assert spec.atlas_utilisation_rascal == 51.7643
    spec.remove_fit_output_rascal()
    assert spec.fit_coeff_rascal is None
    assert spec.matched_peaks_rascal is None
    assert spec.matched_atlas_rascal is None
    assert spec.rms_rascal is None
    assert spec.residual_rascal is None
    assert spec.peak_utilisation_rascal is None
    assert spec.atlas_utilisation_rascal is None

    spec.add_fit_output_refine(
        fit_coeff=[1, 2, 5, 7, 10],
        matched_peaks=[0, 1, 2, 3],
        matched_atlas=[10, 11, 12, 13],
        rms=0.123456,
        residual=0.56789,
        peak_utilisation=87.67894,
        atlas_utilisation=51.7643,
    )
    assert spec.fit_coeff_refine == [1, 2, 5, 7, 10]
    assert spec.matched_peaks_refine == [0, 1, 2, 3]
    assert spec.matched_atlas_refine == [10, 11, 12, 13]
    assert spec.rms_refine == 0.123456
    assert spec.residual_refine == 0.56789
    assert spec.peak_utilisation_refine == 87.67894
    assert spec.atlas_utilisation_refine == 51.7643
    spec.remove_fit_output_refine()
    assert spec.fit_coeff_refine is None
    assert spec.matched_peaks_refine is None
    assert spec.matched_atlas_refine is None
    assert spec.rms_refine is None
    assert spec.residual_refine is None
    assert spec.peak_utilisation_refine is None
    assert spec.atlas_utilisation_refine is None

    spec.add_wavelength(np.arange(1000))
    assert (spec.wave == np.arange(1000)).all()
    spec.remove_wavelength()
    assert spec.wave is None

    spec.add_wavelength_resampled(np.arange(789))
    assert spec.wave_bin == 1
    assert spec.wave_start == 0
    assert spec.wave_end == 788
    assert (spec.wave_resampled == np.arange(789)).all()
    spec.remove_wavelength_resampled()
    assert spec.wave_bin is None
    assert spec.wave_start is None
    assert spec.wave_end is None
    assert spec.wave_resampled is None

    spec.add_count_resampled(np.arange(100), np.arange(200), np.arange(300))
    assert (spec.count_resampled == np.arange(100)).all()
    assert (spec.count_err_resampled == np.arange(200)).all()
    assert (spec.count_sky_resampled == np.arange(300)).all()
    spec.remove_count_resampled()
    assert spec.count_resampled is None
    assert spec.count_err_resampled is None
    assert spec.count_sky_resampled is None

    spec.add_smoothing(smooth=True, slength=11, sorder=3)
    assert spec.smooth
    assert spec.slength == 11
    assert spec.sorder == 3
    spec.remove_smoothing()
    assert spec.smooth is None
    assert spec.slength is None
    assert spec.sorder is None

    spec.add_sensitivity_func(np.poly1d)
    assert spec.sensitivity_func == np.poly1d
    spec.remove_sensitivity_func()
    assert spec.sensitivity_func is None

    spec.add_sensitivity(np.arange(1000))
    assert (spec.sensitivity == np.arange(1000)).all()
    spec.remove_sensitivity()
    assert spec.sensitivity is None

    spec.add_sensitivity_resampled(np.arange(2000))
    assert (spec.sensitivity_resampled == np.arange(2000)).all()
    spec.remove_sensitivity_resampled()
    assert spec.sensitivity_resampled is None

    spec.add_literature_standard(
        wave_literature=np.arange(456), flux_literature=np.ones(789)
    )
    assert (spec.wave_literature == np.arange(456)).all()
    assert (spec.flux_literature == np.ones(789)).all()
    spec.remove_literature_standard()
    assert spec.wave_literature is None
    assert spec.flux_literature is None

    spec.add_flux(np.arange(100), np.arange(200), np.arange(300))
    assert (spec.flux == np.arange(100)).all()
    assert (spec.flux_err == np.arange(200)).all()
    assert (spec.flux_sky == np.arange(300)).all()
    spec.remove_flux()
    assert spec.flux is None
    assert spec.flux_err is None
    assert spec.flux_sky is None

    spec.add_flux_resampled(np.arange(100), np.arange(200), np.arange(300))
    assert (spec.flux_resampled == np.arange(100)).all()
    assert (spec.flux_err_resampled == np.arange(200)).all()
    assert (spec.flux_sky_resampled == np.arange(300)).all()
    spec.remove_flux_resampled()
    assert spec.flux_resampled is None
    assert spec.flux_err_resampled is None
    assert spec.flux_sky_resampled is None
