# -*- coding: utf-8 -*-
import copy
import os
from unittest.mock import patch

import numpy as np

from aspired import spectral_reduction
from aspired.flux_calibration import FluxCalibration

HERE = os.path.dirname(os.path.realpath(__file__))


def test_telluric_square_wave():

    wave = np.arange(1000.0)
    flux_sci = np.ones(1000) * 5.0
    flux_std = np.ones(1000) * 100.0

    flux_sci_continuum = copy.deepcopy(flux_sci)
    flux_std_continuum = copy.deepcopy(flux_std)

    flux_sci[500:550] *= 0.01
    flux_sci[700:750] *= 0.001
    flux_sci[850:950] *= 0.1

    flux_std[500:550] *= 0.01
    flux_std[700:750] *= 0.001
    flux_std[850:950] *= 0.1

    # Get the telluric profile
    fluxcal = FluxCalibration(log_file_name=None)
    telluric_func, _, _ = fluxcal.get_telluric_profile(
        wave,
        flux_std,
        flux_std_continuum,
        mask_range=[[495, 551], [700, 753], [848, 960]],
        return_function=True,
    )

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.science_spectrum_list[0].add_wavelength(wave)
    onedspec.science_spectrum_list[0].add_flux(flux_sci, None, None)
    onedspec.science_spectrum_list[0].add_flux_continuum(flux_sci_continuum)
    # onedspec.fluxcal.spectrum1D.add_wavelength(wave)
    # onedspec.fluxcal.spectrum1D.add_flux(flux_std, None, None)
    # onedspec.fluxcal.spectrum1D.add_flux_continuum(flux_std_continuum)

    onedspec.add_telluric_function(telluric_func, stype="science")
    onedspec.get_telluric_correction()
    onedspec.apply_telluric_correction()

    assert np.isclose(
        np.nansum(onedspec.science_spectrum_list[0].flux),
        np.nansum(flux_sci_continuum),
        rtol=1e-2,
    )

    onedspec.inspect_telluric_correction(
        display=False,
        return_jsonstring=True,
        save_fig=True,
        fig_type="iframe+jpg+png+svg+pdf",
        filename=os.path.join(HERE, "test_output", "test_telluric"),
    )


@patch("plotly.graph_objects.Figure.show")
def test_telluric_real_data(mock_show):
    std_wave = np.load(os.path.join(HERE, "test_data", "std_wave.npy"))
    std_flux = np.load(os.path.join(HERE, "test_data", "std_flux.npy"))
    std_flux_continuum = np.load(
        os.path.join(HERE, "test_data", "std_flux_continuum.npy")
    )
    sci_wave = np.load(os.path.join(HERE, "test_data", "sci_wave.npy"))
    sci_flux = np.load(os.path.join(HERE, "test_data", "sci_flux.npy"))
    sci_flux_continuum = np.load(
        os.path.join(HERE, "test_data", "sci_flux_continuum.npy")
    )

    # Get the telluric profile
    fluxcal = FluxCalibration(log_file_name=None)
    telluric_func, _, _ = fluxcal.get_telluric_profile(
        std_wave, std_flux, std_flux_continuum, return_function=True
    )

    onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    onedspec.science_spectrum_list[0].add_wavelength(sci_wave)
    onedspec.science_spectrum_list[0].add_flux(sci_flux, None, None)
    onedspec.science_spectrum_list[0].add_flux_continuum(sci_flux_continuum)
    onedspec.fluxcal.spectrum1D.add_wavelength(std_wave)
    onedspec.fluxcal.spectrum1D.add_flux(std_flux, None, None)
    onedspec.fluxcal.spectrum1D.add_flux_continuum(std_flux_continuum)

    onedspec.add_telluric_function(telluric_func)
    onedspec.apply_telluric_correction()

    assert np.isclose(
        np.nansum(onedspec.science_spectrum_list[0].flux),
        np.nansum(sci_flux_continuum),
        rtol=1e-2,
    )

    onedspec.inspect_telluric_profile(
        display=True,
        return_jsonstring=True,
        save_fig=True,
        fig_type="iframe+jpg+png+svg+pdf",
        filename=os.path.join(HERE, "test_output", "test_telluric"),
    )
