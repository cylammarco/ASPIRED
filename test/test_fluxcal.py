# -*- coding: utf-8 -*-
import os
from unittest.mock import patch

import numpy as np
from aspired.flux_calibration import FluxCalibration
from aspired.spectrum_oneD import SpectrumOneD

HERE = os.path.dirname(os.path.realpath(__file__))


def test_ing_standard():
    fluxcal = FluxCalibration(log_file_name=None)
    fluxcal.load_standard(target="bd254", library="ing_oke", ftype="flux")
    fluxcal.load_standard(target="bd254", library="ing_oke", ftype="mag")


def test_eso_standard():
    fluxcal = FluxCalibration(log_file_name=None)
    fluxcal.load_standard(target="eg274", library="esoctiostan", ftype="flux")
    fluxcal.load_standard(target="eg274", library="esoctiostan", ftype="mag")


def test_iraf_standard():
    fluxcal = FluxCalibration(log_file_name=None)
    fluxcal.load_standard(
        target="bd75325", library="irafoke1990", ftype="flux"
    )
    fluxcal.load_standard(target="bd75325", library="irafoke1990", ftype="mag")


def test_standard_return_suggection1():
    fluxcal = FluxCalibration(log_file_name=None)
    fluxcal.load_standard(target="hiltner")


def test_standard_return_suggestion2():
    fluxcal = FluxCalibration(log_file_name=None)
    fluxcal.load_standard(target="bd")


@patch("plotly.graph_objects.Figure.show")
def test_sensitivity(mock_show):
    hiltner_spectrum_oned = SpectrumOneD(log_file_name=None)
    sens = FluxCalibration(log_file_name=None)

    # Standard count
    count = np.loadtxt(
        os.path.join(HERE, "test_data", "test_full_run_standard_count.csv"),
        delimiter=",",
        skiprows=1,
    )[:, 0]
    wavelength = np.loadtxt(
        os.path.join(
            HERE, "test_data", "test_full_run_standard_wavelength.csv"
        ),
        skiprows=1,
    )

    hiltner_spectrum_oned.add_count(count)
    hiltner_spectrum_oned.add_wavelength(wavelength)
    sens.from_spectrum_oned(hiltner_spectrum_oned)

    # Load standard star from literature
    sens.load_standard("hiltner102")
    sens.inspect_standard(
        display=False,
        return_jsonstring=True,
        save_fig=True,
        fig_type="iframe+png",
        filename=os.path.join(HERE, "test_output", "fluxcal_inspect_standard"),
    )
    sens.inspect_standard(display=True)

    sens.get_sensitivity()

    # Get back the spectrum_oned and merge
    hiltner_spectrum_oned.merge(sens.get_spectrum_oned())

    # Save a FITS file
    sens.save_fits(
        output="sensitivity",
        filename=os.path.join(HERE, "test_output", "test_sensitivity"),
        overwrite=True,
    )

    # Save a CSV file
    sens.save_csv(
        output="sensitivity",
        filename=os.path.join(HERE, "test_output", "test_sensitivity"),
        overwrite=True,
    )


@patch("plotly.graph_objects.Figure.show")
def test_fluxcalibration(mock_show):
    hiltner_spectrum_oned = SpectrumOneD(log_file_name=None)
    lhs6328_spectrum_oned = SpectrumOneD(log_file_name=None)

    fluxcalibrator = FluxCalibration(log_file_name=None)

    # Science and Standard counts
    standard_count = np.loadtxt(
        os.path.join(HERE, "test_data", "test_full_run_standard_count.csv"),
        delimiter=",",
        skiprows=1,
    )[:, 0]
    science_count = np.loadtxt(
        os.path.join(HERE, "test_data", "test_full_run_science_0_count.csv"),
        delimiter=",",
        skiprows=1,
    )[:, 0]
    wavelength = np.loadtxt(
        os.path.join(
            HERE, "test_data", "test_full_run_standard_wavelength.csv"
        ),
        skiprows=1,
    )

    hiltner_spectrum_oned.add_count(standard_count)
    hiltner_spectrum_oned.add_wavelength(wavelength)

    lhs6328_spectrum_oned.add_count(science_count)
    lhs6328_spectrum_oned.add_wavelength(wavelength)

    # Add the standard spectrum_oned to the flux calibrator
    fluxcalibrator.from_spectrum_oned(hiltner_spectrum_oned)

    # Load standard star from literature
    fluxcalibrator.load_standard("hiltner102")
    fluxcalibrator.get_sensitivity()

    # Get back the spectrum_oned and merge
    fluxcalibrator.apply_flux_calibration(
        lhs6328_spectrum_oned,
        inspect=True,
        display=False,
        return_jsonstring=True,
        save_fig=True,
        fig_type="iframe+png",
        filename=os.path.join(HERE, "test_output", "fluxcal_flux_calibration"),
    )
    fluxcalibrator.apply_flux_calibration(lhs6328_spectrum_oned, display=True)
