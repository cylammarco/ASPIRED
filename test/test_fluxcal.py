import numpy as np
from aspired.fluxcalibration import FluxCalibration
from aspired.spectrum1D import Spectrum1D


def test_sensitivity():

    hiltner_spectrum1D = Spectrum1D(log_file_name=None)
    sens = FluxCalibration(log_file_name=None)

    # Standard count
    count = np.loadtxt('test/test_data/test_full_run_standard_count.csv',
                       delimiter=',',
                       skiprows=1)[:, 0]
    wavelength = np.loadtxt(
        'test/test_data/test_full_run_standard_wavelength.csv', skiprows=1)

    hiltner_spectrum1D.add_count(count)
    hiltner_spectrum1D.add_wavelength(wavelength)
    sens.from_spectrum1D(hiltner_spectrum1D)

    # Load standard star from literature
    sens.load_standard('hiltner102')

    sens.compute_sensitivity()

    # Get back the spectrum1D and merge
    hiltner_spectrum1D.merge(sens.get_spectrum1D())

    # Save a FITS file
    sens.save_fits(output='sensitivity',
                   filename='test/test_output/test_sensitivity',
                   overwrite=True)

    # Save a CSV file
    sens.save_csv(output='sensitivity',
                  filename='test/test_output/test_sensitivity',
                  overwrite=True)


def test_fluxcalibration():

    hiltner_spectrum1D = Spectrum1D(log_file_name=None)
    lhs6328_spectrum1D = Spectrum1D(log_file_name=None)

    fluxcalibrator = FluxCalibration(log_file_name=None)

    # Science and Standard counts
    standard_count = np.loadtxt(
        'test/test_data/test_full_run_standard_count.csv',
        delimiter=',',
        skiprows=1)[:, 0]
    science_count = np.loadtxt(
        'test/test_data/test_full_run_science_0_count.csv',
        delimiter=',',
        skiprows=1)[:, 0]
    wavelength = np.loadtxt(
        'test/test_data/test_full_run_standard_wavelength.csv', skiprows=1)

    hiltner_spectrum1D.add_count(standard_count)
    hiltner_spectrum1D.add_wavelength(wavelength)

    lhs6328_spectrum1D.add_count(science_count)
    lhs6328_spectrum1D.add_wavelength(wavelength)

    # Add the standard spectrum1D to the flux calibrator
    fluxcalibrator.from_spectrum1D(hiltner_spectrum1D)

    # Load standard star from literature
    fluxcalibrator.load_standard('hiltner102')
    fluxcalibrator.compute_sensitivity()

    # Get back the spectrum1D and merge
    fluxcalibrator.apply_flux_calibration(lhs6328_spectrum1D)
