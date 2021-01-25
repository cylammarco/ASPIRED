import numpy as np
from astropy.io import fits
from aspired.wavelengthcalibration import WavelengthCalibration
from aspired.spectrum1D import Spectrum1D
from aspired import spectral_reduction


def test_wavecal():
    # Line list
    atlas = [
        4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
        4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
        6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32,
        6976.18, 7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65,
        7887.40, 7967.34, 8057.258
    ]
    element = ['Xe'] * len(atlas)

    lhs6328_spectrum1D = Spectrum1D(log_file_name=None)
    wavecal = WavelengthCalibration(log_file_name=None)

    # Science arc_spec
    arc_spec = np.loadtxt(
        'test/test_data/test_full_run_science_0_arc_spec.csv',
        delimiter=',',
        skiprows=1)
    lhs6328_spectrum1D.add_arc_spec(arc_spec)
    wavecal.from_spectrum1D(lhs6328_spectrum1D)

    # Find the peaks of the arc
    wavecal.find_arc_lines()

    # Configure the wavelength calibrator
    wavecal.initialise_calibrator()
    wavecal.set_hough_properties(num_slopes=1000,
                                 xbins=200,
                                 ybins=200,
                                 min_wavelength=3500,
                                 max_wavelength=8500)
    wavecal.set_ransac_properties(filter_close=True)

    wavecal.load_user_atlas(elements=element, wavelengths=atlas)
    wavecal.do_hough_transform()

    # Solve for the pixel-to-wavelength solution
    wavecal.fit(max_tries=500, display=False)

    # Just getting the calibrator
    wavecal.get_calibrator()

    # Save a FITS file
    wavecal.save_fits(output='wavecal',
                      filename='test/test_output/test_wavecal',
                      overwrite=True)

    # Save a CSV file
    wavecal.save_csv(output='wavecal',
                     filename='test/test_output/test_wavecal',
                     overwrite=True)


def test_user_supplied_poly_coeff():
    # Load the image
    lhs6328_fits = fits.open('test/test_data/v_e_20180810_12_1_0_0.fits.gz')[0]
    spatial_mask = np.arange(50, 200)
    spec_mask = np.arange(50, 1024)

    #
    # Loading two pre-saved spectral traces from a single FITS file.
    #
    lhs6328 = spectral_reduction.TwoDSpec(lhs6328_fits,
                                          spatial_mask=spatial_mask,
                                          spec_mask=spec_mask,
                                          cosmicray=True,
                                          readnoise=2.34,
                                          log_file_name=None)

    # Trace the spectra
    lhs6328.ap_trace(nspec=2, display=False)

    # Extract the spectra
    lhs6328.ap_extract(apwidth=10, optimal=True, skywidth=10, display=False)

    # Calibrate the 1D spectra
    lhs6328_onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    lhs6328_onedspec.from_twodspec(lhs6328)

    fit_coeff = np.array([
        3.09833375e+03, 5.98842823e+00, -2.83963934e-03, 2.84842392e-06,
        -1.03725267e-09
    ])
    fit_type = 'poly'

    # Note that there are two science traces, so two polyfit coefficients
    # have to be supplied by in a list
    lhs6328_onedspec.add_fit_coeff(fit_coeff, fit_type)
    lhs6328_onedspec.apply_wavelength_calibration()

    # Inspect reduced spectrum
    lhs6328_onedspec.inspect_reduced_spectrum(display=False)

    # Save as a FITS file
    lhs6328_onedspec.save_fits(
        output='wavecal+count',
        filename='test/test_output/user_supplied_wavelength_polyfit_'
                 'coefficients',
        stype='science',
        overwrite=True)


def test_user_supplied_wavelength():
    # Load the image
    lhs6328_fits = fits.open('test/test_data/v_e_20180810_12_1_0_0.fits.gz')[0]
    spatial_mask = np.arange(50, 200)
    spec_mask = np.arange(50, 1024)

    #
    # Loading two pre-saved spectral traces from a single FITS file.
    #
    lhs6328 = spectral_reduction.TwoDSpec(lhs6328_fits,
                                          spatial_mask=spatial_mask,
                                          spec_mask=spec_mask,
                                          cosmicray=True,
                                          readnoise=2.34,
                                          log_file_name=None)

    # Trace the spectra
    lhs6328.ap_trace(nspec=2, display=False)

    # Extract the spectra
    lhs6328.ap_extract(apwidth=10, optimal=True, skywidth=10, display=False)

    # Calibrate the 1D spectra
    lhs6328_onedspec = spectral_reduction.OneDSpec(log_file_name=None)
    lhs6328_onedspec.from_twodspec(lhs6328)

    wavelength = np.genfromtxt(
        'test/test_data/test_full_run_standard_wavelength.csv')
    # Manually supply wavelengths
    lhs6328_onedspec.add_wavelength(wavelength)

    # Inspect reduced spectrum
    lhs6328_onedspec.inspect_reduced_spectrum(display=False)

    # Save as a FITS file
    lhs6328_onedspec.save_fits(
        output='count',
        filename='test/test_output/user_supplied_wavelength',
        stype='science',
        overwrite=True)
