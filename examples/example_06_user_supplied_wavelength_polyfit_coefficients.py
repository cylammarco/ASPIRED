import numpy as np
from astropy.io import fits
from aspired import spectral_reduction

# Load the image
lhs6328_fits = fits.open(
    'sprat_LHS6328_Hiltner102_raw/v_e_20180810_12_1_0_0.fits.gz')[0]

#
# Loading two pre-saved spectral traces from a single FITS file.
#
lhs6328 = spectral_reduction.TwoDSpec(lhs6328_fits,
                                      cosmicray=True,
                                      readnoise=2.34)

# Trace the spectra
lhs6328.ap_trace(nspec=2, display=False)

# Extract the spectra
lhs6328.ap_extract(apwidth=10, optimal=True, skywidth=10, display=False)

# Calibrate the 1D spectra
lhs6328_onedspec = spectral_reduction.OneDSpec()
lhs6328_onedspec.add_twodspec(lhs6328, stype='science')

polyfit_coeff = [
    np.array([
        3.09833375e+03, 5.98842823e+00, -2.83963934e-03, 2.84842392e-06,
        -1.03725267e-09
    ]),
    np.array([
        3.29975984e+03, 5.19493289e+00, -1.87830053e-03, 2.55978647e-06,
        -1.12035864e-09
    ])
]
polyfit_type = ['poly', 'poly']

# Note that there are two science traces, so two polyfit coefficients have to
# be supplied by in a list
lhs6328_onedspec.add_polyfit(polyfit_coeff,
                             polyfit_type=polyfit_type,
                             stype='science')
lhs6328_onedspec.apply_wavelength_calibration(stype='science')

# Inspect reduced spectrum
lhs6328_onedspec.inspect_reduced_spectrum(stype='science')

# Save as a FITS file
lhs6328_onedspec.save_fits(
    output='wavecal+adu',
    filename=
    'example_output/example_06_user_supplied_wavelength_polyfit_coefficients',
    stype='science',
    overwrite=True)
