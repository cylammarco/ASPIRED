import numpy as np
from astropy.io import fits
from aspired import spectral_reduction
from scipy import interpolate as itp

# Load the wavelength calibrated and resampled 1D spectra
lhs6328_fits = fits.open('sprat_LHS6328_Hiltner102_raw/v_e_20180810_12_1_0_2.fits.gz')[3]
hilt102_fits = fits.open('sprat_LHS6328_Hiltner102_raw/v_s_20180810_27_1_0_2.fits.gz')[3]

# Get the wavelength
wave_bin = hilt102_fits.header['CDELT1']
wave_start = hilt102_fits.header['CRVAL1'] + wave_bin / 2.
wave_end = wave_start + (hilt102_fits.header['NAXIS1'] - 1) * wave_bin

wave = np.linspace(wave_start, wave_end, hilt102_fits.header['NAXIS1'])

# Calibrate the 1D spectra
lhs6328_onedspec = spectral_reduction.OneDSpec()

# Note that there are two science traces, so two wavelengths have to be
# supplied by in a list
lhs6328_onedspec.add_spec(lhs6328_fits.data[0], stype='science')
lhs6328_onedspec.add_spec(hilt102_fits.data[0], stype='standard')
lhs6328_onedspec.add_wavelength(wave, stype='science')
lhs6328_onedspec.add_wavelength(wave, stype='standard')

lhs6328_onedspec.load_standard(target='hiltner102', display=False)
lhs6328_onedspec.compute_sensitivity()
lhs6328_onedspec.apply_flux_calibration(stype='science+standard')

# Inspect reduced spectrum
lhs6328_onedspec.inspect_reduced_spectrum(stype='science+standard')

# Save as a FITS file
lhs6328_onedspec.save_fits(
    output='flux+wavecal+fluxraw',
    filename='example_output/example_10_user_supplied_wavelength_calibrated_science_and_standard_1D_spectra',
    stype='science',
    overwrite=True)
