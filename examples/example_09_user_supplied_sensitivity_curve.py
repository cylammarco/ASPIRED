import numpy as np
from astropy.io import fits
from aspired import spectral_reduction
from scipy import interpolate as itp

# Load the Wavelength calibrated 1D spectral image of Hiltner 102
hilt102_fits = fits.open(
    'sprat_LHS6328_Hiltner102_raw/v_s_20180810_27_1_0_2.fits.gz')[3]

# Get the sensitivity curve from the pre-calibrated data
calibrated_data = fits.open(
    'sprat_LHS6328_Hiltner102_raw/v_e_20180810_12_1_0_2.fits.gz')
sensitivity = calibrated_data[5].data[0] / calibrated_data[3].data[0] *\
    calibrated_data[3].header['EXPTIME'] / hilt102_fits.header['EXPTIME']

wave_bin = hilt102_fits.header['CDELT1']
wave_start = hilt102_fits.header['CRVAL1'] + wave_bin / 2.
wave_end = wave_start + (hilt102_fits.header['NAXIS1'] - 1) * wave_bin

wave = np.linspace(wave_start, wave_end, hilt102_fits.header['NAXIS1'])

# interpolate the senstivity curve with wavelength
sensitivity_itp = itp.interp1d(wave, np.log10(sensitivity), fill_value='extrapolate')

# Calibrate the 1D spectra
hilt102_onedspec = spectral_reduction.OneDSpec()

# Note that there are two science traces, so two wavelengths have to be
# supplied by in a list
hilt102_onedspec.add_spec(hilt102_fits.data[0], stype='standard')
hilt102_onedspec.add_wavelength(wave, stype='standard')
hilt102_onedspec.add_sensitivity_itp(sensitivity_itp)

hilt102_onedspec.apply_flux_calibration(stype='standard')

hilt102_onedspec.load_standard(target='hiltner102', display=False)

# Inspect reduced spectrum
hilt102_onedspec.inspect_reduced_spectrum(stype='standard')

# Save as a FITS file
hilt102_onedspec.save_fits(
    output='flux',
    filename='example_output/example_09_user_supplied_sensitivity_curve',
    stype='standard',
    overwrite=True)
