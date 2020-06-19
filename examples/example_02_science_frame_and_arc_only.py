from astropy.io import fits
from aspired import spectral_reduction


# Load the image
arc_fits = fits.open(
    'sprat_LHS6328_Hiltner102_raw/v_a_20180810_13_1_0_1.fits.gz')[0]

# Case 1
#
# Loading two pre-saved spectral traces from a single FITS file.
#
lhs6328_extracted = fits.open('example_output/lhs6328_adu_traces.fits')
lhs6328_adu = lhs6328_extracted[0].data
lhs6328_trace = lhs6328_extracted[3].data
lhs6328_trace_sigma = lhs6328_extracted[4].data

lhs6328_onedspec = spectral_reduction.OneDSpec()

# Add a 2D arc image
lhs6328_onedspec.add_arc(arc_fits, stype='science')

# Add the trace and the line spread function (sigma) to the 2D arc image
lhs6328_onedspec.add_trace(lhs6328_trace, lhs6328_trace_sigma, stype='science')

# Extract the 1D arc by aperture sum of the traces provided
lhs6328_onedspec.extract_arcspec(display=False, stype='science')

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(display=False, stype='science')

# Configure the wavelength calibrator
lhs6328_onedspec.initialise_calibrator(min_wavelength=3500, max_wavelength=8000)
lhs6328_onedspec.set_fit_constraints(stype='science')
lhs6328_onedspec.add_atlas(elements=['Xe'], stype='science')

# Solve for the pixel-to-wavelength solution
lhs6328_onedspec.fit(stype='science')
lhs6328_onedspec.refine_fit(display=False, stype='science')

# Add the extracted 1D spectrum without the uncertainties and sky
lhs6328_onedspec.add_spec(lhs6328_adu, stype='science')

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science')
lhs6328_onedspec.inspect_reduced_spectrum(stype='science')

# Save as a FITS file
lhs6328_onedspec.save_fits(
    output='flux+wavecal+adu',
    filename='example_output/example_02_wavelength_calibrated_spectrum',
    stype='science',
    overwrite=True)

# Case 2
#
# Loading a single pre-saved spectral trace.
#
lhs6328_extracted = fits.open('example_output/lhs6328_adu_traces_1.fits')
lhs6328_adu = lhs6328_extracted[0].data
lhs6328_trace = lhs6328_extracted[3].data
lhs6328_trace_sigma = lhs6328_extracted[4].data

lhs6328_onedspec = spectral_reduction.OneDSpec()

# Add a 2D arc image
lhs6328_onedspec.add_arc(arc_fits, stype='science')

# Add the trace and the line spread function (sigma) to the 2D arc image
lhs6328_onedspec.add_trace(lhs6328_trace, lhs6328_trace_sigma, stype='science')

# Extract the 1D arc by aperture sum of the traces provided
lhs6328_onedspec.extract_arcspec(display=False, stype='science')

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(display=False, stype='science')

# Configure the wavelength calibrator
lhs6328_onedspec.initialise_calibrator(min_wavelength=3500, max_wavelength=8000)
lhs6328_onedspec.set_fit_constraints(stype='science')
lhs6328_onedspec.add_atlas(elements=['Xe'], stype='science')

# Solve for the pixel-to-wavelength solution
lhs6328_onedspec.fit(stype='science')
lhs6328_onedspec.refine_fit(display=False, stype='science')

# Add the extracted 1D spectrum without the uncertainties and sky
lhs6328_onedspec.add_spec(lhs6328_adu, stype='science')

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science')
lhs6328_onedspec.inspect_reduced_spectrum(stype='science')

# Save as a FITS file
lhs6328_onedspec.save_fits(
    output='flux+wavecal+adu',
    filename='example_output/example_02_wavelength_calibrated_spectrum_1',
    stype='science',
    overwrite=True)
