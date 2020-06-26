from astropy.io import fits
from aspired import image_reduction
from aspired import spectral_reduction

# Case 1
#
# Extract two spectra
#

# Science frame
lhs6328_frame = image_reduction.ImageReduction('sprat_LHS6328.list')
lhs6328_frame.reduce()

lhs6328_twodspec = spectral_reduction.TwoDSpec(lhs6328_frame, cosmicray=False)
lhs6328_twodspec.ap_trace(nspec=2, display=True)

lhs6328_twodspec.ap_extract(apwidth=15,
                            skywidth=10,
                            skydeg=1,
                            optimal=True,
                            display=True,
                            jsonstring=False)

lhs6328_twodspec.save_fits(
    filename='example_output/example_01_a_science_traces', overwrite=True)

# Standard frame
standard_frame = image_reduction.ImageReduction('sprat_Hiltner102.list')
standard_frame.reduce()

hilt102_twodspec = spectral_reduction.TwoDSpec(standard_frame, cosmicray=False)

hilt102_twodspec.ap_trace(nspec=1, display=True)

hilt102_twodspec.ap_extract(apwidth=25,
                            skysep=3,
                            skywidth=5,
                            skydeg=1,
                            optimal=True,
                            display=True,
                            jsonstring=False)

hilt102_twodspec.save_fits(
    filename='example_output/example_01_a_standard_trace', overwrite=True)

# Handle 1D Science spectrum
lhs6328_onedspec = spectral_reduction.OneDSpec()
lhs6328_onedspec.add_twodspec(lhs6328_twodspec, stype='science')
lhs6328_onedspec.add_twodspec(hilt102_twodspec, stype='standard')

# Add a 2D arc image
lhs6328_onedspec.add_arc(lhs6328_frame, stype='science')
lhs6328_onedspec.add_arc(standard_frame, stype='standard')

# Extract the 1D arc by aperture sum of the traces provided
lhs6328_onedspec.extract_arcspec(display=True, stype='science+standard')

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(display=True, stype='science+standard')

# Configure the wavelength calibrator
lhs6328_onedspec.initialise_calibrator(min_wavelength=3500,
                                       max_wavelength=8000,
                                       stype='science+standard')
lhs6328_onedspec.set_fit_constraints(stype='science+standard')
lhs6328_onedspec.add_atlas(elements=['Xe'], stype='science+standard')

# Solve for the pixel-to-wavelength solution
lhs6328_onedspec.fit(stype='science+standard', display=True)
lhs6328_onedspec.refine_fit(n_delta=2, display=True, stype='science+standard')
lhs6328_onedspec.refine_fit(display=True, stype='science+standard')

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science+standard')

# Get the standard from the library
lhs6328_onedspec.load_standard(target='hiltner102')
lhs6328_onedspec.inspect_standard()

lhs6328_onedspec.compute_sensitivity(kind='cubic')
lhs6328_onedspec.inspect_sensitivity()

lhs6328_onedspec.apply_flux_calibration(stype='science+standard')

lhs6328_onedspec.inspect_reduced_spectrum(stype='science+standard')

# Save as FITS
lhs6328_onedspec.save_fits(
    output='flux+wavecal+fluxraw+adu',
    filename='example_output/example_01_a_full_reduction',
    stype='science+standard',
    overwrite=True)

# save as CSV
lhs6328_onedspec.save_csv(
    output='flux+wavecal+fluxraw+adu',
    filename='example_output/example_01_a_full_reduction',
    stype='science+standard',
    overwrite=True)

# Case 2
#
# Extract one spectrum
#
lhs6328_frame = image_reduction.ImageReduction('sprat_LHS6328.list')
lhs6328_frame.reduce()

lhs6328_twodspec = spectral_reduction.TwoDSpec(lhs6328_frame, cosmicray=False)
lhs6328_twodspec.ap_trace(nspec=1, display=True)

lhs6328_twodspec.ap_extract(apwidth=15,
                            optimal=True,
                            skywidth=10,
                            skydeg=1,
                            display=True,
                            jsonstring=False)

lhs6328_twodspec.save_fits(
    filename='example_output/example_01_b_science_trace', overwrite=True)

lhs6328_onedspec = spectral_reduction.OneDSpec()
lhs6328_onedspec.add_twodspec(lhs6328_twodspec, stype='science')
# The standard extraction is identical to above, so we are reusing it
lhs6328_onedspec.add_twodspec(hilt102_twodspec, stype='standard')

# Add a 2D arc image
lhs6328_onedspec.add_arc(lhs6328_frame, stype='science')
lhs6328_onedspec.add_arc(standard_frame, stype='standard')

# Extract the 1D arc by aperture sum of the traces provided
lhs6328_onedspec.extract_arcspec(display=True, stype='science+standard')

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(display=True, stype='science+standard')

# Configure the wavelength calibrator
lhs6328_onedspec.initialise_calibrator(min_wavelength=3500,
                                       max_wavelength=8000,
                                       stype='science+standard')
lhs6328_onedspec.set_fit_constraints(stype='science+standard')
lhs6328_onedspec.add_atlas(elements=['Xe'], stype='science+standard')

# Solve for the pixel-to-wavelength solution
lhs6328_onedspec.fit(stype='science+standard')
lhs6328_onedspec.refine_fit(n_delta=2, display=True, stype='science+standard')
lhs6328_onedspec.refine_fit(display=True, stype='science+standard')

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science+standard')
lhs6328_onedspec.inspect_reduced_spectrum(stype='science+standard')

# Get the standard from the library
lhs6328_onedspec.load_standard(target='hiltner102')
lhs6328_onedspec.inspect_standard()

lhs6328_onedspec.compute_sensitivity(kind='cubic')
lhs6328_onedspec.inspect_sensitivity()

lhs6328_onedspec.apply_flux_calibration(stype='science+standard')

lhs6328_onedspec.inspect_reduced_spectrum(stype='science+standard')

# Save as a FITS file
lhs6328_onedspec.save_fits(
    output='flux+wavecal+fluxraw+adu',
    filename='example_output/example_01_b_full_reduction',
    stype='science+standard',
    overwrite=True)

# save as CSV
lhs6328_onedspec.save_csv(
    output='flux+wavecal+fluxraw+adu',
    filename='example_output/example_01_b_full_reduction',
    stype='science+standard',
    overwrite=True)
