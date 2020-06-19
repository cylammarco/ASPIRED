from astropy.io import fits
from aspired import image_reduction
from aspired import spectral_reduction

# Case 1
#
# Extract two spectra
#
lhs6328_frame = image_reduction.ImageReduction('sprat_LHS6328.list')
lhs6328_frame.reduce()

lhs6328_twodspec = spectral_reduction.TwoDSpec(lhs6328_frame, cosmicray=False)
lhs6328_twodspec.ap_trace(nspec=2, display=False)

lhs6328_twodspec.ap_extract(apwidth=15,
                            optimal=True,
                            skywidth=10,
                            skydeg=1,
                            display=False,
                            jsonstring=False)

lhs6328_twodspec.save_fits(filename='example_output/lhs6328_adu_traces',
                           overwrite=True)

lhs6328_onedspec = spectral_reduction.OneDSpec()
lhs6328_onedspec.add_twodspec(lhs6328_twodspec, stype='science')

# Add a 2D arc image
lhs6328_onedspec.add_arc(lhs6328_frame, stype='science')

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
lhs6328_onedspec.refine_fit(
    n_delta=2,
    display=True,
    stype='science')
lhs6328_onedspec.refine_fit(
    display=True,
    stype='science')

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science')
lhs6328_onedspec.inspect_reduced_spectrum(stype='science')

# Save as a FITS file
lhs6328_onedspec.save_fits(output='flux+wavecal+adu',
                           filename='example_output/example_01_full_reduction',
                           stype='science',
                           overwrite=True)

# Case 2
#
# Extract one spectrum
#
lhs6328_frame = image_reduction.ImageReduction('sprat_LHS6328.list')
lhs6328_frame.reduce()

lhs6328_twodspec = spectral_reduction.TwoDSpec(lhs6328_frame, cosmicray=False)
lhs6328_twodspec.ap_trace(nspec=1, display=False)

lhs6328_twodspec.ap_extract(apwidth=15,
                            optimal=True,
                            skywidth=10,
                            skydeg=1,
                            display=False,
                            jsonstring=False)

lhs6328_twodspec.save_fits(filename='example_output/lhs6328_adu_traces',
                           overwrite=True)

lhs6328_onedspec = spectral_reduction.OneDSpec()
lhs6328_onedspec.add_twodspec(lhs6328_twodspec, stype='science')

# Add a 2D arc image
lhs6328_onedspec.add_arc(lhs6328_frame, stype='science')

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
lhs6328_onedspec.refine_fit(
    n_delta=2,
    display=True,
    stype='science')

# Apply the wavelength calibration and display it
lhs6328_onedspec.apply_wavelength_calibration(stype='science')
lhs6328_onedspec.inspect_reduced_spectrum(stype='science')

# Save as a FITS file
lhs6328_onedspec.save_fits(
    output='flux+wavecal+adu',
    filename='example_output/example_01_full_reduction_1',
    stype='science',
    overwrite=True)
