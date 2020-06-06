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
lhs6328_onedspec.extract_arcspec(display=True, stype='science')

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(display=True, stype='science')

lhs6328_onedspec.wavecal_science.save_fits(filename='example_output/example_05_arcspec')

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
lhs6328_onedspec.extract_arcspec(display=True, stype='science')

# Find the peaks of the arc
lhs6328_onedspec.find_arc_lines(display=True, stype='science')

lhs6328_onedspec.wavecal_science.save_fits(filename='example_output/example_05_arcspec_1')

