import numpy as np

from astropy.io import fits
from aspired import spectral_reduction

# Load the image
lhs6328_fits = fits.open(
    'sprat_LHS6328_Hiltner102_raw/v_e_20180810_12_1_0_0.fits.gz')[0]

# Case 1
#
# Loading two pre-saved spectral traces from a single FITS file.
#
lhs6328_two_traces_extracted = fits.open('example_output/lhs6328_adu_traces.fits')
lhs6328_two_traces_trace = lhs6328_two_traces_extracted[3].data
lhs6328_two_traces_trace_sigma = lhs6328_two_traces_extracted[4].data

lhs6328_two_traces = spectral_reduction.TwoDSpec(lhs6328_fits, readnoise=2.34)

# Adding the trace and the line spread function (sigma) to the TwoDSpec object
lhs6328_two_traces.add_trace(lhs6328_two_traces_trace,
                             lhs6328_two_traces_trace_sigma)

lhs6328_two_traces.ap_extract(apwidth=15,
                              optimal=True,
                              skywidth=10,
                              skydeg=1,
                              display=True,
                              jsonstring=False)

lhs6328_two_traces.save_fits(
    output='wavecal+adu',
    filename=
    'example_output/example_04_user_supplied_trace_for_spectral_extraction',
    overwrite=True)

# Case 2
#
# Loading a single pre-saved spectral trace.
#
lhs6328_extracted = fits.open('example_output/lhs6328_adu_traces_1.fits')
lhs6328_trace, lhs6328_trace_sigma = lhs6328_extracted[
    3].data, lhs6328_extracted[4].data

lhs6328 = spectral_reduction.TwoDSpec(lhs6328_fits, readnoise=2.34)

# Adding the trace and the line spread function (sigma) to the TwoDSpec object
lhs6328.add_trace(lhs6328_trace, lhs6328_trace_sigma)

lhs6328.ap_extract(apwidth=15,
                   optimal=True,
                   skywidth=10,
                   skydeg=1,
                   display=True,
                   jsonstring=False)

lhs6328.save_fits(
    output='wavecal+adu',
    filename=
    'example_output/example_04_user_supplied_trace_for_spectral_extraction_1',
    overwrite=True)
