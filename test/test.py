import sys
import numpy as np
from astropy.io import fits
from ASPIRED.aspired import aspired
import plotly.io as pio
pio.renderers.default = 'notebook+jpg'

science_frame = aspired.ImageReduction('examples/sprat_LHS6328.list')
science_frame.reduce()

standard_frame = aspired.ImageReduction('examples/sprat_Hiltner102.list')
standard_frame.reduce()

# Example data from SPRAT
science_data = science_frame.light_master
science_arc = science_frame.arc_master

# Example data from SPRAT
# Hiltner102
standard_data = standard_frame.light_master
standard_arc = standard_frame.arc_master

# Set the spectral and spatial direction
Saxis = 1
Waxis = 0

# spec mask
spec_mask = np.arange(science_frame.fits_data.header['NAXIS1'])
spatial_mask = np.arange(science_frame.fits_data.header['NAXIS2'])

# initialise the two aspired.TwoDSpec()
lhs6328 = aspired.TwoDSpec(
    science_data,
    spatial_mask=spatial_mask,
    spec_mask=spec_mask,
    rn=2.34,
    cr=False,
    gain=2.45,
    seeing=1.2
)

hilt102 = aspired.TwoDSpec(
    standard_data,
    spatial_mask=spatial_mask,
    spec_mask=spec_mask,
    rn=2.34,
    cr=False,
    gain=2.45,
    seeing=1.2
)

# automatically trace the spectrum
lhs6328.ap_trace(nspec=2, display=True)
hilt102.ap_trace(nspec=1, display=True)

# Optimal extracting spectrum by summing over the aperture along the trace
lhs6328.ap_extract(
    apwidth=15,
    optimal=True,
    display=True,
    jsonstring=False)
hilt102.ap_extract(
    apwidth=20, skysep=3, skywidth=5, skydeg=1,
    optimal=True,
    display=True,
    jsonstring=False)

fluxcal = aspired.StandardFlux(
    target='hiltner102',
    group='irafirs',
    cutoff=0.4,
    ftype='flux'
)
fluxcal.load_standard()
fluxcal.inspect_standard(renderer='default')

# Placeholder of wavelength calibration
wavecal_science = aspired.WavelengthPolyFit(lhs6328, science_arc)
wavecal_standard = aspired.WavelengthPolyFit(hilt102, standard_arc)

wavecal_science.find_arc_lines()
wavecal_standard.find_arc_lines()

pfit = np.array((3.90e-12, -1.10e-08,  1.20e-05, -5.73e-03, 5.68e+00,  3.33e+03))
wavecal_science.calibrate_pfit(elements=["Xe"], pfit=pfit, tolerance=5., display=True)
wavecal_science.calibrate_pfit(elements=["Xe"], pfit=pfit, tolerance=5., display=True)
wavecal_standard.calibrate_pfit(elements=["Xe"], pfit=pfit, tolerance=5., display=True)
wavecal_standard.calibrate_pfit(elements=["Xe"], pfit=pfit, tolerance=5., display=True)

# Get the sensitivity curves
lhs6328_reduced = aspired.OneDSpec(
    lhs6328,
    wavecal_science,
    standard=hilt102,
    wave_cal_std=wavecal_standard,
    flux_cal=fluxcal
)
lhs6328_reduced.apply_wavelength_calibration('all')
lhs6328_reduced.compute_sencurve(kind='cubic')
lhs6328_reduced.inspect_sencurve(renderer='default')

lhs6328_reduced.apply_flux_calibration('all')
lhs6328_reduced.inspect_reduced_spectrum('all', renderer='default')

