import os
import sys
import numpy as np
from astropy.io import fits
import plotly.io as pio
pio.renderers.default = 'notebook+jpg'

base_dir = os.path.dirname(__file__)
abs_dir = os.path.abspath(os.path.join(base_dir, '..'))

sys.path.insert(0, abs_dir)

from aspired import image_reduction
from aspired import spectral_reduction


def test_spectral_extraction():
    # Prepare dummy data
    # total signal at a given spectral position = 2 + 5 + 10 + 5 + 2 - 1*5 = 19
    dummy_data = np.ones((100, 100))
    dummy_noise = np.random.random((100, 100))

    dummy_data[47] *= 2.
    dummy_data[48] *= 5.
    dummy_data[49] *= 10.
    dummy_data[50] *= 20.
    dummy_data[51] *= 10.
    dummy_data[52] *= 5.
    dummy_data[53] *= 2.
    dummy_data += dummy_noise

    dummy_arc = np.arange(30, 80, 10)

    # Spectral direction
    Saxis = 1

    # masking
    spec_mask = np.arange(10, 90)
    spatial_mask = np.arange(15, 85)

    # initialise the two spectral_reduction.TwoDSpec()
    dummy_twodspec = spectral_reduction.TwoDSpec(dummy_data,
                                                 spatial_mask=spatial_mask,
                                                 spec_mask=spec_mask)

    # Trace the spectrum, note that the first 15 rows were trimmed from the spatial_mask
    dummy_twodspec.ap_trace(ap_faint=0)
    trace = np.round(np.mean(dummy_twodspec.spectrum_list[0].trace))
    assert trace == 35, 'Trace is at row ' + str(
        trace) + ', but it is expected to be at row 35.'

    # Optimal extracting spectrum by summing over the aperture along the trace
    dummy_twodspec.ap_extract(apwidth=5, optimal=False)
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    assert np.round(count).astype(
        'int') == 47, 'Extracted count is ' + str(count) + ' but it should be 19.'


def test_full_run():
    # Extract two spectra

    # Line list
    atlas = [
        4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
        4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
        6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32, 6976.18,
        7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65, 7887.40,
        7967.34, 8057.258
    ]
    element = ['Xe'] * len(atlas)

    # Science frame
    lhs6328_frame = image_reduction.ImageReduction('test/test_data/sprat_LHS6328.list')
    lhs6328_frame.reduce()

    lhs6328_twodspec = spectral_reduction.TwoDSpec(lhs6328_frame,
                                                   cosmicray=True,
                                                   readnoise=5.7)

    lhs6328_twodspec.ap_trace(nspec=2, display=False)

    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        display=False,
        save_iframe=False)

    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        forced=True,
        variances=lhs6328_twodspec.spectrum_list[1].var,
        display=False,
        save_iframe=False)

    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=False,
        display=False,
        save_iframe=False)

    lhs6328_twodspec.ap_extract(
        apwidth=15,
        skywidth=10,
        skydeg=1,
        optimal=True,
        forced=True,
        variances=1000000.,
        display=False,
        save_iframe=False)
    #lhs6328_twodspec.save_fits(
    #    filename='example_output/example_01_a_science_traces', overwrite=True)

    # Standard frame
    standard_frame = image_reduction.ImageReduction('test/test_data/sprat_Hiltner102.list')
    standard_frame.reduce()

    hilt102_twodspec = spectral_reduction.TwoDSpec(standard_frame,
                                                   cosmicray=True,
                                                   readnoise=5.7)

    hilt102_twodspec.ap_trace(nspec=1, resample_factor=10, display=False)

    hilt102_twodspec.ap_extract(
        apwidth=15,
        skysep=3,
        skywidth=5,
        skydeg=1,
        optimal=True,
        display=False,
        save_iframe=False)

    # Handle 1D Science spectrum
    lhs6328_onedspec = spectral_reduction.OneDSpec()
    lhs6328_onedspec.from_twodspec(lhs6328_twodspec, stype='science')
    lhs6328_onedspec.from_twodspec(hilt102_twodspec, stype='standard')

    # Add a 2D arc image
    lhs6328_onedspec.add_arc(lhs6328_frame, stype='science')
    lhs6328_onedspec.add_arc(standard_frame, stype='standard')

    # Extract the 1D arc by aperture sum of the traces provided
    lhs6328_onedspec.extract_arc_spec(display=False, stype='science+standard')

    # Find the peaks of the arc
    lhs6328_onedspec.find_arc_lines(display=False, stype='science+standard')

    # Configure the wavelength calibrator
    lhs6328_onedspec.initialise_calibrator(stype='science+standard')
    lhs6328_onedspec.set_hough_properties(num_slopes=500,
                                          xbins=100,
                                          ybins=100,
                                          min_wavelength=3500,
                                          max_wavelength=8000,
                                          stype='science+standard')
    lhs6328_onedspec.set_ransac_properties(filter_close=True,
                                           stype='science+standard')

    lhs6328_onedspec.load_user_atlas(elements=element,
                                     wavelengths=atlas,
                                     stype='science+standard')
    lhs6328_onedspec.do_hough_transform()

    # Solve for the pixel-to-wavelength solution
    lhs6328_onedspec.fit(max_tries=50, stype='science+standard', display=False)

    # Apply the wavelength calibration and display it
    lhs6328_onedspec.apply_wavelength_calibration(stype='science+standard')

    lhs6328_onedspec.wavecal_science.spectrum_list[0].create_wavelength_fits()

    # Get the standard from the library
    lhs6328_onedspec.load_standard(target='hiltner102')

    lhs6328_onedspec.compute_sensitivity(kind='cubic', mask_fit_size=1)

    lhs6328_onedspec.apply_flux_calibration(stype='science+standard')

    # Save as FITS
    lhs6328_onedspec.save_fits(
        output='flux_resampled+wavecal+flux+count',
        filename='test/test_output/test_full_run',
        stype='science+standard',
        overwrite=True)

    # save as CSV
    lhs6328_onedspec.save_csv(
        output='flux_resampled+wavecal+flux+count',
        filename='test/test_output/test_full_run',
        stype='science+standard',
        overwrite=True)
