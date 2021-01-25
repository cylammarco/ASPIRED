import numpy as np
from astropy.io import fits
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

    # masking
    spec_mask = np.arange(10, 90)
    spatial_mask = np.arange(15, 85)

    # initialise the two spectral_reduction.TwoDSpec()
    dummy_twodspec = spectral_reduction.TwoDSpec(
        dummy_data,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        log_file_name=None)

    # Trace the spectrum, note that the first 15 rows were trimmed from the
    # spatial_mask
    dummy_twodspec.ap_trace(ap_faint=0)
    trace = np.round(np.mean(dummy_twodspec.spectrum_list[0].trace))
    assert trace == 35, 'Trace is at row ' + str(
        trace) + ', but it is expected to be at row 35.'

    # Optimal extracting spectrum by summing over the aperture along the
    # trace
    dummy_twodspec.ap_extract(apwidth=5, optimal=False)
    count = np.mean(dummy_twodspec.spectrum_list[0].count)
    assert np.round(count).astype('int') == 47, 'Extracted count is ' + str(
        count) + ' but it should be 19.'


def test_user_supplied_trace():

    # Load the image
    lhs6328_fits = fits.open('test/test_data/v_e_20180810_12_1_0_0.fits.gz')[0]

    spatial_mask = np.arange(50, 200)
    spec_mask = np.arange(50, 1024)

    # Loading a single pre-saved spectral trace.
    lhs6328_extracted = fits.open(
        'test/test_data/test_full_run_science_0.fits')
    lhs6328_trace = lhs6328_extracted[1].data
    lhs6328_trace_sigma = lhs6328_extracted[2].data

    lhs6328_twodspec = spectral_reduction.TwoDSpec(lhs6328_fits,
                                                   spatial_mask=spatial_mask,
                                                   spec_mask=spec_mask,
                                                   readnoise=2.34,
                                                   log_file_name=None)

    lhs6328_twodspec.add_trace(trace=lhs6328_trace,
                               trace_sigma=lhs6328_trace_sigma)

    lhs6328_twodspec.ap_extract(apwidth=15,
                                optimal=True,
                                skywidth=10,
                                skydeg=1,
                                display=False)

    lhs6328_twodspec.save_fits(
        output='count',
        filename='test/test_output/user_supplied_trace_for_extraction',
        overwrite=True)
