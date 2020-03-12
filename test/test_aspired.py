import os
import sys
import numpy as np
from astropy.io import fits
import plotly.io as pio
pio.renderers.default = 'notebook+jpg'

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aspired import aspired


def test_imagereduction():
    science_frame = aspired.ImageReduction('examples/sprat_LHS6328.list',
                                           silence=True)
    science_frame.reduce()

    properties_len = len(science_frame.__dict__)
    assert properties_len == 53, 'There should be 53 properties. You have ' + str(
        properties_len) + '.'

    # Check if files are loaded to the right place
    filelist_len = len(science_frame.filelist)
    assert filelist_len == 7, 'There should be 7 files. You have ' + str(
        filelist_len) + '.'

    bias_len = len(science_frame.bias_filename)
    assert bias_len == 0, 'There should be 0 files. You have ' + str(
        bias_len) + '.'

    dark_len = len(science_frame.dark_filename)
    assert dark_len == 1, 'There should be 1 files. You have ' + str(
        dark_len) + '.'

    light_len = len(science_frame.light_filename)
    assert light_len == 5, 'There should be 5 files. You have ' + str(
        light_len) + '.'

    flat_len = len(science_frame.flat_filename)
    assert flat_len == 0, 'There should be 0 files. You have ' + str(
        flat_len) + '.'

    arc_len = len(science_frame.arc_filename)
    assert arc_len == 1, 'There should be 1 files. You have ' + str(
        arc_len) + '.'

    # Check if there is any numeric values in the final image array
    assert np.isfinite(science_frame.light_master).any(
    ), 'The reduced image contains no usable information.'

    # Free memory
    del science_frame
    science_frame = None


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

    # initialise the two aspired.TwoDSpec()
    dummy_twodspec = aspired.TwoDSpec(dummy_data,
                                      spatial_mask=spatial_mask,
                                      spec_mask=spec_mask)

    # Trace the spectrum, note that the first 15 rows were trimmed from the spatial_mask
    dummy_twodspec.ap_trace(ap_faint=0)
    trace = np.round(np.mean(dummy_twodspec.trace))
    assert trace == 35, 'Trace is at row ' + str(
        trace) + ', but it is expected to be at row 35.'

    # Optimal extracting spectrum by summing over the aperture along the trace
    dummy_twodspec.ap_extract(apwidth=5, optimal=False)
    adu = np.mean(dummy_twodspec.adu)
    assert np.round(adu).astype('int') == 47, 'Extracted ADU is ' + str(adu) + ' but it should be 19.'
