import numpy as np
from aspired.wavelengthcalibration import WavelengthCalibration
from aspired.spectrum1D import Spectrum1D


def test_wavecal():
    # Line list
    atlas = [
        4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
        4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
        6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32,
        6976.18, 7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65,
        7887.40, 7967.34, 8057.258
    ]
    element = ['Xe'] * len(atlas)

    lhs6328_spectrum1D = Spectrum1D(log_file_folder='test/test_output/')
    wavecal = WavelengthCalibration(log_file_folder='test/test_output/')

    # Science arc_spec
    arc_spec = np.loadtxt(
        'test/test_data/test_full_run_science_0_arc_spec.csv',
        delimiter=',',
        skiprows=1)
    lhs6328_spectrum1D.add_arc_spec(arc_spec)
    wavecal.from_spectrum1D(lhs6328_spectrum1D)

    # Find the peaks of the arc
    wavecal.find_arc_lines()

    # Configure the wavelength calibrator
    wavecal.initialise_calibrator()
    wavecal.set_hough_properties(num_slopes=1000,
                                 xbins=200,
                                 ybins=200,
                                 min_wavelength=3500,
                                 max_wavelength=8500)
    wavecal.set_ransac_properties(filter_close=True)

    wavecal.load_user_atlas(elements=element, wavelengths=atlas)
    wavecal.do_hough_transform()

    # Solve for the pixel-to-wavelength solution
    wavecal.fit(max_tries=500, display=False)

    # Just getting the calibrator
    wavecal.get_calibrator()

    # Save a FITS file
    wavecal.save_fits(output='wavecal',
                      filename='test/test_output/test_wavecal',
                      overwrite=True)

    # Save a CSV file
    wavecal.save_csv(output='wavecal',
                     filename='test/test_output/test_wavecal',
                     overwrite=True)
