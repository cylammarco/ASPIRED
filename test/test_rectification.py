import copy
import numpy as np
from aspired import image_reduction
from aspired import spectral_reduction

# Line list
atlas = [
    4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
    4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
    6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32, 6976.18,
    7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65, 7887.40,
    7967.34, 8057.258
]
element = ['Xe'] * len(atlas)

spatial_mask = np.arange(35, 200)
spec_mask = np.arange(50, 1024)

# Science frame
lhs6328_frame = image_reduction.ImageReduction(
    'test/test_data/sprat_LHS6328.list',
    log_level='DEBUG',
    log_file_folder='test/test_output/')
lhs6328_frame.reduce()

lhs6328_twodspec = spectral_reduction.TwoDSpec(
    lhs6328_frame,
    spatial_mask=spatial_mask,
    spec_mask=spec_mask,
    cosmicray=True,
    sigclip=1.0,
    readnoise=5.7,
    log_level='DEBUG',
    log_file_folder='test/test_output/')

lhs6328_twodspec.ap_trace(nspec=2, display=False)


def test_rectify():
    twodspec = copy.copy(lhs6328_twodspec)
    twodspec.rectify_image(bin_size=6,
                           n_bin=[2, 4],
                           display=False,
                           save_iframe=True)


def test_rectify_2():
    twodspec = copy.copy(lhs6328_twodspec)
    twodspec.rectify_image(
        bin_size=6,
        n_bin='lala',
        display=False,
        save_iframe=True,
        filename='test/test_output/test_rectifying_image_manual_filename')


def test_rectify_3():
    twodspec = copy.copy(lhs6328_twodspec)
    twodspec.rectify_image(bin_size=6,
                           n_bin=7,
                           display=False,
                           return_jsonstring=True)
