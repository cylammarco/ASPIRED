# -*- coding: utf-8 -*-
import copy
import os
from unittest.mock import patch

import numpy as np
from aspired import image_reduction, spectral_reduction

HERE = os.path.dirname(os.path.realpath(__file__))


# Line list
atlas = [
    4193.5,
    4385.77,
    4500.98,
    4524.68,
    4582.75,
    4624.28,
    4671.23,
    4697.02,
    4734.15,
    4807.02,
    4921.48,
    5028.28,
    5618.88,
    5823.89,
    5893.29,
    5934.17,
    6182.42,
    6318.06,
    6472.841,
    6595.56,
    6668.92,
    6728.01,
    6827.32,
    6976.18,
    7119.60,
    7257.9,
    7393.8,
    7584.68,
    7642.02,
    7740.31,
    7802.65,
    7887.40,
    7967.34,
    8057.258,
]
element = ["Xe"] * len(atlas)

spatial_mask = np.arange(35, 200)
spec_mask = np.arange(50, 1024)

# Science frame
lhs6328_frame = image_reduction.ImageReduction(
    log_level="DEBUG", log_file_folder=os.path.join(HERE, "test_output")
)
lhs6328_frame.add_filelist(
    os.path.join(HERE, "test_data", "sprat_LHS6328.list")
)
lhs6328_frame.load_data()
lhs6328_frame.reduce()

lhs6328_twodspec = spectral_reduction.TwoDSpec(
    lhs6328_frame,
    spatial_mask=spatial_mask,
    spec_mask=spec_mask,
    cosmicray=True,
    sigclip=1.0,
    readnoise=5.7,
    log_level="DEBUG",
    log_file_folder=os.path.join(HERE, "test_output"),
)

lhs6328_twodspec.ap_trace(nspec=2, display=False)


# assert the resampled image has the total photon count within 0.1% of the
# input
def test_rectify():
    twodspec = copy.copy(lhs6328_twodspec)
    twodspec.get_rectification(
        bin_size=6,
        n_bin=[2, 4],
        display=False,
        save_fig=True,
        fig_type="iframe+png",
        filename=os.path.join(HERE, "test_output", "test_rectifying_image"),
    )
    assert (
        abs(np.sum(twodspec.img) / np.sum(twodspec.img_rectified) - 1.0) < 0.01
    )
    twodspec.apply_rectification()


# assert the resampled image has the total photon count within 0.1% of the
# input
@patch("plotly.graph_objects.Figure.show")
def test_rectify_2(mock_show):
    twodspec = copy.copy(lhs6328_twodspec)
    twodspec.get_rectification(
        bin_size=6,
        n_bin="lala",
        display=True,
        save_fig=True,
        fig_type="iframe+png",
        filename=os.path.join(
            HERE, "test_output", "test_rectifying_image_manual_filename"
        ),
    )
    assert (
        abs(np.sum(twodspec.img) / np.sum(twodspec.img_rectified) - 1.0) < 0.01
    )
    twodspec.apply_rectification()


# assert the resampled image has replaced the input image
def test_rectify_3():
    twodspec = copy.copy(lhs6328_twodspec)
    twodspec.get_rectification(
        bin_size=6, n_bin=7, apply=True, display=False, return_jsonstring=True
    )
    assert np.sum(twodspec.img) == np.sum(twodspec.img_rectified)


simulated_image = np.ones((100, 1000))

simulated_image[46] = 2.0
simulated_image[47] = 5.0
simulated_image[48] = 10.0
simulated_image[49] = 30.0
simulated_image[50] = 50.0
simulated_image[51] = 30.0
simulated_image[52] = 10.0
simulated_image[53] = 5.0
simulated_image[54] = 2.0

for i in range(100):
    simulated_image[i, 223 + i] += 100.0
    simulated_image[i, 671 + i] += 150.0


# test simulated 2D spectrum
def test_rectify_simulated_spectrum():
    # add the frame
    simulated_twodspec = spectral_reduction.TwoDSpec(
        simulated_image,
        log_level="DEBUG",
        log_file_folder=os.path.join(HERE, "test_output"),
    )
    simulated_twodspec.ap_trace(nspec=1, display=False)

    simulated_twodspec.get_rectification(
        order=1, upsample_factor=1, use_arc=False
    )
    reconstructed_x1_coeff = simulated_twodspec.rec_coeff.copy()
    simulated_twodspec.get_rectification(
        order=1, upsample_factor=2, use_arc=False
    )
    reconstructed_x2_coeff = simulated_twodspec.rec_coeff.copy()
    simulated_twodspec.get_rectification(
        order=1, upsample_factor=3, use_arc=False
    )
    reconstructed_x3_coeff = simulated_twodspec.rec_coeff.copy()
    simulated_twodspec.get_rectification(
        order=1, upsample_factor=4, use_arc=False
    )
    reconstructed_x4_coeff = simulated_twodspec.rec_coeff.copy()
    simulated_twodspec.get_rectification(
        order=1, upsample_factor=5, use_arc=False
    )
    reconstructed_x5_coeff = simulated_twodspec.rec_coeff.copy()
    simulated_twodspec.get_rectification(
        order=1, upsample_factor=10, use_arc=False
    )
    reconstructed_x10_coeff = simulated_twodspec.rec_coeff.copy()

    assert np.isclose(
        reconstructed_x1_coeff, reconstructed_x2_coeff, atol=0.1, rtol=0.01
    ).all()
    assert np.isclose(
        reconstructed_x1_coeff, reconstructed_x3_coeff, atol=0.1, rtol=0.01
    ).all()
    assert np.isclose(
        reconstructed_x1_coeff, reconstructed_x4_coeff, atol=0.1, rtol=0.01
    ).all()
    assert np.isclose(
        reconstructed_x1_coeff, reconstructed_x5_coeff, atol=0.1, rtol=0.01
    ).all()
    assert np.isclose(
        reconstructed_x1_coeff, reconstructed_x10_coeff, atol=0.1, rtol=0.01
    ).all()
