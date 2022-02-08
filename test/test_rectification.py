# -*- coding: utf-8 -*-
import copy
import os
from unittest.mock import patch

import numpy as np

from aspired import image_reduction
from aspired import spectral_reduction

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
