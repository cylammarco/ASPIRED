# -*- coding: utf-8 -*-
import os

from astropy.io import fits
import numpy as np

from aspired import image_reduction
from aspired import spectral_reduction

HERE = os.path.dirname(os.path.realpath(__file__))

fits_blob = fits.open("test/test_data/v_s_20180810_27_1_0_0.fits.gz")[0]
fits_blob.data = fits_blob.data * (
    65000.0 / np.nanpercentile(fits_blob.data, 99.5)
)
fits_blob.data[fits_blob.data > 65535] = 65535.0
fits_blob.data = fits_blob.data.astype("int")
fits_blob.writeto(
    os.path.join("test", "test_data", "fake_saturated_data.fits"),
    overwrite=True,
)


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


def test_saturated_data():
    # Science frame
    lhs6328_frame = image_reduction.ImageReduction(
        log_level="DEBUG", log_file_folder=os.path.join(HERE, "test_output")
    )
    lhs6328_frame.add_filelist(
        os.path.join(
            HERE, "test_data", "sprat_LHS6328_fake_saturated_data.list"
        )
    )
    lhs6328_frame.load_data()
    lhs6328_frame.reduce()

    lhs6328_frame.save_masks(
        os.path.join(
            HERE, "test_output", "fake_saturated_reduced_image_mask.fits"
        ),
        overwrite=True,
    )

    assert (lhs6328_frame.bad_mask == lhs6328_frame.saturation_mask).all()

    lhs6328_twodspec = spectral_reduction.TwoDSpec(
        lhs6328_frame,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        cosmicray=True,
        readnoise=5.7,
        log_level="DEBUG",
        log_file_folder=os.path.join(HERE, "test_output"),
    )

    lhs6328_twodspec.ap_trace(nspec=1)

    lhs6328_twodspec.ap_extract(model="lowess", lowess_frac=0.8)
