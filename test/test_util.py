# -*- coding: utf-8 -*-
import numpy as np
import pytest
from aspired import util

dummy_image = np.ones((1000, 1000))
dummy_image_nan = np.ones((1000, 1000))
dummy_image_none = np.ones((1000, 1000))
for i in range(100):
    dummy_image[i, i] = 65000
    dummy_image_nan[i, i] = np.nan
    dummy_image_none[i, i] = None

dummy_sum = np.sum(dummy_image)


# Healing the diagonal of pixels with 65000 count
def test_cutoff_mask():
    cutoff_mask, _ = util.create_cutoff_mask(dummy_image, cutoff=60000)
    healed_image = util.bfixpix(dummy_image, cutoff_mask, retdat=True)
    assert np.sum(healed_image) == np.size(dummy_image)


# Healing the diagonal of pixels with 65000 count
def test_cutoff_mask_list():
    cutoff_mask, _ = util.create_cutoff_mask(dummy_image, cutoff=[0, 1000])
    healed_image = util.bfixpix(dummy_image, cutoff_mask, retdat=True)
    assert np.sum(healed_image) == np.size(dummy_image)


# Should do nothing here
def test_cutoff_mask_does_nothing():
    cutoff_mask, _ = util.create_cutoff_mask(dummy_image, cutoff=65500)
    healed_image = util.bfixpix(dummy_image, cutoff_mask, retdat=True)
    assert np.sum(healed_image) == dummy_sum


def test_cutoff_mask_1D_array():
    util.create_cutoff_mask(np.ones(100))


@pytest.mark.xfail()
def test_cutoff_mask_expect_fail_list():
    util.create_cutoff_mask(dummy_image, cutoff=[0, 1000, 60000])


@pytest.mark.xfail()
def test_cutoff_mask_expect_fail_str():
    util.create_cutoff_mask(dummy_image, cutoff="10")


# Repeat with grow set to True
# Healing the diagonal of pixels with 65000 count
def test_cutoff_mask_grow():
    cutoff_mask, _ = util.create_cutoff_mask(
        dummy_image, cutoff=60000, grow=True
    )
    healed_image = util.bfixpix(dummy_image, cutoff_mask, retdat=True)
    assert np.sum(healed_image) == np.size(dummy_image)


# Healing the diagonal of pixels with 65000 count
def test_cutoff_mask_list_grow():
    cutoff_mask, _ = util.create_cutoff_mask(
        dummy_image, cutoff=[0, 1000], grow=True
    )
    healed_image = util.bfixpix(dummy_image, cutoff_mask, retdat=True)
    assert np.sum(healed_image) == np.size(dummy_image)


# Should do nothing here
def test_cutoff_mask_does_nothing_grow():
    cutoff_mask, _ = util.create_cutoff_mask(
        dummy_image, cutoff=65500, grow=True
    )
    healed_image = util.bfixpix(dummy_image, cutoff_mask, retdat=True)
    assert np.sum(healed_image) == dummy_sum


# Test bad mask
# Healing the diagonal of pixels with NAN
def test_bad_mask():
    bad_mask, _ = util.create_bad_pixel_mask(dummy_image_nan)
    healed_image = util.bfixpix(dummy_image, bad_mask, retdat=True)
    assert np.sum(healed_image) == np.size(dummy_image_nan)


# Healing the diagonal of pixels with None
def test_bad_mask_list():
    bad_mask, _ = util.create_bad_pixel_mask(dummy_image_none)
    healed_image = util.bfixpix(dummy_image, bad_mask, retdat=True)
    assert np.sum(healed_image) == np.size(dummy_image_none)


# Should do nothing here
def test_bad_mask_does_nothing():
    bad_mask, _ = util.create_bad_pixel_mask(dummy_image)
    healed_image = util.bfixpix(dummy_image, bad_mask, retdat=True)
    assert np.sum(healed_image) == dummy_sum


def test_bad_mask_1D_array():
    util.create_bad_pixel_mask(np.ones(100))


# Repeat with grow set to True
# Healing the diagonal of pixels with NAN
def test_bad_mask_grow():
    bad_mask, _ = util.create_bad_pixel_mask(
        dummy_image_nan, grow=True, diagonal=True
    )
    healed_image = util.bfixpix(dummy_image, bad_mask, retdat=True)
    assert np.sum(healed_image) == np.size(dummy_image)


# Healing the diagonal of pixels with None
def test_bad_mask_list_grow():
    bad_mask, _ = util.create_bad_pixel_mask(
        dummy_image_none, grow=True, diagonal=True
    )
    healed_image = util.bfixpix(dummy_image, bad_mask, retdat=True)
    assert np.sum(healed_image) == np.size(dummy_image)


# Should do nothing here
def test_bad_mask_does_nothing_grow():
    bad_mask, _ = util.create_bad_pixel_mask(
        dummy_image, grow=True, diagonal=True
    )
    healed_image = util.bfixpix(dummy_image, bad_mask, retdat=True)
    assert np.sum(healed_image) == dummy_sum


# Should return nothing here
def test_bad_mask_does_nothing_grow_none():
    bad_mask, _ = util.create_bad_pixel_mask(
        dummy_image, grow=True, diagonal=True
    )
    healed_image = util.bfixpix(dummy_image, bad_mask, retdat=False)
    assert np.sum(healed_image) is None
