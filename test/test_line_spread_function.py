# -*- coding: utf-8 -*-
import numpy as np
import pytest
from aspired.line_spread_function import (
    build_line_spread_profile,
    get_line_spread_function,
)


def _image(n_disp=50, n_spat=60, centre=30.0, sigma=2.0):
    y = np.arange(n_spat)
    row = 100.0 * np.exp(-0.5 * ((y - centre) / sigma) ** 2) + 5.0
    return np.tile(row, (n_disp, 1))


def test_profile_dispersion_first():
    img = _image()
    prof = build_line_spread_profile(img, np.full(50, 30.0), trace_width=10)
    assert len(prof) == 21
    assert np.argmax(prof) == 10
    assert np.nanmin(prof) == 0


def test_profile_rotated_image():
    img = _image().T
    prof = build_line_spread_profile(img, np.full(50, 30.0), trace_width=10)
    assert len(prof) == 21


def test_profile_sparse_trace_dispersion_axis_b():
    img = _image(n_disp=50, n_spat=60)
    trace = np.column_stack(([0, 49], [30.0, 30.0]))
    # xs max <= _b - 1 -> interpolated to length _b (60) which matches rotated
    prof = build_line_spread_profile(img, trace, trace_width=5)
    assert len(prof) == 11


def test_profile_sparse_trace_axis_a():
    img = _image(n_disp=60, n_spat=40, centre=20.0)
    trace = np.column_stack(([0, 59], [20.0, 20.0]))
    prof = build_line_spread_profile(img, trace, trace_width=5)
    assert len(prof) == 11


def test_profile_bad_trace_length():
    with pytest.raises(ValueError):
        build_line_spread_profile(_image(), np.zeros(7))


def test_profile_edge_padding_and_nan():
    img = _image(centre=2.0)
    img[0, :] = np.nan
    prof = build_line_spread_profile(img, np.full(50, 2.0), trace_width=10)
    assert len(prof) == 21
    assert np.all(np.isfinite(prof))
    # trace beyond the upper edge
    prof = build_line_spread_profile(
        _image(centre=58.0), np.full(50, 58.0), trace_width=10
    )
    assert len(prof) == 21


def test_profile_all_nan():
    img = np.full((50, 60), np.nan)
    with pytest.warns(RuntimeWarning):
        prof = build_line_spread_profile(img, np.full(50, 30.0), 5)
    assert np.all(prof == 0.0)


def test_lsf_fit_default_bounds():
    prof = build_line_spread_profile(_image(), np.full(50, 30.0), 10)
    model = get_line_spread_function(np.full(50, 30.0), prof)
    assert abs(model.mean_0.value - 30.0) < 1.0


@pytest.mark.parametrize("value", [None, [0.0, 200.0]])
def test_lsf_fit_bounds_variants(value):
    prof = build_line_spread_profile(_image(), np.full(50, 30.0), 10)
    bounds = {"amplitude": value, "mean": value, "stddev": value}
    if value is not None:
        bounds["stddev"] = [0.5, 5.0]
        bounds["mean"] = [0.0, 100.0]
    model = get_line_spread_function(np.full(50, 30.0), prof, bounds=bounds)
    assert model.stddev_0.value > 0


def test_lsf_all_nan_returns_initial_model():
    prof = np.full(21, np.nan)
    with pytest.warns(RuntimeWarning):
        model = get_line_spread_function(np.full(5, 30.0), prof)
    assert model is not None
