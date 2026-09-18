import numpy as np
from aspired import spectral_reduction


def gaussian(pixels, central_pixel):
    """
    Parameters
    ----------
    log_age: array
        the age to return the SFH.
    peak_age: float
        the time of the maximum star formation.
    Returns
    -------
    The relative SFH at the given log_age location.
    """
    stdv = 1.0
    variance = stdv**2.0
    g = (
        np.exp(-((pixels - central_pixel) ** 2.0) / 2 / variance)
        / np.sqrt(2 * np.pi)
        / stdv
    )
    return g


# background noise
bg_level = 5.0
total_flux = 10000.0

# Prepare asymmetric line spread function dummy data
asym_profile = gaussian(np.arange(100), 50)
# make asymmetric by shifting and adding a skewed tail
tail = np.exp(-0.5 * ((np.arange(100) - 52) / 3.0) ** 2)
asym_profile = asym_profile + 0.3 * tail
asym_profile /= np.sum(asym_profile)

dummy_asym_gaussian_data = (
    np.ones((100, 1000)).T * asym_profile
).T * total_flux
dummy_asym_gaussian_data = (
    np.random.normal(dummy_asym_gaussian_data, scale=bg_level) + bg_level
)


def test_asymmetric_lsf_extraction_normalisation():
    spec_mask = np.arange(10, 900)
    spatial_mask = np.arange(15, 85)
    twod = spectral_reduction.TwoDSpec(
        dummy_asym_gaussian_data,
        spatial_mask=spatial_mask,
        spec_mask=spec_mask,
        log_file_name=None,
        log_level="CRITICAL",
        saxis=1,
        flip=False,
        cosmicray_sigma=5.0,
        readnoise=0.1,
        gain=1.0,
        seeing=1.0,
        exptime=1.0,
    )

    twod.ap_trace(rescale=True, fit_deg=0)

    # Empirical LOWESS profile should be correctly normalised
    twod.ap_extract(apwidth=8, optimal=True, model="lowess", lowess_frac=0.05)
    count_lowess = np.mean(twod.spectrum_list[0].count)
    err_lowess = np.mean(twod.spectrum_list[0].count_err)
    assert np.isclose(count_lowess, total_flux, rtol=0.05, atol=err_lowess)

    # Marsh89 optimal extraction should also be correctly normalised
    twod.ap_extract(apwidth=8, optimal=True, algorithm="marsh89")
    count_marsh = np.mean(twod.spectrum_list[0].count)
    err_marsh = np.mean(twod.spectrum_list[0].count_err)
    assert np.isclose(count_marsh, total_flux, rtol=0.05, atol=err_marsh)
