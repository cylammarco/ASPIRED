Wavelength Calibration
======================

ASPIRED implements wavelength calibration using `RASCAL <https://rascal.readthedocs.io/>`_. The simple setup should be sufficient for most low resolution spectrographs without crrsion unit in the instrument. For advanced usage of RASCAL or other wavelength calibration methods, it is possible to pass in polynomial coefficients. Currently, it supports ``numpy.polynomial.polynomial.polyval``, ``np.polynomial.legendre.legval`` and ``np.polynomial.chebyshev.chebval``.

Masking
-------
Wavelength ranges can be masked when computing the sensitivity curve, for example, over the range of Telluric absorption lines. The deafult masking ranges are 6850-6960, 7150-7400 and 7575-7700 A.
