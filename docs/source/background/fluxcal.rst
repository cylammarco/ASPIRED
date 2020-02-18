Flux Calibration
================

Standard Stars
--------------
The flux and magnitude of the standard stars available in `iraf <https://github.com/iraf-community/iraf>`_ and on `ESO <https://www.eso.org/sci/observing/tools/standards/spectra.html>`_ are all included in this pakcage. We call these values the *template* hereafter.

Sensitivity Curve
-----------------
.. note::

  Sensitivity curves can only be computed if both the standard and observations are wavelength calibrated.

The sensitivity curve is the ratio of the real flux from the standard and the ADU from the observation. The higher resolution among the template and the observation is resampled to match the lower resolution one. Then, the template is divided by the oserved spectrum to generate the sensitivity curve which is interpolated by the ``scipy.interpolate.interp1d()``. A `Satvisky Golay smoothing <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_ can be applied before the interpolation, which is not used by default.



