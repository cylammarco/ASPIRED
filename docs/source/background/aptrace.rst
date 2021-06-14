Aperture Tracing
================

Cross-Correlate
---------------

The ``ap_trace`` works by dividing the 2D spectrum into subspectra, and then each part is summed along the spectral direction before cross-correlating with the adjacent subspectra, the shift and scaling of the spectrum/a along the spectral direction can be found simultaneously. The middle of the 2D spectrum is used as the zero point of the procedure. Here is the detailed description of the algorithm.

1. The input 2D spectrum is divided into ``nwindow`` subspectra.

2. Each subspectrum is summed along the spectral direction in order to improve the signal(s) of the spectrum/a along the spatial direction -- we call this a *spatial spectrum*.

3. Each spatial spectrum is upscale by a factor of ``resample_factor`` to allow for sub-pixel correlation. This utilises the ``scipy.signal.resample()`` function to fit the spatial profile with spline. Click `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html>`_ for the API. This factor should be as large as you can trust the centroiding to ``1/resample_factor`` pixel.

4. The i-th spatial spectrum is then cross-correlated i+1-th spatial spectrum. The shift (and scale) at where the maximum occurs within the tolerance limit ``tol`` will be stored.

5. While the spatial spectra are being cross-correlatd. They are aligned and stacked for peak finding and gaussian fitting. Peak finding is performed with ``scipy.signal.find_peaks()`` and returned the list of peaks sorted by their ``prominence``. Only centroiding has to be accurate at this stage, so a gaussian function is sufficient. The standard deviation of the gaussian is only served as a first guess of the profile when performing optimal extraction; it would not be used in the case of top-hat extraction.

.. image: ../_static/fig_01_tracing.jpg

Manual
------

The ``add_trace`` function allows users to supply their own traces. Which would be particularly useful for faint and/or confused sources that are beyond the capability of the automated tracing functions. The ``trace`` indicates the pixel of the spectrum in the spatial direction. It must be supplied with
1. the same size as number of pixels in the spectral direction, and a matching number of ``trace_sigma``; **or**
2. the same shape as the ``x_pix`` provided to indicate the pixel locations of the trace in the spectral direction.
