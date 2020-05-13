Aperture Extraction
===================

A spectrum can be extracted from each trace found. It supports tophat and optimal extraction. In both cases, the sky background is fitted in one dimention only. The uncertainty at each pixel is also computed, but the values are only meaningful if correct gain and read noise are provided.

Tophat
------
The extraction loops over each pixel (x) position along the spectral direction. At pixel x, a slice with ``apwidth`` on either side of the ``trace`` (y), rounded to the nearest pixel, is summed to get the combined signal, sky and noise ADU. The slices of length ``skyidth`` with a spearation of ``skysep`` from the spectral slice are used to model the sky background fitted with a polynomial with an order of ``skydeg``.

Optimal
-------
A Gaussian profile is used for optimal extraction. Then at each step of the iteration, the new weighted model and data are compared until the difference is less than 0.01%. Detailed descriptions can be found in `Horne 1986 <https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract>`_.

Flat-Relative Optimal
---------------------
Currently under investigation. Details can be referred to `Zeichmeister et al 2014 <https://ui.adsabs.harvard.edu/abs/2014A%26A...561A..59Z/abstract>`_.
