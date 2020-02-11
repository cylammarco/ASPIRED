Aperture Extraction
===================

A spectrum can be extracted from each trace found. It supports tophat and optimal extraction. In both cases, the sky background is fitted in one dimention only. The uncertainty at each pixel is also computed, but the values are only meaningful if correct gain and read noise are provided.

Tophat
------
The extraction loops over each pixel position along the spectral direction.

Optimal
-------

