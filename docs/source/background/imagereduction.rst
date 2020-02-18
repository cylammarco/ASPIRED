Image Reduction
===============

This class performs basic image reduction routines that accepts light, dark, bias and flat grams to perform fieldflattening. It also handles the headers acutomatically and can be used to initialise the ``TwoDSpec``.
The FITS keywords follow the set in the `FITS Stadnard Document <https://fits.gsfc.nasa.gov/fits_standard.html>`_ recommended by the `IAU FITS Working Group <https://fits.gsfc.nasa.gov/iaufwg/iaufwg.html>`_, also included are a few non-standard keywords that are commonly used among various observatories. The reduced image can be exported as static images, iframes containing an interative `plotly <https://plot.ly/graphing-libraries/>`_ figures, or a JSON string that can be rendered by any variant of plotly in the support programming languages (namely JavaScript, Python and R).

The process of field-flattening is as follows:

1. Supply a filelist containing 2 or 3 columns in CSV format, space gets truncated automatically. The first column is one of ``light``, ``dark``, ``flat``, ``bias`` or ``arc``. The second column is the file location, an absolute is recommended. The last column is optional, which indicates the HDU number of the target FITS image, defaulted to ``0``.

2. First step of reduction is bias subtraction. If multiple bias frames are provided, a mean- or median-combine would be performed (ignoring NAN), with a 5-sigma clipping.

3. Then, it is the dark subtraction. If multiple dark frames are provided, a mean- or median-combine would be performed (ignoring NAN), with a 5-sigma clipping. The exposure time of the dark frame would be computed if it can be found from the header. Otherwise, 1 second will be used. The ADU of the dark frame will be scaled based on the ration of the exposure time of the light and dark frames.

4. To correct for vignetting and other instrumental sensitivity effects, flat frames are used to weigh the pixels. If multiple flat frames are provided, a mean- or median-combine would be performed (ignoring NAN), with a 5-sigma clipping. The master flat is then subtracted by the bias and dark grames, before dividing by the mean value of the frame.

5. After the prcoessing, the data is stored as an ``astropy.io.fits.PrimaryHDU`` object including the header of the first light frame and the additional data reduction information. The reduced data can be exported as a FITS file, and/or be directly used further down the data reduction pipeline without being written to disk.
