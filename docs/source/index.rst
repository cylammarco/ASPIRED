How to Use The ASPIRED Documentation
====================================

To start, you're probably going to need to follow the :ref:`Installation` guide to
get ASPIRED installed on your computer.
After you finish that, you can probably learn most of what you need from the
tutorials listed below (you might want to start with
:ref:`quickstart` and go from there).
If you need more details about specific functionality, the User Guide below
should have what you need.

We welcome bug reports, patches, feature requests, and other comments via `the GitHub
issue tracker <https://github.com/cylammarco/ASPIRED/issues>`_.


User Guide
==========

.. toctree::
   :maxdepth: 2
   :caption: Installation

   installation/pip

.. toctree::
   :maxdepth: 2
   :caption: Behind the Scene

   background/imagereduction
   background/aptrace
   background/apextract
   background/wavecal
   background/fluxcal

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/quickstart
   tutorials/whtisis

.. toctree::
   :maxdepth: 1
   :caption: Image Reduction API

   modules/imagereductionmodule

.. toctree::
   :maxdepth: 1
   :caption: Spectral Reduction API

   modules/spectralreductionmodule

.. toctree::
   :maxdepth: 1
   :caption: Standard List

   modules/standardlistmodule


Basic Usage
===========

The bare minimum example code to perform a complete spectral data reduction with both wavelength and flux calibrated:

.. code-block:: python

    from astropy.io import fits
    from aspired import spectral_reduction

    # Open the FITS file as a fits.hdu.image.PrimaryHDU
    science = fits.open('/path/to/science_FITS_file')
    science2D = spectral_reduction.TwoDSpec(science)
    science2D.ap_trace()
    science2D.ap_extract()

    standard = fits.open('/path/to/standard_FITS_file')
    standard2D = spectral_reduction.TwoDSpec(standard)
    standard2D.ap_trace()
    standard2D.ap_extract()

    # Load the standard flux
    fluxcal = spectral_reduction.StandardFlux(target='Name1', group='Name2')
    fluxcal.load_standard()

    # Wavelength calibration
    wavecal_science = spectral_reduction.WavelengthPolyFit(science2D, science_arc)
    wavecal_science.find_arc_lines()
    wavecal_science.fit(elements=["Chemical Symbol"])

    wavecal_standard = spectral_reduction.WavelengthPolyFit(standard2D, standard_arc)
    wavecal_standard.find_arc_lines()
    wavecal_standard.fit(elements=["Chemical Symbol"])

    # Applying flux and wavelength calibration
    science_reduced = spectral_reduction.OneDSpec(
        science2D,
        wavecal_science,
        standard2D,
        wavecal_standard,
        fluxcal
    )
    science_reduced.apply_wavelength_calibration()
    science_reduced.compute_sencurve()
    science_reduced.inspect_reduced_spectrum()


Some more complete examples are available in the tutorials.


License & Attribution
=====================

Copyright 2019-2020

If you make use of ASPIRED in your work, please cite our paper
(`arXiv <https://arxiv.org/abs/1912.05885>`_,
`ADS <https://ui.adsabs.harvard.edu/abs/2019arXiv191205885L/abstract>`_,
`BibTeX <https://ui.adsabs.harvard.edu/abs/2019arXiv191205885L/exportcitation>`_).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
