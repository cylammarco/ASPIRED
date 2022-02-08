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
   background/rectification
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
   :maxdepth: 2
   :caption: Spectral Reduction API

   modules/spectralreductionmodule

.. toctree::
   :maxdepth: 1
   :caption: Standard List

   modules/standardlistmodule

.. toctree::
   :maxdepth: 1
   :caption: Utility

   modules/utility


Basic Usage
===========

The bare minimum example code to perform a complete spectral data reduction with both wavelength and flux calibrated:

.. code-block:: python

   from astropy.io import fits
   from aspired import spectral_reduction

   # Open the FITS file as a fits.hdu.image.PrimaryHDU
   science = fits.open('/path/to/science_FITS_file')
   standard = fits.open('/path/to/standard_FITS_file')

   # Handle the 2D operations
   science2D = spectral_reduction.TwoDSpec(science)
   science2D.ap_trace()
   science2D.ap_extract()
   science2D.add_arc()
   science2D.extract_arc_spec()

   standard2D = spectral_reduction.TwoDSpec(standard)
   standard2D.ap_trace()
   standard2D.ap_extract()
   standard2D.add_arc()
   standard2D.extract_arc_spec()

   # Handle the 1D operations
   onedspec = spectral_reduction.OneDSpec()
   onedspec.from_twodspec(science2D, stype='science')
   onedspec.from_twodspec(standard2D, stype='standard')
   onedspec.find_arc_lines()

   # Wavelength calibration
   onedspec.initialise_calibrator()
   onedspec.add_atlas(['Chemical Symbol 1', 'Chemical Symbol 2'])
   onedspec.fit()
   onedspec.apply_wavelength_calibration()

   # Flux calibration
   onedspec.load_standard(target='target name')
   onedspec.get_sensitivity()
   onedspec.apply_flux_calibration()

   # Apply atmospheric extinction correction
   onedspec.set_atmospheric_extinction()
   onedspec.apply_atmospheric_extinction_correction()

   # Inspect the reduced data product
   onedspec.inspect_reduced_spectrum()

Some more complete examples are available in the tutorials.


License and Attribution
=======================

Copyright 2019-2021

If you make use of ASPIRED in your work, please cite our paper
(`arXiv <https://arxiv.org/abs/2012.03505>`_,
`ADS <https://ui.adsabs.harvard.edu/abs/2020arXiv201203505L/abstract>`_,
`BibTeX <https://ui.adsabs.harvard.edu/abs/2020arXiv201203505L/exportcitation>`_);

and the specific Software version should it be relevant

`Zenodo for RASCAL <https://zenodo.org/record/4124170#.YTN2rY4zYrQ>`_
`Zenodo for ASPIRED <https://zenodo.org/record/4463569#.YTN2sY4zYrQ>`_)


Acknowledgement
===============

This research made use of Astropy,\footnote{http://www.astropy.org} a community-developed core Python package for Astronomy \citep{astropy:2013, astropy:2018}.

This software has also made some use of the `Astro-Python <https://crossfield.ku.edu/python/>`_ code.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
