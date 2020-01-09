.. ASPIRED documentation master file, created by
   sphinx-quickstart on Thu Jan  9 11:38:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ASPIRED documentation!
======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Basic Usage
===========

The bare minimum example code to to get a wavelength calibration:

.. code-block:: python


Some more complete examples are available in the :ref:`quickstart` tutorial.


How to Use This Guide
=====================

To start, you're probably going to need to follow the :ref:`installation` guide to
get RASCAL installed on your computer.
After you finish that, you can probably learn most of what you need from the
tutorials listed below (you might want to start with
:ref:`quickstart` and go from there).
If you need more details about specific functionality, the User Guide below
should have what you need.

We welcome bug reports, patches, feature requests, and other comments via `the GitHub
issue tracker <https://github.com/jveitchmichaelis/rascal/issues>`_.


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
   tutorials/lt-sprat
   tutorials/wht-isis

.. toctree::
   :maxdepth: 2
   :caption: List of Modules

   modules/imagereduction
   modules/twodspec
   modules/onedspec
   modules/wavelengthpolyfit
   modules/standardflux

.. toctree::
   :maxdepth: 1
   :caption: API

   autoapi/index


License & Attribution
=====================

Copyright 2019-2020

If you make use of ASPIRED in your work, please cite our paper
(`arXiv <https://arxiv.org/abs/1912.05885>`_,
`ADS <https://https://ui.adsabs.harvard.edu/abs/2019arXiv191205885L/abstract>`_,
`BibTeX <https://ui.adsabs.harvard.edu/abs/2019arXiv191205885L/exportcitation>`_).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
