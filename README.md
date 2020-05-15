# Automated SpectroPhotometric Image REDuction (ASPIRED)
[![Build Status](https://travis-ci.com/cylammarco/ASPIRED.svg?branch=dev)](https://travis-ci.com/cylammarco/ASPIRED)
[![Readthedocs Status](https://readthedocs.org/projects/aspired/badge/?version=latest&style=flat)](https://aspired.readthedocs.io/en/latest/)

We aim to provide a suite of publicly available spectral data reduction software to facilitate rapid scientific outcomes from time-domain observations. For time resolved observations, an automated pipeline frees astronomers from performance of the routine data analysis tasks to concentrate on interpretation, planning future observations and communication with international collaborators. Part of the OPTICON project coordinates the operation of a network of largely self-funded European robotic and conventional telescopes, coordinating common science goals and providing the tools to deliver science-ready photometric and spectroscopic data. As part of this activity is  [SPRAT](https://ui.adsabs.harvard.edu/abs/2014SPIE.9147E..8HP/abstract), a compact, reliable, low-cost and high-throughput spectrograph and appropriate for deployment on a wide range of 1-4m class telescopes. Installed on the Liverpool Telescope since 2014, the deployable slit and grating mechanism and optical fibre based calibration system make the instrument self-contained.

ASPIRED is written for use with python 3.6, 3.7 and 3.8 (will revisit [3.9](https://www.python.org/dev/peps/pep-0596/) in Fall 2020), and is intentionally developed as a self-consistent reduction pipeline with its own diagnostics and error handling. The pipeline should be able to reduce 2D spectral data from raw image to wavelength and flux calibrated 1D spectrum automatically without any user input (quicklook quality). However, the real goal is to provide a set of easily configurable routines to build pipelines for long slit spectrographs on different telescopes (science quality). We use SPRAT as a test case for this development, but our aim is to support a much wider range of instruments. By delivering near real-time data reduction we will facilitate automated or interactive decision making, allowing "on-the-fly" modification of observing strategies and rapid triggering of other facilities.

More background information can be referring to the [arXiv article](https://ui.adsabs.harvard.edu/abs/2019arXiv191205885L/abstract), which will appeare in the [Astronomical Society of the Pacific Conference Series (ASPCS)](http://www.aspbooks.org/a/volumes/upcoming/?book_id=606). This is in concurrent development with the automated wavelength calibration software [RASCAL](https://github.com/jveitchmichaelis/rascal), further information can be referred to this [arXiv article](https://ui.adsabs.harvard.edu/abs/2019arXiv191205883V/abstract) and it will appear in the same volume of ASPCS.

## Use cases
We are to cover as many use cases as possible. If you would like to apply some reduction techniques that we have overseen, please use the [issue tracker](https://github.com/cylammarco/ASPIRED/issues) to request new features. The following is the list of scenarios that we can handle:

### Image
1. Dataset with light frame and any combination (including none) of dark(s), bias(s) and flat(s) frames.

### Spectrum - full reduction
2. Dataset with science and standard field-flattened images and the respective arc image.

### Spectrum - flux calibrated
3. Dataset with science field-flattened image and the arc image only.

### Spectrum - not wavelength and flux calibrated
4. Dataset with science field-flattened image only.

### Spectrum - other cases for full or partial reduction
5. User supplied trace(s).
6. [Enhancement required] User supplied wavelength calibration polynomial coefficients.
7. [Not started] User supplied line list.
8. [Not started] User supplied pixel-to-wavelength mapping (not fitted).
9. [Enhancement required] User supplied sensitivity curve (wavelength).
10. [Not started] User supplied wavelength calibrated standard.
11. [Not started] Flux calibration for user supplied wavelength calibrated science and standard spectra.

### Output
12. Save diagnostic plots in [these formats](https://plotly.com/python/renderers/#setting-the-default-renderer).
13. [Enhancement required]Save data in FITS or ascii.

## Dependencies
* python >= 3.6
* numpy
* scipy
* astropy
* ccdproc
* astroscrappy
* plotly
* spectres
* rascal

## Installation
Instructions can be found [here](https://aspired.readthedocs.io/en/latest/installation/pip.html).

## Reporting issues/feature requests
Please use the [issue tracker](https://github.com/cylammarco/ASPIRED/issues) to report any issues or support questions.

## Getting started
The [quickstart guide](https://aspired.readthedocs.io/en/latest/tutorials/quickstart.html) will show you how to reduce the example dataset.

## Contributing Code/Documentation
If you are interested in contributing code to the project, thank you! For those unfamiliar with the process of contributing to an open-source project, you may want to read through Github’s own short informational section on how to submit a [contribution](https://opensource.guide/how-to-contribute/#how-to-submit-a-contribution) or send me a message.

## Funding bodies
1. European Union’s Horizon 2020 research and innovation programme (grant agreement No. 730890)
(January 2019 - March 2020)
