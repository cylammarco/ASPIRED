# Automated SpectroPhotometric Image REDuction (ASPIRED)
![Python package](https://github.com/cylammarco/ASPIRED/workflows/Python%20package/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/cylammarco/ASPIRED/badge.svg?branch=main)](https://coveralls.io/github/cylammarco/ASPIRED?branch=main)
[![Readthedocs Status](https://readthedocs.org/projects/aspired/badge/?version=latest&style=flat)](https://aspired.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/aspired.svg)](https://badge.fury.io/py/aspired)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4127294.svg)](https://doi.org/10.5281/zenodo.4127294)

We aim to provide a suite of publicly available spectral data reduction software to facilitate rapid scientific outcomes from time-domain observations. For time resolved observations, an automated pipeline frees astronomers from performance of the routine data analysis tasks to concentrate on interpretation, planning future observations and communication with international collaborators. Part of the OPTICON project coordinates the operation of a network of largely self-funded European robotic and conventional telescopes, coordinating common science goals and providing the tools to deliver science-ready photometric and spectroscopic data. As part of this activity is  [SPRAT](https://ui.adsabs.harvard.edu/abs/2014SPIE.9147E..8HP/abstract), a compact, reliable, low-cost and high-throughput spectrograph and appropriate for deployment on a wide range of 1-4m class telescopes. Installed on the Liverpool Telescope since 2014, the deployable slit and grating mechanism and optical fibre based calibration system make the instrument self-contained.

ASPIRED is written for use with python 3.6, 3.7 and 3.8 (will revisit [3.9](https://www.python.org/dev/peps/pep-0596/) in Fall 2020), and is intentionally developed as a self-consistent reduction pipeline with its own diagnostics and error handling. The pipeline should be able to reduce 2D spectral data from raw image to wavelength and flux calibrated 1D spectrum automatically without any user input (quicklook quality). However, the real goal is to provide a set of easily configurable routines to build pipelines for long slit spectrographs on different telescopes (science quality). We use SPRAT as a test case for this development, but our aim is to support a much wider range of instruments. By delivering near real-time data reduction we will facilitate automated or interactive decision making, allowing "on-the-fly" modification of observing strategies and rapid triggering of other facilities.

More background information can be referring to the [arXiv article](https://ui.adsabs.harvard.edu/abs/2019arXiv191205885L/abstract), which will appeare in the [Astronomical Society of the Pacific Conference Series (ASPCS)](http://www.aspbooks.org/a/volumes/upcoming/?book_id=606). This is in concurrent development with the automated wavelength calibration software [RASCAL](https://github.com/jveitchmichaelis/rascal), further information can be referred to this [arXiv article](https://ui.adsabs.harvard.edu/abs/2019arXiv191205883V/abstract) and it will appear in the same volume of ASPCS.

Example notebooks can be found [here](https://github.com/cylammarco/ASPIRED-example).

## Use cases
We are to cover as many use cases as possible. If you would like to apply some reduction techniques that we have overseen, please use the [issue tracker](https://github.com/cylammarco/ASPIRED/issues) to request new features. The following is the list of scenarios that we can handle:

### Image
0. [x] Dataset with light frame and any combination (including none) of dark(s), bias(s) and flat(s) frames.

### Spectrum - full reduction
1. [x] Dataset with science and standard field-flattened images and the respective arc image.

### Spectrum - ADU spectrum extraction only (No flux calibration)
2. [x] Dataset with science field-flattened image and the arc image only.

### Spectrum - wavelength calibration only (Pre wavelength-calibrated)
3. [x] Dataset with science and standard field-flattened images only.

### Spectrum - other cases for full or partial reduction
4. [x] User supplied trace(s) for light spectrum extraction (only to extract ADU/s of the spectra).
5. [x] User supplied trace(s) for arc extraction (only to get wavelength calibration polynomial).
6. [x] User supplied wavelength calibration polynomial coefficients.
7. [x] User supplied line list.
8. [x] User supplied pixel-to-wavelength mapping (not fitted).
9. [x] User supplied sensitivity curve.
10. [x] User supplied wavelength calibrated standard.
11. [x] Flux calibration for user supplied wavelength calibrated science and standard 1D spectra.

### Output
11. [x] Save diagnostic plots in [these formats](https://plotly.com/python/renderers/#setting-the-default-renderer).
12. [x] Save data in FITS (with header information) or ascii (csv).

## Dependencies
* python >= 3.6
* numpy
* scipy
* [astropy](https://github.com/astropy/astropy)
* [astroscrappy](https://github.com/astropy/astroscrappy)
* [ccdproc](https://github.com/astropy/ccdproc)
* [plotly](https://github.com/plotly/plotly.py) >= 4.0
* [rascal](https://github.com/jveitchmichaelis/rascal) >= 0.2
* [spectres](https://github.com/ACCarnall/SpectRes) >= 2.1.1

## Installation
Instructions can be found [here](https://aspired.readthedocs.io/en/latest/installation/pip.html).

## Reporting issues/feature requests
Please use the [issue tracker](https://github.com/cylammarco/ASPIRED/issues) to report any issues or support questions.

## Getting started
The [quickstart guide](https://aspired.readthedocs.io/en/latest/tutorials/quickstart.html) will show you how to reduce the example dataset.

## Contributing Code/Documentation
If you are interested in contributing code to the project, thank you! For those unfamiliar with the process of contributing to an open-source project, you may want to read through Github’s own short informational section on how to submit a [contribution](https://opensource.guide/how-to-contribute/#how-to-submit-a-contribution) or send me a message.

Style -- as long as it passes PEP8, it's fine. We suggest you to use yapf default setting. What is not covered is the imports, we cluster imports into three groups in this order: Python built-in libraries, third party imports, and then local application imports. `import` goes before `from`, and then follow alphabatical order.

## Funding bodies
1. European Union’s Horizon 2020 research and innovation programme (grant agreement No. 730890)
(January 2019 - March 2020)
2. Polish NCN grant Daina No. 2017/27/L/ST9/03221
(May - June 2020)
