# Automated SpectroPhotometric Image REDuction (ASPIRED) -- A Python-based spectral data reduction toolkit
![Python package](https://github.com/cylammarco/ASPIRED/workflows/Python%20package/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/cylammarco/ASPIRED/badge.svg?branch=main)](https://coveralls.io/github/cylammarco/ASPIRED?branch=main)
[![Readthedocs Status](https://readthedocs.org/projects/aspired/badge/?version=latest&style=flat)](https://aspired.readthedocs.io/en/latest/)
[![arXiv](https://img.shields.io/badge/arXiv-2111.02127-00ff00.svg)](https://arxiv.org/abs/2111.02127)
[![PyPI version](https://badge.fury.io/py/aspired.svg)](https://badge.fury.io/py/aspired)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4127294.svg)](https://doi.org/10.5281/zenodo.4127294)
[![AJ](https://img.shields.io/badge/Journal-AJ-informational)](https://doi.org/10.3847/1538-3881/acd75c)
[![Downloads](https://pepy.tech/badge/aspired)](https://pepy.tech/project/aspired)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

We aim to provide a suite of publicly available spectral data reduction software to facilitate rapid scientific outcomes from time-domain observations. For time resolved observations, an automated pipeline frees astronomers from performance of the routine data analysis tasks to concentrate on interpretation, planning future observations and communication with international collaborators. Part of the OPTICON project coordinates the operation of a network of largely self-funded European robotic and conventional telescopes, coordinating common science goals and providing the tools to deliver science-ready photometric and spectroscopic data. As part of this activity is [SPRAT](https://ui.adsabs.harvard.edu/abs/2014SPIE.9147E..8HP/abstract), a compact, reliable, low-cost and high-throughput spectrograph and appropriate for deployment on a wide range of 1-4m class telescopes. Installed on the Liverpool Telescope since 2014, the deployable slit and grating mechanism and optical fibre based calibration system make the instrument self-contained.

ASPIRED is written for use with python 3.7, 3.8, 3.9, 3.10 and 3.11, and is intentionally developed as a self-consistent reduction pipeline with its own diagnostics and error handling. The pipeline should be able to reduce 2D spectral data from raw image to wavelength and flux calibrated 1D spectrum automatically without any user input (quicklook quality). However, the real goal is to provide a set of easily configurable routines to build pipelines for long slit spectrographs on different telescopes (science quality). We use SPRAT as a test case for this development, but our aim is to support a much wider range of instruments. By delivering near real-time data reduction we will facilitate automated or interactive decision making, allowing "on-the-fly" modification of observing strategies and rapid triggering of other facilities.

Further information can be referred to this [AJ article](https://iopscience.iop.org/article/10.3847/1538-3881/acd75c).

Early stage development efforts can be referred to this [ASPC article](https://ui.adsabs.harvard.edu/abs/2020ASPC..527..655L/abstract) and this [arXiv article](https://ui.adsabs.harvard.edu/abs/2020arXiv201203505L/abstract). This is in concurrent development with the automated wavelength calibration software [RASCAL](https://github.com/jveitchmichaelis/rascal), further information can be referred to this [ASPC article](https://ui.adsabs.harvard.edu/abs/2020ASPC..527..627V/abstract) and it will appear in the same volume of ASPCS.

Example notebooks and scripts can be found at [aspired-example](https://github.com/cylammarco/ASPIRED-example). More examples can be found at the github repository of the journal article [here](https://github.com/cylammarco/ASPIRED-apj-article), e.g.:

![alt text](https://github.com/cylammarco/ASPIRED-apj-article/blob/main/fig_09_reduction_compared.png?raw=true)

## Use cases
We are to cover as many use cases as possible. If you would like to apply some reduction techniques that we have overseen, please use the [issue tracker](https://github.com/cylammarco/ASPIRED/issues) to request new features. The following is the list of scenarios that we can handle:

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
10. [x] Flux calibration for user supplied wavelength calibrated science and standard 1D spectra.
11. [x] Apply atmospheric extinction correction.
12. [x] Apply basic telluric absorption correction.

### Output
13. [x] Save diagnostic plots in [these formats](https://plotly.com/python/renderers/#setting-the-default-renderer).
14. [x] Save data in FITS (with header information) or ascii (csv).

See the examples of these use cases at [aspired-example](https://github.com/cylammarco/ASPIRED-example/).

## Dependencies
* python >= 3.8 (It should work on 3.6 and 3.7 if you can sort out the astropy (>=5.0) requirement of >=3.8)
* [numpy](https://numpy.org/doc/stable/index.html) >= 1.21
* [scipy](https://scipy.org/) >= 1.7
* [astropy](https://github.com/astropy/astropy) >= 4.3
* [astroscrappy](https://github.com/astropy/astroscrappy) >= 1.1
* [ccdproc](https://github.com/astropy/ccdproc)
* [plotly](https://github.com/plotly/plotly.py) >= 5.0
* [rascal](https://github.com/jveitchmichaelis/rascal) >= 0.3.10, < 4.0
* [spectresc](https://github.com/cylammarco/SpectResC) >= 1.0.2
* [statsmodels](https://www.statsmodels.org/stable/index.html) >= 0.13

## Installation
Instructions can be found [here](https://aspired.readthedocs.io/en/latest/installation/pip.html).

## Reporting issues/feature requests
Please use the [issue tracker](https://github.com/cylammarco/ASPIRED/issues) to report any issues or support questions.

## Getting started
The [quickstart guide](https://aspired.readthedocs.io/en/latest/tutorials/quickstart.html) will show you how to reduce the example dataset.

## Contributing Code/Documentation
If you are interested in contributing code to the project, thank you! For those unfamiliar with the process of contributing to an open-source project, you may want to read through Github’s own short informational section on how to submit a [contribution](https://opensource.guide/how-to-contribute/#how-to-submit-a-contribution) or send me a message.

Style -- black. See the .pre-commit-config.yaml for the other requirements.

## Funding bodies
1. European Union’s Horizon 2020 research and innovation programme (grant agreement No. 730890)
(January 2019 - March 2020)
2. Polish NCN grant Daina No. 2017/27/L/ST9/03221
(May - June 2020)
3. European Research Council Starting Grant (grant agreement No. 852097)
(Sept 2020 - Current)

## Citation
If you make use of the ASPIRED toolkit, we would appreciate if you can refernce the two articles and two pieces of software listed below:

1. [ASPIRED AJ article](https://ui.adsabs.harvard.edu/abs/10.3847/1538-3881/acd75c/abstract)
2. [ASPIRED Zenodo](https://doi.org/10.5281/zenodo.4127294)
3. [RASCAL arXiv article](https://ui.adsabs.harvard.edu/abs/2020ASPC..527..627V/abstract)
4. [RASCAL Zenodo](https://doi.org/10.5281/zenodo.4117516)
