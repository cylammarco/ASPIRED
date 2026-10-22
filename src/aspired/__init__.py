#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ASPIRED."""

from importlib.metadata import PackageNotFoundError, version

__project__ = "ASPIRED"
__author__ = "Marco C Lam"
__copyright__ = "2017-2026, Marco C Lam"
__version__ = "0.6.0"

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass

__status__ = "Production"
__credits__ = [
    "Marco C Lam",
    "Robert J Smith",
    "Iair Arcavi",
    "Iain A Steele",
    "Josh Veitch-Michaelis",
    "Lukasz Wyrzykowski",
]

from . import image_reduction, spectral_reduction, standard_list

__all__ = [
    "image_reduction",
    "spectral_reduction",
    "standard_list",
    "util",
    "flux_calibration",
    "wavelength_calibration",
]
