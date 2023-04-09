#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ASPIRED"""

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed

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
