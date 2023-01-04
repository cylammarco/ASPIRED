#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ASPIRED"""

from pkg_resources import get_distribution, DistributionNotFound

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

from . import image_reduction
from . import spectral_reduction
from . import standard_list

__all__ = [
    "image_reduction",
    "spectral_reduction",
    "standard_list",
    "util",
]