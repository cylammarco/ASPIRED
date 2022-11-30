# -*- coding: utf-8 -*-
import json
import os
import sys

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(base_dir, ".."))

from standard_list import *


# ignore for now
# (1) a dictionary to match names to unique names (the search is one-to-one)
# (2) a dictionary to unique names to names (the search is one-to-many)

# essential
# (3) a dictionary to match libraries to unique names (the search is one-to-many)
# (4) a dictionary to match unique names to libraries (the search is one-to-many)

# the order in which the standard libraries goes through in automated mode
library_rank = {
    "ing_oke": 1,
    "ing_mas": 2,
    "irafctionewcal": 3,
    "irafspec16cal": 4,
    "irafiidscal": 5,
    "ing_fg": 6,
    "ing_og": 7,
    "ing_sto": 8,
    "irafoke1990": 9,
    "irafspec50cal": 10,
    "irafspechayescal": 11,
    "irafctiocal": 12,
    "irafredcal": 13,
    "irafirscal": 14,
    "irafbstdscal": 15,
    "irafblackbody": 16,
    "esoxshooter": 17,
    "esowdstan": 18,
    "esohststan": 19,
    "esookestan": 20,
    "esoctiostan": 21,
}

# Create dictionary for libraries to unique names
lib_to_name = {}

# Loop through the library names
for lib_name in library_list:
    # Loop through the standard stars UNIQUE names in each library
    for star in eval(lib_name):
        if lib_name not in lib_to_name:
            lib_to_name[lib_name] = [star]
        else:
            lib_to_name[lib_name].append(star)

# Create dictionary for unique names to libraries
name_to_lib = {}

# Loop through the library names
for lib, stars in lib_to_name.items():
    for s in stars:
        if s not in name_to_lib:
            name_to_lib[s] = [lib]
        else:
            name_to_lib[s].append(lib)

# Sort the list of the dictionaries
for star, libs in name_to_lib.items():
    name_to_lib[star] = sorted(
        name_to_lib[star], key=lambda i: library_rank[i]
    )

with open("lib_to_uname.json", "w") as f:
    json.dump(lib_to_name, f)

with open("uname_to_lib.json", "w") as f:
    json.dump(name_to_lib, f)
