# -*- coding: utf-8 -*-
import glob
import json
import os
import sys

try:
    base_dir = os.path.abspath(os.path.dirname(__file__))
except:
    base_dir = os.path.abspath(os.path.dirname(__name__))

sys.path.insert(0, os.path.join(base_dir, ".."))

from aspired.standard_list import library_list

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


filelist = glob.glob("*filename-to-identifier.json")


# Create dictionary for libraries to filenames
lib_to_filename = {}

# And the mapping of designations and filename
filename_to_designation = {}
designation_to_filename = {}

for json_filename in filelist:
    data = json.load(open(json_filename))
    lib_to_filename[json_filename.split("-")[0]] = list(data.keys())
    for k, v in data.items():
        filename_to_designation[k] = [d.replace(" ", "").lower() for d in v]


# Loop through the library names
for filename, designation in filename_to_designation.items():
    for d in designation:
        if d.lower() not in designation_to_filename:
            designation_to_filename[d.lower()] = [filename.lower()]
        else:
            designation_to_filename[d.lower()].append(filename.lower())


# Create dictionary for filenames to libraries
filename_to_lib = {}

# Loop through the library names
for lib, stars in lib_to_filename.items():
    for s in stars:
        if s.lower() not in filename_to_lib:
            filename_to_lib[s.lower()] = [lib.lower()]
        else:
            filename_to_lib[s.lower()].append(lib.lower())

# Sort the list of the dictionaries
for star, libs in filename_to_lib.items():
    filename_to_lib[star] = sorted(
        filename_to_lib[star], key=lambda i: library_rank[i]
    )

with open("lib_to_filename.json", "w") as f:
    json.dump(lib_to_filename, f)

with open("filename_to_lib.json", "w") as f:
    json.dump(filename_to_lib, f)

with open("designation_to_filename.json", "w") as f:
    json.dump(designation_to_filename, f)
