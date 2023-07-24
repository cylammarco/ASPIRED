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
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad

# ignore for now
# (1) a dictionary to match names to unique names (the search is one-to-one)
# (2) a dictionary to unique names to names (the search is one-to-many)

# essential
# (3) a dictionary to match libraries to unique names (the search is one-to-many)
# (4) a dictionary to match unique names to libraries (the search is one-to-many)

# the order in which the standard libraries goes through in automated mode
library_rank = {
    "ingoke": 1,
    "ingmas": 2,
    "irafctionewcal": 3,
    "irafspec16cal": 4,
    "ingog": 5,
    "esoxshooter": 6,
    "esowdstan": 7,
    "esohststan": 8,
    "esookestan": 9,
    "esoctiostan": 10,
    "irafctiocal": 11,
    "irafiidscal": 12,
    "irafredcal": 13,
    "irafbstdscal": 14,
    "irafirscal": 15,
    "ingsto": 16,
    "irafspec50cal": 17,
    "irafoke1990": 18,
    "irafspechayescal": 19,
    "ingfg": 20,
    "irafblackbody": 21,
}


filelist = glob.glob("*filename-to-identifier.json")


# Create dictionary for libraries to filenames
lib_to_filename = {}
lib_to_designation = {}

# And the mapping of designations and filename
lib_filename_to_designation_position = {}
designation_to_lib_filename = {}
designation_to_position = {}

for json_filename in filelist:
    data = json.load(open(json_filename, encoding="ascii"))
    lib_name = json_filename.split("-")[0]
    lib_to_filename[lib_name] = list(data.keys())
    for k, v in data.items():
        if lib_name == "irafblackbody":
            ra = 0.0
            dec = 0.0
        else:
            _ra, _dec = Simbad.query_object(v[0])["RA", "DEC"].items()
            c = SkyCoord(
                _ra[-1].data.data[0] + " " + _dec[-1].data.data[0],
                unit=(u.hourangle, u.deg),
            )
            ra = c.ra.value
            dec = c.dec.value
        lib_to_designation[lib_name] = [d.replace(" ", "").lower() for d in v]
        lib_filename_to_designation_position[lib_name + "/" + k] = [
            [d.replace(" ", "").lower(), ra, dec] for d in v
        ]


# Loop through the library names
for (
    lib_filename,
    designation_position,
) in lib_filename_to_designation_position.items():
    libname, filename = lib_filename.split("/")
    for d, _r, _d in designation_position:
        if d.lower() not in designation_to_lib_filename:
            designation_to_lib_filename[d.lower()] = {}
            designation_to_position[d.lower()] = {}
        designation_to_lib_filename[d.lower()][libname] = filename
        designation_to_position[d.lower()][libname] = [_r, _d]

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

with open("lib_to_designation.json", "w") as f:
    json.dump(lib_to_designation, f)

with open("filename_to_lib.json", "w") as f:
    json.dump(filename_to_lib, f)

with open("lib_filename_to_designation_position.json", "w") as f:
    json.dump(lib_filename_to_designation_position, f)

with open("designation_to_lib_filename.json", "w") as f:
    json.dump(designation_to_lib_filename, f)

with open("designation_to_position.json", "w") as f:
    json.dump(designation_to_position, f)
