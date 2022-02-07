#!/usr/bin/env python3

import codecs
import os
import re

__packagename__ = "aspired"

META_PATH = os.path.join("pyproject.toml")

HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^{meta} = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    meta_match_list = re.search(
        r"^{meta} = ([\[\"](?<=\[).+?(?=\])[\"\]])".format(meta=meta),
        meta_file,
        re.M,
    )
    if meta_match:
        return meta_match.group(1)
    elif meta_match_list:
        return eval(meta_match_list.group(1))
    else:
        raise RuntimeError(
            "Unable to find {meta} string or list.".format(meta=meta)
        )


_authors = find_meta("authors")
author_names = [i.split("<")[0].rstrip() for i in _authors]
author_emails = [i.split("<")[1][:-1].rstrip() for i in _authors]

_maintainers = find_meta("authors")
maintainer_names = [i.split("<")[0].rstrip() for i in _maintainers]
maintainer_emails = [i.split("<")[1][:-1].rstrip() for i in _maintainers]

__author__ = author_names
__email__ = author_emails
__maintainer__ = maintainer_names
__maintainer_email__ = maintainer_emails
__license__ = find_meta("license")
__description__ = find_meta("description")
__version__ = find_meta("version")
__status__ = "Production"

__credits__ = [
    "Iair Arcavi",
    "Paul R McWhirter",
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
