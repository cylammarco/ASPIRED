# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8-sig"

# Produce API reference automatically for every public and private methods
apidoc_args = [
    "--force",  # Overwrite existing files
    "--no-toc",  # Don't create a table of contents file
    "--output-dir=.",  # Directory to place all output
]

autodoc_mock_imports = [
    "astropy",
    "astroscrappy",
    "ccdproc",
    "numpy",
    "rascal",
    "plotly",
    "scipy",
    "spectres",
    "statsmodels",
]

autoclass_content = "both"

mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# -- Options for HTML output -------------------------------------------------

# Readthedocs.
master_doc = "index"

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    html_static_path = ["_static"]
else:
    html_theme = "alabaster"
    html_static_path = ["_static"]
