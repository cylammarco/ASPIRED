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
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'ASPIRED'
copyright = '2020, Marco Lam'
author = 'Marco Lam'
__version__ = '0.0.1'

# The full version, including alpha/beta/rc tags
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel', 'sphinx.ext.coverage', 'autoapi.extension'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Produce API reference automatically for every public and private methods
autoapi_dirs = ['../../aspired']
autodoc_default_options = {
    'members': None,
    'undoc-members': None,
    'private-members': None
}
autoclass_content = 'both'
autoapi_python_class_content = 'both'

# -- Options for HTML output -------------------------------------------------

# Readthedocs.
master_doc = 'index'

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
else:
    html_theme = 'alabaster'
    html_static_path = ["_static"]
