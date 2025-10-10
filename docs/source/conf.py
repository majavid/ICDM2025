# docs/source/conf.py
import os
import sys

# Make the package importable for autodoc
sys.path.insert(0, os.path.abspath(".."))  # ../ -> docs/
sys.path.insert(0, os.path.abspath("../.."))  # ../../ -> repo root

project = "ICDM2025"
author = "Mohammad Ali Javidian"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
}

# Napoleon (Google/NumPy docstring support)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

html_theme = "furo"
html_title = "ICDM2025"

# General
templates_path = ["_templates"]
exclude_patterns = []
