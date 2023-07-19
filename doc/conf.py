"""Configuration file for the Sphinx documentation builder."""

import os
import sys
import importlib.util
from sphinx.ext.napoleon.docstring import GoogleDocstring
import sphinx.ext.napoleon

sys.path.insert(0, os.path.abspath(".."))


html_static_path = ['_static']

html_css_files = [
    'custom.css',
]


def load_imsim_version():
    """Extract version of imsim without importing the whole imsim module"""

    spec = importlib.util.spec_from_file_location(
        "imsim_version",
        os.path.join(os.path.dirname(__file__), "..", "imsim", "_version.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GoogleDocstringExt(GoogleDocstring):
    """Extensions to the  Napoleon GoogleDocstring class

    Source: https://michaelgoerz.net/notes/extending-sphinx-napoleon-docstring-sections.html
    """

    def _parse_keys_section(self, _section):
        return self._format_fields("Keys", self._consume_fields())

    def _parse_attributes_section(self, _section):
        return self._format_fields("Attributes", self._consume_fields())

    def _parse_class_attributes_section(self, _section):
        return self._format_fields("Class Attributes", self._consume_fields())

    def _parse(self) -> None:
        self._sections["keys"] = self._parse_keys_section
        self._sections["class attributes"] = self._parse_class_attributes_section
        super()._parse()


sphinx.ext.napoleon.GoogleDocstring = GoogleDocstringExt


project = "imSim"
copyright = "2016-2022, LSSTDESC"
author = "LSSTDESC"


extensions = [
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"


# The reST default role (used for this markup: `text`)
default_role = "any"


imsim_version = load_imsim_version()
version = ".".join(map(str, imsim_version.__version_info__[:2]))
release = imsim_version.__version__
