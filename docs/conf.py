# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

from maize.core.interface import Parameter

sys.path.insert(0, os.path.abspath(".."))

project = "maize-contrib"
copyright = "2023, Molecular AI Group"
author = "Thomas Löhr"
release = "0.2.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

autosummary_generate = True
add_module_names = True
autodoc_member_order = "bysource"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "maize": ("https://molecularai.github.com/maize", None),
}


# Show default values for parameters in the docstring
# TODO This would be nicer to have in the signature, but we don't
# seem to have access to the actual *value* to hack this in
def include_interface_defaults(app, what, name, obj, options, lines):
    if isinstance(obj, Parameter) and obj.default is not None:
        lines[-1] += f" (:code:`default = {obj.default}`)"


# These handlers ensure that the value for `required_callables`
# and `required_packages` is always emitted
def process_required_callables_sig(app, what, name, obj, options, signature, return_annotation):
    env_spec = name.endswith("required_callables") or name.endswith("required_packages")
    if what == "attribute" and env_spec:
        options["no-value"] = False
    else:
        options["no-value"] = True


def include_interfaces(app, what, name, obj, skip, options):
    env_spec = name.endswith("required_callables") or name.endswith("required_packages")
    param_with_default = isinstance(obj, Parameter) and obj.default is not None
    if what == "attribute" and (env_spec or param_with_default):
        return False
    return None


def setup(app):
    app.connect("autodoc-process-docstring", include_interface_defaults)
    app.connect("autodoc-process-signature", process_required_callables_sig)
    app.connect("autodoc-skip-member", include_interfaces)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- AZ Colors
_COLORS = {
    "mulberry": "rgb(131,0,81)",
    "lime-green": "rgb(196,214,0)",
    "navy": "rgb(0,56,101)",
    "graphite": "rgb(63,68,68)",
    "light-blue": "rgb(104,210,223)",
    "magenta": "rgb(208,0,111)",
    "purple": "rgb(60,16,83)",
    "gold": "rgb(240,171,0)",
    "platinum": "rgb(157,176,172)",
}

html_title = "maize contrib"
html_logo = "maize-contrib-logo.svg"
html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": _COLORS["navy"],
        "color-brand-content": _COLORS["navy"],
        "color-api-name": _COLORS["purple"],
        "color-api-pre-name": _COLORS["purple"],
    },
}
html_static_path = ["_static"]
