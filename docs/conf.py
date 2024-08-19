# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import time
from typing import Any, Callable, Literal, cast

from sphinx.ext.autodoc import (
    ModuleLevelDocumenter,
    ClassDocumenter,
    ObjectMember,
    Documenter,
    inherited_members_option,
)
from sphinx.ext.autodoc.importer import get_class_members
from sphinx.domains.python import PyClasslike

from maize.core.node import Node
from maize.core.interface import Interface, Parameter, Input, Output

sys.path.insert(0, os.path.abspath(".."))

project = "maize-contrib"
copyright = f"{time.localtime().tm_year}, Molecular AI Group"
author = "Thomas Löhr"
release = "0.5.5"

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
autosummary_ignore_module_all = False
toc_object_entries = False
add_module_names = True
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "maize": ("https://molecularai.github.com/maize", None),
}


InterfaceType = Literal["requirements", "inputs", "outputs", "parameters"]


class MaizeNodeDirective(PyClasslike):
    pass


class MaizeNodeDocumenter(ModuleLevelDocumenter):
    directivetype = "node"
    objtype = "node"
    priority = ClassDocumenter.priority + 5

    option_spec = {"inherited-members": inherited_members_option}

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        return isinstance(member, Node)

    def get_object_members(self, want_all: bool) -> tuple[bool, list[ObjectMember]]:
        members = get_class_members(
            self.object, self.objpath, self.get_attr, self.config.autodoc_inherit_docstrings
        )
        members = {
            name: member
            for name, member in members.items()
            if isinstance(member.object, Interface)
            or name in ["required_callables", "required_packages"]
        }
        if self.options.inherited_members:
            return False, list(members.values())
        else:
            return False, [m for m in members.values() if m.class_ == self.object]

    def get_interfaces(self, interface: InterfaceType) -> tuple[bool, list[ObjectMember]]:
        members_check_module, members = self.get_object_members(want_all=True)
        kinds: dict[str, Callable[[ObjectMember], bool]] = {
            "requirements": lambda mem: mem.__name__ in ["required_callables", "required_packages"]
            and mem.object,
            "inputs": lambda mem: isinstance(mem.object, Input),
            "outputs": lambda mem: isinstance(mem.object, Output),
            "parameters": lambda mem: isinstance(mem.object, Parameter) and mem.__name__ not in Node.get_parameters(),
            "standard": lambda mem: isinstance(mem.object, Parameter) and mem.__name__ in Node.get_parameters(),
        }
        return members_check_module, [member for member in members if kinds[interface](member)]

    def document_members(self, all_members: bool = False) -> None:
        # Required to avoid duplicate docs
        pass

    def _document_interfaces(self, interface: InterfaceType) -> None:
        # set current namespace for finding members
        self.env.temp_data["autodoc:module"] = self.modname
        if self.objpath:
            self.env.temp_data["autodoc:class"] = self.objpath[0]

        members_check_module, members = self.get_interfaces(interface=interface)

        if members:
            self.add_line(f"**{interface.upper()}**", self.get_sourcename())

        # document non-skipped members
        memberdocumenters: list[tuple[Documenter, bool]] = []
        for mname, member, isattr in self.filter_members(members, want_all=True):
            classes = [
                cls
                for cls in self.documenters.values()
                if cls.can_document_member(member, mname, isattr, self)
            ]
            if not classes:
                # don't know how to document this member
                continue
            # prefer the documenter with the highest priority
            classes.sort(key=lambda cls: cls.priority)
            # give explicitly separated module name, so that members
            # of inner classes can be documented
            full_mname = f"{self.modname}::" + ".".join((*self.objpath, mname))
            documenter = classes[-1](self.directive, full_mname, self.indent)
            memberdocumenters.append((documenter, isattr))

        member_order = self.options.member_order or self.config.autodoc_member_order
        memberdocumenters = self.sort_members(memberdocumenters, member_order)

        for documenter, isattr in memberdocumenters:
            documenter.generate(
                all_members=True,
                real_modname=self.real_modname,
                check_module=members_check_module and not isattr,
            )

        # reset current objects
        self.env.temp_data["autodoc:module"] = None
        self.env.temp_data["autodoc:class"] = None

    def generate(
        self,
        more_content: Any | None = None,
        real_modname: str | None = None,
        check_module: bool = False,
        all_members: bool = False,
    ) -> None:
        super().generate(more_content, real_modname, check_module, all_members=True)
        for interface_type in ["requirements", "inputs", "outputs", "parameters", "standard"]:
            self._document_interfaces(cast(InterfaceType, interface_type))


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
    app.add_directive_to_domain("py", "node", MaizeNodeDirective)
    app.connect("autodoc-process-docstring", include_interface_defaults)
    app.connect("autodoc-process-signature", process_required_callables_sig)
    app.connect("autodoc-skip-member", include_interfaces)
    app.add_autodocumenter(MaizeNodeDocumenter)


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
