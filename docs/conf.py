"""Sphinx configuration for the PyPhi documentation."""

import os
from importlib.metadata import metadata

# Keep the import-time welcome banner out of autodoc's import of pyphi.
os.environ["PYPHI_WELCOME_OFF"] = "1"

project = "PyPhi"
author = "Will Mayner"
copyright = "2014–2026, Will Mayner and contributors"
release = metadata("pyphi")["Version"]
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "superpowers/**",
    "**/.ipynb_checkpoints",
    "examples/IIT_4.0_demo.ipynb",
    "getting-started/first-computation.ipynb",
    # Paired tutorial notebooks are download artifacts; the .md is the rendered
    # source. Exclude the .ipynb so Sphinx does not see two files per document.
    "tutorials/*.ipynb",
]

# --- MyST / executable pages ------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
]
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_raise_on_error = True
# Drop stderr stream output (e.g. the tqdm/ipywidgets notice) from rendered
# pages; genuine cell errors still fail the build via raise_on_error above.
nb_output_stderr = "remove"

# --- API reference ----------------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_google_docstring = False

# pyphi.relations defines both a function and a class whose names differ
# only in case (relation/Relation, relations/Relations). On case-insensitive
# filesystems the default per-object stub filenames collide, so remap the
# functions to distinct filenames.
autosummary_filename_map = {
    "pyphi.relations.relation": "pyphi.relations.relation-function",
    "pyphi.relations.relations": "pyphi.relations.relations-function",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# --- HTML output ------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/phi-favicon.svg"
html_theme_options = {
    "github_url": "https://github.com/wmayner/pyphi",
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "logo": {
        "image_light": "_static/pyphi-logo-text-noborder-776x196.png",
        "image_dark": "_static/pyphi-logo-text-white-noborder-776x196.png",
    },
}
