"""Sphinx configuration for the PyPhi documentation."""

import os
import sys
from importlib.metadata import metadata
from pathlib import Path

# Keep the import-time welcome banner out of autodoc's import of pyphi.
os.environ["PYPHI_WELCOME_OFF"] = "1"

# The Selenized code-block styles live in _ext; register them under the names
# the theme options use.
sys.path.insert(0, str(Path(__file__).parent / "_ext"))
from pygments import styles as _pygments_styles

for _cls, _name in (
    ("SelenizedLightStyle", "selenized-light"),
    ("SelenizedDarkStyle", "selenized-dark"),
):
    # Both maps: get_style_by_name reads the first, get_all_styles the second.
    _pygments_styles._STYLE_NAME_TO_MODULE_MAP[_name] = ("selenized", _cls)
    _pygments_styles.STYLES[_cls] = ("selenized", _name, ())

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
    "sphinx_tippy",
    "sphinx_llms_txt",
    "sphinx_sitemap",
    "sphinxext.opengraph",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "superpowers/**",
    "**/.ipynb_checkpoints",
    # Paired notebooks are download artifacts; the .md is the rendered source.
    # Exclude the .ipynb so Sphinx does not see two files per document.
    "getting-started/*.ipynb",
    "tutorials/*.ipynb",
]

# --- MyST / executable pages ------------------------------------------------

# Figures drawn while executing pages use the site's matplotlib settings; the
# kernel inherits this environment.
os.environ.setdefault("MATPLOTLIBRC", str(Path(__file__).parent / "matplotlib.rc"))

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
]
nb_execution_mode = "cache"
# The demo notebook is committed with its outputs (refresh with
# ``just notebook-outputs``); the build renders them without executing.
nb_execution_excludepatterns = ["examples/IIT_4.0_demo.ipynb"]
nb_execution_timeout = 300
nb_execution_raise_on_error = True
# Drop stderr stream output (e.g. the tqdm/ipywidgets notice) from rendered
# pages; genuine cell errors still fail the build via raise_on_error above.
nb_output_stderr = "remove"

# --- Hover tooltips ----------------------------------------------------------

# Cross-references in the article body (glossary terms, API objects, other
# pages) show their target's first paragraph on hover.
tippy_anchor_parent_selector = "article.bd-article"
tippy_skip_anchor_classes = ("headerlink", "sd-stretched-link", "download")
# MathJax is not loaded on pages without math, and the tooltip code would
# call it there.
tippy_enable_mathjax = False
# No network fetches at build time: no Wikipedia or DOI tooltips.
tippy_enable_wikitips = False
tippy_enable_doitips = False
tippy_props = {"placement": "auto-start", "maxWidth": 480, "interactive": True, "delay": [200, 100]}

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
    "pyphi.mcp.content.TOPICS": "pyphi.mcp.content.TOPICS-attribute",
    "pyphi.mcp.content.topics": "pyphi.mcp.content.topics-function",
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
html_title = "PyPhi"
html_static_path = ["_static"]
html_js_files = ["sidebar-shortcut.js"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400;1,600&family=IBM+Plex+Serif:wght@500;600&display=swap",
    "custom.css",
]
html_favicon = "_static/phi-favicon.svg"
html_theme_options = {
    "github_url": "https://github.com/wmayner/pyphi",
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "pygments_light_style": "selenized-light",
    "pygments_dark_style": "selenized-dark",
    "logo": {
        "image_light": "_static/pyphi-logo-text-noborder-776x196.png",
        "image_dark": "_static/pyphi-logo-text-white-noborder-776x196.png",
    },
    "announcement": (
        "PyPhi 2.0 is released: "
        '<a href="https://pyphi.readthedocs.io/en/stable/whats-new-in-2.0.html">'
        "what's new</a>, and the "
        '<a href="https://pyphi.readthedocs.io/en/stable/migration/migration-2.0.html">'
        "migration guide</a> for 1.x users."
    ),
    "switcher": {
        "json_url": "https://pyphi.readthedocs.io/en/latest/_static/switcher.json",
        "version_match": os.environ.get("READTHEDOCS_VERSION", "latest"),
    },
    "check_switcher": False,
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # An icon instead of the search field: the header has six sections and
    # three controls to fit on one row.
    "navbar_persistent": ["search-button"],
    "primary_sidebar_end": ["sidebar-cite"],
    # The sidebar shows the whole site: sections, their pages, and under the
    # API reference the curated modules (depth 3); module members stay behind
    # each module's page.
    "navigation_depth": 3,
    "show_nav_level": 1,
    "footer_start": ["copyright"],
    "footer_end": [],
}
html_show_sphinx = False
html_sidebars = {"**": ["sidebar-collapse", "search-field", "sidebar-nav-all"]}

# --- Machine-readable copies of the site --------------------------------------
# Read the Docs sets READTHEDOCS_CANONICAL_URL per version; the stable URL is
# the fallback for local builds. The base URL is what the sitemap and llms.txt
# resolve page links against.
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL", "https://pyphi.readthedocs.io/en/stable/"
)
sitemap_url_scheme = "{link}"
# Link previews (Open Graph) for pages shared in chat and social clients.
ogp_site_url = html_baseurl
ogp_site_name = "PyPhi"
ogp_image = html_baseurl + "_static/pyphi-logo-text-776x196.png"
ogp_description_length = 200
llms_txt_title = "PyPhi"
llms_txt_summary = (
    "PyPhi computes the quantities of Integrated Information Theory (IIT 4.0): "
    "system integrated information, cause-effect structures, and related "
    "measures, for discrete dynamical systems given as transition probability "
    "matrices."
)
# The narrative pages only: the generated API stubs would triple the size of
# llms-full.txt (the intersphinx inventory, objects.inv, resolves API names),
# and the demo notebook's source is JSON with embedded outputs.
llms_txt_exclude = ["reference/_autosummary/*", "examples/IIT_4.0_demo"]
