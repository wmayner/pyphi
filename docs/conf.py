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
    "examples/serialize_demo.ipynb",
]

# Substitutions used by the retained reStructuredText pages (conventions,
# examples, tiebreaking) pending their rewrite into the new section layout.
rst_prolog = "".join(
    [
        # Math
        r"""
.. |big_phi| replace:: :math:`\Phi`
.. |big_phi > 0| replace:: :math:`\Phi > 0`
.. |big_phi = 0| replace:: :math:`\Phi = 0`
.. |big_phi_max| replace:: :math:`\Phi^{\textrm{max}}`
.. |small_phi| replace:: :math:`\varphi`
.. |small_phi_s| replace:: :math:`\varphi_s`
.. |small_phi > 0| replace:: :math:`\varphi > 0`
.. |small_phi = 0| replace:: :math:`\varphi = 0`
.. |small_phi_max| replace:: :math:`\varphi^{\textrm{max}}`
.. |small_phi = 1/6| replace:: :math:`\varphi = \frac{1}{6}`
.. |small_phi = 1/10| replace:: :math:`\varphi = \frac{1}{10}`
.. |big_alpha| replace:: :math:`\mathcal{A}`
.. |big_alpha > 0| replace:: :math:`\mathcal{A} > 0`
.. |alpha| replace:: :math:`\alpha`
.. |alpha > 0| replace:: :math:`\alpha > 0`
.. |L1| replace:: :math:`L_1`
.. |S| replace:: :math:`S`
.. |A| replace:: :math:`A`
.. |A'| replace:: :math:`A'`
.. |B| replace:: :math:`B`
.. |C| replace:: :math:`C`
.. |D| replace:: :math:`D`
.. |E| replace:: :math:`E`
.. |F| replace:: :math:`F`
.. |AB| replace:: :math:`AB`
.. |AC| replace:: :math:`AC`
.. |AE| replace:: :math:`AE`
.. |BC| replace:: :math:`BC`
.. |CD| replace:: :math:`CD`
.. |DE| replace:: :math:`DE`
.. |FG| replace:: :math:`FG`
.. |ABC| replace:: :math:`ABC`
.. |BCD| replace:: :math:`BCD`
.. |CDE| replace:: :math:`CDE`
.. |DEF| replace:: :math:`DEF`
.. |A -> B| replace:: :math:`A \rightarrow B`
.. |(AB / DE) x (∅ / C)| replace:: :math:`\frac{AB}{DE} \times \frac{\varnothing}{C}`
.. |(A / CD) x (∅ / E)| replace:: :math:`\frac{A}{CD} \times \frac{\varnothing}{E}`
.. |(∅ / C) x (A / D)| replace:: :math:`\frac{\varnothing}{C} \times \frac{A}{D}`
.. |t| replace:: :math:`t`
.. |t-1| replace:: :math:`t-1`
.. |t+1| replace:: :math:`t+1`
.. |n+1| replace:: :math:`n+1`
.. |1,0,0| replace:: :math:`\{1,0,0\}`
.. |0,1,0| replace:: :math:`\{0,1,0\}`
.. |0,0,1| replace:: :math:`\{0,0,1\}`
.. |N_0 = 0, N_1 = 0, N_2 = 1| replace:: :math:`N_0 = 0, N_1 = 0, N_2 = 1`
.. |ith| replace:: :math:`i^{\textrm{th}}`
.. |jth| replace:: :math:`j^{\textrm{th}}`
.. |(i,j)| replace:: :math:`(i,j)`
.. |r| replace:: :math:`r`
.. |n| replace:: :math:`n`
.. |N| replace:: :math:`N`
.. |n x n| replace:: :math:`N \times N`
.. |2^n x 2^n| replace:: :math:`2^N \times 2^N`
.. |2^m x 2| replace:: :math:`2^m \times 2`
.. |m| replace:: :math:`m`
.. |i| replace:: :math:`i`
.. |j| replace:: :math:`j`
.. |i,jth| replace:: :math:`(i,j)^{\textrm{th}}`
.. |k| replace:: :math:`k`
.. |CM[i][j] = 1| replace:: :math:`[CM]_{i,j} = 1`
.. |CM[i][j] = 0| replace:: :math:`[CM]_{i,j} = 0`
.. |CM| replace:: :math:`CM`
.. |X| replace:: :math:`X`
.. |X_t-1| replace:: :math:`X_{t-1}`
.. |X_t-1 = {OR}| replace:: :math:`X_{t-1} = \{OR\}`
.. |X_t-1 = {OR = 1}| replace:: :math:`X_{t-1} = \{OR = 1\}`
.. |X_t-1 = {OR, AND}| replace:: :math:`X_{t-1} = \{OR, AND\}`
.. |X_t-1 = C| replace:: :math:`X_{t-1} = C`
.. |Y| replace:: :math:`Y`
.. |Y_t| replace:: :math:`Y_t`
.. |Y_t = {AND}| replace:: :math:`Y_t = \{AND\}`
.. |Y_t = {OR}| replace:: :math:`Y_t = \{OR\}`
.. |Y_t = {OR = 1}| replace:: :math:`Y_t = \{OR = 1\}`
.. |Y_t = {OR, AND}| replace:: :math:`Y_t = \{OR, AND\}`
.. |Y_t = {OR, AND = 10}| replace:: :math:`Y_t = \{OR, AND = 10\}`
.. |Y_t = D| replace:: :math:`Y_t = D`
.. |{OR, AND} -> {OR, AND}| replace:: :math:`\{OR, AND\} \rightarrow \{OR, AND\}`
.. |A' = S - {A}| replace:: :math:`A' = S - \{A\}`
.. |C(A)| replace:: :math:`C(A)`
.. |Pr(B | C(A), A=0) != Pr(B | C(A), A=1)| replace:: :math:`\Pr(B \mid C(A), A = 0) \neq \Pr(B \mid C(A), A = 1)`
""",
        # Constants
        r"""
.. |CAUSE| replace:: :const:`~pyphi.direction.Direction.CAUSE`
.. |EFFECT| replace:: :const:`~pyphi.direction.Direction.EFFECT`
.. |EPSILON| replace:: :const:`~pyphi.constants.EPSILON`
""",
        # Configuration
        r"""
.. |PICK_SMALLEST_PURVIEW| replace:: :attr:`~pyphi.conf.PyphiConfig.PICK_SMALLEST_PURVIEW`
.. |PARTITION_TYPE| replace:: :attr:`~pyphi.conf.PyphiConfig.PARTITION_TYPE`
.. |PRECISION| replace:: :attr:`~pyphi.conf.PyphiConfig.PRECISION`
""",
        # Modules
        r"""
.. |compute| replace:: :mod:`~pyphi.compute`
.. |compute.distance| replace:: :mod:`pyphi.compute.distance`
.. |compute.network| replace:: :mod:`pyphi.compute.network`
.. |compute.subsystem| replace:: :mod:`pyphi.compute.subsystem`

.. |models.subsystem| replace:: :mod:`pyphi.models.subsystem`
.. |models.mechanism| replace:: :mod:`pyphi.models.mechanism`
.. |models.cuts| replace:: :mod:`pyphi.models.partitions`

.. |network| replace:: :mod:`~pyphi.network`
.. |subsystem| replace:: :mod:`~pyphi.subsystem`
.. |convert| replace:: :mod:`~pyphi.convert`
.. |examples| replace:: :mod:`~pyphi.examples`
.. |node| replace:: :mod:`~pyphi.node`
.. |utils| replace:: :mod:`~pyphi.utils`
.. |validate| replace:: :mod:`~pyphi.validate`
.. |config| replace:: :mod:`~pyphi.config`
""",
        # Functions
        r"""
.. |compute.conceptual_info()| replace:: :func:`~pyphi.compute.subsystem.conceptual_info`
.. |compute.sia()| replace:: :func:`~pyphi.compute.subsystem.sia`
.. |compute.phi()| replace:: :func:`~pyphi.compute.subsystem.phi`

.. |compute.subsystems()| replace:: :func:`~pyphi.compute.network.subsystems`
.. |compute.possible_complexes()| replace:: :func:`~pyphi.compute.network.possible_complexes`
.. |compute.complexes()| replace:: :func:`~pyphi.compute.network.complexes`
.. |compute.all_complexes()| replace:: :func:`~pyphi.compute.network.all_complexes`
.. |compute.condensed()| replace:: :func:`~pyphi.compute.network.condensed`

.. |Subsystem.clear_caches()| replace:: :func:`~pyphi.subsystem.Subsystem.clear_caches`

.. |configure_logging()| replace:: :func:`~pyphi.config.configure_logging`

.. |le_index2state()| replace:: :func:`~pyphi.convert.le_index2state`
.. |be_index2state()| replace:: :func:`~pyphi.convert.be_index2state`
""",
        # Classes
        r"""
.. |Network| replace:: :class:`~pyphi.network.Network`

.. |Subsystem| replace:: :class:`~pyphi.subsystem.Subsystem`

.. |SystemIrreducibilityAnalysis| replace:: :class:`~pyphi.models.subsystem.SystemIrreducibilityAnalysis`
.. |SIA| replace:: :class:`~pyphi.models.subsystem.SystemIrreducibilityAnalysis`
.. |CauseEffectStructure| replace:: :class:`~pyphi.models.subsystem.CauseEffectStructure`

.. |Concept| replace:: :class:`~pyphi.models.mechanism.Concept`

.. |Cut| replace:: :class:`~pyphi.models.partitions.Cut`
.. |Cuts| replace:: :class:`~pyphi.models.partitions.Cut`
.. |Part| replace:: :class:`~pyphi.models.partitions.Part`
.. |Parts| replace:: :class:`~pyphi.models.partitions.Part`
.. |JointBipartition| replace:: :class:`~pyphi.models.partitions.JointBipartition`

.. |RepertoireIrreducibilityAnalysis| replace:: :class:`~pyphi.models.mechanism.RepertoireIrreducibilityAnalysis`
.. |MaximallyIrreducibleCauseOrEffect| replace:: :class:`~pyphi.models.mechanism.MaximallyIrreducibleCauseOrEffect`
.. |MICE| replace:: :class:`~pyphi.models.mechanism.MaximallyIrreducibleCauseOrEffect`
.. |MIC| replace:: :class:`~pyphi.models.mechanism.MaximallyIrreducibleCause`
.. |MIE| replace:: :class:`~pyphi.models.mechanism.MaximallyIrreducibleEffect`

.. |Node| replace:: :class:`~pyphi.node.Node`
.. |Nodes| replace:: :class:`~pyphi.node.Node`

.. |Transition| replace:: :class:`~pyphi.actual.Transition`

.. |AcSystemIrreducibilityAnalysis| replace:: :class:`~pyphi.models.actual_causation.AcSystemIrreducibilityAnalysis`
.. |AcRepertoireIrreducibilityAnalysis| replace:: :class:`~pyphi.models.actual_causation.AcRepertoireIrreducibilityAnalysis`
.. |DirectedAccount| replace:: :class:`~pyphi.models.actual_causation.DirectedAccount`
.. |Account| replace:: :class:`~pyphi.models.actual_causation.Account`
.. |Event| replace:: :class:`~pyphi.models.actual_causation.Event`
.. |CausalLink| replace:: :class:`~pyphi.models.actual_causation.CausalLink`
.. |CausalLinks| replace:: :class:`~pyphi.models.actual_causation.CausalLink`

.. |ConditionallyDependentError| replace:: :class:`~pyphi.exceptions.ConditionallyDependentError`

.. |NodeLabels| replace:: :class:`~pyphi.labels.NodeLabels`
""",
        # Attributes
        r"""
.. |Subsystem.cm| replace:: :attr:`~pyphi.subsystem.Subsystem.cm`
""",
        # Methods
        r"""
.. |Subsystem.concept()| replace:: :meth:`~pyphi.subsystem.Subsystem.concept`
.. |Subsystem.mic()| replace:: :meth:`~pyphi.subsystem.Subsystem.mic`
.. |Subsystem.mie()| replace:: :meth:`~pyphi.subsystem.Subsystem.mie`
.. |Subsystem.expand_repertoire()| replace:: :meth:`~pyphi.subsystem.Subsystem.expand_repertoire`
.. |expand_repertoire()| replace:: :meth:`~pyphi.subsystem.Subsystem.expand_repertoire`
.. |Subsystem.find_mip()| replace:: :meth:`~pyphi.subsystem.Subsystem.find_mip`
.. |find_mip()| replace:: :meth:`~pyphi.subsystem.Subsystem.find_mip`
.. |Subsystem.find_mice()| replace:: :meth:`~pyphi.subsystem.Subsystem.find_mice`
.. |find_mice()| replace:: :meth:`~pyphi.subsystem.Subsystem.find_mice`
""",
    ]
)

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

# --- API reference ----------------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_use_rtype = False
napoleon_use_ivar = True

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
html_logo = "_static/pyphi-logo-text-776x196.png"
html_favicon = "_static/phi_144x144.png"
html_theme_options = {
    "github_url": "https://github.com/wmayner/pyphi",
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
}
