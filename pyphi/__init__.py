# __init__.py

#      _|_|_|
#        _|
#  _|_|_|_|_|_|_|    _|_|_|_|    _|      _|  _|_|_|_|    _|      _|  _|_|_|_|_|
#  _|    _|    _|    _|      _|  _|      _|  _|      _|  _|      _|      _|
#  _|    _|    _|    _|_|_|_|_|  _|_|_|_|_|  _|_|_|_|_|  _|_|_|_|_|      _|
#  _|    _|    _|    _|              _|      _|          _|      _|      _|
#  _|_|_|_|_|_|_|    _|              _|      _|          _|      _|  _|_|_|_|_|
#        _|
#      _|_|_|

"""
=====
PyPhi
=====

PyPhi is a Python library for computing integrated information.

Usage
~~~~~

A :class:`~pyphi.substrate.Substrate` is the causal model: a set of units and
the probability of each unit's next state given the current state of all of
them. A :class:`~pyphi.system.System` is a candidate subset of a substrate's
units in a state; integrated information is a property of systems.

:func:`~pyphi.analyze.analyze` is the entry point. Given a substrate and a
state it returns an :class:`~pyphi.analyze.Analysis` whose ``phi`` is φₛ, the
system integrated information (whether the system exists as one whole), whose
``ces`` is the Φ-structure the system specifies, and whose ``big_phi`` is Φ,
the structure integrated information (how much structure it specifies). The
two quantities are different and must not be reported as one another. Under
IIT 3.0, ``phi`` is that formalism's Φ and ``big_phi`` is not defined.

The default formalism is IIT 4.0 (2026), which includes the
intrinsic-information requirement. Select another per call with
``formalism="IIT_4_0_2023"`` or ``"IIT_3_0"``, or apply a preset
(:data:`pyphi.iit3`, :data:`pyphi.iit4_2023`, :data:`pyphi.iit4_2026`) with
``pyphi.config.override``.

:meth:`Substrate.complexes <pyphi.substrate.Substrate.complexes>` finds the
complexes of a substrate in a state. :func:`~pyphi.sweep.sweep` runs one
computation over many states, subsets, and formalisms.
:func:`~pyphi.cost.estimate_analysis` counts the work of an analysis before
it runs. :func:`~pyphi.serialize.save` and :func:`~pyphi.serialize.load`
persist results. To search across macro grains, pass ``grains=True`` to
:func:`~pyphi.analyze.analyze` or use :mod:`pyphi.macro`.

Configuration
~~~~~~~~~~~~~

Options are read from a ``pyphi_config.yml`` in the working directory, with
the layers ``formalism``, ``infrastructure``, and ``numerics``; without one,
the defaults apply. At runtime, assign an option (``pyphi.config.precision =
6``) or scope a change with ``pyphi.config.override(...)``. The shipped
defaults are in the repository's `pyphi_config.yml
<https://raw.githubusercontent.com/wmayner/pyphi/main/pyphi_config.yml>`_;
:mod:`pyphi.conf` documents every option.

Citation and support
~~~~~~~~~~~~~~~~~~~~

If you use this software in your research, please cite the software paper,
and the paper of the formalism your results were computed under:

    Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G.
    (2018). PyPhi: A toolbox for integrated information theory.
    PLOS Computational Biology 14(7): e1006343.
    https://doi.org/10.1371/journal.pcbi.1006343

Online documentation is available at `<https://pyphi.readthedocs.io/>`_,
with BibTeX entries on its citing page.

For general discussion, you are welcome to join the `pyphi-users group
<https://groups.google.com/forum/#!forum/pyphi-users>`_.

To report issues, please use the issue tracker on the `GitHub repository
<https://github.com/wmayner/pyphi>`_. Bug reports and pull requests are
welcome.
"""

import importlib
import logging
import os
import pkgutil
import sys
from types import ModuleType

# Populate the registries. Each built-in measure, partition scheme,
# tie-resolution strategy, relation computation, distinction normalization,
# and formalism is registered by a decorator (or an explicit ``.register``
# call) that runs when its defining module is imported. Importing these
# modules makes every built-in registrant available. Third-party plugins
# register when the user imports them.
import pyphi.measures.ces
import pyphi.measures.distribution
import pyphi.models.state_specification  # noqa: F401

from . import formalism  # noqa: F401
from . import partition  # noqa: F401
from . import relations  # noqa: F401
from . import resolve_ties  # noqa: F401

# Lift main interfaces to the top-level namespace.
from .actual import Transition
from .actual import TransitionSystem
from .analyze import Analysis as Analysis
from .analyze import analyze as analyze
from .conf import config
from .conf import iit3
from .conf import iit4_2023
from .conf import iit4_2026
from .core.tpm import FactoredTPM as FactoredTPM
from .core.tpm import JointTPM as JointTPM
from .cost import AnalysisEstimate as AnalysisEstimate
from .cost import estimate_analysis as estimate_analysis
from .direction import Direction
from .estimate import estimate_substrate as estimate_substrate
from .estimate import phi_posterior as phi_posterior
from .landscape import LandscapeSection as LandscapeSection
from .landscape import Perturbation as Perturbation
from .landscape import landscape_section as landscape_section
from .landscape import perturb as perturb
from .landscape import weight_axis as weight_axis
from .log import enable_logging as enable_logging
from .optimize import OptimizationResult as OptimizationResult
from .optimize import optimize as optimize
from .optimize import weight_axes as weight_axes
from .serialize import load
from .serialize import save
from .substrate import Substrate
from .sweep import SweepResult as SweepResult
from .sweep import sweep as sweep
from .system import System

# The conf bootstrap applies ``pyphi_config.yml`` before the formalism
# registry exists, so cross-field constraints cannot be checked at load time.
# Re-validate here, now that formalisms are registered, so an invalid YAML
# config fails at import rather than at compute time.
if config.infrastructure.validate_config:
    from .conf.constraints import check_config_constraints

    check_config_constraints(config)

# Silent by default: a library attaches only a NullHandler to its own logger
# and leaves real handlers to the application (or pyphi.enable_logging).
logging.getLogger("pyphi").addHandler(logging.NullHandler())

# Names of the depth-1 submodules, listed (not imported). Public submodules are
# available as attributes via the lazy ``__getattr__`` below.
_SUBMODULE_NAMES = frozenset(name for _, name, _ in pkgutil.iter_modules(__path__))


def __getattr__(name: str) -> ModuleType:
    """Lazily import a public submodule on first attribute access (PEP 562).

    Keeps ``pyphi.examples``, ``pyphi.macro``, and the like working after a
    bare ``import pyphi`` without importing the whole package eagerly, so
    ``import pyphi`` is fast and is not broken by an unrelated submodule that
    fails to import.
    """
    if name in _SUBMODULE_NAMES and not name.startswith("_"):
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Submodules that require optional dependencies at import time; kept out of
# ``__all__`` so ``from pyphi import *`` works on a base install. They remain
# importable as attributes and are listed by ``dir(pyphi)``.
_OPTIONAL_DEP_SUBMODULES = frozenset({"visualize", "mcp"})

# Submodule names shadowed at module scope by a top-level function of the
# same name (``pyphi.analyze``, ``pyphi.optimize``, ``pyphi.sweep`` are the
# ``analyze``/``optimize``/``sweep`` functions, not the submodules), so
# ``__all__`` documents the function, not the submodule, for these names.
_SHADOWED_BY_FUNCTION = frozenset({"analyze", "optimize", "sweep"})

# Submodule names that collide with a standard-library module name
# (``pyphi.types``, ``pyphi.warnings``): excluded from ``__all__`` so
# ``from pyphi import *`` cannot rebind a caller's ``types``/``warnings``
# import to PyPhi's own submodule. They remain importable as attributes
# (``pyphi.types``, ``pyphi.warnings``) and are listed by ``dir(pyphi)``.
_STDLIB_NAME_COLLISIONS = frozenset(_SUBMODULE_NAMES) & sys.stdlib_module_names

__all__ = [
    "Analysis",
    "AnalysisEstimate",
    "Direction",
    "FactoredTPM",
    "JointTPM",
    "LandscapeSection",
    "OptimizationResult",
    "Perturbation",
    "Substrate",
    "SweepResult",
    "System",
    "Transition",
    "TransitionSystem",
    "analyze",
    "config",
    "enable_logging",
    "estimate_analysis",
    "estimate_substrate",
    "iit3",
    "iit4_2023",
    "iit4_2026",
    "landscape_section",
    "load",
    "optimize",
    "perturb",
    "phi_posterior",
    "save",
    "sweep",
    "weight_axes",
    "weight_axis",
] + [
    name
    for name in sorted(_SUBMODULE_NAMES)
    if not name.startswith("_")
    and name not in _OPTIONAL_DEP_SUBMODULES
    and name not in _SHADOWED_BY_FUNCTION
    and name not in _STDLIB_NAME_COLLISIONS
]


def __dir__() -> list[str]:
    """Include lazily importable submodules in ``dir(pyphi)``."""
    return sorted(
        set(globals())
        | set(__all__)
        | {name for name in _SUBMODULE_NAMES if not name.startswith("_")}
    )


# Written to stderr, not stdout: stdout is the MCP server's JSON-RPC channel,
# and a banner emitted there lands in the protocol stream ahead of the first
# message.
if not (config.infrastructure.welcome_off or "PYPHI_WELCOME_OFF" in os.environ):
    print(
        """
Welcome to PyPhi!

If you use PyPhi in your research, please cite the paper:

  Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G.
  (2018). PyPhi: A toolbox for integrated information theory.
  PLOS Computational Biology 14(7): e1006343.
  https://doi.org/10.1371/journal.pcbi.1006343

Documentation is available online (or with the built-in `help()` function):
  https://pyphi.readthedocs.io

To report issues, please use the issue tracker on the GitHub repository:
  https://github.com/wmayner/pyphi

For general discussion, you are welcome to join the pyphi-users group:
  https://groups.google.com/forum/#!forum/pyphi-users

To suppress this message, either:
  - Set `welcome_off: true` under the `infrastructure:` section of your
    `pyphi_config.yml` file, or
  - Set the environment variable PYPHI_WELCOME_OFF to any value in your shell:
        export PYPHI_WELCOME_OFF='yes'
""",
        file=sys.stderr,
    )


#: Environment variables whose presence indicates PyPhi was imported by an AI
#: coding agent rather than a person. ``PYPHI_AGENT`` lets any harness opt in.
_AGENT_ENV_VARS = ("CLAUDECODE", "PYPHI_AGENT")


def _running_under_agent() -> bool:
    return any(name in os.environ for name in _AGENT_ENV_VARS)


# Printed to stderr, never stdout: stdout is the MCP server's JSON-RPC channel,
# and anything written there corrupts the protocol stream.
if (
    _running_under_agent()
    and not config.infrastructure.agent_note_off
    and "PYPHI_AGENT_NOTE_OFF" not in os.environ
):
    print(
        """\
PyPhi — notes for AI assistants

  Where the PyPhi MCP server is connected, use its tools for exploration and
  for interpreting results: they report which formalism produced each number,
  refuse runs too large to finish, and keep φ_s and Φ distinct. The server
  holds results only in memory, so anything that has to be reproducible belongs
  in a script — where these same facts still apply.

  φ_s and Φ are different quantities under IIT 4.0. `analyze(...).phi` is φ_s,
  the system integrated information, which decides whether the system exists
  as one whole. `.big_phi` is Φ, the structure integrated information: the sum
  of φ over the Φ-structure's distinctions and relations. Reporting one as the
  other is the most common mistake made with this library.

  States are little-endian — the first node is the least-significant bit.

  Read the bundled reference before interpreting any result: through the MCP
  server's get_iit_reference("theory") and ("gotchas") where it is connected,
  or directly with

      python -c "from pyphi.mcp import content; print(content.load('gotchas'))"

  Suppress this note with PYPHI_AGENT_NOTE_OFF=1, or `agent_note_off: true`
  under `infrastructure:` in pyphi_config.yml.
""",
        file=sys.stderr,
    )
