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

If you use this software in your research, please cite the paper:

    Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G.
    (2018). PyPhi: A toolbox for integrated information theory.
    PLOS Computational Biology 14(7): e1006343.
    https://doi.org/10.1371/journal.pcbi.1006343

Online documentation is available at `<https://pyphi.readthedocs.io/>`_.

For general discussion, you are welcome to join the `pyphi-users group
<https://groups.google.com/forum/#!forum/pyphi-users>`_.

To report issues, please use the issue tracker on the `GitHub repository
<https://github.com/wmayner/pyphi>`_. Bug reports and pull requests are
welcome.


Usage
~~~~~

The |Substrate| object is the main object on which computations are performed. It
represents the substrate of interest.

The |System| object is the secondary object; it represents a system of a
substrate. |big_phi| is a function of systems.

The |compute| module is the main entry-point for the library. It contains
methods for calculating concepts, cause-effect structures, complexes, etc. See
its documentation for details.


Configuration (optional)
~~~~~~~~~~~~~~~~~~~~~~~~

There are several package-level options that control aspects of the
computation.

These are loaded from a YAML configuration file, ``pyphi_config.yml``. **This
file must be in the directory where PyPhi is run**. If there is no such file,
the default configuration will be used.

You can download an example configuration file `here
<https://raw.githubusercontent.com/wmayner/pyphi/master/pyphi_config.yml>`_.

See the documentation for the |config| module for a description of the options
and their defaults.
"""

import importlib
import os
import pkgutil
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
from .conf import config
from .conf import iit3
from .conf import iit4_2023
from .conf import iit4_2026
from .core.tpm import FactoredTPM as FactoredTPM
from .core.tpm import JointDistribution as JointDistribution
from .core.tpm.joint_distribution import JointTPM
from .direction import Direction
from .serialize import load
from .serialize import save
from .substrate import Substrate
from .system import System

# Names of the depth-1 submodules, listed (not imported). Public submodules are
# available as attributes via the lazy ``__getattr__`` below.
_SUBMODULE_NAMES = frozenset(name for _, name, _ in pkgutil.iter_modules(__path__))


def __getattr__(name: str) -> ModuleType:
    """Lazily import a public submodule on first attribute access (PEP 562).

    Keeps ``pyphi.examples``, ``pyphi.compute``, and the like working after a
    bare ``import pyphi`` without importing the whole package eagerly, so
    ``import pyphi`` is fast and is not broken by an unrelated submodule that
    fails to import.
    """
    if name in _SUBMODULE_NAMES and not name.startswith("_"):
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Direction",
    "FactoredTPM",
    "JointDistribution",
    "JointTPM",
    "Substrate",
    "System",
    "Transition",
    "TransitionSystem",
    "config",
    "iit3",
    "iit4_2023",
    "iit4_2026",
    "load",
    "save",
] + [name for name in sorted(_SUBMODULE_NAMES) if not name.startswith("_")]


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
  - Set `WELCOME_OFF: true` in your `pyphi_config.yml` file, or
  - Set the environment variable PYPHI_WELCOME_OFF to any value in your shell:
        export PYPHI_WELCOME_OFF='yes'
"""
    )
