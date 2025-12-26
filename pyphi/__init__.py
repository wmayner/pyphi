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

The |Network| object is the main object on which computations are performed. It
represents the network of interest.

The |Subsystem| object is the secondary object; it represents a subsystem of a
network. |big_phi| is a function of subsystems.

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

# Lift main interfaces to top-level namespace
from .actual import Transition
from .conf import config
from .direction import Direction
from .network import Network
from .subsystem import Subsystem
from .tpm import ExplicitTPM

# Skip modules that require optional dependencies
_skip_import = ["visualize", "graphs"]


def _import_submodules(package, recursive=True):
    """Import all submodules of a module, recursively, including subpackages.

    Args:
        package (str | module): package to traverse (name or module object).

    Keyword Args:
        recursive (bool): Whether to import submodules recursively.

    Returns:
        dict[str, types.ModuleType]: A dictionary mapping names to submodules.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        if name not in _skip_import:
            full_name = package.__name__ + "." + name
            results[full_name] = importlib.import_module(full_name)
            if recursive and is_pkg:
                results.update(_import_submodules(full_name))
    return results


# Recursively import all submodules.
# This is necessary in order to populate the Registries, since modules must be
# imported for the registration decorators to be executed.
_submodules = _import_submodules(__name__)

# Populate __all__ with public modules of depth 1
_submodule_names = set([name.split(".")[1] for name in _submodules.keys()])
__all__ = [
    "config",
    "Direction",
    "ExplicitTPM",
    "Network",
    "Subsystem",
    "Transition",
] + [name for name in _submodule_names if not name.startswith("_")]


if not (config.WELCOME_OFF or "PYPHI_WELCOME_OFF" in os.environ):
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
