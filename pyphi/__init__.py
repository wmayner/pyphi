#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

See the documentation for the |examples| module for information on how to use
it.

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

from .__about__ import *  # pylint: disable=wildcard-import

# Initialize config object
from .conf import config

from .direction import Direction
from . import (actual, constants, convert, db, examples, jsonify, macro,
               models, network, node, subsystem, utils, validate)
from .network import Network
from .subsystem import Subsystem
from .actual import Transition


__all__ = ['Network', 'Subsystem', 'actual', 'config', 'constants', 'convert',
           'db', 'examples', 'jsonify', 'macro', 'models', 'network', 'node',
           'subsystem', 'utils', 'validate']
