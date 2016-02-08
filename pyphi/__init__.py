#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

#    _______
#   |__   __|
#  ____| |____
# |  __   __  |     ___         ___   __    _
# | |  | |  | |    / _ \ __ __ / _ \ / /   (_)
# | |__| |__| |   / ___// // // ___// _ \ / /
# |____   ____|  /_/    \_, //_/   /_//_//_/
#    __| |__           /___/
#   |_______|

"""
=====
PyPhi
=====

PyPhi is a Python library for computing integrated information.

See the documentation for the |examples| module for information on how to use
it.

To report issues, please use the issue tracker on the `GitHub repository
<https://github.com/wmayner/pyphi>`_. Bug reports and pull requests are
welcome.


Usage
~~~~~

The |Network| object is the main object on which computations are performed. It
represents the network of interest.

The |Subsystem| object is the secondary object; it represents a subsystem of a
network. |big_phi| is defined on subsystems.

The |compute| module is the main entry-point for the library. It contains
methods for calculating concepts, constellations, complexes, etc. See its
documentation for details.


Configuration (optional)
~~~~~~~~~~~~~~~~~~~~~~~~

There are several module-level options that control aspects of the computation.

These are loaded from a YAML configuration file, ``pyphi_config.yml``. **This
file must be in the directory where PyPhi is run**. If there is no such file,
the default configuration will be used.

You can download an example configuration file `here
<https://raw.githubusercontent.com/wmayner/pyphi/master/pyphi_config.yml>`_.

See the documentation for the |config| module for a description of the options
and their defaults.
"""

from .__about__ import *
from . import (config, constants, convert, db, examples, jsonify, macro,
               models, network, node, subsystem, utils, validate)
from .network import Network
from .subsystem import Subsystem

__all__ = ['Network', 'Subsystem', 'config', 'constants', 'convert', 'db',
           'examples', 'jsonify', 'macro', 'models', 'network', 'node',
           'subsystem', 'utils', 'validate']

import logging
import logging.config


# Configure logging module.
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': config.LOGGING_CONFIG['format']
        }
    },
    'handlers': {
        'file': {
            'level': config.LOGGING_CONFIG['file']['level'],
            'class': 'logging.FileHandler',
            'filename': config.LOGGING_CONFIG['file']['filename'],
            'formatter': 'standard',
        },
        'stdout': {
            'level': config.LOGGING_CONFIG['stdout']['level'],
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': [h for h in ['file', 'stdout'] if
                     config.LOGGING_CONFIG[h]['enabled']]
    }
})

log = logging.getLogger(__name__)

# Log the currently loaded version and configuration.
if config.LOG_CONFIG_ON_IMPORT:
    log.info('PyPhi version {}'.format(__version__))
    if config.file_loaded:
        log.info('Loaded configuration from '
                 '`./{}`'.format(config.PYPHI_CONFIG_FILENAME))
    else:
        log.info('Using default configuration (no config file provided)')
    log.info('Current PyPhi configuration:\n'
             '{}'.format(config.get_config_string()))
