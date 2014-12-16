#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    _______
#   |__   __|
#  ____| |____
# |  __   __  |    _____         ___    __     _
# | |  | |  | |   / ___/ __ __  / _ \  / /    (_)
# | |__| |__| |  / /__  / // / / ___/ / _ \  / /
# |____   ____|  \___/  \_, / /_/    /_//_/ /_/
#    __| |__           /___/
#   |_______|

"""
=====
PyPhi
=====

PyPhi is a Python library for computing integrated information.

See the documentation for :mod:`pyphi.examples` for information on how to use
it.

To report issues, please use the issue tracker on the `GitHub repository
<https://github.com/wmayner/pyphi>`_. Bug reports and pull requests are
welcome.

Configuration (optional)
~~~~~~~~~~~~~~~~~~~~~~~~

There are several module-level options that control aspects of the computation.

These are loaded from a YAML configuration file, ``pyphi_config.yml``. **This
file must be in the directory where PyPhi is run**. If there is no such file,
the default configuration will be used.

You can download an example configuration file `here
<https://raw.githubusercontent.com/wmayner/pyphi/master/pyphi_config.yml>`_.

See the documentation for :mod:`pyphi.constants` for a description of the
options and their defaults.
"""

__title__ = 'pyphi'
__version__ = '0.1.6'
__description__ = 'Python library for computing integrated information.',
__author__ = 'Will Mayner'
__author_email__ = 'wmayner@gmail.com'
__author_website__ = 'http://willmayner.com'
__copyright__ = 'Copyright 2014 Will Mayner'


from .network import Network
from .subsystem import Subsystem
from . import compute, constants, db, examples

import os
import yaml
import logging
import logging.config


# Configure logging module.
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': constants.LOGGING_CONFIG['format']
        }
    },
    'handlers': {
        'file': {
            'level': constants.LOGGING_CONFIG['file']['level'],
            'class': 'logging.FileHandler',
            'filename': constants.LOGGING_CONFIG['file']['filename'],
            'formatter': 'standard',
        },
        'stdout': {
            'level': constants.LOGGING_CONFIG['stdout']['level'],
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': [h for h in ['file', 'stdout'] if
                     constants.LOGGING_CONFIG[h]['enabled']]
    }
})

log = logging.getLogger()

# Log the currently loaded configuration.
if constants.config_file_was_loaded:
    log.info('Loaded configuration from ' + constants.PYPHI_CONFIG_FILE)
else:
    log.info("Using default configuration (no config file provided)")
log.info('Current PyPhi configuration:\n' + constants.get_config_string())
