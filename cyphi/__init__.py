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
CyPhi
=====

CyPhi is a Python library for computing integrated information.

See the documentation for :mod:`cyphi.examples` for information on how to use
it.


Configuration
~~~~~~~~~~~~~

There are several module-level options that control aspects of the computation.
These are loaded from a YAML configuration file, ``cyphi_config.yml``, which
must be in the directory where CyPhi is run. See the documentation for
:mod:`cyphi.constants` for a description of the options and their defaults.
"""

__title__ = 'cyphi'
__version__ = '0.1.1'
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
import logging.config


# Default user-configured logging options.
logging_config = {
    'format': "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    # `level` can be "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
    'file': {
        'enabled': True,
        'level': "INFO",
        'filename': "cyphi.log"
    },
    'stdout': {
        'enabled': True,
        'level': "INFO",
    }
}


# Load user's logging configuration if available.
if os.path.exists(constants.CYPHI_CONFIG_FILE):
    with open(constants.CYPHI_CONFIG_FILE) as f:
        config = yaml.load(f)
        logging_config = config['LOGGING_CONFIG']


# Configure logging module.
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': logging_config['format']
        }
    },
    'handlers': {
        'file': {
            'level': logging_config['file']['level'],
            'class': 'logging.FileHandler',
            'filename': logging_config['file']['filename'],
            'formatter': 'standard',
        },
        'stdout': {
            'level': logging_config['stdout']['level'],
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': [h for h in ['file', 'stdout'] if
                     logging_config[h]['enabled']]
    }
})
