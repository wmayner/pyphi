#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# config.py
"""
The configuration is loaded upon import from a YAML file in the directory where
PyPhi is run: ``pyphi_config.yml``. If no file is found, the default
configuration is used.

The various options are listed here with their defaults.


System resources
~~~~~~~~~~~~~~~~

These settings control how much processing power and memory is available for
PyPhi to use. The default values may not be appropriate for your use-case or
machine, so **please check these settings before running anything**. Otherwise,
there is a risk that simulations might crash (potentially after running for a
long time!), resulting in data loss.

- PyPhi employs several in-memory LRU-caches to speed up computation. However,
  these can quickly use up all the memory on a system; to avoid thrashing, this
  options limits the percentage of a system's RAM that the LRU caches can use.
  **NOTE:** If you are running multiple instances of PyPhi on the same machine,
  then the number of PyPhi instances multiplied by this number should be less
  than ~80.

    >>> import pyphi
    >>> pyphi.config.MAXIMUM_CACHE_MEMORY_PERCENTAGE
    50

- Control whether system cuts are evaluated in parallel, which requires more
  memory. If cuts are evaluated sequentially, only two |BigMip| instances need
  to be in memory at once.

    >>> pyphi.config.PARALLEL_CUT_EVALUATION
    True

- Control the number of CPU cores to evaluate unidirectional cuts. Negative
  numbers count backwards from the total number of available cores, with ``-1``
  meaning "use all available cores".

    >>> pyphi.config.NUMBER_OF_CORES
    -1


Caching
~~~~~~~

PyPhi is equipped with a transparent caching system for the |BigMip| and
|Concept| objects, which stores them as they are computed to avoid having to
recompute them later. This makes it easy to play around interactively with the
program, or to accumulate results with minimal effort. For larger projects,
however, it is recommended that you manage the results explicitly, rather than
relying on the cache. For this reason it is disabled by default.

- Control whether precomputed results are stored and read from a database or
  from a local filesystem-based cache in the current directory. Set this to
  'fs' for the filesystem, 'db' for the database. Caching results on the
  filesystem is the easiest to use but least robust caching system. Caching
  results in a database is more robust and allows for caching individual
  concepts, but requires installing MongoDB.

    >>> pyphi.config.CACHING_BACKEND
    'fs'

- Control whether |BigMip| objects are cached and automatically retreived.

    >>> pyphi.config.CACHE_BIGMIPS
    False

- Control whether |Concept| objects are cached and automatically retrieved.

    >>> pyphi.config.CACHE_CONCEPTS
    False

.. note::
    Concept caching only has an effect when a database is used as the the
    caching backend.

- If the caching backend is set to use the filesystem, the cache will be stored
  in this directory. This directory can be copied and moved around if you want
  to reuse results _e.g._ on a another computer, but it must be in the same
  directory from which PyPhi is being run.

    >>> pyphi.config.FS_CACHE_DIRECTORY
    '__pyphi_cache__'

- Control how much caching information is printed. Takes a value between 0 and
  11. Note that printing during a loop iteration can slow down the loop
  considerably.

    >>> pyphi.config.FS_CACHE_VERBOSITY
    0

- Set the configuration for the MongoDB database backend. This only has an
  effect if the caching backend is set to use the database.

    >>> pyphi.config.MONGODB_CONFIG['host']
    'localhost'
    >>> pyphi.config.MONGODB_CONFIG['port']
    27017
    >>> pyphi.config.MONGODB_CONFIG['database_name']
    'pyphi'
    >>> pyphi.config.MONGODB_CONFIG['collection_name']
    'test'

- Control whether TPMs should be normalized as part of concept normalization.
  TPM normalization increases the chances that a precomputed concept can be
  used again, but is expensive.

    >>> pyphi.config.NORMALIZE_TPMS
    True


Logging
~~~~~~~

These are the settings for PyPhi logging. You can control the format of the
logs and the name of the log file. Logs can be written to standard output, a
file, both, or none. See the `documentation on Python's logger
<https://docs.python.org/3.4/library/logging.html>`_ for more information.

- Control whether logs are written to a file.

    >>> pyphi.config.LOGGING_CONFIG['file']['enabled']
    True

- Control the name of the logfile.

    >>> pyphi.config.LOGGING_CONFIG['file']['filename']
    'pyphi.log'

- Control the concern level of file logging. Can be one of ``'DEBUG'``,
  ``'INFO'``, ``'WARNING'``, ``'ERROR'``, or ``'CRITICAL'``.

    >>> pyphi.config.LOGGING_CONFIG['file']['level']
    'INFO'

- Control whether logs are written to standard output.

    >>> pyphi.config.LOGGING_CONFIG['stdout']['enabled']
    True

- Control the concern level of standard output logging. Same possible values as
  file logging.

    >>> pyphi.config.LOGGING_CONFIG['stdout']['level']
    'INFO'

- If parallel computation is enabled, it will have its own, separate messages,
  which are always sent to standard output. This setting controls their
  verbosity, an integer from 0 to 100.

    >>> pyphi.config.PARALLEL_VERBOSITY
    20


Network state validation
~~~~~~~~~~~~~~~~~~~~~~~~

- Control whether PyPhi checks if the network's current state is possible,
  given the TPM and the past state. Nodes with no inputs are assumed to be able
  to have any state, and will not constrain the possible states.

    >>> pyphi.config.VALIDATE_NETWORK_STATE
    False


Theoretical concerns
~~~~~~~~~~~~~~~~~~~~

- If set to ``True``, this defines the Phi value of subsystems containing only
  a single node with a self-loop to be ``0.5``. If set to False, their
  |big_phi| will be actually be computed (to be zero, in this implementation).

    >>> pyphi.config.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI
    False


Numerical precision
~~~~~~~~~~~~~~~~~~~

- Computations in PyPhi rely on finding the Earth Mover's Distance. This is
  done via an external C++ library that uses flow-optimization to find a good
  approximation of the EMD. Consequently, systems with zero |big_phi| will
  sometimes be computed to have a small but non-zero amount. This setting
  controls the number of decimal places to which PyPhi will consider EMD
  calculations accurate. Values of |big_phi| lower than ``10e-PRECISION`` will
  be considered insignificant and treated as zero. The default value is about
  as accurate as the EMD computations get.

    >>> pyphi.config.PRECISION
    6

"""

import pprint
import os
import sys
import yaml


# TODO: document mongo config
# Defaults for configurable constants.
config = {
    # Controls whether cuts are evaluated in parallel, which requires more
    # memory. If cuts are evaluated sequentially, only two BigMips need to be
    # in memory at a time.
    'PARALLEL_CUT_EVALUATION': True,
    # The number of CPU cores to use in parallel cut evaluation. -1 means all
    # available cores, -2 means all but one available cores, etc.
    'NUMBER_OF_CORES': -1,
    # The verbosity of parallel computation (integer from 0 to 100). See
    # documentation for `joblib.Parallel`.
    'PARALLEL_VERBOSITY': 20,
    # Controls whether the concept caching system is used.
    'CACHE_CONCEPTS': False,
    # Controls whether BigMips are cached and retreived.
    'CACHE_BIGMIPS': False,
    # Controls whether TPMs should be normalized as part of concept
    # normalization. TPM normalization increases the chances that a precomputed
    # concept can be used again, but is expensive.
    'NORMALIZE_TPMS': True,
    # The maximum percentage of RAM that PyPhi should use for caching.
    'MAXIMUM_CACHE_MEMORY_PERCENTAGE': 50,
    # The caching system to use. "fs" means cache results in a subdirectory of
    # the current directory; "db" means connect to a database and store the
    # results there.
    'CACHING_BACKEND': 'fs',
    # Directory for the persistent joblib Memory cache.
    'FS_CACHE_DIRECTORY': '__pyphi_cache__',
    # joblib.Memory verbosity.
    'FS_CACHE_VERBOSITY': 0,
    # MongoDB configuration.
    'MONGODB_CONFIG': {
        'host': 'localhost',
        'port': 27017,
        'database_name': 'pyphi',
        'collection_name': 'cache'
    },
    # These are the settings for PyPhi logging.
    'LOGGING_CONFIG': {
        'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        # `level` can be "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
        'file': {
            'enabled': True,
            'level': 'INFO',
            'filename': 'pyphi.log'
        },
        'stdout': {
            'enabled': True,
            'level': 'INFO'
        }
    },
    # Controls whether network states are validated upon network creation.
    'VALIDATE_NETWORK_STATE': False,
    # In some applications of this library, the user may prefer to define
    # single-node subsystems as having 0.5 Phi.
    'SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI': False,
    # The number of decimal points to which phi values are considered accurate
    'PRECISION': 6
}


# Get a reference to this module's dictionary..
this_module = sys.modules[__name__]


def load_config(config):
    """Load a configuration."""
    this_module.__dict__.update(config)


def get_config_string():
    """Return a string representation of the currently loaded configuration."""
    return pprint.pformat(config, indent=2)


def print_config():
    """Print the current configuration."""
    print('Current PyPhi configuration:\n', get_config_string())


# The name of the file to load configuration data from.
PYPHI_CONFIG_FILENAME = 'pyphi_config.yml'

# Try to load the config file, falling back to the default configuration.
file_loaded = False
if os.path.exists(PYPHI_CONFIG_FILENAME):
    with open(PYPHI_CONFIG_FILENAME) as f:
        config.update(yaml.load(f))
        file_loaded = True

# Load the configuration.
load_config(config)
