#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# config.py

"""
The configuration is loaded upon import from a YAML file in the directory where
PyPhi is run: ``pyphi_config.yml``. If no file is found, the default
configuration is used.

The various options are listed here with their defaults

    >>> import pyphi
    >>> defaults = pyphi.config.DEFAULTS

It is also possible to manually load a YAML configuration file within your
script:

    >>> pyphi.config.load_config_file('pyphi_config.yml')

Or load a dictionary of configuration values:

    >>> pyphi.config.load_config_dict({'SOME_CONFIG': 'value'})


Theoretical approximations
~~~~~~~~~~~~~~~~~~~~~~~~~~

This section deals with assumptions that speed up computation at the cost of
theoretical accuracy.

- ``pyphi.config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS``:
  In certain cases, making a cut can actually cause a previously reducible
  concept to become a proper, irreducible concept. Assuming this can never
  happen can increase performance significantly, however the obtained results
  are not strictly accurate.

    >>> defaults['ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS']
    False

- ``pyphi.config.CUT_ONE_APPROXIMATION``:
  When determining the MIP for |big_phi|, this restricts the set of system cuts
  that are considered to only those that cut the inputs or outputs of a single
  node. This restricted set of cuts scales linearly with the size of the
  system; the full set of all possible bipartitions scales exponentially. This
  approximation is more likely to give theoretically accurate results with
  modular, sparsely-connected, or homogeneous networks.

    >>> defaults['CUT_ONE_APPROXIMATION']
    False

- ``pyphi.config.MEASURE``: The measure to use when computing distances
  between repertoires and concepts. The default is ``EMD``; the Earth Movers's
  Distance. ``KLD`` is the Kullback-Leibler Divergence. If ``L1`` is chosen,
  the ``L1`` distance is initially used instead of the EMD when computing MIPs
  but, if a mechanism and purview are found to be irreducible, the |small_phi|
  value of the MIP is recalculated using the EMD.

    >>> defaults['MEASURE']
    'EMD'


System resources
~~~~~~~~~~~~~~~~

These settings control how much processing power and memory is available for
PyPhi to use. The default values may not be appropriate for your use-case or
machine, so **please check these settings before running anything**. Otherwise,
there is a risk that simulations might crash (potentially after running for a
long time!), resulting in data loss.

- ``pyphi.config.PARALLEL_CONCEPT_EVALUATION``: Control whether concepts are
  evaluated in parallel when computing constellations.

    >>> defaults['PARALLEL_CONCEPT_EVALUATION']
    False

- ``pyphi.config.PARALLEL_CUT_EVALUATION``: Control whether system cuts are
  evaluated in parallel, which requires more memory. If cuts are evaluated
  sequentially, only two |BigMip| instances need to be in memory at once.

    >>> defaults['PARALLEL_CUT_EVALUATION']
    True

- ``pyphi.config.PARALLEL_COMPLEX_EVALUATION``: Control whether systems are
  evaluated in parallel when computing complexes.

    >>> defaults['PARALLEL_COMPLEX_EVALUATION']
    False

  .. warning::

    Only one of ``PARALLEL_CONCEPT_EVALUATION``, ``PARALLEL_CUT_EVALUATION``,
    and ``PARALLEL_COMPLEX_EVALUATION`` can be set to ``True`` at a time. For
    maximal efficiency, you should parallelize the highest level computations
    possible: eg. parallelize complex evaluation instead of cut evaluation, but
    only if you are actually computing complexes. You should only parallelize
    concept evaluation if you are just computing constellations.

- ``pyphi.config.NUMBER_OF_CORES``: Control the number of CPU cores used to
  evaluate unidirectional cuts. Negative numbers count backwards from the total
  number of available cores, with ``-1`` meaning "use all available cores."

    >>> defaults['NUMBER_OF_CORES']
    -1

- ``pyphi.config.MAXIMUM_CACHE_MEMORY_PERCENTAGE``: PyPhi employs several
  in-memory caches to speed up computation. However, these can quickly use a
  lot of memory for large networks or large numbers of them; to avoid
  thrashing, this options limits the percentage of a system's RAM that the
  caches can collectively use.

    >>> defaults['MAXIMUM_CACHE_MEMORY_PERCENTAGE']
    50

Caching
~~~~~~~

PyPhi is equipped with a transparent caching system for |BigMip| objects which
stores them as they are computed to avoid having to recompute them later. This
makes it easy to play around interactively with the program, or to accumulate
results with minimal effort. For larger projects, however, it is recommended
that you manage the results explicitly, rather than relying on the cache. For
this reason it is disabled by default.

- ``pyphi.config.CACHE_BIGMIPS``: Control whether |BigMip| objects are cached
  and automatically retreived.

    >>> defaults['CACHE_BIGMIPS']
    False

- ``pyphi.config.CACHE_POTENTIAL_PURVIEWS``: Controls whether the potential
  purviews of mechanisms of a network are cached. Caching speeds up
  computations by not recomputing expensive reducibility checks, but uses
  additional memory.

    >>> defaults['CACHE_POTENTIAL_PURVIEWS']
    True

- ``pyphi.config.CACHING_BACKEND``: Control whether precomputed results are
  stored and read from a database or from a local filesystem-based cache in the
  current directory. Set this to 'fs' for the filesystem, 'db' for the
  database. Caching results on the filesystem is the easiest to use but least
  robust caching system. Caching results in a database is more robust and
  allows for caching individual concepts, but requires installing MongoDB.

    >>> defaults['CACHING_BACKEND']
    'fs'

- ``pyphi.config.FS_CACHE_VERBOSITY``: Control how much caching information is
  printed. Takes a value between 0 and 11. Note that printing during a loop
  iteration can slow down the loop considerably.

    >>> defaults['FS_CACHE_VERBOSITY']
    0

- ``pyphi.config.FS_CACHE_DIRECTORY``: If the caching backend is set to use the
  filesystem, the cache will be stored in this directory. This directory can be
  copied and moved around if you want to reuse results _e.g._ on a another
  computer, but it must be in the same directory from which PyPhi is being run.

    >>> defaults['FS_CACHE_DIRECTORY']
    '__pyphi_cache__'

- ``pyphi.config.MONGODB_CONFIG``: Set the configuration for the MongoDB
  database backend. This only has an effect if the caching backend is set to
  use the database.

    >>> defaults['MONGODB_CONFIG']['host']
    'localhost'
    >>> defaults['MONGODB_CONFIG']['port']
    27017
    >>> defaults['MONGODB_CONFIG']['database_name']
    'pyphi'
    >>> defaults['MONGODB_CONFIG']['collection_name']
    'cache'

- ``pyphi.config.REDIS_CACHE``: Specifies whether to use Redis to cache Mice.

    >>> defaults['REDIS_CACHE']
    False

- ``pyphi.config.REDIS_CONFIG``: Configure the Redis database backend. These
    are the defaults in the provided ``redis.conf`` file.

    >>> defaults['REDIS_CONFIG']['host']
    'localhost'
    >>> defaults['REDIS_CONFIG']['port']
    6379

Logging
~~~~~~~

These settings control how PyPhi handles log messages. Logs can be written to
standard output, a file, both, or none. If these simple default controls are
not flexible enough for you, you can override the entire logging configuration.
See the `documentation on Python's logger
<https://docs.python.org/3.4/library/logging.html>`_ for more information.

- ``pyphi.config.LOG_STDOUT_LEVEL``: Controls the level of log messages written
  to standard output. Can be one of ``'DEBUG'``, ``'INFO'``,
  ``'WARNING'``, ``'ERROR'``, ``'CRITICAL'``, or ``None``. ``DEBUG`` is the
  least restrictive level and will show the most log messages. ``CRITICAL`` is
  the most restrictive level and will only display information about
  unrecoverable errors. If set to ``None``, logging to standard output will be
  disabled entirely.

    >>> defaults['LOG_STDOUT_LEVEL']
    'WARNING'

- ``pyphi.config.LOG_FILE_LEVEL: Controls the level of log messages written to
  the log file. This option has the same possible values as
  ``LOG_STDOUT_LEVEL``.

    >>> defaults['LOG_FILE_LEVEL']
    'INFO'

- ``pyphi.config.LOG_FILE``: Control the name of the logfile.

    >>> defaults['LOG_FILE']
    'pyphi.log'

- ``pyphi.config.LOG_CONFIG_ON_IMPORT``: Controls whether the current
  configuration is printed when PyPhi is imported.

    >>> defaults['LOG_CONFIG_ON_IMPORT']
    True

- ``pyphi.config.PROGRESS_BARS``: Controls whether to show progress bars on
  the console.

    >>> defaults['PROGRESS_BARS']
    True


Numerical precision
~~~~~~~~~~~~~~~~~~~

- ``pyphi.config.PRECISION``: Computations in PyPhi rely on finding the Earth
  Mover's Distance. This is done via an external C++ library that uses
  flow-optimization to find a good approximation of the EMD. Consequently,
  systems with zero |big_phi| will sometimes be computed to have a small but
  non-zero amount. This setting controls the number of decimal places to which
  PyPhi will consider EMD calculations accurate. Values of |big_phi| lower than
  ``10e-PRECISION`` will be considered insignificant and treated as zero. The
  default value is about as accurate as the EMD computations get.

    >>> defaults['PRECISION']
    6


Miscellaneous
~~~~~~~~~~~~~

- ``pyphi.config.VALIDATE_SUBSYSTEM_STATES``: Control whether PyPhi checks if
  the subsystems's state is possible (reachable from some past state), given
  the subsystem's TPM (**which is conditioned on background conditions**). If
  this is turned off, then **calculated** |big_phi| **values may not be
  valid**, since they may be associated with a subsystem that could never be in
  the given state.

    >>> defaults['VALIDATE_SUBSYSTEM_STATES']
    True


- ``pyphi.config.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI``: If set to ``True``,
  this defines the Phi value of subsystems containing only a single node with a
  self-loop to be ``0.5``. If set to False, their |big_phi| will be actually be
  computed (to be zero, in this implementation).

    >>> defaults['SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI']
    False


- ``pyphi.config.REPR_VERBOSITY``: Controls the verbosity of ``__repr__``
  methods on PyPhi objects. Can be set to ``0``, ``1``, or ``2``. If set to
  ``1``, calling ``repr`` on PyPhi objects will return pretty-formatted and
  legible strings, excluding repertoires. If set to ``2``, ``repr`` calls also
  include repertoires.

  Although this breaks the convention that ``__repr__`` methods should return a
  representation which can reconstruct the object, readable representations are
  convenient since the Python REPL calls ``repr`` to represent all objects in
  the shell and PyPhi is often used interactively with the REPL. If set to
  ``0``, ``repr`` returns more traditional object representations.

    >>> defaults['REPR_VERBOSITY']
    2

- ``pyphi.config.PARTITION_MECHANISMS``: If ``True``, |small_phi|-MIP
  computations will only consider bipartitions that strictly partition the
  mechanism. That is, for the mechanism ``(A, B)`` and purview ``(B, C, D)``
  the partition ::

    AB   []
    -- X --
    B    CD

  is not considered, but ::

    A    B
    -- X --
    B    CD

  is. The following is also valid::

    AB   []
    -- X ---
    []   BCD

  In addition, this option introduces wedge tripartitions of the form ::

    A    B   []
    -- X - X --
    B    C   D

  where the mechanism in the third part is always empty.

  Finally, in the case of a |small_phi|-tie when computing MICE, this
  setting choses the MIP with smallest purview instead the largest (which is
  the default behavior.)

    >>> defaults['PARTITION_MECHANISMS']
    False


- ``pyphi.config.PARTITION_MECHANISMS``: If ``True``, |small_phi|-MIP
  computations will only consider bipartitions that strictly partition the
  mechanism. That is, for the mechanism ``(A, B)`` and purview ``(B, C, D)``
  the partition ::

    AB   []
    -- X --
    B    CD

  is not considered, but ::

    A    B
    -- X --
    B    CD

  is. The following is also valid::

    AB   []
    -- X ---
    []   BCD

  Additionally, in the case of a |small_phi|-tie when computing MICE, this
  setting choses the MIP with smallest purview instead the largest (which is
  the default behavior.)

    >>> defaults['PARTITION_MECHANISMS']
    False


-------------------------------------------------------------------------------
"""

import contextlib
import logging
import logging.config
import os
import pprint
import sys

import yaml

from . import __about__

# TODO: document mongo config
# Defaults for configurable constants.
DEFAULTS = {
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assumptions that speed up computation at the cost of theoretical
    # accuracy.
    'ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS': False,
    # Only check single nodes cuts for the MIP. 2**n cuts instead of n.
    'CUT_ONE_APPROXIMATION': False,
    # The measure to use when computing phi ('EMD', 'KLD', 'L1')
    'MEASURE': 'EMD',
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Controls whether concepts are evaluated in parallel.
    'PARALLEL_CONCEPT_EVALUATION': False,
    # Controls whether cuts are evaluated in parallel, which requires more
    # memory. If cuts are evaluated sequentially, only two BigMips need to be
    # in memory at a time.
    'PARALLEL_CUT_EVALUATION': True,
    # Controls whether systems are evaluated in parallel when searching for
    # complexes.
    'PARALLEL_COMPLEX_EVALUATION': False,
    # The number of CPU cores to use in parallel cut evaluation. -1 means all
    # available cores, -2 means all but one available cores, etc.
    'NUMBER_OF_CORES': -1,
    # The maximum percentage of RAM that PyPhi should use for caching.
    'MAXIMUM_CACHE_MEMORY_PERCENTAGE': 50,
    # Controls whether BigMips are cached and retreived.
    'CACHE_BIGMIPS': False,
    # Controls whether the potential purviews of the mechanisms of a network
    # are cached. Speeds up calculations, but takes up additional memory.
    'CACHE_POTENTIAL_PURVIEWS': True,
    # The caching system to use. "fs" means cache results in a subdirectory of
    # the current directory; "db" means connect to a database and store the
    # results there.
    'CACHING_BACKEND': 'fs',
    # joblib.Memory verbosity.
    'FS_CACHE_VERBOSITY': 0,
    # Directory for the persistent joblib Memory cache.
    'FS_CACHE_DIRECTORY': '__pyphi_cache__',
    # MongoDB configuration.
    'MONGODB_CONFIG': {
        'host': 'localhost',
        'port': 27017,
        'database_name': 'pyphi',
        'collection_name': 'cache'
    },
    # Use Redis to cache Mice
    'REDIS_CACHE': False,
    # Redis configuration
    'REDIS_CONFIG': {
        'host': 'localhost',
        'port': 6379,
    },
    # The file to log to
    'LOG_FILE': 'pyphi.log',
    # The log level to write to `LOG_FILE`
    'LOG_FILE_LEVEL': 'INFO',
    # The log level to write to stdout
    'LOG_STDOUT_LEVEL': 'WARNING',
    # Controls whether the current configuration is logged upon import.
    'LOG_CONFIG_ON_IMPORT': True,
    # Enable/disable progress bars
    'PROGRESS_BARS': True,
    # The number of decimal points to which phi values are considered accurate.
    'PRECISION': 6,
    # Controls whether a subsystem's state is validated when the subsystem is
    # created.
    'VALIDATE_SUBSYSTEM_STATES': True,
    # In some applications of this library, the user may prefer to define
    # single-node subsystems as having 0.5 Phi.
    'SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI': False,
    # Use prettier __str__-like formatting in `repr` calls.
    'REPR_VERBOSITY': 2,
    # Only consider bipartitions which strictly partition the mechanism.
    'PARTITION_MECHANISMS': False,
}

# Get a reference to this module's dictionary so we can set the configuration
# directly in the `pyphi.config` namespace
this_module = sys.modules[__name__]


def load_config_dict(config):
    """Load configuration values.

    Args:
        config (dict): The dict of config to load.
    """
    this_module.__dict__.update(config)


def load_config_file(filename):
    """Load config from a YAML file."""
    with open(filename) as f:
        load_config_dict(yaml.load(f))


def load_config_default():
    """Load default config values."""
    load_config_dict(DEFAULTS)


def get_config_string():
    """Return a string representation of the currently loaded configuration."""
    config = {key: this_module.__dict__[key] for key in DEFAULTS.keys()}
    return pprint.pformat(config, indent=2)


def print_config():
    """Print the current configuration."""
    print('Current PyPhi configuration:\n', get_config_string())


def configure_logging():
    """Configure PyPhi logging based on the loaded configuration.

    Note: if PyPhi config options that control logging are changed after they
    are loaded (eg. in testing), the Python logging configuration will stay
    the same unless you manually reconfigure the logging by calling this
    function.

    TODO: call this in `config.override`?
    """
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
            }
        },
        'handlers': {
            'file': {
                'level': this_module.LOG_FILE_LEVEL,
                'filename': this_module.LOG_FILE,
                'class': 'logging.FileHandler',
                'formatter': 'standard',
            },
            'stdout': {
                'level': this_module.LOG_STDOUT_LEVEL,
                'class': 'pyphi.logging.TqdmHandler',
                'formatter': 'standard',
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': (['file'] if this_module.LOG_FILE_LEVEL else []) +
                        (['stdout'] if this_module.LOG_STDOUT_LEVEL else [])
        }
    })


class override(contextlib.ContextDecorator):
    """Decorator and context manager to override config values.

    The initial configuration values are reset after the decorated function
    returns or the context manager completes it block, even if the function
    or block raises an exception. This is intended to be used by testcases
    which require specific configuration values.

    Example:
        >>> from pyphi import config
        >>>
        >>> @config.override(PRECISION=20000)
        ... def test_something():
        ...     assert config.PRECISION == 20000
        ...
        >>> test_something()
        >>> with config.override(PRECISION=100):
        ...     assert config.PRECISION == 100
        ...
    """
    def __init__(self, **new_conf):
        self.new_conf = new_conf

    def __enter__(self):
        """Save original config values; override with new ones."""
        self.initial_conf = {opt_name: this_module.__dict__[opt_name]
                             for opt_name in self.new_conf}
        load_config_dict(self.new_conf)

    def __exit__(self, *exc):
        """Reset config to initial values; reraise any exceptions."""
        load_config_dict(self.initial_conf)
        return False


PYPHI_CONFIG_FILENAME = 'pyphi_config.yml'

# Load the default config
load_config_default()

# Then try and load the config file
file_loaded = False
if os.path.exists(PYPHI_CONFIG_FILENAME):
    load_config_file(PYPHI_CONFIG_FILENAME)
    file_loaded = True

# Setup logging
configure_logging()

# Log the PyPhi version and loaded configuration
if this_module.LOG_CONFIG_ON_IMPORT:
    log = logging.getLogger(__name__)

    log.info('PyPhi version {}'.format(__about__.__version__))
    if file_loaded:
        log.info('Loaded configuration from '
                 '`./{}`'.format(PYPHI_CONFIG_FILENAME))
    else:
        log.info('Using default configuration (no config file provided)')

    log.info('Current PyPhi configuration:\n {}'.format(get_config_string()))
