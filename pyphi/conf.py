#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conf.py

'''
Loading a configuration
~~~~~~~~~~~~~~~~~~~~~~~

Various aspects of PyPhi's behavior can be configured.

When PyPhi is imported, it checks for a YAML file named ``pyphi_config.yml`` in
the current directory and automatically loads it if it exists; otherwise the
default configuration is used.

The various settings are listed here with their defaults.

    >>> import pyphi
    >>> defaults = pyphi.config.DEFAULTS

It is also possible to manually load a configuration file:

    >>> pyphi.config.load_config_file('pyphi_config.yml')

Or load a dictionary of configuration values:

    >>> pyphi.config.load_config_dict({'SOME_CONFIG': 'value'})

Many settings can also be changed on the fly by simply assigning them a new
value:

    >>> pyphi.config.PROGRESS_BARS = True


Approximations and theoretical options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These settings control the algorithms PyPhi uses.

- ``ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS``:
  In certain cases, making a cut can actually cause a previously reducible
  concept to become a proper, irreducible concept. Assuming this can never
  happen can increase performance significantly, however the obtained results
  are not strictly accurate.

    >>> defaults['ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS']
    False

- ``CUT_ONE_APPROXIMATION``:
  When determining the MIP for |big_phi|, this restricts the set of system cuts
  that are considered to only those that cut the inputs or outputs of a single
  node. This restricted set of cuts scales linearly with the size of the
  system; the full set of all possible bipartitions scales exponentially. This
  approximation is more likely to give theoretically accurate results with
  modular, sparsely-connected, or homogeneous networks.

    >>> defaults['CUT_ONE_APPROXIMATION']
    False

- ``MEASURE``:
  The measure to use when computing distances between repertoires and concepts.
  Users can dynamically register new measures with the
  ``pyphi.distance.measures.register`` decorator; see :mod:`~pyphi.distance`
  for examples. A full list of currently installed measures is available by
  calling ``print(pyphi.distance.measures.all())``. Note that some measures
  cannot be used for calculating |big_phi| because they are asymmetric.

    >>> defaults['MEASURE']
    'EMD'

- ``PARTITION_TYPE``:
  Controls the type of partition used for |small_phi| computations.

  If set to ``'BI'``, partitions will have two parts.

  If set to ``'TRI'``, partitions will have three parts. In addition,
  computations will only consider partitions that strictly partition the
  mechanism the mechanism. That is, for the mechanism ``(A, B)`` and purview
  ``(B, C, D)`` the partition::

    A,B    ∅
    ─── ✕ ───
     B    C,D

  is not considered, but::

     A     B
    ─── ✕ ───
     B    C,D

  is. The following is also valid::

    A,B     ∅
    ─── ✕ ─────
     ∅    B,C,D

  In addition, this setting introduces "wedge" tripartitions of the form::

     A     B     ∅
    ─── ✕ ─── ✕ ───
     B     C     D

  where the mechanism in the third part is always empty.

  In addition, in the case of a |small_phi|-tie when computing MICE, The
  ``'TRIPARTITION'`` setting choses the MIP with smallest purview instead of
  the largest (which is the default).

  Finally, if set to ``'ALL'``, all possible partitions will be tested.

    >>> defaults['PARTITION_TYPE']
    'BI'

- ``PICK_SMALLEST_PURVIEW``:
  When computing MICE, it is possible for several MIPs to have the same
  |small_phi| value. If this setting is set to ``True`` the MIP with the
  smallest purview is chosen; otherwise, the one with largest purview is
  chosen.

    >>> defaults['PICK_SMALLEST_PURVIEW']
    False

- ``USE_SMALL_PHI_DIFFERENCE_FOR_CONSTELLATION_DISTANCE``:
  If set to ``True``, the distance between constellations (when computing a
  |BigMip|) is calculated using the difference between the sum of |small_phi|
  in the constellations instead of the extended EMD.


- ``SYSTEM_CUTS``:
  If set to ``'3.0_STYLE'``, then traditional IIT 3.0 cuts will be used when
  computing |big_phi|. If set to ``'CONCEPT_STYLE'``, then experimental
  concept- style system cuts will be used instead.

    >>> defaults['SYSTEM_CUTS']
    '3.0_STYLE'


System resources
~~~~~~~~~~~~~~~~

These settings control how much processing power and memory is available for
PyPhi to use. The default values may not be appropriate for your use-case or
machine, so **please check these settings before running anything**. Otherwise,
there is a risk that simulations might crash (potentially after running for a
long time!), resulting in data loss.

- ``PARALLEL_CONCEPT_EVALUATION``:
  Controls whether concepts are evaluated in parallel when computing
  constellations.

    >>> defaults['PARALLEL_CONCEPT_EVALUATION']
    False

- ``PARALLEL_CUT_EVALUATION``:
  Controls whether system cuts are evaluated in parallel, which is faster but
  requires more memory. If cuts are evaluated sequentially, only two |BigMip|
  instances need to be in memory at once.

    >>> defaults['PARALLEL_CUT_EVALUATION']
    True

- ``PARALLEL_COMPLEX_EVALUATION``:
  Controls whether systems are evaluated in parallel when computing complexes.

    >>> defaults['PARALLEL_COMPLEX_EVALUATION']
    False

  .. warning::
    Only one of ``PARALLEL_CONCEPT_EVALUATION``, ``PARALLEL_CUT_EVALUATION``,
    and ``PARALLEL_COMPLEX_EVALUATION`` can be set to ``True`` at a time. For
    maximal efficiency, you should parallelize the highest level computations
    possible, *e.g.*, parallelize complex evaluation instead of cut evaluation,
    but only if you are actually computing complexes. You should only
    parallelize concept evaluation if you are just computing constellations.

- ``NUMBER_OF_CORES``:
  Controls the number of CPU cores used to evaluate unidirectional cuts.
  Negative numbers count backwards from the total number of available cores,
  with ``-1`` meaning "use all available cores."

    >>> defaults['NUMBER_OF_CORES']
    -1

- ``MAXIMUM_CACHE_MEMORY_PERCENTAGE``:
  PyPhi employs several in-memory caches to speed up computation. However,
  these can quickly use a lot of memory for large networks or large numbers of
  them; to avoid thrashing, this setting limits the percentage of a system's
  RAM that the caches can collectively use.

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

- ``CACHE_BIGMIPS``:
  Controls whether |BigMip| objects are cached and automatically retrieved.

    >>> defaults['CACHE_BIGMIPS']
    False

- ``CACHE_POTENTIAL_PURVIEWS``:
  Controls whether the potential purviews of mechanisms of a network are
  cached. Caching speeds up computations by not recomputing expensive
  reducibility checks, but uses additional memory.

    >>> defaults['CACHE_POTENTIAL_PURVIEWS']
    True

- ``CACHING_BACKEND``:
  Controls whether precomputed results are stored and read from a local
  filesystem-based cache in the current directory or from a database. Set this
  to ``'fs'`` for the filesystem, ``'db'`` for the database.

    >>> defaults['CACHING_BACKEND']
    'fs'

- ``FS_CACHE_VERBOSITY``:
  Controls how much caching information is printed if the filesystem cache is
  used. Takes a value between ``0`` and ``11``.

    >>> defaults['FS_CACHE_VERBOSITY']
    0

  .. warning::
      Printing during a loop iteration can slow down the loop considerably.

- ``FS_CACHE_DIRECTORY``:
  If the filesystem is used for caching, the cache will be stored in this
  directory. This directory can be copied and moved around if you want to reuse
  results *e.g.* on a another computer, but it must be in the same directory
  from which Python is being run.

    >>> defaults['FS_CACHE_DIRECTORY']
    '__pyphi_cache__'

- ``MONGODB_CONFIG``:
  Set the configuration for the MongoDB database backend (only has an effect if
  ``CACHING_BACKEND`` is ``'db'``).

    >>> defaults['MONGODB_CONFIG']['host']
    'localhost'
    >>> defaults['MONGODB_CONFIG']['port']
    27017
    >>> defaults['MONGODB_CONFIG']['database_name']
    'pyphi'
    >>> defaults['MONGODB_CONFIG']['collection_name']
    'cache'

- ``REDIS_CACHE``:
  Specifies whether to use Redis to cache |Mice|.

    >>> defaults['REDIS_CACHE']
    False

- ``REDIS_CONFIG``:
  Configure the Redis database backend. These are the defaults in the provided
  ``redis.conf`` file.

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

.. important::
    After PyPhi has been imported, changing these settings will have no effect
    unless you call |configure_logging| afterwards.

- ``LOG_STDOUT_LEVEL``:
  Controls the level of log messages written to standard output. Can be one of
  ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``, ``'CRITICAL'``, or
  ``None``. ``'DEBUG'`` is the least restrictive level and will show the most
  log messages. ``'CRITICAL'`` is the most restrictive level and will only
  display information about fatal errors. If set to ``None``, logging to
  standard output will be disabled entirely.

    >>> defaults['LOG_STDOUT_LEVEL']
    'WARNING'

- ``LOG_FILE_LEVEL``:
  Controls the level of log messages written to the log file. This setting has
  the same possible values as ``LOG_STDOUT_LEVEL``.

    >>> defaults['LOG_FILE_LEVEL']
    'INFO'

- ``LOG_FILE``:
  Controls the name of the log file.

    >>> defaults['LOG_FILE']
    'pyphi.log'

- ``LOG_CONFIG_ON_IMPORT``:
  Controls whether the configuration is printed when PyPhi is imported.

    >>> defaults['LOG_CONFIG_ON_IMPORT']
    True

  .. tip::
      If this is enabled and ``LOG_FILE_LEVEL`` is ``INFO`` or higher, then
      the log file can serve as an automatic record of which configuration
      settings you used to obtain results.

- ``PROGRESS_BARS``:
  Controls whether to show progress bars on the console.

    >>> defaults['PROGRESS_BARS']
    True

  .. tip::
    If you are iterating over many systems rather than doing one long-running
    calculation, consider disabling this for speed.

Numerical precision
~~~~~~~~~~~~~~~~~~~

- ``PRECISION``:
  If ``MEASURE`` is ``EMD``, then the Earth Mover's Distance is calculated with
  an external C++ library that a numerical optimizer to find a good
  approximation. Consequently, systems with analytically zero |big_phi| will
  sometimes be numerically found to have a small but non-zero amount. This
  setting controls the number of decimal places to which PyPhi will consider
  EMD calculations accurate. Values of |big_phi| lower than ``10e-PRECISION``
  will be considered insignificant and treated as zero. The default value is
  about as accurate as the EMD computations get.

    >>> defaults['PRECISION']
    6


Miscellaneous
~~~~~~~~~~~~~

- ``VALIDATE_SUBSYSTEM_STATES``:
  Controls whether PyPhi checks if the subsystems's state is possible
  (reachable with nonzero probability from some past state), given the
  subsystem's TPM (**which is conditioned on background conditions**). If this
  is turned off, then **calculated** |big_phi| **values may not be valid**,
  since they may be associated with a subsystem that could never be in the
  given state.

    >>> defaults['VALIDATE_SUBSYSTEM_STATES']
    True

- ``VALIDATE_CONDITIONAL_INDEPENDENCE``:
  Controls whether PyPhi checks if a system's TPM is conditionally
  independent.

    >>> defaults['VALIDATE_CONDITIONAL_INDEPENDENCE']
    True

- ``SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI``:
  If set to ``True``, the Phi value of single micro-node subsystems is the
  difference between their unpartitioned constellation (a single concept) and
  the null concept. If set to False, their Phi is defined to be zero. Single
  macro-node subsystems may always be cut, regardless of circumstances.

    >>> defaults['SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI']
    False

- ``REPR_VERBOSITY``:
  Controls the verbosity of ``__repr__`` methods on PyPhi objects. Can be set
  to ``0``, ``1``, or ``2``. If set to ``1``, calling ``repr`` on PyPhi objects
  will return pretty-formatted and legible strings, excluding repertoires. If
  set to ``2``, ``repr`` calls also include repertoires.

  Although this breaks the convention that ``__repr__`` methods should return a
  representation which can reconstruct the object, readable representations are
  convenient since the Python REPL calls ``repr`` to represent all objects in
  the shell and PyPhi is often used interactively with the REPL. If set to
  ``0``, ``repr`` returns more traditional object representations.

    >>> defaults['REPR_VERBOSITY']
    2

- ``PRINT_FRACTIONS``:
  Controls whether numbers in a ``repr`` are printed as fractions. Numbers are
  still printed as decimals if the fraction's denominator would be large. This
  only has an effect if ``REPR_VERBOSITY > 0``.

    >>> defaults['PRINT_FRACTIONS']
    True

The ``config`` API
~~~~~~~~~~~~~~~~~~
'''

# pylint: disable=too-few-public-methods

import contextlib
import logging
import logging.config
import os
import pprint
import sys
from copy import copy

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
    # Controls whether systems are checked for conditional independence.
    'VALIDATE_CONDITIONAL_INDEPENDENCE': True,
    # In some applications of this library, the user may prefer to define
    # single micro-node subsystems as having Phi.
    'SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI': False,
    # Use prettier __str__-like formatting in `repr` calls.
    'REPR_VERBOSITY': 2,
    # Print numbers as fractions if the denominator isn't too big.
    'PRINT_FRACTIONS': True,
    # Controls the number of parts in a partition.
    'PARTITION_TYPE': 'BI',
    # Controls how to pick MIPs in the case of phi-ties.
    'PICK_SMALLEST_PURVIEW': False,
    # Use the difference in sum of small phi for the constellation distance
    'USE_SMALL_PHI_DIFFERENCE_FOR_CONSTELLATION_DISTANCE': False,
    # The type of system cuts to use
    'SYSTEM_CUTS': '3.0_STYLE',
}


class Config:

    def __str__(self):
        return pprint.pformat(self.__dict__, indent=2)

    def load_config_dict(self, dct):
        '''Load a dictionary of configuration values.'''
        self.__dict__.update(dct)

    def load_config_file(self, filename):
        '''Load config from a YAML file.'''
        with open(filename) as f:
            self.__dict__.update(yaml.load(f))

    def snapshot(self):
        return copy(self.__dict__)

    def override(self, **new_config):
        '''Decorator and context manager to override configuration values.

        The initial configuration values are reset after the decorated function
        returns or the context manager completes it block, even if the function
        or block raises an exception. This is intended to be used by tests
        which require specific configuration values.

        Example:
            >>> from pyphi import config
            >>> @config.override(PRECISION=20000)
            ... def test_something():
            ...     assert config.PRECISION == 20000
            ...
            >>> test_something()
            >>> with config.override(PRECISION=100):
            ...     assert config.PRECISION == 100
            ...
        '''
        return _override(self, **new_config)

    # TODO: call this in `config.override`?
    def configure_logging(self):
        '''Reconfigure PyPhi logging based on the current configuration.'''
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(name)s] %(levelname)s '
                              '%(processName)s: %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'level': self.LOG_FILE_LEVEL,
                    'filename': self.LOG_FILE,
                    'class': 'logging.FileHandler',
                    'formatter': 'standard',
                },
                'stdout': {
                    'level': self.LOG_STDOUT_LEVEL,
                    'class': 'pyphi.log.ProgressBarHandler',
                    'formatter': 'standard',
                }
            },
            'root': {
                'level': 'DEBUG',
                'handlers': (['file'] if self.LOG_FILE_LEVEL else []) +
                            (['stdout'] if self.LOG_STDOUT_LEVEL else [])
            }
        })


class _override(contextlib.ContextDecorator):
    '''See ``Config.override`` for usage.'''

    def __init__(self, config, **new_conf):
        self.config = config
        self.new_conf = new_conf
        self.initial_conf = config.snapshot()

    def __enter__(self):
        '''Save original config values; override with new ones.'''
        self.config.load_config_dict(self.new_conf)

    def __exit__(self, *exc):
        '''Reset config to initial values; reraise any exceptions.'''
        self.config.load_config_dict(self.initial_conf)
        return False


def print_config():
    '''Print the current configuration.'''
    print('Current PyPhi configuration:\n', str(config))


PYPHI_CONFIG_FILENAME = 'pyphi_config.yml'

config = Config()


def initialize():
    '''Initialize PyPhi config.'''
    # Load the default config
    config.load_config_dict(DEFAULTS)

    # Then try and load the config file
    file_loaded = False
    if os.path.exists(PYPHI_CONFIG_FILENAME):
        config.load_config_file(PYPHI_CONFIG_FILENAME)
        file_loaded = True

    # Setup logging
    config.configure_logging()

    # Log the PyPhi version and loaded configuration
    if config.LOG_CONFIG_ON_IMPORT:
        log = logging.getLogger(__name__)

        log.info('PyPhi v%s', __about__.__version__)
        if file_loaded:
            log.info('Loaded configuration from '
                     '`./%s`', PYPHI_CONFIG_FILENAME)
        else:
            log.info('Using default configuration (no config file provided)')

        log.info('Current PyPhi configuration:\n %s', str(config))


initialize()
