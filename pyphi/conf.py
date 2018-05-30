#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conf.py

"""
Loading a configuration
~~~~~~~~~~~~~~~~~~~~~~~

Various aspects of PyPhi's behavior can be configured.

When PyPhi is imported, it checks for a YAML file named ``pyphi_config.yml`` in
the current directory and automatically loads it if it exists; otherwise the
default configuration is used.

.. only:: never

    This py.test fixture resets PyPhi config back to defaults after running
    this doctest. This will not be shown in the output markup.

    >>> getfixture('restore_config_afterwards')

The various settings are listed here with their defaults.

    >>> import pyphi
    >>> defaults = pyphi.config.defaults()

Print the ``config`` object to see the current settings:

    >>> print(pyphi.config)  # doctest: +SKIP
    { 'ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS': False,
      'CACHE_SIAS': False,
      'CACHE_POTENTIAL_PURVIEWS': True,
      'CACHING_BACKEND': 'fs',
      ...

Setting can be changed on the fly by assigning them a new value:

    >>> pyphi.config.PROGRESS_BARS = False

It is also possible to manually load a configuration file:

    >>> pyphi.config.load_file('pyphi_config.yml')

Or load a dictionary of configuration values:

    >>> pyphi.config.load_dict({'PRECISION': 1})


Approximations and theoretical options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These settings control the algorithms PyPhi uses.

- :attr:`~pyphi.conf.PyphiConfig.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS`
- :attr:`~pyphi.conf.PyphiConfig.CUT_ONE_APPROXIMATION`
- :attr:`~pyphi.conf.PyphiConfig.MEASURE`
- :attr:`~pyphi.conf.PyphiConfig.PARTITION_TYPE`
- :attr:`~pyphi.conf.PyphiConfig.PICK_SMALLEST_PURVIEW`
- :attr:`~pyphi.conf.PyphiConfig.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE`
- :attr:`~pyphi.conf.PyphiConfig.SYSTEM_CUTS`
- :attr:`~pyphi.conf.PyphiConfig.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI`
- :attr:`~pyphi.conf.PyphiConfig.VALIDATE_SUBSYSTEM_STATES`
- :attr:`~pyphi.conf.PyphiConfig.VALIDATE_CONDITIONAL_INDEPENDENCE`


Parallelization and system resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These settings control how much processing power and memory is available for
PyPhi to use. The default values may not be appropriate for your use-case or
machine, so **please check these settings before running anything**. Otherwise,
there is a risk that simulations might crash (potentially after running for a
long time!), resulting in data loss.

- :attr:`~pyphi.conf.PyphiConfig.PARALLEL_CONCEPT_EVALUATION`
- :attr:`~pyphi.conf.PyphiConfig.PARALLEL_CUT_EVALUATION`
- :attr:`~pyphi.conf.PyphiConfig.PARALLEL_COMPLEX_EVALUATION`
- :attr:`~pyphi.conf.PyphiConfig.NUMBER_OF_CORES`
- :attr:`~pyphi.conf.PyphiConfig.MAXIMUM_CACHE_MEMORY_PERCENTAGE`

  .. important::
    Only one of ``PARALLEL_CONCEPT_EVALUATION``, ``PARALLEL_CUT_EVALUATION``,
    and ``PARALLEL_COMPLEX_EVALUATION`` can be set to ``True`` at a time.

    **For most networks,** ``PARALLEL_CUT_EVALUATION`` **is the most
    efficient.** This is because the algorithm is exponential time in the
    number of nodes, so the most of the time is spent on the largest subsystem.

    You should only parallelize concept evaluation if you are just computing a
    |CauseEffectStructure|.


Memoization and caching
~~~~~~~~~~~~~~~~~~~~~~~

PyPhi provides a number of ways to cache intermediate results.

- :attr:`~pyphi.conf.PyphiConfig.CACHE_SIAS`
- :attr:`~pyphi.conf.PyphiConfig.CACHE_REPERTOIRES`
- :attr:`~pyphi.conf.PyphiConfig.CACHE_POTENTIAL_PURVIEWS`
- :attr:`~pyphi.conf.PyphiConfig.CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA`
- :attr:`~pyphi.conf.PyphiConfig.CACHING_BACKEND`
- :attr:`~pyphi.conf.PyphiConfig.FS_CACHE_VERBOSITY`
- :attr:`~pyphi.conf.PyphiConfig.FS_CACHE_DIRECTORY`
- :attr:`~pyphi.conf.PyphiConfig.MONGODB_CONFIG`
- :attr:`~pyphi.conf.PyphiConfig.REDIS_CACHE`
- :attr:`~pyphi.conf.PyphiConfig.REDIS_CONFIG`


Logging
~~~~~~~

These settings control how PyPhi handles log messages. Logs can be written to
standard output, a file, both, or none. If these simple default controls are
not flexible enough for you, you can override the entire logging configuration.
See the `documentation on Python's logger
<https://docs.python.org/3.4/library/logging.html>`_ for more information.

- :attr:`~pyphi.conf.PyphiConfig.LOG_STDOUT_LEVEL`
- :attr:`~pyphi.conf.PyphiConfig.LOG_FILE_LEVEL`
- :attr:`~pyphi.conf.PyphiConfig.LOG_FILE`
- :attr:`~pyphi.conf.PyphiConfig.PROGRESS_BARS`
- :attr:`~pyphi.conf.PyphiConfig.REPR_VERBOSITY`
- :attr:`~pyphi.conf.PyphiConfig.PRINT_FRACTIONS`


Numerical precision
~~~~~~~~~~~~~~~~~~~

- :attr:`~pyphi.conf.PyphiConfig.PRECISION`


The ``config`` API
~~~~~~~~~~~~~~~~~~
"""

# pylint: disable=protected-access

import contextlib
import logging
import logging.config
import os
import pprint
from copy import copy

import yaml

from . import __about__

log = logging.getLogger(__name__)


class Option:
    """A descriptor implementing PyPhi configuration options.

    Args:
        default: The default value of this ``Option``.

    Keyword Args:
        values (list): Allowed values for this option. A ``ValueError`` will
            be raised if ``values`` is not ``None`` and the option is set to
            be a value not in the list.
        on_change (function): Optional callback that is called when the value
            of the option is changed. The ``Config`` instance is passed as
            the only argument to the callback.
        doc (str): Optional docstring for the option.
    """

    def __init__(self, default, values=None, on_change=None, doc=None):
        self.default = default
        self.values = values
        self.on_change = on_change
        self.doc = doc

        # Set during ``Config`` class creation
        self.name = None

        self.__doc__ = self._docstring()

    def _docstring(self):
        default = '``default={}``'.format(repr(self.default))

        values = (', ``values={}``'.format(repr(self.values))
                  if self.values is not None else '')

        on_change = (', ``on_change={}``'.format(self.on_change.__name__)
                     if self.on_change is not None else '')

        return '{}{}{}\n{}'.format(default, values, on_change, self.doc or '')

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        return obj._values[self.name]

    def __set__(self, obj, value):
        self._validate(value)
        obj._values[self.name] = value
        self._callback(obj)

    def _validate(self, value):
        """Validate the new value."""
        if self.values and value not in self.values:
            raise ValueError(
                '{} is not a valid value for {}'.format(value, self.name))

    def _callback(self, obj):
        """Trigger any callbacks."""
        if self.on_change is not None:
            self.on_change(obj)


class ConfigMeta(type):
    """Metaclass for ``Config``.

    Responsible for setting the name of each ``Option`` when a subclass of
    ``Config`` is created; because ``Option`` objects are defined on the class,
    not the instance, their name should only be set once.

    Python 3.6 handles this exact need with the special descriptor method
    ``__set_name__`` (see PEP 487). We should use that once we drop support
    for 3.4 & 3.5.
    """

    def __init__(cls, cls_name, bases, namespace):
        super().__init__(cls_name, bases, namespace)
        for name, opt in cls.options().items():
            opt.name = name


class Config(metaclass=ConfigMeta):
    """Base configuration object.

    See ``PyphiConfig`` for usage.
    """

    def __init__(self):
        self._values = {}
        self._loaded_files = []

        # Set the default value of each ``Option``
        for name, opt in self.options().items():
            opt._validate(opt.default)
            self._values[name] = opt.default

        # Call hooks for each Option
        # (This must happen *after* all default values are set so that
        # logging can be properly configured.
        for opt in self.options().values():
            opt._callback(self)

    def __str__(self):
        return pprint.pformat(self._values, indent=2)

    def __setattr__(self, name, value):
        if name.startswith('_') or name in self.options().keys():
            super().__setattr__(name, value)
        else:
            raise ValueError('{} is not a valid config option'.format(name))

    @classmethod
    def options(cls):
        """Return a dictionary the ``Option`` objects for this config"""
        return {k: v for k, v in cls.__dict__.items() if isinstance(v, Option)}

    def defaults(self):
        """Return the default values of this configuration."""
        return {k: v.default for k, v in self.options().items()}

    def load_dict(self, dct):
        """Load a dictionary of configuration values."""
        for k, v in dct.items():
            setattr(self, k, v)

    def load_file(self, filename):
        """Load config from a YAML file."""
        filename = os.path.abspath(filename)

        with open(filename) as f:
            self.load_dict(yaml.load(f))

        self._loaded_files.append(filename)

    def snapshot(self):
        """Return a snapshot of the current values of this configuration."""
        return copy(self._values)

    def override(self, **new_values):
        """Decorator and context manager to override configuration values.

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
        """
        return _override(self, **new_values)


class _override(contextlib.ContextDecorator):
    """See ``Config.override`` for usage."""

    def __init__(self, conf, **new_values):
        self.conf = conf
        self.new_values = new_values
        self.initial_values = conf.snapshot()

    def __enter__(self):
        """Save original config values; override with new ones."""
        self.conf.load_dict(self.new_values)

    def __exit__(self, *exc):
        """Reset config to initial values; reraise any exceptions."""
        self.conf.load_dict(self.initial_values)
        return False


def configure_logging(conf):
    """Reconfigure PyPhi logging based on the current configuration."""
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
                'level': conf.LOG_FILE_LEVEL,
                'filename': conf.LOG_FILE,
                'class': 'logging.FileHandler',
                'formatter': 'standard',
            },
            'stdout': {
                'level': conf.LOG_STDOUT_LEVEL,
                'class': 'pyphi.log.TqdmHandler',
                'formatter': 'standard',
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': (['file'] if conf.LOG_FILE_LEVEL else []) +
                        (['stdout'] if conf.LOG_STDOUT_LEVEL else [])
        }
    })


class PyphiConfig(Config):
    """``pyphi.config`` is an instance of this class."""

    ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS = Option(False, doc="""
    In certain cases, making a cut can actually cause a previously reducible
    concept to become a proper, irreducible concept. Assuming this can never
    happen can increase performance significantly, however the obtained results
    are not strictly accurate.  """)

    CUT_ONE_APPROXIMATION = Option(False, doc="""
    When determining the MIP for |big_phi|, this restricts the set of system
    cuts that are considered to only those that cut the inputs or outputs of a
    single node. This restricted set of cuts scales linearly with the size of
    the system; the full set of all possible bipartitions scales
    exponentially. This approximation is more likely to give theoretically
    accurate results with modular, sparsely-connected, or homogeneous
    networks.""")

    MEASURE = Option('EMD', doc="""
    The measure to use when computing distances between repertoires and
    concepts. A full list of currently installed measures is available by
    calling ``print(pyphi.distance.measures.all())``. Note that some measures
    cannot be used for calculating |big_phi| because they are asymmetric.

    Custom measures can be added using the ``pyphi.distance.measures.register``
    decorator. For example::

        from pyphi.distance import measures

        @measures.register('ALWAYS_ZERO')
        def always_zero(a, b):
            return 0

    This measures can then be used by setting
    ``config.MEASURE = 'ALWAYS_ZERO'``.

    If the measure is asymmetric you should register it using the
    ``asymmetric`` keyword argument. See :mod:`~pyphi.distance` for examples.
    """)

    PARALLEL_CONCEPT_EVALUATION = Option(False, doc="""
    Controls whether concepts are evaluated in parallel when computing
    cause-effect structures.""")

    PARALLEL_CUT_EVALUATION = Option(True, doc="""
    Controls whether system cuts are evaluated in parallel, which is faster but
    requires more memory. If cuts are evaluated sequentially, only two
    |SystemIrreducibilityAnalysis| instances need to be in memory at once.""")

    PARALLEL_COMPLEX_EVALUATION = Option(False, doc="""
    Controls whether systems are evaluated in parallel when computing
    complexes.""")

    NUMBER_OF_CORES = Option(-1, doc="""
    Controls the number of CPU cores used to evaluate unidirectional cuts.
    Negative numbers count backwards from the total number of available cores,
    with ``-1`` meaning 'use all available cores.'""")

    MAXIMUM_CACHE_MEMORY_PERCENTAGE = Option(50, doc="""
    PyPhi employs several in-memory caches to speed up computation. However,
    these can quickly use a lot of memory for large networks or large numbers
    of them; to avoid thrashing, this setting limits the percentage of a
    system's RAM that the caches can collectively use.""")

    CACHE_SIAS = Option(False, doc="""
    PyPhi is equipped with a transparent caching system for
    |SystemIrreducibilityAnalysis| objects which stores them as they are
    computed to avoid having to recompute them later. This makes it easy to
    play around interactively with the program, or to accumulate results with
    minimal effort. For larger projects, however, it is recommended that you
    manage the results explicitly, rather than relying on the cache. For this
    reason it is disabled by default.""")

    CACHE_REPERTOIRES = Option(True, doc="""
    PyPhi caches cause and effect repertoires. This greatly improves speed, but
    can consume a significant amount of memory. If you are experiencing memory
    issues, try disabling this.""")

    CACHE_POTENTIAL_PURVIEWS = Option(True, doc="""
    Controls whether the potential purviews of mechanisms of a network are
    cached. Caching speeds up computations by not recomputing expensive
    reducibility checks, but uses additional memory.""")

    CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA = Option(False, doc="""
    Controls whether a |Subsystem|'s repertoire and MICE caches are cleared
    with |Subsystem.clear_caches()| after computing the
    |SystemIrreducibilityAnalysis|. If you don't need to do any more
    computations after running |compute.sia()|, then enabling this may help
    conserve memory.""")

    CACHING_BACKEND = Option('fs', doc="""
    Controls whether precomputed results are stored and read from a local
    filesystem-based cache in the current directory or from a database. Set
    this to ``'fs'`` for the filesystem, ``'db'`` for the database.""")

    FS_CACHE_VERBOSITY = Option(0, doc="""
    Controls how much caching information is printed if the filesystem cache is
    used. Takes a value between ``0`` and ``11``.""")

    FS_CACHE_DIRECTORY = Option('__pyphi_cache__', doc="""
    If the filesystem is used for caching, the cache will be stored in this
    directory. This directory can be copied and moved around if you want to
    reuse results *e.g.* on a another computer, but it must be in the same
    directory from which Python is being run.""")

    MONGODB_CONFIG = Option({
        'host': 'localhost',
        'port': 27017,
        'database_name': 'pyphi',
        'collection_name': 'cache'
    }, doc="""
    Set the configuration for the MongoDB database backend (only has an
    effect if ``CACHING_BACKEND`` is ``'db'``).""")

    REDIS_CACHE = Option(False, doc="""
    Specifies whether to use Redis to cache |MICE|.""")

    REDIS_CONFIG = Option({
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'test_db': 1,
    }, doc="""
    Configure the Redis database backend. These are the defaults in the
    provided ``redis.conf`` file.""")

    LOG_FILE = Option('pyphi.log', on_change=configure_logging, doc="""
    Controls the name of the log file.""")

    LOG_FILE_LEVEL = Option('INFO', on_change=configure_logging, doc="""
    Controls the level of log messages written to the log
    file. This setting has the same possible values as
    ``LOG_STDOUT_LEVEL``.""")

    LOG_STDOUT_LEVEL = Option('WARNING', on_change=configure_logging, doc="""
    Controls the level of log messages written to standard
    output. Can be one of ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``,
    ``'CRITICAL'``, or ``None``. ``'DEBUG'`` is the least restrictive level and
    will show the most log messages. ``'CRITICAL'`` is the most restrictive
    level and will only display information about fatal errors. If set to
    ``None``, logging to standard output will be disabled entirely.""")

    PROGRESS_BARS = Option(True, doc="""
    Controls whether to show progress bars on the console.

      .. tip::
        If you are iterating over many systems rather than doing one
        long-running calculation, consider disabling this for speed.""")

    PRECISION = Option(6, doc="""
    If ``MEASURE`` is ``EMD``, then the Earth Mover's Distance is calculated
    with an external C++ library that a numerical optimizer to find a good
    approximation. Consequently, systems with analytically zero |big_phi| will
    sometimes be numerically found to have a small but non-zero amount. This
    setting controls the number of decimal places to which PyPhi will consider
    EMD calculations accurate. Values of |big_phi| lower than ``10e-PRECISION``
    will be considered insignificant and treated as zero. The default value is
    about as accurate as the EMD computations get.""")

    VALIDATE_SUBSYSTEM_STATES = Option(True, doc="""
    Controls whether PyPhi checks if the subsystems's state is possible
    (reachable with nonzero probability from some previous state), given the
    subsystem's TPM (**which is conditioned on background conditions**). If
    this is turned off, then **calculated** |big_phi| **values may not be
    valid**, since they may be associated with a subsystem that could never be
    in the given state.""")

    VALIDATE_CONDITIONAL_INDEPENDENCE = Option(True, doc="""
    Controls whether PyPhi checks if a system's TPM is conditionally
    independent.""")

    SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI = Option(False, doc="""
    If set to ``True``, the |big_phi| value of single micro-node subsystems is
    the difference between their unpartitioned |CauseEffectStructure| (a single
    concept) and the null concept. If set to False, their |big_phi| is defined
    to be zero. Single macro-node subsystems may always be cut, regardless of
    circumstances.""")

    REPR_VERBOSITY = Option(2, values=[0, 1, 2], doc="""
    Controls the verbosity of ``__repr__`` methods on PyPhi objects. Can be set
    to ``0``, ``1``, or ``2``. If set to ``1``, calling ``repr`` on PyPhi
    objects will return pretty-formatted and legible strings, excluding
    repertoires. If set to ``2``, ``repr`` calls also include repertoires.

    Although this breaks the convention that ``__repr__`` methods should return
    a representation which can reconstruct the object, readable representations
    are convenient since the Python REPL calls ``repr`` to represent all
    objects in the shell and PyPhi is often used interactively with the
    REPL. If set to ``0``, ``repr`` returns more traditional object
    representations.""")

    PRINT_FRACTIONS = Option(True, doc="""
    Controls whether numbers in a ``repr`` are printed as fractions. Numbers
    are still printed as decimals if the fraction's denominator would be
    large. This only has an effect if ``REPR_VERBOSITY > 0``.""")

    PARTITION_TYPE = Option('BI', doc="""
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

    In addition, in the case of a |small_phi|-tie when computing a |MIC| or
    |MIE|, The ``'TRIPARTITION'`` setting choses the MIP with smallest purview
    instead of the largest (which is the default).

    Finally, if set to ``'ALL'``, all possible partitions will be tested.""")

    PICK_SMALLEST_PURVIEW = Option(False, doc="""
    When computing a |MIC| or |MIE|, it is possible for several MIPs to have
    the same |small_phi| value. If this setting is set to ``True`` the MIP with
    the smallest purview is chosen; otherwise, the one with largest purview is
    chosen.""")

    USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE = Option(False, doc="""
    If set to ``True``, the distance between cause-effect structures (when
    computing a |SystemIrreducibilityAnalysis|) is calculated using the
    difference between the sum of |small_phi| in the cause-effect structures
    instead of the extended EMD.""")

    SYSTEM_CUTS = Option('3.0_STYLE', values=['3.0_STYLE', 'CONCEPT_STYLE'],
                         doc="""
    If set to ``'3.0_STYLE'``, then traditional IIT 3.0 cuts will be used when
    computing |big_phi|. If set to ``'CONCEPT_STYLE'``, then experimental
    concept-style system cuts will be used instead.""")

    def log(self):
        """Log current settings."""
        log.info('PyPhi v%s', __about__.__version__)
        if self._loaded_files:
            log.info('Loaded configuration from %s', self._loaded_files)
        else:
            log.info('Using default configuration (no config file provided)')
        log.info('Current PyPhi configuration:\n %s', str(config))


PYPHI_CONFIG_FILENAME = 'pyphi_config.yml'

config = PyphiConfig()

# Try and load the config file
if os.path.exists(PYPHI_CONFIG_FILENAME):
    config.load_file(PYPHI_CONFIG_FILENAME)

# Log the PyPhi version and loaded configuration
config.log()
