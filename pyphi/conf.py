# conf.py
"""
Configuring PyPhi
~~~~~~~~~~~~~~~~~

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
      'CACHE_POTENTIAL_PURVIEWS': True,
      ...

Setting can be changed on the fly by assigning them a new value:

    >>> pyphi.config.PROGRESS_BARS = False

It is also possible to manually load a configuration file:

    >>> pyphi.config.load_file('pyphi_config.yml')

Or load a dictionary of configuration values:

    >>> pyphi.config.load_dict({'PRECISION': 1})


The ``config`` API
~~~~~~~~~~~~~~~~~~
"""

# pylint: disable=protected-access

import contextlib
import functools
import logging
import logging.config
import os
import pprint
import shutil
import tempfile
from copy import copy
from importlib.metadata import version
from pathlib import Path
from typing import Mapping
from warnings import warn

import toolz
import yaml

from . import constants
from .deferred.ray import ray, NO_RAY
from .warnings import MissingOptionalDependenciesWarning, PyPhiWarning


log = logging.getLogger(__name__)

_VALID_LOG_LEVELS = [None, "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]

# Flag to prevent writing to the managed config until we've tried to load an
# existing one
_LOADED = False


class ConfigurationError(ValueError):
    pass


class ConfigurationWarning(UserWarning):
    pass


# TODO(4.0) deprecate options
def deprecated(option):
    # Don't warn until config is loaded
    # TODO onchange is not triggered?
    if _LOADED:
        warn(
            f"The {option} configuration option is deprecated and will be removed in a future version.",
            FutureWarning,
            stacklevel=2,
        )


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

    def __init__(self, default, values=None, type=None, on_change=None, doc=None):
        self.default = default
        self.values = values
        self.type = type
        self.on_change = on_change
        self.doc = doc
        self.__doc__ = self._docstring()

    def __set_name__(self, owner, name):
        self.name = name

    def _docstring(self):
        default = "``default={}``".format(repr(self.default))

        values = (
            ", ``values={}``".format(repr(self.values))
            if self.values is not None
            else ""
        )

        on_change = (
            ", ``on_change={}``".format(self.on_change.__name__)
            if self.on_change is not None
            else ""
        )

        return "{}{}{}\n{}".format(default, values, on_change, self.doc or "")

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        return obj._values[self.name]

    def __set__(self, obj, value):
        previous = obj._values[self.name]
        try:
            self._validate(value)
            obj._values[self.name] = value
            self._callback(obj)
        except ConfigurationError as e:
            obj._values[self.name] = previous
            raise e

    def _validate(self, value):
        """Validate the new value."""
        if self.type is not None and not isinstance(value, self.type):
            raise ConfigurationError(
                "{} must be of type {} for {}; got {}".format(
                    value, self.type, self.name, type(value)
                )
            )
        if self.values and value not in self.values:
            raise ConfigurationError(
                "{} ({}) is not a valid value for {}; must be one of:\n    {}".format(
                    value,
                    type(value),
                    self.name,
                    "\n    ".join(["{} ({})".format(v, type(v)) for v in self.values]),
                )
            )

    def _callback(self, obj):
        """Trigger any callbacks."""
        if self.on_change is not None:
            self.on_change(obj, self)


class Config:
    """Base configuration object.

    See ``PyphiConfig`` for usage.
    """

    def __init__(self, on_change=None):
        self._values = {}
        self._loaded_files = []
        self._on_change = on_change

        # Set the default value of each ``Option``
        for name, opt in self.options().items():
            opt._validate(opt.default)
            self._values[name] = opt.default

        # Call hooks for each Option
        # (This must happen *after* all default values are set so that
        # logging can be properly configured.
        for opt in self.options().values():
            opt._callback(self)

            # Insert config-wide hook
            def hook(func):
                if func is None:

                    def config_callback(*args, **kwargs):
                        self._callback(self)

                    return config_callback

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    func(*args, **kwargs)
                    self._callback(self)

                return wrapper

            opt.on_change = hook(opt.on_change)

        # Call config-wide hook
        self._callback(self)

    def __repr__(self):
        return pprint.pformat(self._values, indent=2)

    def __str__(self):
        return repr(self)

    def __setattr__(self, name, value):
        if name.startswith("_") or name in self.options().keys():
            super().__setattr__(name, value)
        else:
            raise ConfigurationError("{} is not a valid config option".format(name))

    def __getitem__(self, name):
        return self._values[name]

    def __eq__(self, other):
        return self._values == other._values

    def _callback(self, obj):
        """Config-wide callback, called when any option is changed."""
        if self._on_change is not None:
            self._on_change(obj)

    @classmethod
    def options(cls):
        """Return a dictionary of the ``Option`` objects for this config."""
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

        with open(filename, mode="rt") as f:
            self.load_dict(yaml.safe_load(f))

        self._loaded_files.append(filename)

    def to_yaml(self, filename):
        """Write config to a YAML file."""
        with open(filename, mode="wt") as f:
            yaml.dump(self._values, f)
        return filename

    def snapshot(self):
        """Return a snapshot of the current values of this configuration."""
        return copy(self._values)

    to_dict = snapshot

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

    def diff(self, other):
        """Return differences between this configuration and another.

        Returns:
            tuple[dict]: A tuple of two dictionaries. The first contains the
            differing values of this configuration; the second contains those of
            the other.
        """
        different_items = toolz.diff(
            self.to_dict().items(), other.to_dict().items(), default=None
        )
        left, right = zip(*different_items)
        return dict(left), dict(right)


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


def configure_logging(conf, opt):
    """Reconfigure PyPhi logging based on the current configuration."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(name)s] %(levelname)s "
                    "%(processName)s: %(message)s"
                }
            },
            "handlers": {
                "file": {
                    "level": conf.LOG_FILE_LEVEL,
                    "filename": conf.LOG_FILE,
                    "class": "logging.FileHandler",
                    "formatter": "standard",
                },
                "stdout": {
                    "level": conf.LOG_STDOUT_LEVEL,
                    "class": "pyphi.log.TqdmHandler",
                    "formatter": "standard",
                },
            },
            "root": {
                "level": "DEBUG",
                "handlers": (["file"] if conf.LOG_FILE_LEVEL else [])
                + (["stdout"] if conf.LOG_STDOUT_LEVEL else []),
            },
        }
    )


def on_change_distinction_phi_normalization(obj, opt):
    if _LOADED:
        warn(
            """
    IMPORTANT: Changes to `DISTINCTION_PHI_NORMALIZATION` will not be reflected in
    new MICE computations for existing Subsystem objects if the MICE have been
    previously computed, since they are cached.

    Make sure to call `subsystem.clear_caches()` before re-computing MICE with
    the new setting.
            """,
            category=PyPhiWarning,
            stacklevel=6,
        )


def on_change_parallel_global(obj, opt):
    if NO_RAY and obj[opt.name]:
        warn(
            message=(
                f"""
    '{opt.name}' option: """
                + MissingOptionalDependenciesWarning.MSG.format(dependencies="parallel")
            ),
            category=MissingOptionalDependenciesWarning,
            stacklevel=6,
        )


def on_change_parallel_suboption(obj, opt):
    if NO_RAY and obj[opt.name].get("parallel"):
        warn(
            message=(
                f"""
    '{opt.name}' option: """
                + MissingOptionalDependenciesWarning.MSG.format(dependencies="parallel")
            ),
            category=MissingOptionalDependenciesWarning,
            stacklevel=6,
        )
        return


# TODO(configuration) actual causation parallel config
class PyphiConfig(Config):
    """``pyphi.config`` is an instance of this class."""

    IIT_VERSION = Option(
        4.0,
        doc="""
    The version of the theory to use.""",
    )

    ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS = Option(
        False,
        type=bool,
        doc="""
    In certain cases, making a cut can actually cause a previously reducible
    concept to become a proper, irreducible concept. Assuming this can never
    happen can increase performance significantly, however the obtained results
    are not strictly accurate.""",
    )

    REPERTOIRE_DISTANCE = Option(
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        doc="""
    The measure to use when computing distances between repertoires and
    concepts. A full list of currently installed measures is available by
    calling ``print(pyphi.distance.measures.all())``. Note that some measures
    cannot be used for calculating |big_phi| because they are asymmetric.

    Custom measures can be added using the ``pyphi.distance.measures.register``
    decorator. For example::

        from pyphi.metrics.distribution import measures

        @measures.register('ALWAYS_ZERO')
        def always_zero(a, b):
            return 0

    This measure can then be used by setting
    ``config.REPERTOIRE_DISTANCE = 'ALWAYS_ZERO'``.

    If the measure is asymmetric you should register it using the
    ``asymmetric`` keyword argument. See :mod:`~pyphi.distance` for examples.
    """,
    )

    REPERTOIRE_DISTANCE_INFORMATION = Option(
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        doc="""
    The repertoire distance used for evaluating information specified by a
    mechanism (i.e., finding the maximal state with respect to a purview).
    """,
    )

    CES_DISTANCE = Option(
        "SUM_SMALL_PHI",
        doc="""
    The measure to use when computing distances between cause-effect structures.

    See documentation for ``config.REPERTOIRE_DISTANCE`` for more information on
    configuring measures.
    """,
    )

    ACTUAL_CAUSATION_MEASURE = Option(
        "PMI",
        doc="""
    The measure to use when computing the pointwise information between state
    probabilities in the actual causation module.

    See documentation for ``config.REPERTOIRE_DISTANCE`` for more information on
    configuring measures.
    """,
    )

    PARALLEL = Option(
        False,
        type=bool,
        on_change=on_change_parallel_global,
        doc="""
    Global switch to turn off parallelization: if ``False``, parallelization is
    never used, regardless of parallelization settings for individual options;
    otherwise parallelization is determined by those settings.

    IMPORTANT: Parallelization requires extra dependencies; please install PyPhi
    with `pyphi[parallel]` to enable parallelization.""",
    )

    PARALLEL_COMPLEX_EVALUATION = Option(
        dict(
            parallel=False,
            sequential_threshold=2**4,
            chunksize=2**6,
            progress=True,
        ),
        type=Mapping,
        on_change=on_change_parallel_suboption,
        doc="""
    Controls parallel evaluation of candidate systems within a network.""",
    )

    PARALLEL_CUT_EVALUATION = Option(
        dict(
            parallel=False,
            sequential_threshold=2**10,
            chunksize=2**12,
            progress=True,
        ),
        type=Mapping,
        on_change=on_change_parallel_suboption,
        doc="""
    Controls parallel evaluation of system partitions.""",
    )

    PARALLEL_CONCEPT_EVALUATION = Option(
        dict(
            parallel=False,
            sequential_threshold=2**6,
            chunksize=2**8,
            progress=True,
        ),
        type=Mapping,
        on_change=on_change_parallel_suboption,
        doc="""
    Controls parallel evaluation of candidate mechanisms.""",
    )

    PARALLEL_PURVIEW_EVALUATION = Option(
        dict(
            parallel=False,
            sequential_threshold=2**6,
            chunksize=2**8,
            progress=True,
        ),
        type=Mapping,
        on_change=on_change_parallel_suboption,
        doc="""
    Controls parallel evaluation of candidate purviews.""",
    )

    PARALLEL_MECHANISM_PARTITION_EVALUATION = Option(
        dict(
            parallel=False,
            sequential_threshold=2**10,
            chunksize=2**12,
            progress=True,
        ),
        type=Mapping,
        on_change=on_change_parallel_suboption,
        doc="""
    Controls parallel evaluation of mechanism partitions.""",
    )

    PARALLEL_RELATION_EVALUATION = Option(
        dict(
            parallel=False,
            sequential_threshold=2**10,
            chunksize=2**12,
            progress=True,
        ),
        type=Mapping,
        on_change=on_change_parallel_suboption,
        doc="""
    Controls parallel evaluation of relations.

    Only applies if RELATION_COMPUTATION = 'CONCRETE'.
    """,
    )

    NUMBER_OF_CORES = Option(
        -1,
        type=int,
        doc="""
    Controls the number of CPU cores used in parallel evaluation. Negative
    numbers count backwards from the total number of available cores, with
    ``-1`` meaning all available cores.""",
    )

    MAXIMUM_CACHE_MEMORY_PERCENTAGE = Option(
        50,
        type=int,
        doc="""
    PyPhi employs several in-memory caches to speed up computation. However,
    these can quickly use a lot of memory for large networks or large numbers
    of them; to avoid thrashing, this setting limits the percentage of a
    system's RAM that the caches can collectively use.""",
    )

    RAY_CONFIG = Option(
        dict(),
        type=dict,
        doc="""
    Keyword arguments to ``ray.init()``. Controls the initialization of the Ray
    cluster used for parallelization / distributed computation.""",
    )

    CACHE_REPERTOIRES = Option(
        True,
        type=bool,
        doc="""
    PyPhi caches cause and effect repertoires. This greatly improves speed, but
    can consume a significant amount of memory. If you are experiencing memory
    issues, try disabling this.""",
    )

    CACHE_POTENTIAL_PURVIEWS = Option(
        True,
        type=bool,
        doc="""
    Controls whether the potential purviews of mechanisms of a network are
    cached. Caching speeds up computations by not recomputing expensive
    reducibility checks, but uses additional memory.""",
    )

    CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA = Option(
        False,
        type=bool,
        doc="""
    Controls whether a |Subsystem|'s repertoire and MICE caches are cleared
    with |Subsystem.clear_caches()| after computing the
    |SystemIrreducibilityAnalysis|. If you don't need to do any more
    computations after running |compute.sia()|, then enabling this may help
    conserve memory.""",
    )

    REDIS_CACHE = Option(
        False,
        type=bool,
        doc="""
    Specifies whether to use Redis to cache |MICE|.""",
    )

    REDIS_CONFIG = Option(
        {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "test_db": 1,
        },
        type=dict,
        doc="""
    Configure the Redis database backend. These are the defaults in the
    provided ``redis.conf`` file.""",
    )

    WELCOME_OFF = Option(
        False,
        type=bool,
        doc="""
    Specifies whether to suppress the welcome message when PyPhi is imported.

    Alternatively, you may suppress the message by setting the environment
    variable ``PYPHI_WELCOME_OFF`` to any value in your shell:

    .. code-block:: bash

        export PYPHI_WELCOME_OFF='yes'

    The message will not print if either this option is ``True`` or the
    environment variable is set.""",
    )

    LOG_FILE = Option(
        "pyphi.log",
        type=(str, Path),
        on_change=configure_logging,
        doc="""
    Controls the name of the log file.""",
    )

    LOG_FILE_LEVEL = Option(
        "INFO",
        values=_VALID_LOG_LEVELS,
        on_change=configure_logging,
        doc="""
    Controls the level of log messages written to the log
    file. This setting has the same possible values as
    ``LOG_STDOUT_LEVEL``.""",
    )

    LOG_STDOUT_LEVEL = Option(
        "WARNING",
        values=_VALID_LOG_LEVELS,
        on_change=configure_logging,
        doc="""
    Controls the level of log messages written to standard
    output. Can be one of ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``,
    ``'CRITICAL'``, or ``None``. ``'DEBUG'`` is the least restrictive level and
    will show the most log messages. ``'CRITICAL'`` is the most restrictive
    level and will only display information about fatal errors. If set to
    ``None``, logging to standard output will be disabled entirely.""",
    )

    PROGRESS_BARS = Option(
        True,
        type=bool,
        doc="""
    Controls whether to show progress bars on the console.

      .. tip::
        If you are iterating over many systems rather than doing one
        long-running calculation, consider disabling this for speed.""",
    )

    PRECISION = Option(
        13,
        type=int,
        # TODO(4.0) update docstring
        doc="""
    If ``REPERTOIRE_DISTANCE`` is ``EMD``, then the Earth Mover's Distance is
    calculated with an external C++ library that a numerical optimizer to find a
    good approximation. Consequently, systems with analytically zero |big_phi|
    will sometimes be numerically found to have a small but non-zero amount.
    This setting controls the number of decimal places to which PyPhi will
    consider EMD calculations accurate. Values of |big_phi| lower than
    ``10**(-PRECISION)`` will be considered insignificant and treated as zero.
    The default value is about as accurate as the EMD computations get.""",
    )

    VALIDATE_SUBSYSTEM_STATES = Option(
        True,
        type=bool,
        doc="""
    Controls whether PyPhi checks if the subsystems's state is possible
    (reachable with nonzero probability from some previous state), given the
    subsystem's TPM (**which is conditioned on background conditions**). If
    this is turned off, then **calculated** |big_phi| **values may not be
    valid**, since they may be associated with a subsystem that could never be
    in the given state.""",
    )

    VALIDATE_CONDITIONAL_INDEPENDENCE = Option(
        True,
        type=bool,
        doc="""
    Controls whether PyPhi checks if a system's TPM is conditionally
    independent.""",
    )

    SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI = Option(
        False,
        type=bool,
        doc="""
    If set to ``True``, the |big_phi| value of single micro-node subsystems is
    the difference between their unpartitioned |CauseEffectStructure| (a single
    concept) and the null concept. If set to False, their |big_phi| is defined
    to be zero. Single macro-node subsystems may always be cut, regardless of
    circumstances.""",
    )

    LABEL_SEPARATOR = Option(
        "",
        type=str,
        doc="""
    Separator to use between labels in the string representation of a set of nodes.""",
    )

    REPR_VERBOSITY = Option(
        2,
        type=int,
        values=[0, 1, 2],
        doc="""
    Controls the verbosity of ``__repr__`` methods on PyPhi objects. Can be set
    to ``0``, ``1``, or ``2``. If set to ``1``, calling ``repr`` on PyPhi
    objects will return pretty-formatted and legible strings, excluding
    repertoires. If set to ``2``, ``repr`` calls also include repertoires.

    Although this breaks the convention that ``__repr__`` methods should return
    a representation which can reconstruct the object, readable representations
    are convenient since the Python REPL calls ``repr`` to represent all
    objects in the shell and PyPhi is often used interactively with the
    REPL. If set to ``0``, ``repr`` returns more traditional object
    representations.""",
    )

    PRINT_FRACTIONS = Option(
        True,
        type=bool,
        doc="""
    Controls whether numbers in a ``repr`` are printed as fractions. Numbers
    are still printed as decimals if the fraction's denominator would be
    large. This only has an effect if ``REPR_VERBOSITY > 0``.""",
    )

    PARTITION_TYPE = Option(
        "ALL",
        doc="""
    Controls the type of partition used for |small_phi| computations.

    If set to ``'BI'``, partitions will have two parts.

    If set to ``'TRI'``, partitions will have three parts. In addition,
    computations will only consider partitions that strictly partition the
    mechanism. That is, for the mechanism ``(A, B)`` and purview ``(B, C, D)``
    the partition::

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

    Finally, if set to ``'ALL'``, all possible partitions will be tested.

    You can experiment with custom partitioning strategies using the
    ``pyphi.partition.partition_types.register`` decorator. For example::

        from pyphi.models import KPartition, Part
        from pyphi.partition import partition_types

        @partition_types.register('SINGLE_NODE')
        def single_node_partitions(mechanism, purview, node_labels=None):
           for element in mechanism:
               element = tuple([element])
               others = tuple(sorted(set(mechanism) - set(element)))

               part1 = Part(mechanism=element, purview=())
               part2 = Part(mechanism=others, purview=purview)

               yield KPartition(part1, part2, node_labels=node_labels)

    This generates the set of partitions that cut connections between a single
    mechanism element and the entire purview. The mechanism and purview of each
    |Part| remain undivided - only connections *between* parts are severed.

    You can use this new partititioning scheme by setting
    ``config.PARTITION_TYPE = 'SINGLE_NODE'``.

    See :mod:`~pyphi.partition` for more examples.""",
    )

    SYSTEM_PARTITION_INCLUDE_COMPLETE = Option(
        False,
        type=bool,
        doc="""
    Whether to include the complete partition in partition set.

    Currently only applies to "SET_UNI/BI".
    """,
    )

    # TODO(4.0) finish documenting
    SYSTEM_PARTITION_TYPE = Option(
        "SET_UNI/BI",
        doc="""
    Controls the system partitioning scheme.
    """,
    )

    DISTINCTION_PHI_NORMALIZATION = Option(
        "NUM_CONNECTIONS_CUT",
        on_change=on_change_distinction_phi_normalization,
        values=["NONE", "NUM_CONNECTIONS_CUT"],
        doc="""
    Controls how distinction |small_phi| values are normalized for determining the MIP.
    """,
    )

    RELATION_COMPUTATION = Option(
        "CONCRETE",
        values=["CONCRETE", "ANALYTICAL"],
        doc="""
    Controls how relations are computed.
    """,
    )

    STATE_TIE_RESOLUTION = Option(
        "PHI",
        doc="""
    Controls how ties among states are resolved.

    NOTE: Operation is `max`.
    """,
    )

    MIP_TIE_RESOLUTION = Option(
        ["NORMALIZED_PHI", "NEGATIVE_PHI"],
        doc="""
    Controls how ties among mechanism partitions are resolved.

    NOTE: Operation is `min`; with the default values, the minimum normalized
    phi is taken, then in case of ties, the maximal un-normalized phi is taken.
    """,
    )

    PURVIEW_TIE_RESOLUTION = Option(
        "PHI",
        doc="""
    Controls how ties among purviews are resolved.

    NOTE: Operation is `max`.
    """,
    )

    SYSTEM_CUTS = Option(
        "3.0_STYLE",
        values=["3.0_STYLE", "CONCEPT_STYLE"],
        doc="""
    If set to ``'3.0_STYLE'``, then traditional IIT 3.0 cuts will be used when
    computing |big_phi|. If set to ``'CONCEPT_STYLE'``, then experimental
    concept-style system cuts will be used instead.""",
    )

    SHORTCIRCUIT_SIA = Option(
        True,
        type=bool,
        doc="""
    Controls whether SIA calculations short-circuit if an a-priori reducibility
    condition is found.""",
    )

    def log(self):
        """Log current settings."""
        log.info("PyPhi v%s", version("pyphi"))
        if self._loaded_files:
            log.info("Loaded configuration from %s", self._loaded_files)
        else:
            log.info("Using default configuration (no configuration file " "provided)")
        log.info("Current PyPhi configuration:\n %s", str(self))


def _validate_combinations(config, options, combinations, valid_if_included=True):
    values = tuple(map(config._values.get, options))
    if valid_if_included ^ (values in combinations):
        msg = "\n".join(
            [
                "invalid combination:",
                "{options}",
                "must {valid_if_in}form one of the following combinations:",
                "{combinations}",
                "got:",
                "{values}",
            ]
        )
        text = {
            name: "  " + "\n  ".join(map(str, value))
            for name, value in dict(
                options=options,
                combinations=combinations,
                values=values,
            ).items()
        }
        raise ConfigurationError(
            msg.format(valid_if_in=("" if valid_if_included else "NOT "), **text)
        )


def validate_combinations(
    config, options, valid_combinations=set(), invalid_combinations=set()
):
    _validate_combinations(
        config, options, combinations=valid_combinations, valid_if_included=True
    )
    _validate_combinations(
        config, options, combinations=invalid_combinations, valid_if_included=False
    )


def validate(config):
    pass


PYPHI_USER_CONFIG_PATH = Path("pyphi_config.yml")
PYPHI_MANAGED_CONFIG_PATH = (
    constants.DISK_CACHE_LOCATION / "config" / PYPHI_USER_CONFIG_PATH
)
PYPHI_MANAGED_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_yaml(data, path):
    try:
        # delete=True in case there's an error, but ignore if we fail to delete
        # after successfully renaming the file
        with tempfile.NamedTemporaryFile(mode="wt", delete=True) as f:
            yaml.dump(data, f)
            # Use `shutil.move()` instead of `rename()` to properly deal with
            # atomic writes across filesystems
            shutil.move(f.name, path)
    except FileNotFoundError:
        pass
    return path


def write_to_cache(config):
    atomic_write_yaml(config.snapshot(), PYPHI_MANAGED_CONFIG_PATH)


def on_change_global(config):
    validate(config)
    if _LOADED:
        # Write any changes to disk for remote processes to load
        write_to_cache(config)


# Instantiate the config object
config = PyphiConfig(on_change=on_change_global)


def on_driver():
    if ray.is_initialized():
        try:
            # Ignore warning log
            current_level = ray.runtime_context.logger.level
            ray.runtime_context.logger.setLevel("ERROR")
            ray.get_runtime_context().get_task_id()
            ray.runtime_context.logger.setLevel(current_level)
            return False
        except AssertionError:
            pass
    return True


def driver_config():
    """Handle configuration for the main instance."""
    # We're a main instance; load the user config
    try:
        config.load_file(PYPHI_USER_CONFIG_PATH)
    except FileNotFoundError:
        pass
    # Ensure write to disk in case no config was loaded (i.e. onchange was not
    # triggered)
    write_to_cache(config)


def remote_config():
    """Handle configuration for remote instances."""
    # We're in a remote instance; load the PyPhi-managed config
    config.load_file(PYPHI_MANAGED_CONFIG_PATH)
    # Disable progress bars on remote processes
    config.PROGRESS_BARS = False


if NO_RAY or on_driver():
    driver_config()
else:
    remote_config()

# We've loaded/written; now we can allow loading
_LOADED = True

# Log the PyPhi version and loaded configuration
config.log()


def fallback(*args):
    """Return the first argument that is not ``None``."""
    for arg in args:
        if arg is not None:
            return arg


PARALLEL_KWARGS = [
    "reduce_func",
    "reduce_kwargs",
    "parallel",
    "ordered",
    "total",
    "chunksize",
    "sequential_threshold",
    "max_depth",
    "max_size",
    "max_leaves",
    "branch_factor",
    "shortcircuit_func",
    "shortcircuit_callback",
    "shortcircuit_callback_args",
    "inflight_limit",
    "progress",
    "desc",
    "map_kwargs",
]


def parallel_kwargs(option_kwargs, **user_kwargs):
    """Return the kwargs for a parallel function call.

    Applies user overrides to the global configuration.
    """
    kwargs = copy(option_kwargs)
    if not config.PROGRESS_BARS:
        kwargs["progress"] = False
    if not config.PARALLEL:
        kwargs["parallel"] = False
    kwargs.update(
        {
            user_kwarg: value
            for user_kwarg, value in user_kwargs.items()
            if user_kwarg in PARALLEL_KWARGS
        }
    )
    return kwargs
