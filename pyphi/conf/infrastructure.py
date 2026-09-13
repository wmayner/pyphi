"""Infrastructure layer of the PyPhi config.

Holds knobs that govern how PyPhi runs (parallelism, caching, logging,
display, validation) but not what it computes. Snapshotted onto every result
object alongside the formalism config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Any

from pyphi.conf._helpers import yaml_repr

_VALID_REPR_VERBOSITY = frozenset({0, 1, 2, 3, 4})

_PARALLEL_LEVEL_FIELDS = (
    "parallel_complex_evaluation",
    "parallel_partition_evaluation",
    "parallel_distinction_evaluation",
    "parallel_purview_evaluation",
    "parallel_mechanism_partition_evaluation",
    "parallel_relation_evaluation",
    "parallel_macro_system_evaluation",
)

# The complete key set of every per-level parallel dict (the keys of
# ``_default_parallel_dict``).
_PARALLEL_LEVEL_KEYS = frozenset(
    {"parallel", "sequential_threshold", "chunksize", "progress"}
)


def _default_parallel_dict(
    sequential_threshold: int, chunksize: int, *, progress: bool = True
) -> dict[str, Any]:
    return {
        "parallel": False,
        "sequential_threshold": sequential_threshold,
        "chunksize": chunksize,
        "progress": progress,
    }


def _check_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool; got {type(value).__name__}")


def _check_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be int; got {type(value).__name__}")


@dataclass(frozen=True)
class InfrastructureConfig:
    """Infrastructure-scoped configuration.

    Knobs in this layer don't change PyPhi's mathematical output — they
    affect performance, caching policy, logging, presentation, and
    validation. Frozen dataclass; replace via :func:`dataclasses.replace`
    or top-level write on the global config.
    """

    parallel: bool = False
    """Master switch for parallel computation (default ``False``). ``False``
    runs every level sequentially; ``True`` permits the levels whose own
    ``parallel_*_evaluation`` mapping has ``parallel`` set. See
    :doc:`/howto/parallel`."""
    # Each level's sequential_threshold is the dispatch gate (workloads
    # below it run sequentially) and encodes that level's typical per-item
    # cost: parallel dispatch amortizes at roughly 0.5-1 s of total work
    # (measured in benchmarks/b18_dispatch_gate.py). System partitions,
    # purviews, distinctions, and complexes cost ~1 ms - 10 s per item, so
    # small counts already pay. Mechanism partitions (~50 µs) showed no
    # parallel benefit below 8192 items, and lazy relation construction
    # (~µs, cost dominated by pickling results back) none at any measured
    # size; their thresholds sit at the edge of the measured range. The
    # chunksize governs chunk granularity only.
    parallel_complex_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**4, 2**6, progress=True)
    )
    """Parallelism for the candidate systems of a complexes search: a mapping with the
    keys ``parallel`` (whether this level may run in parallel; ``False`` by
    default), ``sequential_threshold`` (workloads with fewer items than this run
    sequentially regardless), ``chunksize`` (items per task), and ``progress``
    (show a progress bar). A partial mapping merges over the level's defaults;
    unknown keys are rejected. The master switch ``parallel`` must also be on.
    See :doc:`/howto/parallel`."""
    parallel_partition_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**6, 2**12, progress=False)
    )
    """Parallelism for the system partitions of a system irreducibility analysis: a
    mapping with the keys ``parallel`` (whether this level may run in parallel;
    ``False`` by default), ``sequential_threshold`` (workloads with fewer items
    than this run sequentially regardless), ``chunksize`` (items per task), and
    ``progress`` (show a progress bar). A partial mapping merges over the
    level's defaults; unknown keys are rejected. The master switch ``parallel``
    must also be on. See :doc:`/howto/parallel`."""
    parallel_distinction_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**6, 2**8, progress=True)
    )
    """Parallelism for the mechanisms of a cause-effect structure: a mapping with the
    keys ``parallel`` (whether this level may run in parallel; ``False`` by
    default), ``sequential_threshold`` (workloads with fewer items than this run
    sequentially regardless), ``chunksize`` (items per task), and ``progress``
    (show a progress bar). A partial mapping merges over the level's defaults;
    unknown keys are rejected. The master switch ``parallel`` must also be on.
    See :doc:`/howto/parallel`."""
    parallel_purview_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**6, 2**8, progress=True)
    )
    """Parallelism for the candidate purviews of a mechanism: a mapping with the keys
    ``parallel`` (whether this level may run in parallel; ``False`` by default),
    ``sequential_threshold`` (workloads with fewer items than this run
    sequentially regardless), ``chunksize`` (items per task), and ``progress``
    (show a progress bar). A partial mapping merges over the level's defaults;
    unknown keys are rejected. The master switch ``parallel`` must also be on.
    See :doc:`/howto/parallel`."""
    parallel_mechanism_partition_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**13, 2**12, progress=True)
    )
    """Parallelism for the partitions of a mechanism and purview: a mapping with the
    keys ``parallel`` (whether this level may run in parallel; ``False`` by
    default), ``sequential_threshold`` (workloads with fewer items than this run
    sequentially regardless), ``chunksize`` (items per task), and ``progress``
    (show a progress bar). A partial mapping merges over the level's defaults;
    unknown keys are rejected. The master switch ``parallel`` must also be on.
    See :doc:`/howto/parallel`."""
    parallel_relation_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**13, 2**12, progress=True)
    )
    """Parallelism for the relations of a Φ-structure under concrete relation
    computation: a mapping with the keys ``parallel`` (whether this level may
    run in parallel; ``False`` by default), ``sequential_threshold`` (workloads
    with fewer items than this run sequentially regardless), ``chunksize``
    (items per task), and ``progress`` (show a progress bar). A partial mapping
    merges over the level's defaults; unknown keys are rejected. The master
    switch ``parallel`` must also be on. See :doc:`/howto/parallel`."""
    parallel_macro_system_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**4, 2**6, progress=True)
    )
    """Parallelism for the candidate macro systems of a grain search: a mapping with
    the keys ``parallel`` (whether this level may run in parallel; ``False`` by
    default), ``sequential_threshold`` (workloads with fewer items than this run
    sequentially regardless), ``chunksize`` (items per task), and ``progress``
    (show a progress bar). A partial mapping merges over the level's defaults;
    unknown keys are rejected. The master switch ``parallel`` must also be on.
    See :doc:`/howto/parallel`."""
    parallel_workers: int = -1
    """Number of worker processes or threads; ``-1`` (default) uses every
    core."""
    parallel_backend: str = "local"
    """Where parallel work runs: ``"local"`` (default; a pool of processes),
    ``"thread"`` (a pool of threads), or ``"auto"`` (threads on a
    free-threaded interpreter, processes otherwise). Cluster schedulers are
    configured through :mod:`pyphi.parallel` and the ``cluster`` extra."""

    memory_ceiling_percentage: int = 50
    """The share of the memory this process may use that the in-memory
    caches may occupy (default ``50``), above which they evict least
    recently used entries to admit new ones. The denominator is the
    process's cgroup allowance where it has one (a scheduler-managed job, a
    container) and total physical memory otherwise."""
    memory_ceiling_bytes: int | None = None
    """An absolute ceiling on the process's resident memory, above which the
    in-memory caches hold their occupancy steady by evicting least recently
    used entries (default ``None``). Replaces ``memory_ceiling_percentage``
    when set, for an allowance no cgroup reports. It is compared against
    the whole process's resident memory, of which the caches are usually a
    small part, so size it from what the process may use: the caches get
    the ceiling less the interpreter, the substrate, and working space."""
    cache_repertoires: bool = True
    """Cache the repertoires computed within a system (default ``True``).
    ``False`` recomputes them on every use, which is slower but bounds
    memory."""
    cache_potential_purviews: bool = True
    """Cache each mechanism's connectivity-pruned candidate purviews on the
    substrate (default ``True``)."""
    cache_macro_construction: bool = True
    """Cache the mapping-independent intermediates of macro-unit
    construction (the discounted transition matrix and the per-grain
    sequence-class distributions) per substrate, so candidate units that
    share a footprint, update grain, and apportionment reuse them (default
    ``True``). Results are identical either way."""
    clear_system_caches_after_computing_sia: bool = False
    """Clear a system's caches after each system irreducibility analysis
    (default ``False``). Frees memory in sweeps over many systems at the
    cost of recomputing what a later analysis of the same system would
    have reused."""
    disk_cache_results: bool = False
    """Persist top-level results (system irreducibility analyses and
    cause-effect structures) to a content-addressed cache on disk, in
    ``__pyphi_cache__`` under the working directory, and reuse them across
    runs (default ``False``). The key includes the configuration and the
    code version, so a changed setting or release never returns a stale
    result. See :doc:`/howto/cache`."""

    progress_bars: bool = True
    """Show progress bars during long computations (default ``True``)."""
    repr_verbosity: int = 2
    """How much a result's ``repr`` shows, ``0`` to ``4``: ``0`` the one-line
    compact form; ``1`` the card without expensive embedded grids such as a
    substrate's TPM; ``2`` (default) the standard card; ``3`` the card plus
    all mathematical content, such as partition cut grids and selection
    margins; ``4`` that plus a provenance section recording how, when, and
    by what code the result was computed."""
    repr_max_table_rows: int = 50
    """Maximum rows shown in a collection table (distinctions, relations,
    account links) in a result's text or HTML rendering (default ``50``).
    Larger collections are truncated with a "… N more" line; the full data
    is always available from the object itself and ``to_pandas()``."""
    print_fractions: bool = True
    """When ``True`` (default), a probability in text output that is close to
    a simple fraction (denominator at most 128) is printed as that fraction;
    otherwise probabilities print as decimals at the configured
    precision."""
    label_separator: str = ""
    """The string placed between unit labels when a set of units is written
    as one label (default ``""``, so units A and B print as ``AB``; ``","``
    gives ``A,B``)."""
    welcome_off: bool = False
    """Suppress the welcome message printed when PyPhi is imported (default
    ``False``). The environment variable ``PYPHI_WELCOME_OFF`` has the same
    effect. Controls only the welcome; the agent note has its own switch."""
    agent_note_off: bool = False
    """Suppress the note printed to standard error when PyPhi is imported
    under an AI coding agent, detected through the ``CLAUDECODE`` or
    ``PYPHI_AGENT`` environment variables (default ``False``). The
    environment variable ``PYPHI_AGENT_NOTE_OFF`` has the same effect.
    Independent of ``welcome_off``: the two messages have different
    audiences and channels."""

    validate_system_states: bool = True
    """When ``True`` (default), constructing a system checks that its state
    can be reached under the substrate's dynamics and raises
    :class:`~pyphi.exceptions.StateUnreachableForwardsError` otherwise,
    since a state with no possible predecessor has no defined analysis."""
    validate_conditional_independence: bool = True
    """When ``True`` (default), constructing a substrate checks that its
    units are conditionally independent given the previous state and
    raises :class:`~pyphi.exceptions.ConditionallyDependentError`
    otherwise. See :doc:`/theory/conditional-independence`."""

    validate_connectivity: bool = True
    """When ``True`` (default), a substrate's connectivity matrix is checked
    against the connections its TPM implies, and a matrix that omits a real
    connection (which would silently marginalize a true dependency and
    under-count φ) is rejected with a ``ValueError`` naming the missing
    connections. Declaring an unused connection stays legal. Set ``False``
    for a deliberately permissive matrix."""

    validate_phi_bounds: bool = False
    """When ``True``, every result checks its φ against the certified upper
    bound of Zaeemzadeh et al. (2024) and raises ``BoundViolationError`` on
    an overshoot within the bound's domain, which would prove a formalism
    bug. Off by default, since it adds bound arithmetic to the hot path;
    intended for continuous integration and debugging. Outside the
    certified domain (non-binary units, other measures) the check is
    skipped."""

    validate_config: bool = True
    """When ``True`` (default), combinations of options are checked eagerly
    on ``override`` and when a configuration file is loaded, so that an IIT
    version paired with a measure or scheme it does not define is rejected
    with a ``ConfigurationError`` naming the two fields and a fix (see
    :mod:`pyphi.conf.constraints`). Set ``False`` to experiment with
    unsupported combinations."""

    __repr__ = yaml_repr

    def __post_init__(self) -> None:
        # A frozen config must not share mutable containers with callers,
        # presets, or snapshots; the per-level parallel mappings are stored
        # as immutable (and hashable) FrozenMaps. A partial mapping merges
        # over that level's tuned defaults (so setting one key never resets
        # the others to a call-site fallback); unknown keys are rejected.
        from pyphi.data_structures import FrozenMap

        level_defaults = None
        for name in _PARALLEL_LEVEL_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a Mapping; got {type(value).__name__}")
            if isinstance(value, FrozenMap) and set(value) == _PARALLEL_LEVEL_KEYS:
                continue  # already normalized (the common case: snapshots)
            if level_defaults is None:
                level_defaults = {
                    f.name: f.default_factory()  # type: ignore[misc]
                    for f in fields(self)
                    if f.name in _PARALLEL_LEVEL_FIELDS
                }
            default = level_defaults[name]
            unknown = sorted(set(value) - _PARALLEL_LEVEL_KEYS)
            if unknown:
                from pyphi.conf._field_routing import ConfigurationError

                raise ConfigurationError(
                    f"{name} has unknown key(s) {unknown}; valid keys: {sorted(default)}"
                )
            object.__setattr__(self, name, FrozenMap({**default, **dict(value)}))
        _check_bool("parallel", self.parallel)
        _check_int("parallel_workers", self.parallel_workers)
        _check_int(
            "memory_ceiling_percentage",
            self.memory_ceiling_percentage,
        )
        if self.memory_ceiling_bytes is not None:
            _check_int("memory_ceiling_bytes", self.memory_ceiling_bytes)
        _check_bool("cache_repertoires", self.cache_repertoires)
        _check_bool("cache_potential_purviews", self.cache_potential_purviews)
        _check_bool("cache_macro_construction", self.cache_macro_construction)
        _check_bool(
            "clear_system_caches_after_computing_sia",
            self.clear_system_caches_after_computing_sia,
        )
        _check_bool("disk_cache_results", self.disk_cache_results)
        _check_bool("progress_bars", self.progress_bars)
        _check_bool("print_fractions", self.print_fractions)
        _check_bool("welcome_off", self.welcome_off)
        _check_bool("agent_note_off", self.agent_note_off)
        _check_bool("validate_system_states", self.validate_system_states)
        _check_bool(
            "validate_conditional_independence", self.validate_conditional_independence
        )
        _check_bool("validate_connectivity", self.validate_connectivity)
        _check_bool("validate_phi_bounds", self.validate_phi_bounds)
        _check_bool("validate_config", self.validate_config)
        if self.repr_verbosity not in _VALID_REPR_VERBOSITY:
            raise ValueError(
                f"repr_verbosity={self.repr_verbosity!r} not in "
                f"{sorted(_VALID_REPR_VERBOSITY)}"
            )
        if not isinstance(self.label_separator, str):
            raise ValueError(
                f"label_separator must be str; got {type(self.label_separator).__name__}"
            )
        if not isinstance(self.parallel_backend, str):
            raise ValueError(
                "parallel_backend must be str; got "
                f"{type(self.parallel_backend).__name__}"
            )
