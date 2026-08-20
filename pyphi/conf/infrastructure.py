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
    parallel_partition_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**6, 2**12, progress=False)
    )
    parallel_distinction_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**6, 2**8, progress=True)
    )
    parallel_purview_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**6, 2**8, progress=True)
    )
    parallel_mechanism_partition_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**13, 2**12, progress=True)
    )
    parallel_relation_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**13, 2**12, progress=True)
    )
    parallel_macro_system_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**4, 2**6, progress=True)
    )
    parallel_workers: int = -1
    parallel_backend: str = "local"

    # Share of the memory this process may use that in-memory caches may
    # occupy. The denominator is the process's cgroup allowance where it has
    # one — a scheduler-managed job, a container — and total physical memory
    # otherwise, so the share bounds a confined process rather than measuring
    # against a machine it cannot fill.
    memory_ceiling_percentage: int = 50
    # An absolute ceiling on process resident memory, above which in-memory
    # caches hold their occupancy steady, admitting new entries by evicting
    # least recently used ones. Replaces `memory_ceiling_percentage` when
    # set, for an allowance no cgroup reports.
    #
    # Size it from what the process may use, not from expected cache size: it
    # is compared against total resident memory, of which the caches are
    # usually a small part. Sampled through a 21-unit scoped cause-effect
    # structure shard, they held 70-130 MB against 2.6 GB resident, the rest
    # being the interpreter, the substrate TPM, and numpy working space. The
    # allowance the caches actually get is this ceiling less that baseline.
    memory_ceiling_bytes: int | None = None
    cache_repertoires: bool = True
    cache_potential_purviews: bool = True
    # When True (default), the macro TPM construction caches its
    # mapping-independent Steps 1-2 intermediates (the discounted transition
    # matrix and the per-grain sequence-class distributions) per substrate, so
    # candidate units sharing a footprint, update grain, and apportionment
    # structure reuse them. Results are identical either way; set False to
    # disable the cache entirely (no reads or writes).
    cache_macro_construction: bool = True
    clear_system_caches_after_computing_sia: bool = False
    disk_cache_results: bool = False

    progress_bars: bool = True
    repr_verbosity: int = 2
    # Maximum number of rows shown in a collection table (distinctions,
    # relations, account links) in a result's repr/HTML. Larger collections
    # are truncated with a "… N more" indicator; the full data is always
    # available via the object's iterables and ``to_pandas()``.
    repr_max_table_rows: int = 50
    print_fractions: bool = True
    label_separator: str = ""
    welcome_off: bool = False
    # Suppress the note printed to stderr when PyPhi is imported under an AI
    # coding agent. Independent of ``welcome_off``: the two messages have
    # different audiences and different channels.
    agent_note_off: bool = False

    validate_system_states: bool = True
    validate_conditional_independence: bool = True

    # When True (default), a substrate's connectivity matrix is checked against
    # the edges its TPM implies, and an under-specified CM (one that omits a
    # real edge, silently marginalizing a true dependency and under-counting
    # phi) is rejected with a ``ValueError`` naming the missing edge(s).
    # Over-specification (declaring an unused edge) stays legal. Set False to
    # opt out (e.g. for a deliberately permissive CM).
    validate_connectivity: bool = True

    # When True, every result-construction site checks its phi against the
    # theorem-certified Zaeemzadeh (2024) upper bound and raises
    # BoundViolationError on an in-domain overshoot (a proof of a formalism
    # bug). Off by default — it adds per-construction bound arithmetic to the
    # hot path; intended for CI and debugging. Outside the certified domain
    # (non-binary, non-GID, etc.) the check is silently skipped.
    validate_phi_bounds: bool = False

    # When True (default), config-combination constraints are checked eagerly
    # on ``override`` and ``load_yaml`` (see :mod:`pyphi.conf.constraints`),
    # rejecting silently-wrong combinations (e.g. an IIT version paired with a
    # measure it does not define) with a ``ConfigurationError`` naming the two
    # conflicting fields and a fix. Set False to opt out (e.g. for
    # experimentation with unsupported combinations).
    validate_config: bool = True

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
