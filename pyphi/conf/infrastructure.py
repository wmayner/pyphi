"""Infrastructure layer of the PyPhi config.

Holds knobs that govern how PyPhi runs (parallelism, caching, logging,
display, validation) but not what it computes. Snapshotted onto every result
object alongside the formalism config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


def _default_parallel_dict(
    sequential_threshold: int, chunksize: int, *, progress: bool = True
) -> dict[str, Any]:
    return {
        "parallel": False,
        "sequential_threshold": sequential_threshold,
        "chunksize": chunksize,
        "progress": progress,
    }


@dataclass(frozen=True)
class InfrastructureConfig:
    """Infrastructure-scoped configuration.

    Knobs in this layer don't change PyPhi's mathematical output — they
    affect performance, caching policy, logging, presentation, and
    validation. Frozen dataclass; replace via :func:`dataclasses.replace`
    or top-level write on the global config.
    """

    parallel: bool = False
    parallel_complex_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**4, 2**6, progress=True)
    )
    parallel_cut_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**10, 2**12, progress=False)
    )
    parallel_concept_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**6, 2**8, progress=True)
    )
    parallel_purview_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**6, 2**8, progress=True)
    )
    parallel_mechanism_partition_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**10, 2**12, progress=True)
    )
    parallel_relation_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**10, 2**12, progress=True)
    )
    parallel_workers: int = -1
    parallel_backend: str = "local"

    maximum_cache_memory_percentage: int = 50
    cache_repertoires: bool = True
    cache_potential_purviews: bool = True
    clear_subsystem_caches_after_computing_sia: bool = False

    log_file: str | Path = "pyphi.log"
    log_file_level: str | None = "INFO"
    log_stdout_level: str | None = "WARNING"

    progress_bars: bool = True
    repr_verbosity: int = 2
    print_fractions: bool = True
    label_separator: str = ""
    welcome_off: bool = False

    validate_subsystem_states: bool = True
    validate_conditional_independence: bool = True
    validate_json_version: bool = True
