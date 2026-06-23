"""Shared performance-harness helpers.

The single source of truth imported by the perf-counter regression gate
(``test/test_perf_counters.py``), its regeneration script
(``scripts/gen_perf_counts.py``), and the ASV benchmark suite (via the
``benchmarks/benchmarks/_fixtures.py`` path shim). Keeping the FRAMES list and
the grain dispatch here means they cannot drift between consumers.

Call counts are exact and reproducible for a given computation; profiler
overhead inflates wall time only, never the counts.
"""

from __future__ import annotations

import cProfile
import pstats
from collections.abc import Callable

from pyphi import System

from .compute import _compute_mechanism_mips
from .compute import _compute_repertoires
from .compute import _compute_sia
from .zoo import ALL_FIXTURES

FIXTURES_BY_NAME = {f.name: f for f in ALL_FIXTURES}
GRAINS = ("repertoires", "mechanism_mips", "phi_structure", "sia")

# Hot frames as (file_substring, funcname). Tuned against live cProfile stats in
# Task 7; this is the canonical list every consumer imports.
FRAMES: list[tuple[str, str]] = [
    ("system.py", "find_mip"),
    ("system.py", "cause_repertoire"),
    ("system.py", "effect_repertoire"),
    ("relations.py", "relations"),
    ("conf/", "override"),
]


def count_calls(
    thunk: Callable[[], object],
    frames: list[tuple[str, str]],
) -> dict[str, int]:
    """Run ``thunk`` under cProfile; return total call counts for ``frames``."""
    profiler = cProfile.Profile()
    profiler.enable()
    thunk()
    profiler.disable()
    stats = pstats.Stats(profiler)
    counts = {f"{sub}:{func}": 0 for sub, func in frames}
    # stats.stats maps (filename, lineno, funcname) -> (cc, nc, tt, ct, callers).
    for (filename, _lineno, funcname), (_cc, nc, *_rest) in stats.stats.items():  # type: ignore[attr-defined]
        for sub, func in frames:
            if func == funcname and sub in filename:
                counts[f"{sub}:{func}"] += nc
    return counts


def _is_iit3(fixture) -> bool:
    iit = fixture.config_overrides.get("iit")
    if iit is not None and hasattr(iit, "version"):
        return iit.version == "IIT_3_0"
    return fixture.config_overrides.get("FORMALISM") == "IIT_3_0"


def build_system(fixture) -> System:
    substrate = fixture.build_substrate()
    nodes = fixture.node_indices or substrate.node_indices
    return System(substrate, fixture.state, nodes)


def applies(fixture, grain: str) -> bool:
    if grain in fixture.skip_layers:
        return False
    return not (grain == "phi_structure" and _is_iit3(fixture))


def run_grain(fixture, grain: str) -> None:
    """Run one grain of one fixture inside its config context (no-op stash)."""
    with fixture.config_context():
        system = build_system(fixture)
        if grain == "repertoires":
            _compute_repertoires(system, lambda _a: "")
        elif grain == "mechanism_mips":
            _compute_mechanism_mips(system, lambda _a: "")
        elif grain == "phi_structure":
            system.ces()
        elif grain == "sia":
            _compute_sia(system, lambda _a: "", 3.0 if _is_iit3(fixture) else 4.0)
        else:
            raise ValueError(f"unknown grain {grain!r}")
