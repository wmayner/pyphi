"""Shared performance-harness helpers.

The single source of truth imported by the perf-counter regression gate
(``test/integration/test_perf_counters.py``), its regeneration script
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
from .perf_fixtures import PERF_ONLY_FIXTURES
from .zoo import ALL_FIXTURES

# The golden zoo plus the large fixtures that carry no stored φ values (see
# ``perf_fixtures``). Golden regression reads ``ALL_FIXTURES`` directly and is
# unaffected by the additions here.
FIXTURES_BY_NAME = {f.name: f for f in (*ALL_FIXTURES, *PERF_ONLY_FIXTURES)}
GRAINS = (
    "repertoires",
    "mechanism_mips",
    "phi_structure",
    "sia",
    "specified_state",
)

# Hot frames as (file_substring, funcname), tuned against live cProfile stats.
# Each consumer imports this canonical list. Two regression classes are
# covered, and a new frame belongs to one of them:
#
# Redundant work — the same operation performed more often than necessary.
# Counted at PyPhi frames: the repertoire kernel, the mechanism-MIP search,
# relations enumeration, and ``config.override`` (the per-partition override
# blow-up that motivated the gate).
#
# Cost per operation — the same number of operations, each more expensive.
# Counted at the frames that dictionary collision handling passes through.
# Cache keys are hashed containers, so a key type whose hash stops separating
# distinct keys leaves every count above unchanged while making each cache
# operation a linear scan. ``FrozenMap.__eq__`` is where that scan spends its
# time for the key type PyPhi memoizes repertoires on; ``Mapping.__eq__``
# covers any key type that inherits its equality instead, including one not
# yet written. ``FrozenMap.__getitem__`` tracks per-key iteration cost. All
# three counts are deterministic and independent of ``PYTHONHASHSEED``.
FRAMES: list[tuple[str, str]] = [
    ("repertoire_algebra.py", "cause_repertoire"),
    ("repertoire_algebra.py", "effect_repertoire"),
    ("system.py", "find_mip"),
    ("relations.py", "relations"),
    ("conf/", "override"),
    ("_collections_abc", "__eq__"),
    ("frozen_map.py", "__eq__"),
    ("frozen_map.py", "__getitem__"),
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
    # Frame substrings use forward slashes (e.g. ``conf/``); normalize the
    # profiled filename so matching is platform-independent (Windows paths use
    # backslashes).
    for (filename, _lineno, funcname), (_cc, nc, *_rest) in stats.stats.items():  # type: ignore[attr-defined]
        normalized = filename.replace("\\", "/")
        for sub, func in frames:
            if func == funcname and sub in normalized:
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
    if _is_iit3(fixture):
        # IIT 3.0 defines neither relations nor a specified state.
        return grain not in ("phi_structure", "specified_state")
    return True


def _compute_specified_state(system: System) -> None:
    """Search the system's specified cause and effect states (Eq. 53).

    Sweeps every state of the system as both mechanism and purview, so its
    cost grows with the size of the whole system rather than with any
    mechanism's size — the one layer here that does.
    """
    from pyphi.conf import config
    from pyphi.formalism.iit4 import system_intrinsic_information
    from pyphi.measures.distribution import resolve_mechanism_measure

    system_intrinsic_information(
        system,
        specification_measure=resolve_mechanism_measure(
            config.formalism.iit.specification_measure
        ),
    )


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
        elif grain == "specified_state":
            _compute_specified_state(system)
        else:
            raise ValueError(f"unknown grain {grain!r}")
