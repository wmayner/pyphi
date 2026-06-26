"""Cross-version benchmark harness for IIT 3.0 vs 4.0 at two points in time.

The same script lives in both checkouts (the main checkout and the
pre-refactor worktree). It auto-detects which pyphi generation it's running
against by trying imports, then dispatches to version-appropriate measurement
thunks, network fixtures, and config overrides.

Generations:
- "pre"  — pyphi before the 2.0 refactor: pyphi.compute.sia, pyphi.new_big_phi.phi_structure,
          flat `IIT_VERSION` config, Subsystem objects
- "post" — pyphi after the 2.0 refactor: pyphi.formalism.iit3.sia, pyphi.formalism.iit4.sia,
          layered `config.formalism.iit.version` config, System objects

Per-trial output JSON is written to `results/{generation}/` so a single
`analyze.py` can read both result sets and produce cross-temporal comparisons.

See README.md for rationale.
"""

from __future__ import annotations

import contextlib
import cProfile
import dataclasses
import json
import pstats
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pyphi


# ---------------------------------------------------------------------------
# Generation detection — try post-refactor imports first, fall back to pre.
# ---------------------------------------------------------------------------

try:
    # 2.0 layout: dispatcher in pyphi.formalism reads config.formalism.iit.version
    from pyphi import formalism as _formalism
    from pyphi import examples
    from pyphi.conf import config
    from pyphi.conf.presets import iit3 as _iit3_preset
    from pyphi.conf.presets import iit4_2023 as _iit4_2023_preset
    from pyphi.conf.presets import iit4_2026 as _iit4_2026_preset

    GENERATION = "post"
except ImportError:
    # Pre-2.0 layout: separate compute.sia and new_big_phi.phi_structure entries
    from pyphi import compute as _compute  # type: ignore[attr-defined]
    from pyphi.new_big_phi import phi_structure as _phi_structure  # type: ignore[attr-defined]
    from pyphi import examples
    from pyphi import config

    GENERATION = "pre"


RESULTS_DIR = Path(__file__).parent / "results" / GENERATION


# ---------------------------------------------------------------------------
# Sequential execution — keep all work in-process so cProfile captures it
# ---------------------------------------------------------------------------

if GENERATION == "pre":
    # Pre-refactor: the parallel engine is the `MapReduce` class, and several
    # call sites pass a truthy config dict as the `parallel` keyword, bypassing
    # the `if self.parallel:` check so subprocesses spawn even when the global
    # flag is False. Patch `MapReduce.run` to force sequential mode.
    from pyphi.parallel import MapReduce  # type: ignore[attr-defined]

    @contextlib.contextmanager
    def force_sequential_mapreduce() -> Iterator[None]:
        original_run = MapReduce.run

        def patched_run(self: Any) -> Any:
            self.parallel = False
            return original_run(self)

        MapReduce.run = patched_run  # type: ignore[method-assign]
        try:
            yield
        finally:
            MapReduce.run = original_run  # type: ignore[method-assign]
else:
    # Post-refactor: the parallel engine is the `map_reduce()` function; every
    # call site passes `parallel` from config, so the harness's
    # `parallel=False` override already forces serial in-process execution
    # (`map_reduce(..., parallel=False)` runs the map in this process). No
    # monkeypatch needed; this context manager is a no-op kept for symmetry.
    @contextlib.contextmanager
    def force_sequential_mapreduce() -> Iterator[None]:
        yield


# ---------------------------------------------------------------------------
# Network fixtures (generation-specific)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class NetworkFixture:
    name: str
    builder: Callable[[], Any]
    n_nodes: int


def _logistic3_k8_system() -> Any:
    """3-node fully-connected logistic substrate (k=8, weights 0.3) at (0,0,0).

    The cap-biting network from the Eq-23 oracle: phi_2023 ~ 0.037,
    phi_2026 ~ 0.003, so the 2026 ii(s) cap binds at a non-trivial value
    (unlike the standard examples, where the 2026 variant short-circuits to 0).
    """
    import itertools

    import numpy as np

    from pyphi import Substrate, System

    k = 8.0
    weights = np.full((3, 3), 0.3)
    cm = np.ones((3, 3), dtype=int)
    tpm = np.zeros((8, 3))
    for i, s in enumerate(itertools.product([-1, 1], repeat=3)):
        for j in range(3):
            inp = sum(weights[ki, j] * s[ki] for ki in range(3))
            tpm[i, j] = 1.0 / (1.0 + np.exp(-k * inp))
    return System(Substrate(tpm, cm), (0, 0, 0))


if GENERATION == "pre":
    NETWORKS: dict[str, NetworkFixture] = {
        "basic": NetworkFixture("basic", examples.basic_subsystem, 3),
        "fig4": NetworkFixture("fig4", examples.fig4_subsystem, 3),
        "xor": NetworkFixture("xor", examples.xor_subsystem, 3),
        "macro": NetworkFixture("macro", examples.macro_subsystem, 4),
        "residue": NetworkFixture("residue", examples.residue_subsystem, 5),
        "rule154": NetworkFixture("rule154", examples.rule154_subsystem, 5),
    }
else:
    NETWORKS = {
        "basic": NetworkFixture("basic", examples.basic_system, 3),
        "fig4": NetworkFixture("fig4", examples.fig4_system, 3),
        "xor": NetworkFixture("xor", examples.xor_system, 3),
        "macro": NetworkFixture("macro", examples.macro_system, 4),
        "residue": NetworkFixture("residue", examples.residue_system, 5),
        "rule154": NetworkFixture("rule154", examples.rule154_system, 5),
        "logistic3_k8": NetworkFixture("logistic3_k8", _logistic3_k8_system, 3),
    }


# ---------------------------------------------------------------------------
# Measurements (generation-specific entry points)
# ---------------------------------------------------------------------------

if GENERATION == "pre":
    MEASUREMENTS = ("iit3_sia", "iit4_phi_structure")
else:
    MEASUREMENTS = ("iit3_sia", "iit4_sia_2023", "iit4_sia_2026")


def measurement_callable(name: str, target: Any) -> Callable[[], Any]:
    """Return a zero-arg thunk that runs the named measurement on the target.

    target is a Subsystem (pre) or System (post).
    """
    if GENERATION == "pre":
        if name == "iit3_sia":
            return lambda: _compute.sia(target)
        if name == "iit4_phi_structure":
            return lambda: _phi_structure(target)
    else:
        # All post-refactor measurements go through the dispatcher; the
        # difference is the `version` config override applied before the call.
        if name in ("iit3_sia", "iit4_sia_2023", "iit4_sia_2026"):
            return lambda: _formalism.sia(target)
    raise ValueError(f"unknown measurement for generation={GENERATION}: {name}")


# ---------------------------------------------------------------------------
# Config overrides (generation-specific schema)
# ---------------------------------------------------------------------------

def overrides_for(measurement: str) -> dict[str, Any]:
    """Return the config.override(**kwargs) dict for a given measurement.

    Per-generation pinning:
    - "pre"  uses the flat config (IIT_VERSION, SYSTEM_PARTITION_TYPE, ...)
    - "post" uses field-routed kwargs that map to nested config (version, ...)

    Each version uses its native distance metric — they aren't interchangeable.
    """
    if GENERATION == "pre":
        base: dict[str, Any] = {
            "PARALLEL": False,
            "PROGRESS_BARS": False,
            "WELCOME_OFF": True,
            "CACHE_REPERTOIRES": True,
            "CACHE_POTENTIAL_PURVIEWS": True,
            "SHORTCIRCUIT_SIA": True,
            "CES_DISTANCE": "SUM_SMALL_PHI",
            "PARTITION_TYPE": "ALL",
            "SYSTEM_PARTITION_TYPE": "DIRECTED_BI",
            "SYSTEM_CUTS": "3.0_STYLE",
        }
        if measurement == "iit3_sia":
            return {**base, "IIT_VERSION": 3.0, "REPERTOIRE_DISTANCE": "EMD"}
        if measurement == "iit4_phi_structure":
            return {
                **base,
                "IIT_VERSION": 4.0,
                "REPERTOIRE_DISTANCE": "GENERALIZED_INTRINSIC_DIFFERENCE",
            }
        raise ValueError(measurement)

    # GENERATION == "post"
    # Use the canonical formalism presets from pyphi.conf.presets, which
    # configure the per-formalism sub-namespaces wholesale (partition scheme,
    # tie resolution, measures, etc.) to paper-faithful values. Add
    # infrastructure pins on top for benchmark reproducibility.
    infra = {
        "parallel": False,
        "progress_bars": False,
        "welcome_off": True,
    }
    if measurement == "iit3_sia":
        return {**infra, **_iit3_preset}
    if measurement == "iit4_sia_2023":
        return {**infra, **_iit4_2023_preset}
    if measurement == "iit4_sia_2026":
        return {**infra, **_iit4_2026_preset}
    raise ValueError(measurement)


def snapshot_config() -> dict[str, Any]:
    """Snapshot the config values that vary across the comparison.

    Both generations report a flat dict of name → value for the names that
    matter for the comparison. Read inside the override context so the
    snapshot reflects what was actually active during the trial.
    """
    if GENERATION == "pre":
        names = [
            "IIT_VERSION", "PARALLEL", "REPERTOIRE_DISTANCE", "CES_DISTANCE",
            "SYSTEM_PARTITION_TYPE", "PARTITION_TYPE", "SHORTCIRCUIT_SIA",
        ]
        return {n: getattr(config, n, None) for n in names}
    # post
    iit = config.formalism.iit
    return {
        "version": iit.version,
        "mechanism_phi_measure": iit.mechanism_phi_measure,
        "system_phi_measure": iit.system_phi_measure,
        "ces_measure": iit.ces_measure,
        "mechanism_partition_scheme": iit.mechanism_partition_scheme,
        "system_partition_scheme": iit.system_partition_scheme,
        "sia_tie_resolution": list(iit.sia_tie_resolution),
        "parallel": config.infrastructure.parallel,
    }


# ---------------------------------------------------------------------------
# Phase anchors — function names to extract cumulative time for
# ---------------------------------------------------------------------------

# Stored as (module_substr, function_name) — substring match against pstats
# filename keys. List is data, easy to extend. Functions absent from a given
# trial just omit from the output. Both generations share this list.
PHASE_ANCHORS: list[tuple[str, str]] = [
    # ---- Pre-refactor pyphi locations ----
    ("pyphi/compute/subsystem.py", "_ces"),
    ("pyphi/compute/subsystem.py", "ces"),
    ("pyphi/compute/subsystem.py", "_sia"),
    ("pyphi/compute/subsystem.py", "evaluate_cut"),
    ("pyphi/new_big_phi/__init__.py", "phi_structure"),
    ("pyphi/new_big_phi/__init__.py", "sia"),
    ("pyphi/new_big_phi/__init__.py", "system_intrinsic_information"),
    ("pyphi/new_big_phi/__init__.py", "_integration_value_for_state"),
    ("pyphi/new_big_phi/__init__.py", "resolve_system_state"),
    ("pyphi/new_big_phi/__init__.py", "evaluate_partition"),
    # ---- Post-refactor pyphi locations ----
    ("pyphi/formalism/iit3", "sia"),
    ("pyphi/formalism/iit3", "_sia"),
    ("pyphi/formalism/iit3", "ces"),
    ("pyphi/formalism/iit3", "evaluate_cut"),
    ("pyphi/formalism/iit4", "sia"),
    ("pyphi/formalism/iit4", "ces"),
    ("pyphi/formalism/iit4", "evaluate_partition"),
    ("pyphi/formalism/queries.py", "sia"),
    # ---- Shared across generations ----
    ("pyphi/subsystem.py", "cause_repertoire"),
    ("pyphi/subsystem.py", "effect_repertoire"),
    ("pyphi/subsystem.py", "concept"),
    ("pyphi/subsystem.py", "mic"),
    ("pyphi/subsystem.py", "mie"),
    ("pyphi/subsystem.py", "_find_mip_single_state"),
    ("pyphi/subsystem.py", "find_mice"),
    ("pyphi/subsystem.py", "intrinsic_information"),
    ("pyphi/subsystem.py", "forward_repertoire"),
    ("pyphi/system.py", "intrinsic_information"),
    # Relations (4.0 only)
    ("pyphi/relations.py", "all_relations"),
    ("pyphi/relations.py", "compute_relations"),
    # Distance metrics
    ("pyphi/metrics/distribution.py", "generalized_intrinsic_difference"),
    ("pyphi/metrics/distribution.py", "intrinsic_difference"),
    ("pyphi/metrics/distribution.py", "emd"),
    ("pyphi/metrics/distribution.py", "hamming_emd"),
    ("pyphi/metrics/distribution.py", "effect_emd"),
    ("pyphi/metrics/ces.py", "ces_distance"),
    ("pyphi/measures", "ces_distance"),
]


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_trial(
    measurement: str,
    network_name: str,
    seed: int,
    trial: int,
) -> dict[str, Any]:
    """Run a single trial, returning a dict ready for JSON serialization.

    Caches are freshly built per trial: each call constructs a new
    target object via the fixture's builder, so no state leaks between trials.
    """
    fixture = NETWORKS[network_name]
    overrides = overrides_for(measurement)

    with config.override(**overrides), force_sequential_mapreduce():
        target = fixture.builder()
        # Pre-generation Subsystem has clear_caches; post-generation System may not.
        if hasattr(target, "clear_caches"):
            target.clear_caches()
        thunk = measurement_callable(measurement, target)

        config_snapshot = snapshot_config()

        profiler = cProfile.Profile()
        wall_start_ns = time.perf_counter_ns()
        profiler.enable()
        try:
            result = thunk()
        finally:
            profiler.disable()
        wall_end_ns = time.perf_counter_ns()

    pstats_path = unique_path(
        RESULTS_DIR,
        f"{network_name}_{measurement}_seed{seed}_trial{trial}",
        ".pstats",
    )
    profiler.dump_stats(str(pstats_path))

    phase_times = extract_phase_times(pstats_path)

    record: dict[str, Any] = {
        "generation": GENERATION,
        "measurement": measurement,
        "network": network_name,
        "n_nodes": fixture.n_nodes,
        "seed": seed,
        "trial": trial,
        "wall_ns": wall_end_ns - wall_start_ns,
        "wall_s": (wall_end_ns - wall_start_ns) / 1e9,
        "phase_times_s": phase_times,
        "pstats_path": pstats_path.name,
        "config_snapshot": _coerce_jsonable(config_snapshot),
        "pyphi_version": getattr(pyphi, "__version__", "unknown"),
        "result_phi": _safe_phi(result),
    }
    return record


def _coerce_jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Best-effort coercion of config snapshot values to JSON-friendly types."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out


def _safe_phi(result: Any) -> float | None:
    """Best-effort extraction of phi for sanity-checking; never raises."""
    try:
        if hasattr(result, "phi"):
            return float(result.phi)
        if hasattr(result, "sia") and hasattr(result.sia, "phi"):
            return float(result.sia.phi)
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def extract_phase_times(pstats_path: Path) -> dict[str, float]:
    """Pull cumulative time (seconds) for each anchor function from pstats."""
    stats = pstats.Stats(str(pstats_path))
    raw: dict[tuple[str, int, str], tuple[int, int, float, float, dict]] = (
        getattr(stats, "stats")
    )
    out: dict[str, float] = {}
    for module_substr, func_name in PHASE_ANCHORS:
        cumtime = 0.0
        hits = 0
        for key_tuple, entry in raw.items():
            filename, fname = key_tuple[0], key_tuple[2]
            if fname == func_name and module_substr in filename:
                cumtime += entry[3]
                hits += 1
        if hits > 0:
            key = f"{module_substr}::{func_name}"
            out[key] = cumtime
    return out


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """Return a non-clobbering path: stem.suffix, then stem_v2.suffix, ..."""
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"{stem}{suffix}"
    if not base.exists():
        return base
    n = 2
    while True:
        candidate = directory / f"{stem}_v{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def write_record(record: dict[str, Any]) -> Path:
    """Write a trial record to a non-clobbering JSON file. Returns the path."""
    stem = (
        f"{record['network']}_{record['measurement']}"
        f"_seed{record['seed']}_trial{record['trial']}"
    )
    path = unique_path(RESULTS_DIR, stem, ".json")
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return path
