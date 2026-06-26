"""P17 Part 4 — hot-path config-behavior sweep.

Audits whether the configuration flags the IIT 4.0 hot paths actually read
behave as their docstrings say, with particular attention to the two
config bugs that existed before the 2.0 refactor:

1. ``PARALLEL=False`` still spawned subprocesses, because several call sites
   passed a truthy per-level config dict as the ``parallel`` keyword,
   bypassing the ``if self.parallel`` guard in the old ``MapReduce`` class.
2. ``IIT_3_0`` paired with ``GENERALIZED_INTRINSIC_DIFFERENCE`` raised a raw
   ``AttributeError`` deep in the compute path rather than a clean error.

Each section runs a small experiment and prints a
``(flag, documented, actual, severity)`` row. ``severity`` is ``MATCH`` when
documented behavior equals observed behavior. This is an audit, not a
benchmark: it asserts behavior, it does not time anything.

Run (post checkout):  uv run python -m benchmarks.iit_3_vs_4.config_sweep
"""

from __future__ import annotations

from typing import Any

from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.conf._field_routing import ConfigurationError

_QUIET = {"progress_bars": False, "welcome_off": True}

# A per-level parallel dict requesting parallel mode with no sequential
# threshold, so the parallel path is taken whenever the global gate allows it.
_PARALLEL_LEVEL = {
    "parallel": True,
    "sequential_threshold": 1,
    "chunksize": 1,
    "progress": False,
}
_SEQUENTIAL_LEVEL = {**_PARALLEL_LEVEL, "parallel": False}


def _row(flag: str, documented: str, actual: str, severity: str) -> None:
    print(f"  flag:       {flag}")
    print(f"  documented: {documented}")
    print(f"  actual:     {actual}")
    print(f"  severity:   {severity}\n")


def audit_parallel_gate() -> None:
    """``infrastructure.parallel`` gates every per-level parallel switch.

    Spies on the two branches of :func:`pyphi.parallel.map_reduce` (the
    in-process ``_map_sequential`` and the ``default_scheduler`` that owns
    subprocess dispatch) while computing a ``macro`` SIA under the four
    combinations of the global flag and the partition-evaluation flag.
    """
    from pyphi import parallel
    from pyphi.parallel import scheduler

    counts = {"sequential": 0, "scheduler": 0}
    real_seq = parallel._map_sequential
    real_sched = scheduler.default_scheduler

    def seq_spy(*a: Any, **k: Any) -> Any:
        counts["sequential"] += 1
        return real_seq(*a, **k)

    def sched_spy(*a: Any, **k: Any) -> Any:
        counts["scheduler"] += 1
        return real_sched(*a, **k)

    # (label, global parallel flag, per-level parallel dict)
    scenarios = [
        ("global=F, level=F", False, _SEQUENTIAL_LEVEL),
        ("global=F, level=T", False, _PARALLEL_LEVEL),
        ("global=T, level=T", True, _PARALLEL_LEVEL),
        ("global=T, level=F", True, _SEQUENTIAL_LEVEL),
    ]
    observed: dict[str, bool] = {}
    for label, global_flag, level in scenarios:
        parallel._map_sequential = seq_spy  # type: ignore[assignment]
        scheduler.default_scheduler = sched_spy  # type: ignore[assignment]
        counts["sequential"] = counts["scheduler"] = 0
        try:
            with config.override(
                **presets.iit4_2023,
                **_QUIET,
                parallel=global_flag,
                parallel_partition_evaluation=level,
            ):
                examples.macro_system().sia()
        finally:
            parallel._map_sequential = real_seq  # type: ignore[assignment]
            scheduler.default_scheduler = real_sched  # type: ignore[assignment]
        observed[label] = counts["scheduler"] > 0

    gate_holds = (
        not observed["global=F, level=F"]
        and not observed["global=F, level=T"]
        and observed["global=T, level=T"]
        and not observed["global=T, level=F"]
    )
    actual = (
        "scheduler (subprocess dispatch) entered only when BOTH the global flag "
        "and the per-level flag are True; global=False forces the in-process "
        "sequential branch even when the per-level dict requests parallel "
        f"(scheduler-entered map: {observed})"
    )
    _row(
        "infrastructure.parallel + parallel_partition_evaluation['parallel']",
        "Global switch gates the per-level parallel flags; parallelization "
        "needs both on.",
        actual,
        "MATCH (pre-2.0 'PARALLEL=False still spawns' bug fixed)"
        if gate_holds
        else "MISMATCH",
    )


def audit_shortcircuit_sia() -> None:
    """``formalism.iit.shortcircuit_sia`` is a φ-preserving early return.

    On every strongly-connected standard example the flag must not change φ;
    a pure-noise substrate (no specified cause/effect) exercises the live
    short-circuit path, where the flag toggles the early null-SIA return.
    """
    import numpy as np

    from pyphi import Substrate
    from pyphi import System

    names = ["basic_system", "xor_system", "grid3_system", "macro_system"]
    invariant = True
    with config.override(**presets.iit4_2023, **_QUIET):
        for name in names:
            build = getattr(examples, name)
            phi_on = build().sia().phi
            with config.override(shortcircuit_sia=False):
                phi_off = build().sia().phi
            invariant = invariant and abs(phi_on - phi_off) < 1e-12

    # Pure-noise 2-node substrate: every node outputs 0.5, so the system has
    # no specified cause or effect and the short-circuit path is reached.
    noise = System(Substrate(np.full((4, 2), 0.5), np.ones((2, 2), dtype=int)), (0, 0))
    with config.override(**presets.iit4_2023, **_QUIET, validate_connectivity=False):
        sia_on = noise.sia()
        with config.override(shortcircuit_sia=False):
            sia_off = noise.sia()
    reasons_on = [r.name for r in (getattr(sia_on, "reasons", None) or [])]
    path_live = "NO_CAUSE" in reasons_on or "NO_EFFECT" in reasons_on
    noise_invariant = abs(sia_on.phi - sia_off.phi) < 1e-12

    _row(
        "formalism.iit.shortcircuit_sia",
        "Short-circuit (return a null SIA) when the system has no cause or "
        "effect; an optimization that does not change φ.",
        f"φ identical with the flag on/off on {names} (invariant={invariant}); "
        f"the noise substrate reaches the live short-circuit "
        f"(reasons={reasons_on}, φ-invariant={noise_invariant}).",
        "MATCH" if invariant and path_live and noise_invariant else "MISMATCH",
    )


def audit_cache_flags() -> None:
    """``cache_repertoires`` / ``cache_potential_purviews`` never change φ."""
    build = examples.basic_system
    caches_on = {"cache_repertoires": True, "cache_potential_purviews": True}
    caches_off = {"cache_repertoires": False, "cache_potential_purviews": False}
    with config.override(**presets.iit4_2023, **_QUIET, **caches_on):
        phi_on = build().sia().phi
    with config.override(**presets.iit4_2023, **_QUIET, **caches_off):
        phi_off = build().sia().phi
    _row(
        "infrastructure.cache_repertoires / cache_potential_purviews",
        "Caching policy; affects performance only, never the computed value.",
        f"basic_system φ identical with both caches on ({phi_on:.6f}) and off "
        f"({phi_off:.6f}).",
        "MATCH" if abs(phi_on - phi_off) < 1e-12 else "MISMATCH",
    )


def audit_config_combinations() -> None:
    """Cross-field combos either compute or raise a clean ``ConfigurationError``.

    Sweeps (version x measure x system scheme) on ``basic_system`` and checks
    that no combination raises a raw, untyped exception at override time. Then
    confirms the eager check's value: with ``validate_config=False`` the
    incompatible IIT 3.0 + GID combo is accepted at config time but raises a
    *typed* error at compute time, while the default eager check rejects it up
    front.
    """
    versions = ["IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026"]
    measures = ["GENERALIZED_INTRINSIC_DIFFERENCE", "INTRINSIC_INFORMATION", "EMD"]
    schemes = ["DIRECTED_SET_PARTITION", "DIRECTED_BIPARTITION"]

    raw_exceptions: list[str] = []
    ok = 0
    rejected = 0
    for version in versions:
        for measure in measures:
            for scheme in schemes:
                overrides = {
                    "iit.version": version,
                    "iit.mechanism_phi_measure": measure,
                    "iit.system_phi_measure": measure,
                    "iit.system_partition_scheme": scheme,
                }
                try:
                    with config.override({**_QUIET, **overrides}):
                        pass
                    ok += 1
                except ConfigurationError:
                    rejected += 1
                except Exception as exc:
                    raw_exceptions.append(
                        f"{version}/{measure}/{scheme}: {type(exc).__name__}"
                    )

    # Eager-validation value: IIT 3.0 + GID with the check off.
    bad = {
        "iit.version": "IIT_3_0",
        "iit.mechanism_phi_measure": "GENERALIZED_INTRINSIC_DIFFERENCE",
    }
    compute_error = "none"
    with config.override({**_QUIET, **bad, "validate_config": False}):
        try:
            examples.basic_system().sia()
        except Exception as exc:
            compute_error = type(exc).__name__
    eager_caught = False
    try:
        with config.override({**_QUIET, **bad}):
            pass
    except ConfigurationError:
        eager_caught = True

    raw_suffix = f" {raw_exceptions}" if raw_exceptions else ""
    _row(
        "formalism.iit (version x measure x scheme) / validate_config",
        "Incompatible combinations are rejected eagerly at override time with a "
        "ConfigurationError naming the conflict and a fix.",
        f"{ok} combos compute, {rejected} cleanly rejected, "
        f"{len(raw_exceptions)} raw exceptions{raw_suffix}. "
        f"With validate_config off, IIT 3.0 + GID raises a typed "
        f"{compute_error} at compute time; the default eager check rejects it "
        f"at override time (eager_caught={eager_caught}).",
        "MATCH (pre-2.0 'IIT 3.0 + GID raw AttributeError' bug fixed)"
        if not raw_exceptions and eager_caught
        else "MISMATCH",
    )


def main() -> int:
    print("P17 Part 4 — hot-path config-behavior sweep (post-refactor)\n")
    audit_parallel_gate()
    audit_shortcircuit_sia()
    audit_cache_flags()
    audit_config_combinations()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
