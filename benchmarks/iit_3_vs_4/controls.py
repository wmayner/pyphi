"""Like-for-like controls disentangling the IIT 4.0 cross-temporal gap.

The harness maps pre ``new_big_phi.phi_structure`` (full structure: SIA +
distinctions + relations) against post ``formalism.sia`` (system phi only).
Those are different computations, so the headline "speedup" conflates three
things: entry-point *scope*, system-partition *scheme*, and the refactor
itself. These controls isolate each on the post side; the pre side numbers are
read from the existing pre profiles (the ``sia`` and ``ces`` frames inside
``phi_structure``).

Run (post checkout):  uv run python -m benchmarks.iit_3_vs_4.controls
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets


def _time(fn, repeats: int = 3) -> float:
    """Median wall seconds over `repeats` fresh calls."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e9)
    return float(np.median(samples))


_QUIET = {"progress_bars": False, "welcome_off": True}


def _system(name):
    return {"macro": examples.macro_system, "rule154": examples.rule154_system}[name]()


def control_ces(name: str) -> float:
    """Post-side CES-only time (the bulk of what pre phi_structure does).

    A fresh System is built per call, so its per-object caches do not leak
    between repeats.
    """
    with config.override(**presets.iit4_2023, **_QUIET):
        return _time(lambda: _system(name).ces())


def control_sia_matched_scheme(name: str) -> tuple[float, int]:
    """Post-side SIA with DIRECTED_BIPARTITION (matching pre's DIRECTED_BI).

    Returns (median_wall_s, n_partitions_evaluated).
    """
    from pyphi.formalism import iit4
    from pyphi.partition import system_partitions

    seen: list = []
    real = system_partitions

    def counting(*a, **k):
        parts = list(real(*a, **k))
        seen.append(len(parts))
        return iter(parts)

    overrides = {**presets.iit4_2023, **_QUIET, "system_partition_scheme": "DIRECTED_BIPARTITION"}
    with config.override(**overrides):
        iit4.system_partitions = counting  # type: ignore[assignment]
        try:
            wall = _time(lambda: _system(name).sia(), repeats=3)
            seen.clear()
            _system(name).sia()  # one more to capture the partition count
        finally:
            iit4.system_partitions = real  # type: ignore[assignment]
    return wall, (seen[-1] if seen else -1)


def main() -> int:
    print("Cross-temporal controls (post-refactor side; pre numbers from existing profiles)\n")
    # Pre numbers read from results/pre/*phi_structure*.pstats frames:
    pre = {
        "macro": {"sia_s": 0.026, "sia_parts": 1, "ces_s": 1.349},
        "rule154": {"sia_s": 1.047, "sia_parts": 30, "ces_s": 80.128},
    }
    for name in ("macro", "rule154"):
        print(f"=== {name} ===")
        ces_post = control_ces(name)
        sia_post, parts_post = control_sia_matched_scheme(name)
        p = pre[name]
        print(f"  CES-only:        pre {p['ces_s']:8.3f}s   post {ces_post:8.3f}s   "
              f"-> {p['ces_s']/ces_post:5.2f}x {'faster' if ces_post < p['ces_s'] else 'SLOWER'}")
        print(f"  SIA (DIRECTED_BIPARTITION, matched scheme):")
        print(f"     pre  {p['sia_s']:8.3f}s  ({p['sia_parts']} partitions)")
        print(f"     post {sia_post:8.3f}s  ({parts_post} partitions)")
        if parts_post > 0:
            print(f"     per-partition: pre {p['sia_s']/max(p['sia_parts'],1)*1000:6.2f}ms  "
                  f"post {sia_post/parts_post*1000:6.2f}ms")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
