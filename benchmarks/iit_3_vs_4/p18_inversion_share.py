"""Measure the cause-inversion share of IIT 4.0 SIA wall time.

Two regimes:

1. **Full-substrate** (system == substrate, n = 6): the regime behind the
   P18 negative result.
2. **Embedded** (fixed 6-node system inside substrates of growing size):
   the standard small-system-in-large-substrate workflow. The inversion
   (``_cause_marginal_factored``) builds ``pr_joint`` over the *full
   substrate* dimensions and is recomputed for every partition of the cut
   system, so at fixed system size its cost grows ``a^N`` with substrate
   size ``N`` while the partition-search size stays constant.

The 6-node core weight matrix is identical across all embedded runs (only
background rows/columns are added), so the partition search is the same
size in every run and the share trend isolates the substrate-size effect.

Usage::

    uv run python benchmarks/iit_3_vs_4/p18_inversion_share.py \
        [--seed 6001] [--substrate-sizes 6 8 10 12] [--run-label LABEL]

Results (raw per-run profile rows + shares + the seed) are written to
``results/p18_inversion_share_seed<seed>[_<label>].json``; existing files
are never overwritten (a ``_v2``/``_v3`` suffix is added instead).
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
from pathlib import Path

import numpy as np

from pyphi.provenance import format_stem
from pyphi.provenance import unique_path

CORE_N = 6
CORE_DENSITY = 0.35
COUPLING_MEAN = 1.0
COUPLING_SD = 0.5
BACKGROUND_COUPLING_SCALE = 0.3
TEMPERATURE = 0.25
INVERSION_FUNC = "_cause_marginal_factored"


def _core_weights(seed: int) -> np.ndarray:
    """The 6-node sparse ferromagnetic Ising core (harness recipe)."""
    rng = np.random.default_rng(seed)
    mask = rng.random((CORE_N, CORE_N)) < CORE_DENSITY
    np.fill_diagonal(mask, True)
    return mask * rng.normal(COUPLING_MEAN, COUPLING_SD, size=(CORE_N, CORE_N))


def _embedded_system(core: np.ndarray, substrate_n: int, seed: int):
    """The fixed 6-node core embedded in a ``substrate_n``-node substrate.

    Background rows/columns get weak random couplings (drawn from an
    isolated RNG seeded per substrate size), so the core's induced
    connectivity — and therefore the partition search — is identical in
    every run.
    """
    from pyphi import System
    from pyphi.substrate_generator import build_substrate

    rng = np.random.default_rng(seed + substrate_n)
    weights = rng.normal(
        0.0, COUPLING_SD * BACKGROUND_COUPLING_SCALE, size=(substrate_n, substrate_n)
    )
    np.fill_diagonal(weights, COUPLING_MEAN)
    weights[:CORE_N, :CORE_N] = core
    substrate = build_substrate("ising", weights, temperature=TEMPERATURE)
    return System(substrate, (0,) * substrate_n, node_indices=tuple(range(CORE_N)))


def _profile_sia(system) -> dict:
    """Run ``system.sia()`` under cProfile; return the inversion share."""
    profiler = cProfile.Profile()
    profiler.enable()
    sia = system.sia()
    profiler.disable()

    stats = pstats.Stats(profiler)
    total = max(stats.total_tt, 1e-12)  # pyright: ignore[reportAttributeAccessIssue]
    inversion_cumtime = 0.0
    inversion_ncalls = 0
    top_rows = []
    for (filename, lineno, funcname), (
        ncalls,
        _primitive,
        tottime,
        cumtime,
        _callers,
    ) in stats.stats.items():  # pyright: ignore[reportAttributeAccessIssue]
        if funcname == INVERSION_FUNC:
            inversion_cumtime += cumtime
            inversion_ncalls += ncalls
        top_rows.append((cumtime, tottime, ncalls, f"{filename}:{lineno}:{funcname}"))
    top_rows.sort(reverse=True)
    return {
        "phi": float(sia.phi),
        "total_seconds": total,
        "inversion_cumtime_seconds": inversion_cumtime,
        "inversion_ncalls": inversion_ncalls,
        "inversion_share": inversion_cumtime / total,
        "top_functions": [
            {"cumtime": c, "tottime": t, "ncalls": n, "function": f}
            for c, t, n, f in top_rows[:25]
        ],
    }


def _output_path(seed: int, run_label: str | None) -> Path:
    results_dir = Path(__file__).parent / "results"
    return unique_path(
        results_dir,
        format_stem("p18_inversion_share", {"seed": seed}, run_label),
        ".json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument(
        "--substrate-sizes", type=int, nargs="+", default=[6, 8, 10, 12]
    )
    parser.add_argument("--run-label", default=None)
    args = parser.parse_args()

    from pyphi.conf import config, presets

    core = _core_weights(args.seed)
    runs = []
    # Sequential and without SIA short-circuiting, so every run sweeps the
    # identical full partition set and profiles stay in-process.
    with config.override(
        **presets.iit4_2023,
        parallel=False,
        shortcircuit_sia=False,
        progress_bars=False,
    ):
        from pyphi import System
        from pyphi.substrate_generator import build_substrate

        # Regime 1: full-substrate baseline (system == substrate, n = 6).
        baseline = System(
            build_substrate("ising", core, temperature=TEMPERATURE), (0,) * CORE_N
        )
        print(f"full-substrate n={CORE_N} ...", flush=True)
        row = {"regime": "full_substrate", "substrate_n": CORE_N} | _profile_sia(
            baseline
        )
        print(
            f"  share={row['inversion_share']:.1%} "
            f"({row['inversion_ncalls']} calls, {row['total_seconds']:.1f}s)",
            flush=True,
        )
        runs.append(row)

        # Regime 2: the same 6-node system inside growing substrates.
        for substrate_n in args.substrate_sizes:
            system = _embedded_system(core, substrate_n, args.seed)
            print(f"embedded 6-in-{substrate_n} ...", flush=True)
            row = {"regime": "embedded", "substrate_n": substrate_n} | _profile_sia(
                system
            )
            print(
                f"  share={row['inversion_share']:.1%} "
                f"({row['inversion_ncalls']} calls, {row['total_seconds']:.1f}s)",
                flush=True,
            )
            runs.append(row)

    out = {
        "seed": args.seed,
        "core_n": CORE_N,
        "core_density": CORE_DENSITY,
        "background_coupling_scale": BACKGROUND_COUPLING_SCALE,
        "temperature": TEMPERATURE,
        "substrate_sizes": args.substrate_sizes,
        "config": {
            "preset": "iit4_2023",
            "parallel": False,
            "shortcircuit_sia": False,
        },
        "runs": runs,
    }
    path = _output_path(args.seed, args.run_label)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")  # noqa: T201


if __name__ == "__main__":
    main()
