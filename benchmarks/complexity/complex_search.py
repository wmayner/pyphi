"""Scaling of the IIT 3.0 major-complex search.

Times ``Substrate.maximal_complex`` — the search over candidate subsystems for
the maximal complex — on the stochastic-majority ring family from
:mod:`scaling`, under the IIT 3.0 configuration (matching the historical
majority-gate benchmark). This is the outermost level of the computation: it
evaluates a subsystem's Φ (itself the cause-effect structure swept over system
cuts) for each candidate subsystem.

The expected scaling follows from the inner levels: a size-$k$ subsystem's Φ
costs ~$6^k$ (a ~$3^k$ cause-effect structure times a ~$2^k$ cut sweep), and
summing over the $\\binom{n}{k}$ subsystems of each size gives
$\\sum_k \\binom{n}{k} 6^k = 7^n$. So the major-complex search scales as ~$n^p 7^n$.

Run::

    uv run python benchmarks/complexity/complex_search.py --n-max 5
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

import pyphi

from benchmarks.complexity.scaling import RESULTS_DIR
from benchmarks.complexity.scaling import _versioned_path
from benchmarks.complexity.scaling import ring_substrate


@dataclass
class Trial:
    n: int
    trial: int
    seconds: float


def run(beta: float, trials: int, n_min: int, n_max: int, max_seconds: float) -> list[Trial]:
    pyphi.config.progress_bars = False
    results: list[Trial] = []
    for n in range(n_min, n_max + 1):
        substrate = ring_substrate(n, beta)
        state = tuple([1] * n)
        times: list[float] = []
        for t in range(trials):
            start = time.perf_counter()
            with pyphi.config.override(**pyphi.iit3, progress_bars=False):
                substrate.maximal_complex(state)
            dt = time.perf_counter() - start
            times.append(dt)
            results.append(Trial(n, t, dt))
            if dt > max_seconds:
                break
        median = float(np.median(times))
        print(f"maximal_complex n={n}  median={median:8.3f}s", flush=True)
        if median > max_seconds:
            print(f"  -> stop: exceeded {max_seconds}s", flush=True)
            break
    return results


def save(results: list[Trial], beta: float, trials: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(t) for t in results])
    label = f"complex_beta{beta:g}_trials{trials}"
    meta = {"beta": beta, "trials": trials, "platform": platform.platform()}
    raw = _versioned_path(RESULTS_DIR / f"{label}.raw.json")
    raw.write_text(json.dumps({"meta": meta, "trials": [asdict(t) for t in results]}, indent=2))
    agg = df.groupby("n").agg(seconds_median=("seconds", "median")).reset_index()
    agg_path = _versioned_path(RESULTS_DIR / f"{label}.agg.csv")
    agg.to_csv(agg_path, index=False)
    print(f"\nwrote {raw.name}, {agg_path.name}")
    print(agg.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=200.0)
    args = parser.parse_args()
    results = run(args.beta, args.trials, args.n_min, args.n_max, args.max_seconds)
    save(results, args.beta, args.trials)


if __name__ == "__main__":
    main()
