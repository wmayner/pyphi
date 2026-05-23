"""Storage-backend benchmark for FactoredTPM.

Measures xarray vs. ndarray on the four operations that hot-path
consumers exercise: from_joint, to_joint, condition, and factor access.
Network sizes match the perf-budget fixtures (n in {3, 5, 8, 10}) plus
one k=3 size to preview multi-valued. Reports median + p95 wall time
per (operation, backend, size). Writes a markdown table and the raw
per-trial timings.

Decision rule: if xarray is within <= 2x of ndarray on every
(operation, size) measurement, set xarray as the default in
pyphi.core.tpm.factored._FACTORED_TPM_DEFAULT_BACKEND. Otherwise stay
on ndarray.

Usage:
    uv run python benchmarks/factored_tpm_backend.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM

WARMUP_TRIALS = 5
MEASURE_TRIALS = 50
SIZES = [3, 5, 8, 10]
K3_SIZE = 4


def _time(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _measure(fn: Callable[[], object]) -> dict[str, float | int | list[float]]:
    for _ in range(WARMUP_TRIALS):
        fn()
    samples = [_time(fn) for _ in range(MEASURE_TRIALS)]
    return {
        "median_s": float(np.median(samples)),
        "p95_s": float(np.percentile(samples, 95)),
        "min_s": float(np.min(samples)),
        "n_trials": MEASURE_TRIALS,
        "samples_s": samples,
    }


def _random_binary_joint(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(size=(2,) * n + (n,))


def _random_kary_factored(rng: np.random.Generator, n: int, k: int) -> FactoredTPM:
    factors = []
    for _i in range(n):
        raw = rng.uniform(size=(k,) * n + (k,))
        normalized = raw / raw.sum(axis=-1, keepdims=True)
        factors.append(normalized)
    return FactoredTPM(factors=factors, alphabet_sizes=(k,) * n)


def _bench_size_binary(
    rng: np.random.Generator, n: int, backend: str
) -> dict[str, dict]:
    joint = _random_binary_joint(rng, n)

    def op_from_joint() -> object:
        return FactoredTPM.from_joint(joint, alphabet_sizes=(2,) * n)

    factored = FactoredTPM.from_joint(joint, alphabet_sizes=(2,) * n)
    factored = FactoredTPM(
        factors=factored.factors, alphabet_sizes=(2,) * n, backend=backend
    )

    def op_to_joint() -> object:
        return factored.to_joint()

    def op_condition() -> object:
        return factored.condition({0: 1})

    def op_factor_access() -> object:
        return factored.factor(0)

    return {
        "from_joint": _measure(op_from_joint),
        "to_joint": _measure(op_to_joint),
        "condition": _measure(op_condition),
        "factor_access": _measure(op_factor_access),
    }


def _bench_size_kary(
    rng: np.random.Generator, n: int, k: int, backend: str
) -> dict[str, dict]:
    factored = _random_kary_factored(rng, n, k)
    factored = FactoredTPM(
        factors=factored.factors, alphabet_sizes=(k,) * n, backend=backend
    )

    def op_to_joint() -> object:
        return factored.to_joint()

    def op_condition() -> object:
        return factored.condition({0: 0})

    return {
        "to_joint": _measure(op_to_joint),
        "condition": _measure(op_condition),
    }


def main(out_dir: Path = Path("benchmarks/results")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = 2026
    results: dict = {
        "seed": seed,
        "warmup_trials": WARMUP_TRIALS,
        "measure_trials": MEASURE_TRIALS,
        "binary_sizes": SIZES,
        "k3_size": K3_SIZE,
        "binary": {},
        "kary_k3": {},
    }

    try:
        import xarray  # noqa: F401
        backends: tuple[str, ...] = ("ndarray", "xarray")
        results["xarray_available"] = True
    except ImportError:
        backends = ("ndarray",)
        results["xarray_available"] = False

    for n in SIZES:
        results["binary"][str(n)] = {}
        for backend in backends:
            rng = np.random.default_rng(seed + n)
            results["binary"][str(n)][backend] = _bench_size_binary(rng, n, backend)

    for backend in backends:
        rng = np.random.default_rng(seed + 100 + K3_SIZE)
        results["kary_k3"][backend] = _bench_size_kary(rng, K3_SIZE, 3, backend)

    decision = "ndarray"
    if "xarray" in backends:
        ratios: list[float] = []
        for per_size in results["binary"].values():
            nd = per_size["ndarray"]
            xr_ = per_size["xarray"]
            ratios.extend(
                xr_[op]["median_s"] / nd[op]["median_s"] for op in nd
            )
        if all(r <= 2.0 for r in ratios):
            decision = "xarray"
        results["max_xarray_ratio"] = float(max(ratios)) if ratios else None
    results["decision"] = decision

    json_path = out_dir / "factored-tpm-backend-2026-05-22.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    sys.stdout.write(f"Wrote {json_path}\n")

    md_path = out_dir / "factored-tpm-backend-2026-05-22.md"
    with open(md_path, "w") as f:
        f.write("# FactoredTPM storage-backend benchmark\n\n")
        f.write(f"- Seed: {seed}\n")
        f.write(f"- Warmup trials: {WARMUP_TRIALS}\n")
        f.write(f"- Measure trials: {MEASURE_TRIALS}\n")
        f.write(f"- xarray available: {results['xarray_available']}\n")
        f.write(f"- **Decision: `{decision}`**\n\n")
        if "max_xarray_ratio" in results and results["max_xarray_ratio"] is not None:
            f.write(
                f"Max xarray:ndarray ratio across (op, size): "
                f"`{results['max_xarray_ratio']:.3f}` "
                f"(rule: xarray default iff <= 2.0).\n\n"
            )
        f.write("## Binary networks\n\n")
        f.write("| n | op | backend | median (s) | p95 (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for n_str, per_size in results["binary"].items():
            for backend in backends:
                for op, stats in per_size[backend].items():
                    f.write(
                        f"| {n_str} | {op} | {backend} | "
                        f"{stats['median_s']:.6e} | {stats['p95_s']:.6e} |\n"
                    )
        f.write("\n## k=3 preview\n\n")
        f.write("| n | k | op | backend | median (s) | p95 (s) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for backend in backends:
            for op, stats in results["kary_k3"][backend].items():
                f.write(
                    f"| {K3_SIZE} | 3 | {op} | {backend} | "
                    f"{stats['median_s']:.6e} | {stats['p95_s']:.6e} |\n"
                )
    sys.stdout.write(f"Wrote {md_path}\n")
    sys.stdout.write(f"\nDecision: {decision}\n")


if __name__ == "__main__":
    main()
