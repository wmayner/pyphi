"""Regress observed shard cost on the planner's unit charge.

Reads every result file this directory holds and reports cost per unit by
shard form and by cache ceiling, plus the two candidate coefficients: cost
per unit (the calibration ``units_per_job`` needs) and cost per cache miss
(what a shard pays for packing many distinct mechanisms).

Usage
-----
    uv run python experiments/units_runtime_model/analyze.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"


def load_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob(pattern)):
        document = json.loads(path.read_text())
        for row in document["rows"]:
            row["_file"] = path.name
            row["_params"] = document["params"]
            rows.append(row)
    return rows


def pair_cost_law() -> None:
    rows = load_rows("pair_cost_*.json")
    if not rows:
        return
    print("\n=== per-pair cost, by (|mechanism|, |purview|) ===")
    print(
        f"{'|m|':>3} {'|p|':>3} {'n':>3} {'units':>9} {'evals/pred':>10} "
        f"{'states':>6} {'us/unit':>8}"
    )
    for key in sorted({(r["msize"], r["psize"]) for r in rows}):
        group = [r for r in rows if (r["msize"], r["psize"]) == key]
        first = group[0]
        print(
            f"{key[0]:>3} {key[1]:>3} {len(group):>3} {first['model_units']:>9,} "
            f"{st.median(r['evaluations'] / r['partition_count'] for r in group):>10.3f} "
            f"{st.median(r['n_states'] for r in group):>6.1f} "
            f"{st.median(r['cpu_s'] / r['model_units'] for r in group) * 1e6:>8.2f}"
        )
    # cpu = A·(one purview evaluation) + B·(one partition): the relative
    # weight the model charges as 1 and 1.
    design = np.array([[1.0, r["partition_count"]] for r in rows])
    observed = np.array([r["cpu_s"] for r in rows])
    (per_pair, per_partition), *_ = np.linalg.lstsq(design, observed, rcond=None)
    print(
        f"\ncpu_s ≈ {per_pair * 1e6:.1f} us/pair + {per_partition * 1e6:.2f} us/partition"
    )
    print(
        f"  one purview evaluation costs {per_pair / per_partition:.1f} partitions; "
        "the model charges 1"
    )
    off_by_one = [r for r in rows if r["evaluations"] != r["partition_count"]]
    print(
        f"\npairs whose sweep length differed from the charge: "
        f"{len(off_by_one)} of {len(rows)}"
    )
    multi_state = [r for r in rows if r["n_states"] > 1]
    print(f"pairs with more than one tied specified state: {len(multi_state)}")


def shard_cost() -> None:
    rows = load_rows("shard_cost_*.json")
    if not rows:
        return
    print("\n=== per-shard cost ===")
    header = (
        f"{'arm':>12} {'form':>14} {'N':>3} {'ceiling':>8} {'nmech':>6} "
        f"{'units':>12} {'us/unit':>8} {'miss/unit':>9} {'us/eval':>8} "
        f"{'evict':>9} {'rss MiB':>7}"
    )
    print(header)
    for row in rows:
        ceiling = row.get("ceiling_bytes")
        print(
            f"{row.get('arm', '?'):>12} {row['payload_kind']:>14} "
            f"{row['_params']['units']:>3} "
            f"{'none' if ceiling is None else f'{ceiling >> 20}M':>8} "
            f"{row['n_mechanisms']:>6} {row['units']:>12,.0f} "
            f"{row['cpu_s'] / max(1.0, row['units']) * 1e6:>8.2f} "
            f"{row['cache_misses'] / max(1.0, row['units']):>9.3f} "
            f"{row['cpu_s'] / max(1, row['evaluations']) * 1e6:>8.2f} "
            f"{row.get('cache_evictions', 0):>9,} "
            f"{row.get('rss_after', 0) >> 20:>7}"
        )
    fit(rows)


def fit(rows: list[dict]) -> None:
    """Least squares of CPU seconds on units and on cache misses.

    ``cpu = a·units + b·misses`` separates the work the model counts from
    the recomputation a shard pays for the distinct repertoires it touches.
    """
    usable = [r for r in rows if r["units"] > 0]
    if len(usable) < 3:
        return
    design = np.array([[r["units"], r["cache_misses"]] for r in usable], dtype=float)
    observed = np.array([r["cpu_s"] for r in usable], dtype=float)
    (per_unit, per_miss), *_ = np.linalg.lstsq(design, observed, rcond=None)
    predicted = design @ np.array([per_unit, per_miss])
    print(
        f"\ncpu_s ≈ {per_unit * 1e6:.2f} us/unit + {per_miss * 1e6:.2f} us/miss"
        f"   (n={len(usable)})"
    )
    print(
        "  worst relative error, two-term fit: "
        f"{max(abs(p - o) / o for p, o in zip(predicted, observed, strict=True)):.1%}"
    )
    units_only = float(
        np.linalg.lstsq(design[:, :1], observed, rcond=None)[0][0]  # type: ignore[index]
    )
    errors = [abs(units_only * r["units"] - r["cpu_s"]) / r["cpu_s"] for r in usable]
    print(
        f"  units only: {units_only * 1e6:.2f} us/unit, "
        f"worst relative error {max(errors):.1%}"
    )


if __name__ == "__main__":
    pair_cost_law()
    shard_cost()
