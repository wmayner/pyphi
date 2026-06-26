"""Analyze cross-version IIT 3.0 vs 4.0 benchmark results across both
pre-refactor and post-refactor pyphi.

Reads all JSON files in results/pre/ and results/post/, prints:
  - Side-by-side wall-time table per (network, measurement, generation)
  - Pre/post comparison for measurements that map across generations
  - Phase breakdown per (generation, network) for the latest run
  - Top-N hot functions per pstats file (optional)

Usage:
    uv run python -m benchmarks.iit_3_vs_4.analyze
    uv run python -m benchmarks.iit_3_vs_4.analyze --top 15
    uv run python -m benchmarks.iit_3_vs_4.analyze --network basic
"""

from __future__ import annotations

import argparse
import json
import pstats
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path(__file__).parent / "results"
GENERATIONS = ("pre", "post")


# Cross-generation measurement mapping. Used for pre→post comparison rows.
# IIT 4.0 has multiple post-refactor variants (2023 + 2026), so pre's single
# `iit4_phi_structure` is mapped against the closest analog: the 2023 variant
# matches the original paper's formalism, while 2026 adds the Eq. 23 cap.
PRE_TO_POST = {
    "iit3_sia": "iit3_sia",
    "iit4_phi_structure": "iit4_sia_2023",
}


def load_records(generation: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    directory = RESULTS_ROOT / generation
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"skipping {path.name}: {exc}", file=sys.stderr)
    return records


def fmt_seconds(s: float | None) -> str:
    if s is None:
        return "      -"
    if s >= 1.0:
        return f"{s:>7.3f}s"
    return f"{s * 1000:>6.1f}ms"


def median_lo_hi(values: list[float]) -> tuple[float, float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return (values[0], values[0], values[0])
    sorted_v = sorted(values)
    med = statistics.median(sorted_v)
    return (med, sorted_v[0], sorted_v[-1])


def group_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """(network, measurement) → list of trials, for a single generation."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(r["network"], r["measurement"])].append(r)
    return groups


def print_wall_time_table(
    by_gen: dict[str, dict[tuple[str, str], list[dict[str, Any]]]],
) -> None:
    print()
    print("=" * 100)
    print("WALL-TIME SUMMARY (median across trials; min..max in brackets)")
    print("=" * 100)
    for generation in GENERATIONS:
        groups = by_gen[generation]
        if not groups:
            continue
        print(f"\n{generation.upper()}-REFACTOR")
        # Collect measurements in this generation
        measurements: list[str] = []
        networks: list[str] = []
        for (n, m) in groups:
            if m not in measurements:
                measurements.append(m)
            if n not in networks:
                networks.append(n)
        # Stable order
        measurements.sort()
        networks.sort()

        header = f"  {'network':>10}  "
        for m in measurements:
            header += f"{m:>22}  "
        header += "phi (per measurement)"
        print(header)
        for network in networks:
            row = f"  {network:>10}  "
            phis: list[str] = []
            for m in measurements:
                trials = groups.get((network, m), [])
                walls = [t["wall_s"] for t in trials]
                stat = median_lo_hi(walls)
                if stat is None:
                    row += f"{'-':>22}  "
                    phis.append("-")
                else:
                    med, lo, hi = stat
                    row += f"{fmt_seconds(med):>9} [{fmt_seconds(lo)}-{fmt_seconds(hi)}]  "
                    phi_set = sorted({str(t.get("result_phi")) for t in trials})
                    phis.append(",".join(phi_set))
            row += " | ".join(f"{m}={p}" for m, p in zip(measurements, phis))
            print(row)
    print()


def print_cross_generation_table(
    by_gen: dict[str, dict[tuple[str, str], list[dict[str, Any]]]],
) -> None:
    """For measurements that map across generations, show pre vs post."""
    if "pre" not in by_gen or "post" not in by_gen:
        return
    print("=" * 100)
    print("PRE vs POST (median wall-time; speedup = pre/post; >1 means post is faster)")
    print("=" * 100)
    pre = by_gen["pre"]
    post = by_gen["post"]
    networks = sorted({n for n, _ in pre} | {n for n, _ in post})

    print(f"\n  {'network':>10}  {'measurement':>22}  "
          f"{'pre':>10}  {'post':>10}  {'post/pre':>10}  speedup")
    for network in networks:
        for pre_m, post_m in PRE_TO_POST.items():
            pre_trials = pre.get((network, pre_m), [])
            post_trials = post.get((network, post_m), [])
            if not pre_trials or not post_trials:
                continue
            pre_med = statistics.median(t["wall_s"] for t in pre_trials)
            post_med = statistics.median(t["wall_s"] for t in post_trials)
            ratio = post_med / pre_med if pre_med > 0 else float("inf")
            speedup = pre_med / post_med if post_med > 0 else float("inf")
            label = f"{pre_m} → {post_m}"
            row = (
                f"  {network:>10}  {label:>22}  "
                f"{fmt_seconds(pre_med):>10}  {fmt_seconds(post_med):>10}  "
                f"{ratio:>9.2f}x  {speedup:>5.2f}x faster"
                if speedup >= 1
                else
                f"  {network:>10}  {label:>22}  "
                f"{fmt_seconds(pre_med):>10}  {fmt_seconds(post_med):>10}  "
                f"{ratio:>9.2f}x  {1/speedup:>5.2f}x SLOWER"
            )
            print(row)
    print()


def print_phase_breakdown(
    by_gen: dict[str, dict[tuple[str, str], list[dict[str, Any]]]],
    network_filter: str | None,
) -> None:
    print("=" * 100)
    print("PHASE BREAKDOWN (median cumulative time per phase, across trials)")
    print("=" * 100)
    for generation in GENERATIONS:
        groups = by_gen[generation]
        if not groups:
            continue
        networks = sorted({n for n, _ in groups})
        if network_filter:
            networks = [n for n in networks if n == network_filter]

        for network in networks:
            measurements = sorted(
                {m for n, m in groups if n == network}
            )
            phase_keys: set[str] = set()
            for m in measurements:
                for trial in groups.get((network, m), []):
                    phase_keys.update(trial["phase_times_s"].keys())
            if not phase_keys:
                continue

            print(f"\n[{generation}] network={network}")
            header = f"  {'phase':<72}"
            for m in measurements:
                header += f"  {m:>18}"
            print(header)

            def sort_key(p: str) -> tuple[int, str]:
                if "new_big_phi" in p or "formalism/iit4" in p:
                    bucket = 2
                elif "relations.py" in p:
                    bucket = 3
                elif "formalism/iit3" in p:
                    bucket = 1
                elif "metrics" in p or "measures" in p:
                    bucket = 4
                else:
                    bucket = 0
                return (bucket, p)

            for phase in sorted(phase_keys, key=sort_key):
                row = f"  {phase:<72}"
                for m in measurements:
                    trials = groups[(network, m)]
                    values = [t["phase_times_s"].get(phase, 0.0) for t in trials]
                    med = statistics.median(values) if values else 0.0
                    if med > 0.0001:
                        row += f"  {fmt_seconds(med):>18}"
                    else:
                        row += f"  {'-':>18}"
                print(row)
    print()


def print_top_functions(
    by_gen: dict[str, dict[tuple[str, str], list[dict[str, Any]]]],
    top_n: int,
) -> None:
    print("=" * 100)
    print(f"TOP-{top_n} FUNCTIONS BY CUMULATIVE TIME (one trial per group)")
    print("=" * 100)
    for generation in GENERATIONS:
        groups = by_gen[generation]
        for (network, measurement), trials in sorted(groups.items()):
            if not trials:
                continue
            record = trials[0]
            pstats_path = RESULTS_ROOT / generation / record["pstats_path"]
            if not pstats_path.exists():
                continue
            print(f"\n--- [{generation}] {network} :: {measurement} ---")
            stats = pstats.Stats(str(pstats_path))
            stats.sort_stats("cumulative")
            stats.print_stats(top_n)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cross-temporal IIT 3.0 vs 4.0 benchmark results.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        help="Restrict phase-breakdown table to a single network.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Print top-N cumulative-time functions per (generation, network, measurement). "
        "Verbose; default 0 (skip).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    by_gen: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
    for gen in GENERATIONS:
        records = load_records(gen)
        by_gen[gen] = group_records(records)
        print(f"loaded {sum(len(v) for v in by_gen[gen].values())} {gen}-refactor trials")
    print_wall_time_table(by_gen)
    print_cross_generation_table(by_gen)
    print_phase_breakdown(by_gen, args.network)
    if args.top > 0:
        print_top_functions(by_gen, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
