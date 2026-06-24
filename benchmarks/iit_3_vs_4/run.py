"""CLI entry: iterate (network, measurement, trial) combinations and write results.

Usage:
    uv run python -m benchmarks.iit_3_vs_4.run
    uv run python -m benchmarks.iit_3_vs_4.run --networks basic,fig4 --trials 5
    uv run python -m benchmarks.iit_3_vs_4.run --measurements sia_3_0,sia_4_0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from benchmarks.iit_3_vs_4.harness import (
    MEASUREMENTS,
    NETWORKS,
    run_trial,
    write_record,
)


log = logging.getLogger("iit_3_vs_4.run")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-version IIT 3.0 vs 4.0 benchmark runner.",
    )
    parser.add_argument(
        "--networks",
        type=str,
        default="basic,fig4,xor",
        help="Comma-separated network names. Available: "
        + ",".join(sorted(NETWORKS)),
    )
    parser.add_argument(
        "--measurements",
        type=str,
        default=",".join(MEASUREMENTS),
        help="Comma-separated measurements. Available: " + ",".join(MEASUREMENTS),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Trials per (network, measurement). Default 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Recorded in output filename and JSON for reproducibility. "
        "Currently no randomization is performed, but the field is reserved.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-trial wall time.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
    )

    networks = [n.strip() for n in args.networks.split(",") if n.strip()]
    measurements = [m.strip() for m in args.measurements.split(",") if m.strip()]

    unknown_networks = set(networks) - set(NETWORKS)
    if unknown_networks:
        print(f"unknown networks: {sorted(unknown_networks)}", file=sys.stderr)
        return 2
    unknown_measurements = set(measurements) - set(MEASUREMENTS)
    if unknown_measurements:
        print(f"unknown measurements: {sorted(unknown_measurements)}", file=sys.stderr)
        return 2

    total = len(networks) * len(measurements) * args.trials
    print(
        f"running {total} trials: "
        f"{len(networks)} networks x {len(measurements)} measurements "
        f"x {args.trials} trials"
    )

    wall_start = time.perf_counter()
    completed = 0
    for network in networks:
        for measurement in measurements:
            for trial in range(args.trials):
                t0 = time.perf_counter()
                record = run_trial(
                    measurement=measurement,
                    network_name=network,
                    seed=args.seed,
                    trial=trial,
                )
                path = write_record(record)
                dt = time.perf_counter() - t0
                completed += 1
                print(
                    f"  [{completed}/{total}] {network:>8} {measurement:>18} "
                    f"trial={trial} wall={dt:7.3f}s phi={record['result_phi']} "
                    f"-> {path.name}"
                )

    total_dt = time.perf_counter() - wall_start
    print(f"done. {completed} trials in {total_dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
