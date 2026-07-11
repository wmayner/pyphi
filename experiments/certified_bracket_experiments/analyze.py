"""Summarize a certified-bracket run: for each order and target factor, the
median fraction of distinctions that must be computed before the bracket width
falls within ``target × true Φ``. A high fraction means the bracket is useless
for early-stopping (the honest null).

Usage:
    uv run python experiments/certified_bracket_experiments/analyze.py <results.json>
"""

import gzip
import json
import statistics
import sys


def _load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def fraction_to_close(sweep_rows, order, target, true_phi):
    rows = [r for r in sweep_rows if r["order"] == order]
    rows.sort(key=lambda r: r["k"])
    for r in rows:
        if true_phi > 0 and r["width"] <= target * true_phi:
            return r["fraction_computed"]
    return 1.0


def main() -> None:
    data = _load(sys.argv[1])
    for target in (0.5, 1.0, 2.0):
        for order in ("oracle", "cheap"):
            fracs = [
                fraction_to_close(rec["sweeps"], order, target, rec["true_phi"])
                for rec in data["records"]
                if rec["true_phi"] > 0
            ]
            med = statistics.median(fracs) if fracs else float("nan")
            print(
                f"target={target:>4} order={order:<7} "
                f"median_fraction_to_close={med:.3f}"
            )
    print("soundness_violations:", data["summary"]["soundness_violations"])


if __name__ == "__main__":
    main()
