"""Confirmation experiment for the partial-distinction certified Φ bracket.

Per system and state, computes the full CES once for ground truth, then sweeps
truncations under two computation orders (oracle by φ_d, cheap by |m|·n),
recording at each truncation the certified bracket [L_Φ, U_Φ], its width, the
Σφ_r upper tightness, soundness (true Φ ∈ bracket), and the fraction of
distinctions computed. Answers: does the bracket close usefully before the CES
is complete? Seeded; raw per-record data saved; outputs never overwritten.

Usage:
    uv run python experiments/certified_bracket_experiments/verify_certified_bracket.py --seed 20260711 --trials 120
"""

import argparse
import importlib.metadata
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import pyphi

from experiments.certified_bracket_experiments import bracket as B

EPS = 0.02
OUT_DIR = Path(__file__).parent


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def sweep(distinctions, n, true_phi, order):
    if order == "oracle":
        ordered = sorted(distinctions, key=lambda d: float(d.phi), reverse=True)
    elif order == "cheap":
        ordered = sorted(
            distinctions, key=lambda d: len(d.mechanism) * n, reverse=True
        )
    else:
        raise ValueError(order)
    m = len(ordered)
    rows = []
    for k in range(m + 1):
        computed = ordered[:k]
        uncomputed_sizes = [len(d.mechanism) for d in ordered[k:]]
        br = B.phi_bracket(computed, uncomputed_sizes, n)
        width = br.upper - br.lower
        rows.append(
            {
                "order": order,
                "k": k,
                "fraction_computed": k / m if m else 1.0,
                "lower": br.lower,
                "upper": br.upper,
                "width": width,
                "sum_phi_r_upper": br.sum_phi_r_upper,
                "sum_phi_r_lower": br.sum_phi_r_lower,
                "sound": (br.lower <= true_phi + 1e-9)
                and (br.upper >= true_phi - 1e-9),
                "width_over_true": (width / true_phi) if true_phi > 0 else None,
            }
        )
    return rows


def evaluate(system):
    ces = system.ces()
    distinctions = list(ces.distinctions)
    if not distinctions:
        return None
    n = system.substrate.size
    true_phi = float(ces.sum_phi_distinctions) + float(ces.sum_phi_relations)
    rows = []
    for order in ("oracle", "cheap"):
        rows.extend(sweep(distinctions, n, true_phi, order))
    return {
        "n": n,
        "n_distinctions": len(distinctions),
        "true_sum_phi_d": float(ces.sum_phi_distinctions),
        "true_sum_phi_r": float(ces.sum_phi_relations),
        "true_phi": true_phi,
        "sweeps": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()

    pyphi.config.progress_bars = False
    rng = np.random.default_rng(args.seed)
    records = []
    start = time.time()

    for name in ("pqr_system", "grid3_system", "residue_system", "basic_system"):
        system = getattr(pyphi.examples, name)()
        rec = evaluate(system)
        if rec is not None:
            rec["fixture"] = name
            records.append(rec)

    sizes = rng.choice([2, 3, 3, 4], size=args.trials)
    for t, n in enumerate(sizes):
        n = int(n)
        table = rng.uniform(EPS, 1 - EPS, size=(2**n, n))
        sub = pyphi.Substrate(table, cm=np.ones((n, n), dtype=int))
        states = list(itertools.product((0, 1), repeat=n))
        if n == 4:
            states = [
                states[i] for i in rng.choice(len(states), size=2, replace=False)
            ]
        for state in states:
            rec = evaluate(pyphi.System(sub, state))
            if rec is not None:
                rec.update(trial=t, state=list(state), tpm=table.tolist())
                records.append(rec)

    all_rows = [r for rec in records for r in rec["sweeps"]]
    summary = {
        "n_records": len(records),
        "n_sweep_points": len(all_rows),
        "soundness_violations": sum(1 for r in all_rows if not r["sound"]),
        "wall_time_s": time.time() - start,
    }

    out = {
        "seed": args.seed,
        "trials": args.trials,
        "git_sha": git_sha(),
        "pyphi_version": importlib.metadata.version("pyphi"),
        "config_note": "library defaults (IIT_4_0_2023, GID); precision "
        + str(pyphi.config.numerics.precision),
        "summary": summary,
        "records": records,
    }

    label = f"_{args.run_label}" if args.run_label else ""
    base = OUT_DIR / f"certified_bracket_seed{args.seed}_trials{args.trials}{label}"
    path = base.with_suffix(".json")
    version = 2
    while path.exists():
        path = base.with_name(base.name + f"_v{version}").with_suffix(".json")
        version += 1
    path.write_text(json.dumps(out))
    print(json.dumps(summary, indent=1))
    print("->", path.name)


if __name__ == "__main__":
    main()
