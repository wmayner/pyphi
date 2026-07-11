"""Verify the S(o) empirical certificate for the Sum-phi_r upper bound.

Claims tested, per substrate/state, against exact concrete relations:

  IDENTITY  (Eq. 11, state-keyed): with o ranging over UnitState pairs,
      Z(o) = distinctions whose purview_union contains o, and densities
      q = phi_d / |purview_union| sorted ascending within each o,
          Sum_{|d|>=2} phi_r  ==  Sum_o Sum_i q_(i) (2^(|Z(o)|-i) - 1)
      exactly (machine precision). Self-relations are computed exactly from
      each distinction alone (|z*_c ∩ z*_e| * phi_d / |z*_c ∪ z*_e|).

  BOUND (Eq. 15, empirical profile): the per-o term is a feasible point of
      the paper's per-o problem with S(o) equal to its measured value, so
          Sum phi_r <= [exact self-relation sum] + Sum_o S(o) * g(|Z(o)|),
      g(k) = (2^k - 1 - k)/k  (Chebyshev: ascending q vs descending weights).

  CONSERVATIVITY: the index-keyed variant (the meta-theory verification
      script's convention) must dominate the state-keyed bound.

Each record stores the exact sums, both bounds, and tightness ratios; any
identity residual beyond 1e-9 or bound violation is flagged. Results JSON is
seeded and never overwritten.

Usage:
    uv run python experiments/so_certificate_experiments/verify_so_certificate.py --seed 20260708 --trials 150
"""

import argparse
import importlib.metadata
import itertools
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import pyphi

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


def g(k: int) -> float:
    return (2.0**k - 1.0 - k) / k if k > 0 else 0.0


def certificate_terms(distinctions) -> dict:
    """Exact self-relation sum, state-keyed identity value, and both bounds."""
    self_sum = 0.0
    state_groups: dict = defaultdict(list)  # o (UnitState) -> densities
    index_groups: dict = defaultdict(
        list
    )  # o (unit index) -> densities (index denominators)
    for d in distinctions:
        union = set(d.purview_union)  # UnitState pairs
        cause_units = set(d.cause.purview_units)
        effect_units = set(d.effect.purview_units)
        inter = cause_units & effect_units
        phi = float(d.phi)
        if union:
            self_sum += len(inter) * phi / len(union)
            dens = phi / len(union)
            for o in union:
                state_groups[o].append(dens)
        idx_union = {u.index for u in union}
        if idx_union:
            dens_idx = phi / len(idx_union)
            for i in idx_union:
                index_groups[i].append(dens_idx)

    identity_value = 0.0
    state_bound_cross = 0.0
    for densities in state_groups.values():
        densities.sort()
        k = len(densities)
        identity_value += sum(
            q * (2.0 ** (k - (i + 1)) - 1.0) for i, q in enumerate(densities)
        )
        state_bound_cross += sum(densities) * g(k)

    index_bound_cross = sum(sum(ds) * g(len(ds)) for ds in index_groups.values())

    return {
        "self_sum_exact": self_sum,
        "identity_cross": identity_value,
        "state_bound_cross": state_bound_cross,
        "index_bound_cross": index_bound_cross,
    }


def evaluate(table: np.ndarray, state: tuple) -> dict | None:
    n = table.shape[1]
    sub = pyphi.Substrate(table, cm=np.ones((n, n), dtype=int))
    system = pyphi.System(sub, state)
    ces = system.ces()
    distinctions = list(ces.distinctions)
    if not distinctions:
        return None
    sum_r = float(ces.sum_phi_relations)
    terms = certificate_terms(distinctions)
    reconstructed = terms["self_sum_exact"] + terms["identity_cross"]
    bound_state = terms["self_sum_exact"] + terms["state_bound_cross"]
    bound_index = terms["self_sum_exact"] + terms["index_bound_cross"]
    return {
        "state": list(state),
        "n": n,
        "n_distinctions": len(distinctions),
        "sum_phi_relations": sum_r,
        "identity_reconstruction": reconstructed,
        "identity_residual": reconstructed - sum_r,
        "bound_state_keyed": bound_state,
        "bound_index_keyed": bound_index,
        "bound_holds": sum_r <= bound_state + 1e-9,
        "index_dominates_state": bound_index >= bound_state - 1e-9,
        "tightness_state": bound_state / sum_r if sum_r > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()

    pyphi.config.progress_bars = False
    rng = np.random.default_rng(args.seed)

    records = []
    start = time.time()

    # fixtures first (tightness comparison against the worst-case ceiling)
    for name in ("pqr_system", "grid3_system", "residue_system"):
        system = getattr(pyphi.examples, name)()
        ces = system.ces()
        distinctions = list(ces.distinctions)
        terms = certificate_terms(distinctions)
        sum_r = float(ces.sum_phi_relations)
        rec = {
            "fixture": name,
            "n_distinctions": len(distinctions),
            "sum_phi_relations": sum_r,
            "identity_residual": terms["self_sum_exact"]
            + terms["identity_cross"]
            - sum_r,
            "bound_state_keyed": terms["self_sum_exact"] + terms["state_bound_cross"],
            "bound_index_keyed": terms["self_sum_exact"] + terms["index_bound_cross"],
        }
        rec["bound_holds"] = sum_r <= rec["bound_state_keyed"] + 1e-9
        records.append(rec)

    # random substrates
    sizes = rng.choice([2, 3, 3, 4], size=args.trials)
    for t, n in enumerate(sizes):
        n = int(n)
        table = rng.uniform(EPS, 1 - EPS, size=(2**n, n))
        states = list(itertools.product((0, 1), repeat=n))
        if n == 4:
            states = [states[i] for i in rng.choice(len(states), size=2, replace=False)]
        for state in states:
            rec = evaluate(table, state)
            if rec is not None:
                rec.update(trial=t, tpm=table.tolist())
                records.append(rec)

    residuals = [abs(r["identity_residual"]) for r in records]
    holds = [r["bound_holds"] for r in records]
    dominance = [r.get("index_dominates_state", True) for r in records]
    summary = {
        "n_records": len(records),
        "max_identity_residual": max(residuals),
        "identity_exact_everywhere": max(residuals) < 1e-9,
        "bound_violations": sum(not h for h in holds),
        "index_dominance_violations": sum(not d for d in dominance),
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
    base = OUT_DIR / f"so_certificate_seed{args.seed}_trials{args.trials}{label}"
    path = base.with_suffix(".json")
    version = 2
    while path.exists():
        path = base.with_name(base.name + f"_v{version}").with_suffix(".json")
        version += 1
    path.write_text(json.dumps(out))
    print(json.dumps(summary, indent=1))
    for r in records[:3]:
        if "fixture" in r:
            print(
                f"{r['fixture']}: sum_r={r['sum_phi_relations']:.6f} "
                f"state-bound={r['bound_state_keyed']:.6f} "
                f"index-bound={r['bound_index_keyed']:.6f} "
                f"residual={r['identity_residual']:.2e}"
            )
    print("->", path.name)


if __name__ == "__main__":
    main()
