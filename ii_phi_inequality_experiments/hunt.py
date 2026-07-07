"""Counterexample hunt: does min(ii_c, ii_e) >= phi_s hold under IIT 4.0 (2023)/GID?

Three arms:
  random  — random dense binary substrates, all (or sampled) states.
  driver  — targeted common-driver family: units 1..n-1 are noisy copies of
            unit 0 (positively correlated conditionals through the shared
            driver — the regime where the effect-side unconstrained average
            E[prod] exceeds the complete-partition product prod E[], i.e.
            where the naive complete-partition proof argument fails).
  adversarial — seeded random-restart coordinate descent on the full
            state-by-node probability table, minimizing the margin
            min(ii_c, ii_e) - phi_s over all states.

Each record carries phi (clamped), signed_phi, ii_c, ii_e, both margins, and
the full TPM, so any violation is immediately reproducible. Results are saved
as JSON with the seed and parameters in the filename; existing files are
never overwritten (a _v2/_v3 suffix is appended instead).

Usage:
    uv run python ii_phi_inequality_experiments/hunt.py --arm random --seed 20260708 --trials 300
    uv run python ii_phi_inequality_experiments/hunt.py --arm driver --seed 20260708
    uv run python ii_phi_inequality_experiments/hunt.py --arm adversarial --seed 20260708 --restarts 6 --steps 250
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

EPS = 0.02  # keep probabilities interior so every state is reachable
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


def substrate_from_table(table: np.ndarray) -> pyphi.Substrate:
    n = table.shape[1]
    return pyphi.Substrate(table, cm=np.ones((n, n), dtype=int))


def evaluate(table: np.ndarray, state: tuple) -> dict | None:
    """One (substrate, state) evaluation: phi, ii per direction, margins."""
    system = pyphi.System(substrate_from_table(table), state)
    sia = system.sia()
    ss = getattr(sia, "system_state", None)
    if ss is None or ss.cause is None or ss.effect is None:
        return None  # short-circuited before state specification; no ii to compare
    ii_c = float(ss.cause.intrinsic_information)
    ii_e = float(ss.effect.intrinsic_information)
    phi = float(sia.phi)
    signed = float(sia.signed_phi) if sia.signed_phi is not None else phi
    return {
        "state": list(state),
        "phi": phi,
        "signed_phi": signed,
        "ii_c": ii_c,
        "ii_e": ii_e,
        "margin": min(ii_c, ii_e) - phi,
        "margin_signed": min(ii_c, ii_e) - signed,
    }


def all_states(n: int):
    return list(itertools.product((0, 1), repeat=n))


def arm_random(rng: np.random.Generator, trials: int) -> list[dict]:
    """Random dense substrates; n=2,3 exhaustive states, n=4 sampled states."""
    records = []
    sizes = rng.choice([2, 3, 3, 3, 4], size=trials)  # weight toward n=3
    for t, n in enumerate(sizes):
        n = int(n)
        table = rng.uniform(EPS, 1 - EPS, size=(2**n, n))
        states = all_states(n)
        if n == 4:
            states = [states[i] for i in rng.choice(len(states), size=4, replace=False)]
        for state in states:
            rec = evaluate(table, state)
            if rec is not None:
                rec.update(trial=t, n=n, tpm=table.tolist())
                records.append(rec)
    return records


def driver_table(
    n: int, q_lo: float, q_hi: float, r_lo: float, r_hi: float
) -> np.ndarray:
    """Units 1..n-1 = noisy copies of unit 0; unit 0 = noisy copy of unit n-1.

    p(unit_i' = 1 | u0 = 0) = q_lo, p(unit_i' = 1 | u0 = 1) = q_hi for i >= 1;
    unit 0 reads unit n-1 with (r_lo, r_hi) so the graph is strongly connected.
    """
    rows = []
    for state in all_states(n):
        u0, ulast = state[0], state[-1]
        row = [r_hi if ulast else r_lo]
        row += [q_hi if u0 else q_lo] * (n - 1)
        rows.append(row)
    return np.array(rows)


def arm_driver(rng: np.random.Generator) -> list[dict]:
    """Grid over the common-driver family, n = 3 and 4."""
    records = []
    grid = [0.05, 0.15, 0.3, 0.7, 0.85, 0.95]
    for n in (3, 4):
        for q_lo, q_hi, r_lo, r_hi in itertools.product(grid, grid, grid, grid):
            if abs(q_hi - q_lo) < 0.2:  # weak driver: uninteresting, skip for time
                continue
            table = driver_table(n, q_lo, q_hi, r_lo, r_hi)
            states = all_states(n)
            if n == 4:
                states = [
                    states[i] for i in rng.choice(len(states), size=3, replace=False)
                ]
            for state in states:
                rec = evaluate(table, state)
                if rec is not None:
                    rec.update(
                        n=n,
                        params={"q_lo": q_lo, "q_hi": q_hi, "r_lo": r_lo, "r_hi": r_hi},
                        tpm=table.tolist(),
                    )
                    records.append(rec)
    return records


def worst_margin(table: np.ndarray) -> tuple[float, dict | None]:
    """Minimum margin over all states of a substrate (None if all short-circuit)."""
    worst, worst_rec = np.inf, None
    for state in all_states(table.shape[1]):
        rec = evaluate(table, state)
        if rec is not None and rec["margin"] < worst:
            worst, worst_rec = rec["margin"], rec
    return worst, worst_rec


def arm_adversarial(rng: np.random.Generator, restarts: int, steps: int) -> list[dict]:
    """Random-restart coordinate descent on the n=3 probability table."""
    records = []
    n = 3
    for restart in range(restarts):
        # half the restarts start from the most promising driver corner
        if restart % 2 == 0:
            table = driver_table(n, 0.05, 0.95, 0.3, 0.7)
            table += rng.uniform(-0.02, 0.02, size=table.shape)
        else:
            table = rng.uniform(EPS, 1 - EPS, size=(2**n, n))
        table = np.clip(table, EPS, 1 - EPS)
        current, current_rec = worst_margin(table)
        step = 0.15
        for it in range(steps):
            i = rng.integers(0, table.shape[0])
            j = rng.integers(0, table.shape[1])
            delta = step * rng.choice([-1.0, 1.0])
            trial = table.copy()
            trial[i, j] = np.clip(trial[i, j] + delta, EPS, 1 - EPS)
            value, rec = worst_margin(trial)
            if value < current:
                table, current, current_rec = trial, value, rec
            elif it % 40 == 39:
                step = max(step * 0.5, 0.01)  # cool
        records.append(
            {
                "restart": restart,
                "final_margin": current,
                "record": current_rec,
                "tpm": table.tolist(),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=["random", "driver", "adversarial"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()

    pyphi.config.progress_bars = False
    rng = np.random.default_rng(args.seed)

    start = time.time()
    if args.arm == "random":
        records = arm_random(rng, args.trials)
        params = {"trials": args.trials}
    elif args.arm == "driver":
        records = arm_driver(rng)
        params = {}
    else:
        records = arm_adversarial(rng, args.restarts, args.steps)
        params = {"restarts": args.restarts, "steps": args.steps}

    flat = [r["record"] if args.arm == "adversarial" else r for r in records]
    flat = [r for r in flat if r is not None]
    margins = [r["margin"] for r in flat]
    violations = [r for r in flat if r["margin"] < -1e-13]
    summary = {
        "n_records": len(flat),
        "min_margin": min(margins) if margins else None,
        "n_violations": len(violations),
        "wall_time_s": time.time() - start,
    }

    out = {
        "arm": args.arm,
        "seed": args.seed,
        "params": params,
        "git_sha": git_sha(),
        "pyphi_version": importlib.metadata.version("pyphi"),
        "config_note": "library defaults (IIT_4_0_2023, GID); precision "
        + str(pyphi.config.numerics.precision),
        "summary": summary,
        "records": records,
    }

    label = f"_{args.run_label}" if args.run_label else ""
    base = OUT_DIR / f"hunt_{args.arm}_seed{args.seed}{label}"
    path = base.with_suffix(".json")
    version = 2
    while path.exists():
        path = base.with_name(base.name + f"_v{version}").with_suffix(".json")
        version += 1
    path.write_text(json.dumps(out))
    print(
        f"[{args.arm}] {summary['n_records']} records, "
        f"min margin {summary['min_margin']:.6f}, "
        f"{summary['n_violations']} violations -> {path.name}"
    )
    if violations:
        worst = min(violations, key=lambda r: r["margin"])
        print(
            "WORST VIOLATION:",
            json.dumps(
                {k: worst[k] for k in ("state", "phi", "ii_c", "ii_e", "margin")}
            ),
        )


if __name__ == "__main__":
    main()
