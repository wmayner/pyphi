"""Confirmation experiment: which prior recovers Phi best near the deterministic boundary?

Paired design: for each (system, state, seed, N) we draw ONE set of per-row
next-state counts and feed the SAME counts to all four priors, so the prior is
the only thing that varies (shared-data pairing). Deterministic systems only, so
every row is visited under per-row sampling and the unvisited-row problem is
excluded -- this isolates the prior's boundary behavior.

Estimator = posterior mean under Beta(a, a): P(on) = (k_on + a) / (n + 2a).
  Laplace          a = 1.0    (uniform, shrinks toward 0.5)
  Jeffreys         a = 0.5
  Boundary(0.1)    a = 0.1    (concentrated at 0/1)
  Boundary(0.05)   a = 0.05
"""
import json
import sys

import numpy as np

import pyphi

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from demo import ground_truth_pon, substrate_from_pon  # noqa: E402

OUT = str(__import__("pathlib").Path(__file__).parent)

PRIORS = {"laplace_1.0": 1.0, "jeffreys_0.5": 0.5, "boundary_0.1": 0.1, "boundary_0.05": 0.05}
SYSTEMS = [("basic_substrate", (1, 0, 0)), ("xor_substrate", (1, 1, 0))]
NS = [1, 2, 4, 8, 16, 32]
SEEDS = 24


def run():
    out = {}
    raw = {}
    for sysname, state in SYSTEMS:
        sub = getattr(pyphi.examples, sysname)()
        pon_true = ground_truth_pon(sub)
        phi_true = float(sub.sia(state).phi)
        S, n = pon_true.shape
        # phi_hat[prior][N] -> array over seeds
        phi_hat = {p: {N: np.zeros(SEEDS) for N in NS} for p in PRIORS}
        for seed in range(SEEDS):
            rng = np.random.default_rng(1000 * len(out) + seed)
            for N in NS:
                # one shared draw of counts, fed to every prior
                k_on = np.zeros((S, n), dtype=int)
                for row in range(S):
                    k_on[row] = (rng.random((N, n)) < pon_true[row]).sum(axis=0)
                for p, a in PRIORS.items():
                    pon = (k_on + a) / (N + 2 * a)
                    phi_hat[p][N][seed] = float(substrate_from_pon(pon).sia(state).phi)
        # aggregates
        agg = {"phi_true": phi_true, "n_states": S, "priors": {}}
        for p in PRIORS:
            agg["priors"][p] = {
                "mae": {N: float(np.mean(np.abs(phi_hat[p][N] - phi_true))) for N in NS},
                "bias": {N: float(np.mean(phi_hat[p][N] - phi_true)) for N in NS},
                "frac_zero": {N: float(np.mean(phi_hat[p][N] <= 1e-9)) for N in NS},
            }
        out[sysname] = agg
        raw[sysname] = {p: {str(N): phi_hat[p][N].tolist() for N in NS} for p in PRIORS}
        print(f"\n=== {sysname}  state={state}  phi_true={phi_true:.4f}  ({S} states, {SEEDS} seeds) ===")
        hdr = "  N   " + "".join(f"{p:>16}" for p in PRIORS)
        print("MAE (mean abs error of Phi_hat):")
        print(hdr)
        for N in NS:
            print(f"  {N:<4}" + "".join(f"{agg['priors'][p]['mae'][N]:>16.4f}" for p in PRIORS))
        print("Fraction of runs with Phi_hat == 0 (false 'not integrated'):")
        print(hdr)
        for N in NS:
            print(f"  {N:<4}" + "".join(f"{agg['priors'][p]['frac_zero'][N]:>16.3f}" for p in PRIORS))

    with open(f"{OUT}/prior_experiment.json", "w") as f:
        json.dump({"aggregates": out, "config": {"priors": PRIORS, "Ns": NS, "seeds": SEEDS}}, f, indent=2)
    with open(f"{OUT}/prior_experiment_raw.json", "w") as f:
        json.dump(raw, f)
    print(f"\nsaved -> {OUT}/prior_experiment.json (+ _raw.json)")


if __name__ == "__main__":
    run()
