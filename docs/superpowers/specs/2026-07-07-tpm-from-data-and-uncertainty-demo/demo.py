"""Worked demonstrations for the TPM-uncertainty exploration.

Everything here exercises the real pyphi library. Raw per-trial data is saved
to NPZ/JSON alongside the printed aggregates. All randomness takes an explicit
seed and uses an isolated Generator.
"""
import itertools
import json
import sys

import numpy as np

import pyphi
from pyphi import convert

OUT = str(__import__("pathlib").Path(__file__).parent)


def ground_truth_pon(substrate):
    """State-by-node P(unit_i = ON | previous state), rows in little-endian order."""
    ft = substrate.factored_tpm
    n = ft.n_nodes
    pon = np.zeros((2**n, n))
    for state in itertools.product((0, 1), repeat=n):
        row = sum(b << j for j, b in enumerate(state))  # little-endian row index
        for i in range(n):
            f = ft.factor(i)
            idx = tuple(state[j] if f.shape[j] > 1 else 0 for j in range(n)) + (1,)
            pon[row, i] = f[idx]
    return pon


def substrate_from_pon(pon, node_labels=None):
    """Build a Substrate from a state-by-node P(on) matrix (the estimated-TPM primitive)."""
    n = pon.shape[1]
    sbn = np.asarray(pon, dtype=float)
    multidim = convert.to_multidimensional(sbn)  # [2]*n + [n]
    return pyphi.Substrate(tpm=multidim, node_labels=node_labels)


def phi_of_pon(pon, state, labels=None):
    sub = substrate_from_pon(pon, node_labels=labels)
    sys = pyphi.System.from_substrate(sub, state)
    return float(sub.sia(state).phi) if hasattr(sub, "sia") else float(sys.sia().phi)


def phi_from_substrate(pon, state):
    sub = substrate_from_pon(pon)
    return float(sub.sia(state).phi)


# ---------------------------------------------------------------------------
# Demo A: interventional (uniform perturbation) vs observational (free-running)
# ---------------------------------------------------------------------------
def demo_A(seed=1):
    rng = np.random.default_rng(seed)
    sub = pyphi.examples.basic_substrate()
    state = (1, 0, 0)
    n = 3
    pon_true = ground_truth_pon(sub)
    phi_true = float(sub.sia(state).phi)

    # (1) INTERVENTIONAL: draw current state uniformly over all 2^n, sample next.
    def sample_next(state_row):
        p = pon_true[state_row]
        return tuple((rng.random(n) < p).astype(int))

    Ns = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    interv_phi = []
    for N in Ns:
        counts_on = np.zeros((2**n, n))
        counts_tot = np.zeros(2**n)
        for _ in range(N):
            row = rng.integers(2**n)
            nxt = sample_next(row)
            counts_on[row] += nxt
            counts_tot[row] += 1
        # Laplace / Beta(1,1) posterior mean for unvisited or visited rows
        pon_hat = (counts_on + 1) / (counts_tot[:, None] + 2)
        interv_phi.append(phi_from_substrate(pon_hat, state))

    # (2) OBSERVATIONAL: free-running stochastic trajectory using the true P(on).
    visited = set()
    traj_pairs = []
    st = state
    steps = 2000
    for _ in range(steps):
        row = int(np.ravel_multi_index(st[::-1], (2,) * n)) if False else sum(
            b << i for i, b in enumerate(st)
        )
        visited.add(row)
        nxt = sample_next(row)
        traj_pairs.append((row, nxt))
        st = nxt
    n_states_visited = len(visited)

    # Estimate from observational data with Laplace prior
    counts_on = np.zeros((2**n, n))
    counts_tot = np.zeros(2**n)
    for row, nxt in traj_pairs:
        counts_on[row] += nxt
        counts_tot[row] += 1
    pon_obs = (counts_on + 1) / (counts_tot[:, None] + 2)
    phi_obs = phi_from_substrate(pon_obs, state)

    result = {
        "phi_true": phi_true,
        "Ns": Ns,
        "interventional_phi_hat": interv_phi,
        "observational_steps": steps,
        "observational_states_visited": n_states_visited,
        "observational_total_states": 2**n,
        "observational_phi_hat": phi_obs,
        "observational_visited_rows": sorted(int(v) for v in visited),
        "observational_counts_tot": counts_tot.tolist(),
    }
    np.savez(
        f"{OUT}/demoA_raw_seed{seed}.npz",
        pon_true=pon_true,
        pon_obs=pon_obs,
        counts_tot=counts_tot,
        Ns=np.array(Ns),
        interv_phi=np.array(interv_phi),
    )
    with open(f"{OUT}/demoA_seed{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    print("=== DEMO A: interventional vs observational ===")
    print(f"phi_true = {phi_true:.6f}")
    print(f"interventional phi_hat by N={Ns}:")
    print("  ", [round(x, 4) for x in interv_phi])
    print(
        f"observational: {steps} steps visited only "
        f"{n_states_visited}/{2**n} states; phi_hat = {phi_obs:.6f}"
    )
    print(f"  per-state visit counts: {counts_tot.astype(int).tolist()}")
    return result


# ---------------------------------------------------------------------------
# Demos B & C: Beta-posterior over the TPM -> posterior over Phi and structure
# ---------------------------------------------------------------------------
def _sample_counts(pon_true, n_per_state, rng):
    """Interventional sampling: n_per_state draws from every row (uniform do(u))."""
    S, n = pon_true.shape
    k_on = np.zeros((S, n), dtype=int)
    for row in range(S):
        draws = rng.random((n_per_state, n)) < pon_true[row]
        k_on[row] = draws.sum(axis=0)
    return k_on, np.full(S, n_per_state)


def demo_BC(seed=2, n_per_state=5, M=300):
    rng = np.random.default_rng(seed)
    sub = pyphi.examples.grid3_substrate()
    state = (0, 0, 0)
    pon_true = ground_truth_pon(sub)
    phi_true = float(sub.sia(state).phi)
    S, n = pon_true.shape

    k_on, n_tot = _sample_counts(pon_true, n_per_state, rng)
    # Independent Beta(1+k_on, 1+k_off) posterior per (row, unit) cell.
    alpha = 1 + k_on
    beta = 1 + (n_tot[:, None] - k_on)

    phis = np.zeros(M)
    complexes = []
    for m in range(M):
        pon_s = rng.beta(alpha, beta)
        sub_s = substrate_from_pon(pon_s)
        phis[m] = float(sub_s.sia(state).phi)
        # which subset is the complex (max phi_s over candidates)?
        sias = sub_s.all_sias(state)
        best = max(sias, key=lambda a: float(a.phi))
        complexes.append(tuple(int(i) for i in best.node_indices))

    from collections import Counter

    comp_counts = Counter(complexes)
    result = {
        "system": "grid3_substrate",
        "state": state,
        "phi_true": phi_true,
        "n_per_state": n_per_state,
        "M": M,
        "phi_mean": float(phis.mean()),
        "phi_median": float(np.median(phis)),
        "phi_ci95": [float(np.percentile(phis, 2.5)), float(np.percentile(phis, 97.5))],
        "prob_phi_positive": float((phis > 1e-9).mean()),
        "complex_distribution": {str(k): v / M for k, v in comp_counts.items()},
    }
    np.savez(f"{OUT}/demoBC_raw_seed{seed}.npz", phis=phis, pon_true=pon_true, k_on=k_on)
    with open(f"{OUT}/demoBC_seed{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    print("=== DEMO B/C: posterior over Phi and complex identity (grid3, N=%d/state) ===" % n_per_state)
    print(f"phi_true          = {phi_true:.5f}")
    print(f"posterior mean    = {result['phi_mean']:.5f}")
    print(f"posterior median  = {result['phi_median']:.5f}")
    print(f"95% credible int  = [{result['phi_ci95'][0]:.5f}, {result['phi_ci95'][1]:.5f}]")
    print(f"P(Phi > 0)        = {result['prob_phi_positive']:.3f}")
    print("complex identity distribution (which units form the maximal substrate):")
    for k, v in sorted(comp_counts.items(), key=lambda kv: -kv[1]):
        print(f"    units {k}: {v/M:.3f}")
    # histogram (text)
    hist, edges = np.histogram(phis, bins=12)
    print("Phi posterior histogram:")
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"    [{lo:.3f},{hi:.3f}) {'#' * h}")
    return result


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    if which == "A":
        demo_A()
    elif which == "BC":
        demo_BC()
