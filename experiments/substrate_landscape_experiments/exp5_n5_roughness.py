"""E1/E6 replication at n = 5: landscape roughness and FD-ascent viability.

Sweeps one weight of a 5-node Ising-sigmoid substrate over a fine grid
(recording per-point selection identities: MIP partition, specified
cause/effect states), plus a coarser secondary-axis cross-check, then runs
finite-difference gradient ascent on signed normalized phi_s from two seeded
starts against a same-budget seeded random-search baseline.

Usage: uv run python exp5_n5_roughness.py [--seed N] [--smoke]
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np

import pyphi

pyphi.config.progress_bars = False

from exp_common import TEMPERATURE
from exp_common import make_system
from exp_common import save_json

# 5-node extension of the Fig-1A construction: same Ising-sigmoid units and
# temperature, connected weight matrix, weights[i, j] = i -> j.
# A ring of reciprocal +0.7 couplings (the Fig-1A A<->B motif tiled around
# five nodes) with the Fig-1A -0.2 self-connections. In the all-OFF state
# this substrate has phi_s = 1.404 (positive, unclamped), giving the sweep a
# positive-phi region to cross, as the n=3 E1 sweep had around Fig 1A.
N5_WEIGHTS = np.array(
    [
        # A     B     C     D     E
        [-0.2, 0.7, 0.0, 0.0, 0.7],  # A
        [0.7, -0.2, 0.7, 0.0, 0.0],  # B
        [0.0, 0.7, -0.2, 0.7, 0.0],  # C
        [0.0, 0.0, 0.7, -0.2, 0.7],  # D
        [0.7, 0.0, 0.0, 0.7, -0.2],  # E
    ]
)
STATE5 = (0, 0, 0, 0, 0)


def sia_point(weights):
    sub = make_system(weights, state=STATE5)
    s = pyphi.analyze(sub, STATE5, compute="sia")
    row = {
        "phi": float(s.phi),
        "normalized_phi": float(s.normalized_phi),
        "signed_phi": float(s.signed_phi) if s.signed_phi is not None else None,
        "signed_normalized_phi": (
            float(s.signed_normalized_phi)
            if s.signed_normalized_phi is not None
            else None
        ),
        "partition": part_id(s.partition),
        "cause_state": None,
        "effect_state": None,
        "ii_cause": None,
        "ii_effect": None,
    }
    ss = s.system_state
    if ss is not None and ss.cause is not None:
        row["cause_state"] = tuple(int(x) for x in ss.cause.state)
        row["ii_cause"] = float(ss.cause.intrinsic_information)
    if ss is not None and ss.effect is not None:
        row["effect_state"] = tuple(int(x) for x in ss.effect.state)
        row["ii_effect"] = float(ss.effect.intrinsic_information)
    return row


def part_id(part):
    if not hasattr(part, "set_partition"):
        return type(part).__name__
    return f"{part.set_partition}|{sorted(part.removed_edges())}"


def run_sweep(i, j, lo, hi, n, label):
    grid = np.linspace(lo, hi, n)
    rows = []
    t0 = time.time()
    for w in grid:
        W = N5_WEIGHTS.copy()
        W[i, j] = w
        row = sia_point(W)
        row["w"] = float(w)
        rows.append(row)
    dt = time.time() - t0
    print(f"[{label}] {n} points in {dt:.1f}s ({dt / n:.2f} s/point)")

    # segments by full selection identity (partition, cause_state, effect_state)
    segs = []
    for r in rows:
        key = (r["partition"], r["cause_state"], r["effect_state"])
        if not segs or segs[-1]["key"] != key:
            segs.append({"key": key, "w_start": r["w"], "w_end": r["w"], "n": 1})
        else:
            segs[-1]["w_end"] = r["w"]
            segs[-1]["n"] += 1
    print(f"[{label}] {len(segs)} selection segments:")
    for s in segs:
        print(
            f"  w in [{s['w_start']:+.3f}, {s['w_end']:+.3f}] (n={s['n']}): "
            f"partition={s['key'][0]} cause={s['key'][1]} effect={s['key'][2]}"
        )

    # dead-zone fraction: phi clamped to 0 while signed phi is negative
    n_dead = sum(1 for r in rows if r["phi"] == 0.0 and (r["signed_phi"] or 0.0) < 0)
    n_zero = sum(1 for r in rows if r["phi"] == 0.0)
    print(
        f"[{label}] dead zone: {n_dead}/{n} clamped ({n_dead / n:.1%}); "
        f"phi==0 total: {n_zero}/{n} ({n_zero / n:.1%})"
    )

    # jump magnitudes at segment boundaries
    phis = np.array([r["phi"] for r in rows])
    dphi = np.abs(np.diff(phis))
    med = float(np.median(dphi) + 1e-15)
    jumps = sorted(
        (
            {
                "w_left": float(grid[k]),
                "w_right": float(grid[k + 1]),
                "dphi": float(dphi[k]),
                "ratio_to_median": float(dphi[k] / med),
            }
            for k in range(len(dphi))
            if dphi[k] > 20 * med
        ),
        key=lambda d: -d["dphi"],
    )[:10]
    print(f"[{label}] candidate jumps (>20x median step {med:.2e}):")
    for jr in jumps:
        print(
            f"  w {jr['w_left']:+.3f}->{jr['w_right']:+.3f}: "
            f"|dphi|={jr['dphi']:.4f} ({jr['ratio_to_median']:.0f}x median)"
        )
    return {
        "axis": [i, j],
        "grid": grid.tolist(),
        "rows": rows,
        "n_segments": len(segs),
        "segments": [
            {**s, "key": [s["key"][0], s["key"][1], s["key"][2]]} for s in segs
        ],
        "dead_fraction": n_dead / n,
        "zero_fraction": n_zero / n,
        "jumps": jumps,
        "seconds": dt,
    }


def objective(W):
    s = pyphi.analyze(make_system(W, state=STATE5), STATE5, compute="sia")
    return float(s.signed_normalized_phi)


def fd_gradient(f, W, h=1e-4):
    g = np.zeros_like(W)
    for i, j in itertools.product(range(W.shape[0]), range(W.shape[1])):
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += h
        Wm[i, j] -= h
        g[i, j] = (f(Wp) - f(Wm)) / (2 * h)
    return g


def ascend(f, W0, budget, eta=0.25, label=""):
    """FD gradient ascent capped by an SIA-evaluation budget (exp4 procedure)."""
    n_params = W0.size
    W = W0.copy()
    traj = []
    n_evals = 0
    k = 0
    while n_evals + 2 * n_params + 1 <= budget:
        val = f(W)
        g = fd_gradient(f, W)
        n_evals += 1 + 2 * n_params
        gn = float(np.linalg.norm(g))
        traj.append({"step": k, "value": val, "grad_norm": gn, "weights": W.tolist()})
        if gn < 1e-12:
            print(f"  [{label}] step {k}: value={val:+.5f}, |grad|=0 -> stalled")
            break
        step = eta * g
        for _ in range(6):
            if n_evals >= budget:
                break
            n_evals += 1
            if f(W + step) > val:
                break
            step = step / 2
        W = W + step
        k += 1
    final = f(W)
    n_evals += 1
    traj.append({"step": k, "value": final, "grad_norm": None, "weights": W.tolist()})
    print(
        f"  [{label}] {k} steps: {traj[0]['value']:+.5f} -> {final:+.5f} "
        f"({n_evals} SIA evals)"
    )
    return traj, n_evals


def output_path(base):
    """Never overwrite: append _v2, _v3, ... if the file exists."""
    p = Path(__file__).parent / f"{base}.json"
    v = 2
    while p.exists():
        p = Path(__file__).parent / f"{base}_v{v}.json"
        v += 1
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--smoke", action="store_true", help="10-point validation run")
    ap.add_argument("--primary-points", type=int, default=120)
    ap.add_argument("--secondary-points", type=int, default=40)
    ap.add_argument("--budget", type=int, default=200, help="SIA evals per ascent chain")
    ap.add_argument("--run-label", default="")
    args = ap.parse_args()

    if args.smoke:
        args.primary_points, args.secondary_points, args.budget = 10, 5, 55

    out = {
        "seed": args.seed,
        "n": 5,
        "weights": N5_WEIGHTS.tolist(),
        "state": list(STATE5),
        "temperature": TEMPERATURE,
        "params": vars(args),
    }
    t_start = time.time()

    # --- E1 replication: primary sweep on w[A->B], secondary on w[C->B] ---
    print("== E1@n5: primary sweep w[0,1] (A->B) ==")
    out["primary_sweep"] = run_sweep(
        0, 1, 0.02, 1.40, args.primary_points, "A->B w[0,1]"
    )
    print("== E1@n5: secondary sweep w[2,1] (C->B) ==")
    out["secondary_sweep"] = run_sweep(
        2, 1, -1.0, 0.4, args.secondary_points, "C->B w[2,1]"
    )

    # --- E6 replication: FD ascent on signed normalized phi vs random search ---
    rng = np.random.default_rng(args.seed)
    print("== E6@n5: FD ascent, start 1 = the N5 base point ==")
    traj1, evals1 = ascend(objective, N5_WEIGHTS, args.budget, label="base/snphi")
    print("== E6@n5: FD ascent, start 2 = seeded random weights ==")
    W2 = rng.uniform(-1.0, 1.0, size=(5, 5))
    traj2, evals2 = ascend(objective, W2, args.budget, label="random-start/snphi")

    print("== E6@n5: random-search baseline, same total budget ==")
    n_rand = evals1 + evals2
    vals = []
    best, best_W = -np.inf, None
    for _ in range(n_rand):
        W = rng.uniform(-1.2, 1.2, size=(5, 5))
        v = objective(W)
        vals.append(v)
        if v > best:
            best, best_W = v, W.copy()
    print(f"  [random] {n_rand} evals: best signed_normalized_phi = {best:+.5f}")

    out["ascent_base"] = {"trajectory": traj1, "n_evals": evals1}
    out["ascent_random_start"] = {
        "trajectory": traj2,
        "n_evals": evals2,
        "W0": W2.tolist(),
    }
    out["random_search"] = {
        "n": n_rand,
        "values": vals,
        "best": float(best),
        "best_W": best_W.tolist(),
    }
    out["total_seconds"] = time.time() - t_start

    label = f"_{args.run_label}" if args.run_label else ""
    base = (
        f"exp5_n5_roughness_raw_seed{args.seed}"
        f"_p{args.primary_points}_s{args.secondary_points}_b{args.budget}"
        f"{label}{'_smoke' if args.smoke else ''}"
    )
    path = output_path(base)
    save_json(path, out)
    print(f"saved: {path}  (total {out['total_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
