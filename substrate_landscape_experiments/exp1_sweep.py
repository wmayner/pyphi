"""E1: fine one-parameter sweeps of system phi_s on the Fig-1A substrate.

Sweeps one weight over a fine grid; records phi_s plus the identity of every
discrete selection (MIP partition, specified cause/effect states) so kinks and
jumps in phi_s(w) can be aligned with selection switches.
"""

import sys
import time

import numpy as np
from exp_common import FIG1A_WEIGHTS
from exp_common import STATE
from exp_common import make_system
from exp_common import save_json

import pyphi


def part_id(part):
    if not hasattr(part, "set_partition"):
        return type(part).__name__
    return f"{part.set_partition}|{sorted(part.removed_edges())}"


def sia_point(weights):
    sub = make_system(weights)
    s = pyphi.analyze(sub, STATE, compute="sia")
    row = {
        "phi": float(s.phi),
        "normalized_phi": float(s.normalized_phi),
        "partition": part_id(s.partition),
        "cause_state": None,
        "effect_state": None,
        "phi_cause": float(s.cause.phi) if s.cause is not None else None,
        "phi_effect": float(s.effect.phi) if s.effect is not None else None,
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


def run_sweep(i, j, lo, hi, n, label):
    grid = np.linspace(lo, hi, n)
    rows = []
    t0 = time.time()
    for w in grid:
        W = FIG1A_WEIGHTS.copy()
        W[i, j] = w
        row = sia_point(W)
        row["w"] = float(w)
        rows.append(row)
    dt = time.time() - t0
    print(f"[{label}] {n} points in {dt:.1f}s ({dt / n * 1000:.0f} ms/point)")

    # report segments by (partition, cause_state, effect_state) identity
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

    # detect candidate discontinuities: |Δphi| step much larger than neighbors
    phis = np.array([r["phi"] for r in rows])
    dphi = np.abs(np.diff(phis))
    med = np.median(dphi) + 1e-15
    jumps = [
        (float(grid[k]), float(grid[k + 1]), float(dphi[k]), float(dphi[k] / med))
        for k in np.argsort(dphi)[::-1][:8]
        if dphi[k] > 20 * med
    ]
    print(f"[{label}] candidate jumps (w_left, w_right, |dphi|, ratio-to-median):")
    for jrow in jumps:
        print(f"  {jrow}")
    return {"grid": grid.tolist(), "rows": rows, "segments_printed": len(segs)}


if __name__ == "__main__":
    out = {}
    # Sweep the A->B coupling through and past its nominal 0.7
    out["A_to_B"] = run_sweep(0, 1, 0.02, 1.40, 277, "A->B weight w[0,1]")
    # Sweep the inhibitory C->B coupling through 0 (topology change at 0)
    out["C_to_B"] = run_sweep(2, 1, -1.0, 0.4, 281, "C->B weight w[2,1]")
    save_json(sys.argv[1] if len(sys.argv) > 1 else "exp1_sweep_raw.json", out)
