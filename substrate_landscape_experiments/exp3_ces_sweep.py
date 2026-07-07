"""E4/E5: sweep the cause-effect structure (distinctions + relations) and a
single mechanism's phi_d over the A->B weight; track structural identity.

Records per point: each distinction's (mechanism, phi_d, cause purview, effect
purview), sum of distinction phi, number of distinctions, sum of relation phi
(analytical), and big Phi = sum phi_d + sum phi_r.
"""

import time

import numpy as np

import pyphi

pyphi.config.progress_bars = False

from exp_common import FIG1A_WEIGHTS
from exp_common import STATE
from exp_common import make_system
from exp_common import save_json


def ces_point(weights):
    sub = make_system(weights)
    ces = pyphi.analyze(sub, STATE, compute="ces")
    dists = []
    for d in ces.distinctions:
        dists.append(
            {
                "mechanism": tuple(d.mechanism),
                "phi": float(d.phi),
                "cause_purview": tuple(d.cause.purview),
                "effect_purview": tuple(d.effect.purview),
            }
        )
    return {
        "n_distinctions": len(dists),
        "distinctions": dists,
        "sum_phi_d": float(ces.sum_phi_distinctions),
        "sum_phi_r": float(ces.sum_phi_relations),
        "big_phi": float(ces.big_phi),
    }


def struct_id(row):
    return tuple(
        (d["mechanism"], d["cause_purview"], d["effect_purview"])
        for d in row["distinctions"]
    )


if __name__ == "__main__":
    grid = np.linspace(0.02, 1.40, 139)  # step 0.01
    rows = []
    t0 = time.time()
    for w in grid:
        W = FIG1A_WEIGHTS.copy()
        W[0, 1] = w
        row = ces_point(W)
        row["w"] = float(w)
        rows.append(row)
    print(f"{len(grid)} points in {time.time() - t0:.1f}s")

    # segments by structural identity (which mechanisms exist + their purviews)
    segs = []
    for r in rows:
        key = struct_id(r)
        if not segs or segs[-1]["key"] != key:
            segs.append({"key": key, "w_start": r["w"], "w_end": r["w"]})
        else:
            segs[-1]["w_end"] = r["w"]
    print(f"{len(segs)} structural segments:")
    for s in segs:
        mechs = [f"{m}:c{cp}e{ep}" for (m, cp, ep) in s["key"]]
        print(f"  w in [{s['w_start']:+.3f}, {s['w_end']:+.3f}]: {mechs}")

    bp = np.array([r["big_phi"] for r in rows])
    spd = np.array([r["sum_phi_d"] for r in rows])
    dbp = np.abs(np.diff(bp))
    print("largest big-Phi steps (w_left, |dPhi|):")
    for k in np.argsort(dbp)[::-1][:6]:
        print(
            f"  w={grid[k]:.3f} -> {grid[k + 1]:.3f}   |dPhi|={dbp[k]:.4f}  Phi: {bp[k]:.4f} -> {bp[k + 1]:.4f}"
        )
    save_json("exp3_ces_sweep_raw.json", {"grid": grid.tolist(), "rows": rows})
