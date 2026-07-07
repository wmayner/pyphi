"""E6: gradient vanishing toward the deterministic regime.
E7: finite-difference gradient ascent on signed phi_s over all 9 weights,
from two starts (a generic point and a clamped-zero point), vs. a seeded
random-search baseline with the same evaluation budget.
"""

import time

import numpy as np

import pyphi

pyphi.config.progress_bars = False

from exp_common import FIG1A_WEIGHTS
from exp_common import STATE
from exp_common import make_system
from exp_common import save_json

RNG_SEED = 20260707


def signed_phi(W):
    s = pyphi.analyze(make_system(W), STATE, compute="sia")
    return float(s.signed_phi)


def clamped_phi(W):
    s = pyphi.analyze(make_system(W), STATE, compute="sia")
    return float(s.phi)


out = {"seed": RNG_SEED}

# ---- E6: scale weights toward determinism ----
print("== E6: weight-scaling (determinism) sweep ==")
scales = np.concatenate([np.linspace(0.25, 3.0, 34), np.linspace(3.25, 8.0, 20)])
rows = []
h = 1e-4
for s in scales:
    W = FIG1A_WEIGHTS * s
    phi = signed_phi(W)
    dphi_ds = (
        signed_phi(FIG1A_WEIGHTS * (s + h)) - signed_phi(FIG1A_WEIGHTS * (s - h))
    ) / (2 * h)
    # determinism proxy: mean distance of TPM entries from 0.5
    sub = make_system(W)
    tpm = np.asarray(sub.factored_tpm.to_array())
    det = float(np.mean(np.abs(tpm - 0.5)) * 2)
    rows.append(
        {
            "scale": float(s),
            "signed_phi": phi,
            "dphi_dscale": dphi_ds,
            "determinism": det,
        }
    )
out["E6_scaling"] = rows
for r in rows[::6]:
    print(
        f"  s={r['scale']:.2f}  signed_phi={r['signed_phi']:+.4f}  "
        f"d(phi)/ds={r['dphi_dscale']:+.6f}  determinism={r['determinism']:.3f}"
    )

# ---- E7: FD gradient ascent ----


def fd_gradient(f, W, h=1e-4):
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy()
            Wp[i, j] += h
            Wm[i, j] -= h
            g[i, j] = (f(Wp) - f(Wm)) / (2 * h)
    return g


def ascend(f, W0, n_steps=30, eta=0.25, label=""):
    W = W0.copy()
    traj = []
    n_evals = 0
    for k in range(n_steps):
        val = f(W)
        g = fd_gradient(f, W)
        n_evals += 1 + 18
        gn = float(np.linalg.norm(g))
        traj.append({"step": k, "value": val, "grad_norm": gn, "weights": W.copy()})
        if gn < 1e-12:
            print(f"  [{label}] step {k}: value={val:+.5f}, |grad|=0 -> stalled")
            break
        step = eta * g
        # backtracking: halve until improvement (or give up after 6 halvings)
        for _ in range(6):
            if f(W + step) > val:
                break
            step = step / 2
            n_evals += 1
        W = W + step
    final = f(W)
    traj.append(
        {"step": len(traj), "value": final, "grad_norm": None, "weights": W.copy()}
    )
    print(
        f"  [{label}] {len(traj)} steps: {traj[0]['value']:+.5f} -> {final:+.5f}  ({n_evals} SIA evals)"
    )
    return traj, n_evals


print("== E7a: ascent from the Fig-1A point (signed phi objective) ==")
t0 = time.time()
traj_a, evals_a = ascend(signed_phi, FIG1A_WEIGHTS, n_steps=30, label="fig1a/signed")
print(f"  time: {time.time() - t0:.0f}s")

print("== E7b: ascent from a clamped-zero start, both objectives ==")
W_dead = FIG1A_WEIGHTS.copy()
W_dead[0, 1] = 0.9  # phi = 0 (clamped), signed phi < 0
traj_dead_clamped, evals_dc = ascend(
    clamped_phi, W_dead, n_steps=10, label="dead/clamped"
)
traj_dead_signed, evals_ds = ascend(signed_phi, W_dead, n_steps=30, label="dead/signed")

print("== E7c: random-search baseline, same budget as E7a ==")
rng = np.random.default_rng(RNG_SEED)
best, best_W = -np.inf, None
n_rand = evals_a
vals = []
for _ in range(n_rand):
    W = rng.uniform(-1.2, 1.2, size=(3, 3))
    v = signed_phi(W)
    vals.append(v)
    if v > best:
        best, best_W = v, W.copy()
print(f"  [random] {n_rand} evals: best signed_phi = {best:+.5f}")

out["E7a_ascent_fig1a"] = [{**t, "weights": t["weights"].tolist()} for t in traj_a]
out["E7b_dead_clamped"] = [
    {**t, "weights": t["weights"].tolist()} for t in traj_dead_clamped
]
out["E7b_dead_signed"] = [
    {**t, "weights": t["weights"].tolist()} for t in traj_dead_signed
]
out["E7c_random"] = {
    "n": n_rand,
    "values": vals,
    "best": best,
    "best_W": best_W.tolist(),
}
save_json("exp4_saturation_ascent_raw.json", out)
