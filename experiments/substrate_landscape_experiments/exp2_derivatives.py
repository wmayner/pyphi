"""E2/E3: finite-difference derivative behavior of phi_s.

E2: central-difference derivative of phi_s w.r.t. w[0,1] at a generic point
(w=0.60, inside a single selection regime) across step sizes h — does the
estimate stabilize (=> locally smooth)?

E3: tightly bracket the selection switch near w=0.4525 and measure one-sided
limits of phi and normalized phi (jump vs. kink classification).
"""

from exp_common import FIG1A_WEIGHTS
from exp_common import STATE
from exp_common import make_system
from exp_common import save_json

import pyphi


def phi_at(wv, signed=False):
    W = FIG1A_WEIGHTS.copy()
    W[0, 1] = wv
    s = pyphi.analyze(make_system(W), STATE, compute="sia")
    return float(s.signed_phi) if signed else float(s.phi)


def norm_phi_at(wv):
    W = FIG1A_WEIGHTS.copy()
    W[0, 1] = wv
    s = pyphi.analyze(make_system(W), STATE, compute="sia")
    return float(s.signed_normalized_phi)


out = {}

# ---- E2: derivative stability at a generic point ----
w0 = 0.60
hs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
rows = []
for h in hs:
    d = (phi_at(w0 + h) - phi_at(w0 - h)) / (2 * h)
    rows.append({"h": h, "central_diff": d})
    print(f"h={h:.0e}  dphi/dw = {d:.10f}")
out["E2_generic_point"] = {"w0": w0, "rows": rows}

# also: derivative of signed phi inside the clamped-zero plateau
w1 = 0.72
rows = []
print("-- signed phi derivative inside the phi=0 plateau (w=0.72) --")
for h in [1e-3, 1e-4, 1e-5, 1e-6]:
    d_clamped = (phi_at(w1 + h) - phi_at(w1 - h)) / (2 * h)
    d_signed = (phi_at(w1 + h, signed=True) - phi_at(w1 - h, signed=True)) / (2 * h)
    rows.append({"h": h, "clamped": d_clamped, "signed": d_signed})
    print(
        f"h={h:.0e}  d(phi)/dw = {d_clamped:+.8f}   d(signed_phi)/dw = {d_signed:+.8f}"
    )
out["E2_plateau_point"] = {"w1": w1, "rows": rows}

# ---- E3: bisect the MIP switch near 0.4525 ----
lo, hi = 0.450, 0.455  # phi(lo) ~ 0.678, phi(hi) ~ 0.270
flo, fhi = phi_at(lo), phi_at(hi)
for _ in range(30):
    mid = (lo + hi) / 2
    if abs(phi_at(mid) - flo) < abs(phi_at(mid) - fhi):
        lo = mid
    else:
        hi = mid
wstar = (lo + hi) / 2
eps = 1e-7
left, right = phi_at(wstar - eps), phi_at(wstar + eps)
nleft, nright = norm_phi_at(wstar - eps), norm_phi_at(wstar + eps)
print(f"switch located at w* = {wstar:.9f}")
print(
    f"phi({wstar:.6f} - 1e-7) = {left:.9f}   phi(+1e-7) = {right:.9f}   jump = {left - right:.9f}"
)
print(
    f"normalized phi: left = {nleft:.9f}  right = {nright:.9f}  diff = {nleft - nright:.3e}"
)
# slope of normalized phi on each side (kink check)
h = 1e-4
dl = (norm_phi_at(wstar - eps) - norm_phi_at(wstar - eps - h)) / h
dr = (norm_phi_at(wstar + eps + h) - norm_phi_at(wstar + eps)) / h
print(f"normalized-phi one-sided slopes: left = {dl:+.6f}  right = {dr:+.6f}")
out["E3_mip_switch"] = {
    "w_star": wstar,
    "phi_left": left,
    "phi_right": right,
    "norm_left": nleft,
    "norm_right": nright,
    "norm_slope_left": dl,
    "norm_slope_right": dr,
}

save_json("exp2_derivatives_raw.json", out)
