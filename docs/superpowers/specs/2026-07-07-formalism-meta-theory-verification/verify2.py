"""Verify: exact distinctions -> tighter certified Sum phi_r bound via Zaeemzadeh Eq.15.

Eq.15 (Zaeemzadeh 2024):  Sum phi_r <= Sum phi_d + Sum_o [ LP-max given S(o), |Z(o)| ]
where S(o) = sum over distinctions incident to unit-state o of phi_d/|z*_c U z*_e|,
and the LP-max (Eq.14) = S(o)*((2^|Z(o)|/|Z(o)|)(1 - 2^-|Z(o)|) - 1).

Once distinctions are computed exactly, S(o) and |Z(o)| are known exactly, so Eq.15
with the *empirical* profile is a valid (certified) upper bound -- and far tighter
than the worst-case n*2^(n-1) ceiling that bounds.py's GENERAL bound uses.
"""
from fractions import Fraction
import numpy as np
import pyphi
from pyphi.formalism import FORMALISM_REGISTRY
import pyphi.formalism.iit4.bounds as B

def eq14_lpmax(S_o, Zsize):
    if Zsize == 0:
        return Fraction(0)
    term = Fraction(2)**Zsize / Zsize * (1 - Fraction(1, 2**Zsize)) - 1
    return S_o * term

# NOTE: the 6-unit iit4_2023_fig6* systems are valid here but their relation
# enumeration is slow; grid3/pqr/residue exercise the bracket in seconds.
for ex in ["pqr_system", "grid3_system", "residue_system"]:
    try:
        s = getattr(pyphi.examples, ex)()
    except Exception as e:
        print(ex, "load failed:", e); continue
    n = len(s.node_indices)
    with pyphi.config.override(**{"formalism.iit.version": "IIT_4_0_2023"}):
        f = FORMALISM_REGISTRY.get("IIT_4_0_2023")
        ces = f.build_ces(s)
    dists = list(ces.distinctions)
    sum_d = float(ces.sum_phi_distinctions)
    sum_r = float(ces.sum_phi_relations)
    Phi = float(ces.big_phi)

    # Build the empirical per-unit-state profile from the exact distinctions.
    # Each distinction d has cause purview z*_c and effect purview z*_e (state-tagged
    # units). z*_c U z*_e is its "relation purview"; density = phi_d / |z*_c U z*_e|.
    from collections import defaultdict
    incidence = defaultdict(list)  # o -> list of (phi_d / |union|)
    for d in dists:
        try:
            cp = set(d.cause.purview); ep = set(d.effect.purview)
        except Exception:
            cp = set(d.mice(pyphi.Direction.CAUSE).purview); ep = set(d.mice(pyphi.Direction.EFFECT).purview)
        union = cp | ep
        if not union:
            continue
        density = Fraction(d.phi).limit_denominator(10**9) / len(union)
        for o in union:
            incidence[o].append(density)

    # Eq.15 with empirical S(o), |Z(o)|
    self_rel_ceiling = Fraction(0)
    for d in dists:
        self_rel_ceiling += Fraction(d.phi).limit_denominator(10**9)  # Sum phi_d ceiling on self-relations
    cross = Fraction(0)
    for o, densities in incidence.items():
        S_o = sum(densities)
        Zsize = len(densities)
        cross += eq14_lpmax(S_o, Zsize)
    emp_bound = float(self_rel_ceiling + cross)

    gen = B.sum_phi_relations_upper_bound(n, bound="GENERAL")
    print(f"\n{ex}  (n={n}, #distinctions={len(dists)})")
    print(f"  computed:  Sum phi_d={sum_d:.6f}  Sum phi_r={sum_r:.6f}  Phi={Phi:.6f}")
    print(f"  Sum phi_r <= EMPIRICAL Eq.15 bound = {emp_bound:.6f}   (holds: {sum_r <= emp_bound + 1e-6})")
    print(f"  Sum phi_r <= GENERAL worst-case bound = {float(gen.value):.6f}")
    if emp_bound > 0:
        print(f"  tightening factor (general/empirical) = {float(gen.value)/emp_bound:.1f}x")
    # Two-sided bracket
    L = sum_d  # every phi_r >= 0
    U_emp = sum_d + emp_bound
    print(f"  bracket [Sum phi_d, Sum phi_d + Eq15_emp] = [{L:.4f}, {U_emp:.4f}]  contains Phi={Phi:.4f}: {L <= Phi <= U_emp + 1e-6}")
