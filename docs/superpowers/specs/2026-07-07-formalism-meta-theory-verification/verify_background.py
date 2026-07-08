"""Background-convention divergence for PyPhi's IIT 3.0 (spec §2, Axis 4).

Three cause-side background conventions for a proper-subset system S within
substrate U (W = U \\ S the background units):

  (2014)   condition W at its actual PAST state — requires the past state,
           which PyPhi's API has never taken (System accepts only the
           current state); never implemented in any PyPhi version.
  (legacy) condition W at its CURRENT state for both directions (the
           post-3.0 publication convention; what pyphi 1.x did via
           condition_tpm on external nodes).
  (4.0)    causally marginalize W conditional on the current state
           (IIT 4.0 Eq. 4); what pyphi 2.0's shared kernel does for ALL
           formalisms, including IIT_3_0.

The conventions coincide trivially for full-substrate systems (W empty), so
the 2014 paper's worked ABC numbers and the IIT 3.0 SIA goldens are
unaffected. For proper-subset systems the cause side diverges; this script
demonstrates it at the repertoire level and end-to-end at the Phi level.
"""
import numpy as np
import pyphi
import pyphi.formalism.iit3 as iit3
from pyphi.conf import presets

# Substrate: p(A'=1|b,c) = 0.9 if (b or c) else 0.1 (parents B,C);
#            B' = copy(A); p(C'=1|.) = 0.5 (no parents).
def pA1(b, c):
    return 0.9 if (b or c) else 0.1

rows = []
for c in (0, 1):
    for b in (0, 1):
        for a in (0, 1):
            rows.append((pA1(b, c), float(a), 0.5))
sub3 = pyphi.Substrate(
    np.array(rows),
    cm=np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]]),
    node_labels=("A", "B", "C"),
)
u = (1, 0, 0)
system = pyphi.System(sub3, u, node_indices=(0, 1))  # S={A,B}, W={C}, c_now=0

# --- repertoire level: mechanism {A}, purview {B} ----------------------------
lib = system.cause_repertoire((0,), (1,)).squeeze()

# manual Eq. 4: background posterior over C_prev given u (uniform prior),
# then the weighted cause factor for A.
w = np.array([0.5 * (pA1(0, c) + pA1(1, c)) for c in (0, 1)])
w /= w.sum()
eq4 = np.array([sum(pA1(b, c) * w[c] for c in (0, 1)) for b in (0, 1)])
eq4 /= eq4.sum()

# manual legacy: condition C_prev at c_now = 0.
legacy = np.array([pA1(b, 0) for b in (0, 1)])
legacy /= legacy.sum()

print("cause_repertoire(mech={A}, purview={B}):")
print("  library :", np.round(lib, 6))
print("  Eq. 4   :", np.round(eq4, 6), " match:", np.allclose(lib, eq4))
print("  legacy  :", np.round(legacy, 6), " match:", np.allclose(lib, legacy))
assert np.allclose(lib, eq4) and not np.allclose(lib, legacy)

# effect side: legacy and 4.0 both condition W at the current state.
lib_e = system.effect_repertoire((0,), (1,)).squeeze()
a_now, c_now = u[0], u[2]
p_b1 = float(a_now)  # B' = copy(A)
eff = np.array([1 - p_b1, p_b1])
print("effect side: library", lib_e, " manual", eff, " equal:", np.allclose(lib_e, eff))
assert np.allclose(lib_e, eff)

# --- end-to-end Phi_3.0 under both conventions -------------------------------
# Legacy semantics == analyze the 2-node substrate obtained by conditioning C
# at c_now (literally what pyphi 1.x condition_tpm on externals produced).
rows2 = []
for b in (0, 1):
    for a in (0, 1):
        rows2.append((pA1(b, 0), float(a)))
sub2 = pyphi.Substrate(
    np.array(rows2), cm=np.array([[0, 1], [1, 0]]), node_labels=("A", "B")
)
sys_legacy = pyphi.System(sub2, (1, 0))

with pyphi.config.override(**presets.iit3):
    sia_20 = iit3.sia(system)      # 2.0 semantics: subsystem of 3-node substrate
    sia_leg = iit3.sia(sys_legacy)  # legacy semantics: conditioned 2-node substrate

print("\nIIT 3.0 big Phi (presets.iit3 — EMD, directed bipartition):")
print(f"  2.0 extended-background subsystem : Phi = {float(sia_20.phi):.10f}")
print(f"  legacy conditioned-substrate      : Phi = {float(sia_leg.phi):.10f}")
assert not pyphi.utils.eq(float(sia_20.phi), float(sia_leg.phi))

# --- full-substrate fixtures have no background ------------------------------
for name in ("basic_system", "xor_system"):
    s = getattr(pyphi.examples, name)()
    print(f"{name}: external_indices = {s.external_indices} (conventions coincide)")
    assert s.external_indices == ()
