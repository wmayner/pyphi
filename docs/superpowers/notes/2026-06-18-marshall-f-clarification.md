# Marshall's clarification of f(U^J, W^J) in Eq. 16 — confirmation

**Context.** SP2 of the intrinsic-units work
(`docs/superpowers/specs/2026-06-11-intrinsic-units-criteria-search-design.md`)
implemented the competitor set `f(U^J, W^J)` from a pinned reading of an
ambiguous definition, flagged as an interpretation risk and queued for
the authors. William Marshall has now answered.

## The ambiguity

The paper defines `f` as "all valid systems V' (ones that satisfy
Eqs. 16 and 18) whose micro constituents are a subset of U^J." Read with
"subset" applied to a competitor's *total* constituents, including the
improper subset, the candidate's own one-unit macro *wrapping* is a
competitor. Because macroing typically raises phi_s, the candidate would
then have to beat its own wrapping — impossible whenever macroing helps,
which invalidates the units the theory exists to describe.

## Marshall's answer (verbatim)

Question put to him:

> in f(U^J, W^J), is the subset condition on the competitor system's
> total constituents, or on each of its units' constituents — and is it
> proper?

Answer:

> it should exclude the unit under consideration (but not require a
> strict subset) … currently it reads `phi(v^J) > phi(v') for all v' in
> f(U^J, W^J, tau_J)` and it should be `phi(v^J) > phi(v') for all
> v' != v^J in f(..)` … we don't want to compare to v^J, otherwise it
> would be circular … but we do want to consider systems whose
> constituents are equal to U^J, but perhaps organized differently at a
> meso "spatial" scale.

So: the subset is on the **total constituents** and is **not strict**;
the circularity is removed by excluding **v^J itself**, and same-U^J
reorganizations into smaller meso/micro units do compete.

## Confirming experiment

For each published candidate we computed the candidate's constituent-
system phi_s (the left-hand side of Eq. 16), the phi_s of its one-unit
wrapping, the shipped verdict, and the verdict under two readings:

- **M2 (consistent):** competitors = the shipped `competing_systems`
  set (total constituents subset of U^J, v^J and single-unit wrappings
  excluded).
- **M1 (naive literal):** M2 plus the single-unit wrapping over all of
  U^J.

All values under `config.override(**presets.iit4_2023)`. phi_s matches
the authors' committed result sets to within 1e-13.

| case | phi(v^J) | phi(wrapping) | shipped | M2 | M1 | wrapping in shipped f |
|---|---|---|---|---|---|---|
| min AB (both-on)   | 0.005107 | 0.788334 | VALID          | VALID          | **NOT_MAXIMAL** | 0 |
| sfn AC (w_v=0.0)   | 0.000000 | 0.004214 | NOT_INTEGRATED | NOT_INTEGRATED | NOT_INTEGRATED  | 0 |
| sfnn AC (w_v=0.01) | 0.004864 | 0.005251 | NOT_MAXIMAL    | NOT_MAXIMAL    | NOT_MAXIMAL     | 0 |
| sfs AC (w_v=0.25)  | 0.167586 | 0.061601 | VALID          | VALID          | VALID           | 0 |
| sfs AB (w_v=0.25)  | 0.672812 | 0.291806 | VALID          | VALID          | VALID           | 0 |

Findings:

1. **M2 == shipped for every case.** The shipped per-member-proper
   construction reproduces Marshall's reading verdict-for-verdict, and
   every phi_s matches the committed anchor.
2. **The wrapping never appears in the shipped competitor set.**
3. **M1 flips `min AB` from VALID to NOT_MAXIMAL.** The forcing case is
   specifically `min` — the example built to demonstrate macroing, where
   the wrapping's phi_s (0.788) dwarfs the constituent system's (0.005).
   It does not flip `sfs`, where the constituent system is already
   strongly integrated and beats its own wrapping; but `min` alone is
   enough to show the exclusion is load-bearing, not a free choice.

**Depth-2 structural check** (candidate over {A,B,C} in
`dancing_couples(0.25)`, `SearchBounds(max_depth=2, max_constituents=3)`):
of 26 shipped competitors, 10 are same-U^J meso reorganizations (more
than one unit, total constituents == {A,B,C}) and 0 are single-unit
wrappings spanning {A,B,C}. So at depth 2 the implementation includes
exactly the "organized differently at a meso scale" systems Marshall
calls for, and excludes the wrappings.

## Outcome

- Implementation: **no change** — the shipped `f` already matches the
  confirmed reading.
- Documentation: the spec's Eq 16 pin and the `search.py` module
  docstring restated in Marshall's terms; the f-subset question removed
  from the queued-for-authors list.
- Regression guard: `test/test_macro_search.py::TestWrappingExcludedFromF`
  pins the wrapping exclusion, the `min`-VALID-despite-higher-wrapping
  case, the admitting-the-wrapping control, and the depth-2 behavior.
- Residual (flagged): whether a *single* macro unit spanning all of U^J
  built from a different meso organization should compete. It is
  structurally a wrapping, so the implementation excludes it (the only
  choice consistent with the forcing case); depth >= 2 only, no
  published anchor, queued alongside the SP1 hand-entered-TPM finding.

## Reproducing

```python
import numpy as np
from pyphi import config
from pyphi.conf import presets
from pyphi.macro.criteria import judge_candidate, unit_integration
from pyphi.macro.search import SearchBounds, competing_systems, is_intrinsic_unit
from pyphi.macro.system import MacroSystem
from pyphi.macro.units import MacroUnit, coarse_grain
from pyphi.substrate import Substrate

MIN_TPM = np.array([[0.05, 0.05], [0.05, 0.06], [0.06, 0.05], [0.95, 0.95]])


def dancing_couples(w_v):
    horizontal, vertical = {0: 1, 1: 0, 2: 3, 3: 2}, {0: 2, 1: 3, 2: 0, 3: 1}
    tpm = np.zeros((16, 4))
    for row in range(16):
        s = tuple((row >> k) & 1 for k in range(4))
        for i in range(4):
            tpm[row, i] = (
                0.05 + 0.05 * s[i] + 0.6 * s[horizontal[i]] + w_v * s[vertical[i]]
            )
    return Substrate(tpm, node_labels=("A", "B", "C", "D"))


cases = [
    ("min AB", Substrate(MIN_TPM, node_labels=("A", "B")),
     MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})), ((0, 0),)),
    ("sfs AC", dancing_couples(0.25),
     MacroUnit((0, 2), 1, coarse_grain(2, on_counts={2})), ((0, 0, 0, 0),)),
]
with config.override(**presets.iit4_2023):
    for name, sub, unit, hist in cases:
        shipped = competing_systems(sub, unit, hist)
        phi_vJ = float(unit_integration(sub, unit.constituents, hist))
        wrap = MacroSystem.from_micro(sub, (unit,), hist)
        comps = [(s, float(s.sia().phi)) for s in shipped]
        m2 = judge_candidate(phi_vJ, comps)
        m1 = judge_candidate(phi_vJ, [*comps, (wrap, float(wrap.sia().phi))])
        shipped_v = is_intrinsic_unit(sub, unit, hist)
        print(name, "shipped", shipped_v.reason.value, "M2", m2.reason.value,
              "M1", m1.reason.value, "phi(wrap)", round(float(wrap.sia().phi), 6))
```
