# Triggering coefficients + perception — design

**Project:** P14b sub-project 3 (of 4). The single-stimulus perception layer:
how much of a triggered Φ-structure's cause-effect power was caused by the
stimulus. Reference implementation: the external matching research repo's
`triggering.py` / `perception.py` (pre-2.0 API, subclass-and-mutate pattern —
ported to immutable views, not copied).

**Paper grounding (arXiv 2412.21111v2):** Eq 5 connectedness `c(x,m)`
(positive PMI), Eq 6 its bound, Eq 7 the triggering coefficient `t(x,m) =
c/-log2(q)`; Eq 8 distinction perception `t·φ_d`; Eq 9 relation triggering
(relata-mean); Eq 10 relation perception `t(x,r)·φ_r`; Eq 13 perceptual
richness. **The relation perception uses the full φ_r** (Eq 10), not the
apportioned `φ_r/|r|` — manuscript Eq 12 is wrong on this; the reference code
is correct (see the P14b sub-project 1 formalism analysis and the manuscript
fix-brief). The per-distinction-fold perception `t(x,m)·Φ_d` (Eq 11, used for
the heatmap) reuses sub-project 1's `PhiFold.big_phi_contribution`.

## Scope

In scope (single stimulus):
- `TriggeringCoefficient` + its computation from a `TriggeredTPM`.
- `TriggeredTPM` marginalization primitives (`conditional_probability` /
  `marginal_probability`) — the ones deferred from sub-project 2.
- `Perception(ces, triggered_tpm, stimulus)` — the immutable view exposing
  triggering coefficients, per-component perception, per-fold perception, and
  richness for one stimulus.

Out of scope (deferred):
- Cross-stimulus projection: differentiation, matching (sub-project 4). The
  `Perception` object is the per-stimulus unit sub-project 4 unions.
- `TriggeringCoefficientMax` (max connectedness over input subsets): the paper
  (footnote 2) explicitly does *not* consider which subset of the sensory
  interface caused the state — the coefficient uses the full ∂S. Dropped.

## Index convention (verified)

A `Distinction.mechanism` over a `System` built on substrate indices `(1, 2)`
is in **substrate-global indices** (`(1,)`, `(2,)`), the same space as
`TriggeredTPM.system_indices`. So `mechanism ⊆ system_indices`, and the
marginalization primitives map each mechanism index to its position within
`system_indices` to select the right array axis. No system-local re-indexing.

## Components

### `TriggeredTPM` marginalization primitives (`pyphi/matching/triggered_tpm.py`)

The multidimensional array has axes `(∂S axes…, S axes…)`. Both primitives
take a `mechanism` (substrate indices, ⊆ `system_indices`) and its `state`.

```python
def conditional_probability(self, mechanism, state, stimulus) -> float:
    """Pr(M = state | ∂S = stimulus): index the ∂S axes at `stimulus`, sum out
    the system axes not in `mechanism`, read off `state`."""

def marginal_probability(self, mechanism, state) -> float:
    """Pr(M = state): mean over the ∂S axes (uniform stimulus prior, intrinsic
    perspective), then sum out system axes not in `mechanism`, read `state`."""
```

Implementation: map `mechanism` indices to system-axis positions via their
index in `system_indices`; `numpy` sum over the complementary system axes;
index the result at `state` (little-endian, consistent with the rest of the
package).

### `TriggeringCoefficient` (`pyphi/matching/triggering.py`)

```python
@dataclass(frozen=True)
class TriggeringCoefficient:
    value: float          # t(x, m)  — Eq 7
    connectedness: float  # c(x, m)  — Eq 5
    p: float              # Pr(M=m | ∂S=x)
    q: float              # Pr(M=m)
```

Computed by:

```python
def triggering_coefficient(triggered_tpm, mechanism, state, stimulus)
        -> TriggeringCoefficient:
    p = triggered_tpm.conditional_probability(mechanism, state, stimulus)
    q = triggered_tpm.marginal_probability(mechanism, state)
    connectedness = log2(p / q) if (p > 0 and q > 0 and p >= q) else 0.0
    info = -log2(q) if q > 0 else 0.0           # mechanism self-information
    value = connectedness / info if info > 0 else 0.0
    ...
```

Edge cases (defined, not left to NaN): `q == 1` → `info = 0` → `value = 0`
(a mechanism state certain a priori carries no information to be triggered;
connectedness is already 0 there since `p ≤ q`). `q == 0` → `value = 0`.
`p < q` → `connectedness = 0` (the actual-effect/positive-PMI rule: the
environment did not bring about the state). So `0 ≤ value ≤ 1` always.

### `Perception` (`pyphi/matching/perception.py`)

```python
@dataclass(frozen=True)
class Perception:
    ces: CauseEffectStructure
    triggered_tpm: TriggeredTPM
    stimulus: tuple[int, ...]
```

- `__post_init__` validates the **consistency contract**: the structure must
  be the one triggered by `stimulus`. Concretely, the CES's specified system
  state (over `triggered_tpm.system_indices`) must equal
  `triggered_tpm.argmax_state(stimulus)`. A mismatch raises `ValueError` — it
  means a CES for the wrong stimulus/state was passed.
- `triggering_coefficients` (`cached_property`) — `{mechanism:
  TriggeringCoefficient}`, one per distinction, using `distinction.mechanism`
  and `distinction.mechanism_state` (the state is already encoded in the CES,
  consistent with the triggered response).
- `distinction_perception(distinction) -> float` =
  `t(mechanism).value · distinction.phi` (Eq 8).
- `relation_perception(relation) -> float` = `relation.phi · mean over relata
  of t(relatum.mechanism).value` (Eq 9–10, **full φ_r**).
- `fold_perception(fold: PhiFold) -> float` = `t(seed_mechanism).value ·
  fold.big_phi_contribution` (Eq 11, the per-fold heatmap quantity; the seed
  is the fold's single distinction).
- `richness` (`cached_property`) = `Σ_d distinction_perception(d) + Σ_r
  relation_perception(r)` (Eq 13).
- Convenience (for figures): `distinction_perceptions` /
  `relation_perceptions` returning the per-component value maps.

The CES is never mutated; every quantity is a pure, cached function of
`(ces, triggered_tpm, stimulus)`. Caching uses `cached_property` (verified to
work on frozen dataclasses — it writes to `__dict__`, bypassing the frozen
`__setattr__` guard), so values compute once on first access.

## Data flow

```
substrate ─PerceptualSystem─► triggered_tpm(τ, τ_clamp)  ► TriggeredTPM
                              triggered_states(...)        ► {stimulus: response y}
   (caller computes CES(y) per response state — expensive, user-supplied)
TriggeredTPM + CES(y) + stimulus x ─► Perception(ces, triggered_tpm, x)
   ► {mechanism: TriggeringCoefficient}, per-component perception, richness
```

## Error handling

- `conditional_probability` / `marginal_probability`: a `mechanism` not ⊆
  `system_indices`, or a `state` of wrong length, raises `ValueError`.
- `triggering_coefficient`: defined for all p/q (no NaN); see edge cases.
- `Perception.__post_init__`: CES system state ≠ triggered argmax for the
  stimulus raises `ValueError` (wrong-CES guard).

## Testing

Hand-computed (primary correctness — the math is simple on small inputs):
- On a tiny triggered TPM, pick `(mechanism, state, stimulus)`; compute
  `p`, `q`, `connectedness`, `t` by hand; assert each field of the
  `TriggeringCoefficient`.
- On a small CES, hand-compute `distinction_perception` (`t·φ_d`),
  `relation_perception` (mean-relata-`t·φ_r`), and `richness`.
- `fold_perception` equals `t · fold.big_phi_contribution` for a
  single-distinction fold.

Invariants:
- `0 ≤ t ≤ 1`; `connectedness ≥ 0`; `distinction_perception ≤ φ_d` (since
  `t ≤ 1`); `richness ≥ 0`.
- `t = 0` when the stimulus makes no difference (`p = q`); `t = 1` when
  `p = 1` (deterministic triggering) and `q < 1`.
- A relay/copy substrate where a unit is fully triggered by the stimulus
  yields `t = 1` for that mechanism (end-to-end via `PerceptualSystem`).

Consistency guard:
- Passing a CES whose system state ≠ the stimulus's triggered state raises.

Regression golden:
- Freeze a computed `(CES, triggered TPM, stimulus)` fixture and the resulting
  triggering coefficients + richness as a self-golden (guards future
  regressions). The *old-code* cross-check (resurrecting the matching env) is
  deferred — the marginalization is fully hand-verifiable here; the frozen-Φ
  golden is most valuable in sub-project 4 where cross-stimulus projection is
  harder to hand-check.

## Files

- `pyphi/matching/triggered_tpm.py` — add `conditional_probability`,
  `marginal_probability`
- `pyphi/matching/triggering.py` — new (`TriggeringCoefficient`,
  `triggering_coefficient`)
- `pyphi/matching/perception.py` — new (`Perception`)
- `pyphi/matching/__init__.py` — export the new names
- `test/test_triggering.py` — new
- `test/test_perception.py` — new
- `test/test_triggered_tpm.py` — extend (marginalization primitives)
- `changelog.d/perception.feature.md` — new

## Notes carried from brainstorming

- Relation perception is **full φ_r** (Eq 10), matching the reference code;
  the apportioned `φ_r/|r|` belongs only inside `Φ_d` (sub-project 1). This is
  the resolution of the manuscript Eq 10-vs-12 inconsistency.
- `TriggeringCoefficientMax` is **not** ported (paper uses full ∂S).
- The subclass-and-mutate pattern is replaced by the immutable `Perception`
  view — the CES and its distinctions are never modified.
