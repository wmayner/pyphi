# Certified ii-gated grain scheduling — design

**Date:** 2026-07-19
**Status:** approved design (this document)
**Source exploration:** `docs/superpowers/specs/2026-07-07-grain-discovery.md` §3.2
(lever 2), amended by the refutation in
`experiments/ii_phi_inequality_experiments/FINDINGS.md`.
**ROADMAP:** Wave 7 — Exploration builds, "ii-gated grain scheduling".

## 1. Goal

Skip provably-futile partition sweeps in the macro grain search. For every
candidate system, the system's intrinsic information (ii) — computed without
any partition sweep — is a certified upper bound on φ_s under the shipping
formalism default. Computing cheap ii ceilings first and running the full
`sia()` sweep only for candidates that could still change the outcome
preserves the search result exactly while eliminating most sweeps (measured
on the worked Example 1: 1 full sweep instead of 45, saving 78% of total SIA
time even after paying for every ii computation).

Two sites are gated:

1. **The `complexes()` sweep** (`pyphi/macro/search.py`): every Eq. 18
   assembly is currently fully evaluated before the exclusion cascade runs.
2. **The Eq. 16 competitor evaluations** (`_f_for_unit` / `_judge` /
   `competing_systems`): every competitor in a candidate unit's footprint is
   currently fully evaluated to check that none beats or ties the candidate.

## 2. Soundness basis

Under IIT 4.0 (2026) with `system_phi_measure="INTRINSIC_INFORMATION"`, the
Eq. 23 cap makes φ_s ≤ ii(s) true **by construction**: the sia driver caps
each per-state-pair SIA by that pair's ii terms
(`_apply_ii_cap` in `pyphi/formalism/iit4/__init__.py`, applied whenever the
resolved system measure has `applies_ii_cap=True`). Each chosen pair's cap
term is bounded by its direction's state-maximal ii, so

> φ_s ≤ min(max-over-states ii_cause, max-over-states ii_effect)

holds exactly (not merely at tolerance). The right-hand side is the
**ceiling** the gate uses.

Under the 2023 GID measure the inequality is **refuted**
(`experiments/ii_phi_inequality_experiments/FINDINGS.md`): a minimal n = 2
witness has φ_s = 0.1719 > ii_e = 0.1178 (margin −0.054, three orders of
magnitude beyond precision). Consequences adopted here:

- The certified gate is available **only** when the resolved system measure
  has `applies_ii_cap=True`. There is **no heuristic mode** under other
  measures — requesting the gate there is an error, not an approximation.
- The witness TPM ships in the test suite as the permanent
  would-have-pruned-wrongly guard (§8a).

All gated entry points already require IIT 4.0 (`_require_iit4`), so IIT 3.0
is out of scope by construction; `iit3_exclusion_cascade` is untouched.

## 3. The ceiling helper (`pyphi/macro/search.py`)

A private helper computes ceilings for a batch of constructed systems:

- Per system: call `system_intrinsic_information(system,
  specification_measure=<resolved>, directions=Direction.both())` and take
  the minimum over directions of each direction's intrinsic-information
  value; a missing direction contributes 0.0 (no cause or no effect spec
  certifies φ_s = 0). The specification measure is resolved exactly as the
  sia driver resolves it for the active formalism, so the ceiling is computed
  from the same numbers `sia()` would compute.
- Batches run through the same ordered parallel dispatch as
  `_evaluate_systems` (the `parallel_macro_system_evaluation` infrastructure
  option), preserving determinism and the parallel ≡ sequential invariant.
- The returned `SystemStateSpecification` for each surviving candidate is
  passed into that candidate's `sia()` call (`System.sia(**kwargs)` forwards
  to the driver's `system_state` parameter), so survivors never recompute
  ii. In the worst case — no candidate gated — total cost therefore
  degenerates to today's, because every `sia()` call was computing that same
  system state anyway.

State ties inside `system_intrinsic_information` do not affect the ceiling:
tied states are tolerance-equal at the direction maximum, and the maximum is
what the ceiling reads.

## 4. The gated exclusion cascade (`pyphi/condensation.py`)

A new, additive entry point alongside the untouched `exclusion_cascade`:

```python
@dataclass(frozen=True)
class PendingCandidate:
    footprint: frozenset[int]
    ceiling: float
    payload: Any  # opaque; handed back to evaluate_batch

def gated_exclusion_cascade(
    pending: Sequence[PendingCandidate],
    evaluate_batch: Callable[[Sequence[PendingCandidate]], Sequence[Candidate]],
) -> tuple[CondensationOutcome, tuple[PendingCandidate, ...]]:
    ...  # returns (outcome, gated)
```

`pending` arrives in the same canonical dispatch order the eager cascade
would have received (the sweep-assembly order in `complexes()`); forced
batches preserve that order, and within-tier presentation order follows it,
matching the eager cascade's input-order contract.

**Algorithm.** Maintain `evaluated` (exact-φ `Candidate`s not yet resolved
into a tier), `pending` (ceiling only), `accepted`, `covered` (micro indices
of accepted complexes), `failed` (tie cliques), and `gated`.

Repeat until both `evaluated` and `pending` are empty:

1. **Force the reachable band.** If `evaluated` is empty, force the top
   ceiling band (all pending tolerance-equal to the maximum pending
   ceiling). Otherwise let L be the maximum φ over `evaluated` and
   force-evaluate every pending candidate whose ceiling could reach L at
   precision (ceiling ≥ L or `numerics.eq(ceiling, L)`). Forcing is one
   `evaluate_batch` call in pending order. Newly evaluated φ values may
   raise L; repeat this step until no pending ceiling reaches the current
   L.
2. **Resolve one tier.** Every still-pending candidate now has
   φ ≤ ceiling strictly below L at precision, so it cannot belong to the
   tier and cannot tie with any tier member. Resolve the tier at L exactly
   as the eager cascade does: tolerant membership, drop candidates
   overlapping `covered`, resolve φ-tied survivors via
   `_resolve_phi_tied_group` (tie escalation to Composition, failed-clique
   handling), accept winners, extend `covered`.
3. **Gate.** Move every pending candidate whose footprint intersects
   `covered` to `gated`. This is certified: its φ is strictly below the
   accepting complex's φ_s at precision, so the eager cascade would have
   dropped it by coverage in a strictly later tier without its φ ever
   mattering. Pending candidates disjoint from `covered` remain pending and
   are forced later, when L descends to their band.

**Equivalence.** At every tier the survivor set and tie resolution are
identical to the eager cascade's (forced set ⊇ tier membership; gated
candidates are exactly those the eager cascade drops by coverage), so
`accepted` and `failed_cliques` are reproduced exactly, with the one
documented boundary below. At least one candidate is always evaluated
(the top band is always forced), so results never come from ceilings
alone.

**Equivalence boundary (documented, not closed).** Tier membership is
tolerant (`numerics.eq` against the tier head) and tolerance does not
chain transitively, so the eager cascade's tier boundaries can be
anchored by a candidate that is itself later dropped by coverage. A gated
candidate is never evaluated and cannot anchor. Consequently, when three
candidates' φ values form a sub-tolerance chain (each within
10^(−precision) of the next but the ends not of each other) and the
middle candidate is gated, the gated cascade groups the two ends into one
tier where the eager cascade splits them. The difference is observable
only when the two ends also overlap each other: the gated cascade
escalates their tie to Composition, while the eager cascade accepts the
higher and drops the lower by coverage. The escalation is the
theory-faithful reading — overlapping candidates tied at precision
escalate — and the eager split is an artifact of drawing hard tier
boundaries through non-transitive tolerance. The corner cannot be closed
while gating at all: whether the middle candidate anchors a split is a
fact about its exact φ, which only the skipped sweep could produce.
Near-ties at ~10^(−13) do occur in practice (theoretically-equal values
computed along different construction paths); anchored three-value chains
have not been observed outside constructed examples.

**Runtime soundness check.** For every forced candidate, the cascade checks
φ ≤ ceiling + 10^(−precision) and raises `RuntimeError` on violation. The
inequality holds exactly under the cap; a violation means the certified
premise itself is broken (formalism drift), and the gate must fail loudly
rather than prune wrongly. The check is one float comparison per forced
candidate.

## 5. `complexes()` wiring (`pyphi/macro/search.py`)

With the gate active, `complexes()`:

1. Derives units (unchanged; the Eq. 16 gate of §6 applies inside).
2. Assembles and constructs all candidate systems (unchanged — construction
   is never skipped: the reachability admissibility check and ii both need
   the constructed TPM).
3. Batch-computes ii ceilings for all constructed systems (§3).
4. Runs `gated_exclusion_cascade` with an `evaluate_batch` callback that
   forwards to `_evaluate_systems` (shared memo, ordered parallel dispatch,
   each survivor's `sia()` receiving its precomputed system state).
5. Builds `ComplexesResult` as today from the outcome, plus gated records
   (§7).

With `prune="off"` the current code path runs byte-identically.

## 6. The Eq. 16 competitor gate (`pyphi/macro/search.py`)

In the unit-judging path (`_judge`, `_f_for_unit`, and the shared `_f`), the
candidate decomposition V's own φ_s is computed first; competitors' ceilings
are then computed before their sweeps (batched on the `_f_for_unit` path,
in dispatch order on the sequential `_judge`/`_f` path), and a competitor
whose ceiling is strictly below φ_V at precision is gated — it certifiably
cannot beat or tie V, so the Eq. 16 verdict is unchanged. Competitors whose ceiling reaches φ_V at precision
(potential beats *and* potential ties) are fully evaluated.

- `judge_candidate` verdicts are identical with the gate on or off: gated
  competitors still count in `num_competitors`; any witness (a beating
  competitor) is by construction fully evaluated; ties are never gated.
- `competing_systems()` returns the same systems as today (all competitor
  systems are still constructed); it stops paying for sweeps the verdict
  never needed.
- The batching restructure (φ_V before competitors) replaces the current
  single mixed batch in `_f_for_unit`; with `prune="off"` the current
  single-batch behavior is kept.

## 7. API and reporting

**Knob.** `prune: str | None = None` keyword on the public entry points that
evaluate φ: `complexes`, `intrinsic_units`, `is_intrinsic_unit`,
`competing_systems`, and `valid_systems` (whose unit derivation judges
candidates). Resolution:

- `None` (default): `"certified"` when the resolved system measure has
  `applies_ii_cap=True` (the 2026 shipping default), else `"off"`.
- `"certified"`: the gate, as specified. If the resolved system measure does
  **not** apply the ii cap, raise `pyphi.conf.ConfigurationError` naming the
  measure and the refutation (there is no heuristic arm).
- `"off"`: today's paths, untouched.
- Any other value: `ValueError`.

**Records.** `EvaluationRecord` becomes:

```python
@dataclass(frozen=True)
class EvaluationRecord:
    system: MacroSystem
    phi: float | None
    ii_ceiling: float | None = None
    gated: bool = False
```

Evaluated records look exactly as today (`phi` set; `ii_ceiling` also set
when the gate computed one). Gated records carry `phi=None`,
`gated=True`, and their ceiling — an upper bound, not the value; the
certified fact is that φ_s is strictly below the accepting complex's φ_s at
precision. `ComplexesResult.records` keeps one entry per candidate:
evaluated entries first, in evaluation order (exactly as without the
gate), followed by gated entries in the order they were gated; both
segments are deterministic. Exclusion records on accepted complexes include
the gated candidates they excluded, carrying the ceiling where exact φ is
unavailable.

**Documented contract change:** under the default (`prune=None` resolving to
`"certified"`), `records` is no longer all-exact-φ. Consumers that need
exact φ for every candidate pass `prune="off"`.

## 8. Testing

a. **Refutation witness guard** (permanent): under pinned 2023/GID
   (`IIT_4_CONFIG`-style preset pin with the GID system measure),
   `prune="certified"` raises `ConfigurationError`; and the witness TPM
   (below, from the FINDINGS) is asserted to violate the inequality
   (`sia().phi > min-direction ii` at precision), keeping the guard's
   premise live against measure drift.

   ```
   state-by-node TPM, rows little-endian over (n0, n1), cm = ones,
   state (0, 1):
   [[0.90294049 0.74463958]
    [0.55496427 0.24027935]
    [0.5432427  0.42462247]
    [0.07088583 0.84413472]]
   ```

b. **Equivalence** (certified ≡ off): identical complexes, φ_s values, tie
   sets, and unit verdicts on (i) the worked Example 1 substrate
   (`CG_TPM`), (ii) a substrate with multiple disjoint complexes (exercises
   the lower-tier path where a low-φ candidate must not be gated by the
   global maximum), and (iii) a tie case with overlapping tolerance-tied
   candidates (asserts tie members are never gated and failed cliques
   reproduce). `records` are compared on the evaluated subset only (the
   record shape differs by design).

c. **Determinism**: parallel and sequential runs with the gate on produce
   identical results (mirrors the existing invariant coverage).

d. **Gated-record invariants**: every gated record's ceiling is strictly
   below the φ_s of an overlapping accepted complex at precision;
   `phi is None` iff `gated`.

e. **Bite**: on Example 1, the number of full partition sweeps under
   `"certified"` is strictly less than under `"off"` (counted via the shared
   evaluation path, asserted as strictly-fewer rather than a brittle exact
   count).

f. **Slow lane**: a Hypothesis property over small random substrates
   asserting certified ≡ off end-to-end.

## 9. Docs and bookkeeping

- Changelog fragment (`changelog.d/ii-gate.feature.md`).
- Sweep **all executed docs** (docs/**/*.md and the build-excluded
  `docs/examples/IIT_4.0_demo.ipynb`) for `.records` and grain-search usage
  that the reporting change affects; update the grain/temporal tutorial
  where it narrates search cost.
- Check the MCP content surfaces (`pyphi/mcp/`) for grain-search mentions
  and update if the search cost story is described there.
- ROADMAP: the Wave 7 "ii-gated grain scheduling" row flips to landed in the
  merge flow, citing this spec.

## 10. Out of scope

- Any heuristic gating under measures without the cap (refuted; no arm).
- The m(m−1)/singleton size ceilings, the τ spectral cap, and symmetry-aware
  deduplication (separate levers in the exploration document).
- Reusing the gated cascade for micro-level condensation (the placement in
  `condensation.py` permits it later; nothing is built for it now).
- `SearchBounds.estimate` cost model updates (the estimate remains an upper
  bound on work; the gate only reduces actual work).
