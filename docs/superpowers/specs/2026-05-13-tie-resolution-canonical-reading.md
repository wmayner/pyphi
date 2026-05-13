# Tie-resolution canonical reading

**Status:** Source-of-truth specification
**Inputs:** IIT 4.0 (Albantakis et al. 2023) main text and S1 Text
("Resolving ties in the IIT algorithm"); IIT 4.0 2026 (Mayner et al.);
IIT 3.0 (Oizumi et al. 2014); IIT actual causation (Albantakis et al.
2019).
**Audience:** Implementation specs (cascade primitive, per-level
resolvers) and reviewers verifying PyPhi against the papers.

## Purpose

This document is the canonical reading PyPhi targets when resolving
ties at every level of the IIT computation. It encodes:

- The postulate-cascade meta-rule from S1 Text.
- Per-level cascade specifications.
- Resolved design decisions for theoretical questions that S1 didn't
  fully specify (cross-purview distinction ties, clamp interaction,
  intrinsically-identical CESes, config defaults).
- File:line references to current PyPhi implementations of each
  rule, so the gap between spec and code is auditable.

Implementation specs (`2026-05-13-cascade-execution-model.md` and
per-cascade docs) reference this document and do not redefine the
rules.

## The postulate-cascade meta-rule (S1 Text, p. 2)

> "In general, ties that occur at an intermediate step in the
> algorithm are resolved based on the principle of maximal existence
> by considering the **subsequent postulates (essential requirements
> for existence) in order**."

Postulate order (per Box 1 of the main paper):

1. **Existence** — has cause-effect power.
2. **Intrinsicality** — for itself.
3. **Information** — selects a specific cause-effect state (max ii).
4. **Integration** — irreducible (max φ_s, MIP).
5. **Exclusion** — definite (maximal substrate; argmax φ_s).
6. **Composition** — structured (max Φ over distinctions + relations).

**Cascade rule**: when a tie occurs at postulate K, escalate to
postulate K+1, K+2, ... until a single winner emerges. If all
postulates through Composition tie, the substrate fails the
postulates and does not qualify as a complex — with one exception:
when the resulting CESes are intrinsically identical, the tie is
extrinsic (labeling only) and acceptable (see §
Intrinsically-identical CESes).

## Per-level cascade specifications

### 1. Repertoire-distance ties

Mathematically equivalent. Treated as no-tie at PyPhi's
`PRECISION`-aware comparison. No cascade.

### 2. Specified state per purview at the mechanism level

**Paper rule** (Eq 36 + S1):
1. Identify all states tied at max ii(m, z) for purview Z (Eq 36).
   ii ties may exist; PyPhi today: `intrinsic_information` surfaces
   `.ties`.
2. For each tied state, compute the distinction's per-state MIP
   value φ_d via Eq 42-44. The state that wins is the argmax (per
   the principle of maximal existence at Integration).
3. Cross-purview: see § 7.

**PyPhi current**: `pyphi/formalism/iit4/formalism.py:207-267`
`_find_mip_iit4` does this correctly via `resolve_ties.states` after
computing per-state MIPs.

**Cascade encoding**:
```
PURVIEW_SPECIFIED_STATE_CASCADE = [
    Level("Information",  argmax, key=intrinsic_information),
    Level("Integration",  argmax, key=per_state_phi_d),
    # Tie-break at Composition handled at distinction-aggregation level (§7).
]
```

### 3. Mechanism MIP ties

**Paper rule** (Eq 23 mechanism scope, applied via cascade): two
partitions tied at min normalized φ_d → maximum existence selects the
partition with the largest **unnormalized** integrated information.

**PyPhi current**: `resolve_ties.partitions` with config
`["NORMALIZED_PHI", "NEGATIVE_PHI"]`. Correct.

**Cascade encoding**:
```
MECHANISM_MIP_CASCADE = [
    Level("Integration",  argmin, key=normalized_phi_d),
    Level("Integration",  argmax, key=phi_d),  # same postulate, different existence-strength
    # No further escalation: distinct mechanism MIPs at same normalized & unnormalized phi are extrinsically tied.
]
```

The "Integration" appears twice because both keys instantiate the
same postulate at different granularities: normalized φ identifies
candidates that minimize integration relative to TPM capacity;
unnormalized φ among those candidates selects the one with greatest
absolute integrated information.

### 4. Specified state at the system level (Eq 20 parenthetical + S1)

**Paper rule** (S1 Text, p. 2):
> "if multiple states comply with equation (12), we select the one
> for which the system specifies the maximal integrated information
> φ_s(T_e, T_c, s, θ') over its minimum partition θ'.
> ...
> Remaining ties in which multiple cause-effect states specify the
> same φ_s ... we choose the cause-effect state that maximizes the
> system's structure integrated information Φ. As above, in the rare
> case that two or more states also tie in Φ, the system does not
> comply with the information postulate and thus does not qualify as
> a complex (unless the cause-effect structures are actually
> identical from the intrinsic perspective, in which case the tie
> would be extrinsic and not a violation of the information
> postulate)."

**Cascade**:
1. Tied at max ii → compute per-state φ_s via per-state max-min over
   partitions (P11.95b spec, per-state reading); pick argmax.
2. Tied at max φ_s → compute per-state Φ over the per-state CES;
   pick argmax.
3. Tied at max Φ → the substrate fails the information postulate
   **unless the CESes are intrinsically identical** (see § below);
   in the failing case, the substrate does not qualify as a complex.

**PyPhi current**: WRONG — `integration_value` uses cruelest-cut
(min signed_phi per partition) at
`pyphi/formalism/iit4/__init__.py:420-458`. Replace per Phase C.1.

**Cascade encoding**:
```
SYSTEM_STATE_CASCADE = [
    Level("Information",  argmax, key=ii),
    Level("Integration",  argmax, key=per_state_phi_s),
    Level("Composition",  argmax, key=per_state_big_phi),
    Level("Postulate-violation", action=NotAComplex,
          exception=intrinsically_identical_ces),
]
```

### 5. System MIP ties

Same rule as mechanism MIP at the system scope (Eq 23 + S1 maximum-
existence rule). PyPhi today: `resolve_ties.sias` with
`["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]` — correct.

The `PARTITION_LEX` tertiary key is a PyPhi-specific canonicalization
that fires only when the paper's two-level rule itself ties (same
normalized and unnormalized φ_s with different partitions). This is
a pragmatic determinism choice, not a paper rule; preserved as a
config option in the cascade primitive.

**Cascade encoding**:
```
SYSTEM_MIP_CASCADE = [
    Level("Integration",  argmin, key=normalized_phi_s),
    Level("Integration",  argmax, key=phi_s),
    Level("Determinism",  argmin, key=partition_lex_key),  # PyPhi canonical
]
```

### 6. Substrate exclusion (Eq 25 + S1)

**Paper rule** (S1 Text, p. 2):
> "If overlapping systems tie for max φ_s, we apply the maximum
> existence principle taking their respective Φ values (composition)
> into consideration and choose the system with maximal structure
> integrated information Φ as the complex. In the rare case that two
> or more such systems also tie in Φ, these systems do not comply
> with the exclusion postulate. For this reason, they do not qualify
> as complexes and we choose the next best system (based on φ_s)
> that is unique."

**Cascade**:
1. Candidates tied at max φ_s → compute Φ per candidate; argmax.
2. Tied at Φ AND overlapping → exclusion postulate fails; skip to
   next-best (by φ_s, descending) candidate that is **unique** (sole
   winner at its φ_s level), provided it doesn't overlap accepted
   complexes.
3. Non-overlapping ties at any level: all accepted as separate
   complexes.

**PyPhi current**: WRONG — greedy condensation in
`pyphi/substrate.py:513-541` skips both Φ-escalation and
skip-to-next-unique. Replace per Phase C.2.

**Cascade encoding** (with overlap awareness):
```
SUBSTRATE_EXCLUSION_CASCADE = [
    Level("Integration",  argmax, key=phi_s),
    # If overlapping ties remain:
    Level("Composition",  argmax, key=big_phi),
    # If still overlapping ties:
    Level("Postulate-violation", action=SkipToNextUniqueByPhiS),
]
```

The cascade's overlap check determines when escalation fires — if
tied substrates are disjoint, they are all complexes and escalation
is unnecessary. Φ computation is expensive, so the cascade must be
**lazy**: only compute Φ for the tied candidates that overlap.

### 7. Distinction-level cause/effect state (Eqs 36, 45 + S1)

**Paper rule** (S1 Text, p. 3):
> "The cause or effect state of a mechanism within the system for a
> candidate purview is first selected based on its intrinsic
> information ii(m, z) (36). Next, we compare the integrated
> information φ(m, Z) (42) of all maximal cause or effect states
> across all possible purviews (including all possible ties in
> ii(m, z) within a candidate purview) to identify the maximally
> irreducible cause or effect z*_c/e of the mechanism within the
> system (45). By the maximum existence postulate, potential ties in
> max φ_d(m, Z) and thus in the cause-effect state z*_c/e of a
> distinction may be resolved at the level of the cause-effect
> structure, by selecting the z*_c/e that maximizes the system's
> structure integrated information Φ. Accordingly, in case of state
> ties within the same purview, we select the state that is congruent
> with the system's cause-effect state s'. In case of ties across
> different purviews, the maximal cause-effect state will generally
> correspond to the one that supports the most relations with other
> distinctions, which typically favors larger purviews."

**Cascade**:
1. Identify (z, Z) candidates tied at max φ_d(m, Z).
2. **Same purview** (Z fixed, multiple z tied): pick the z that is
   **congruent with s'** (the system's specified state from § 4).
3. **Cross-purview** (multiple (z, Z) pairs tied): pick the one that
   supports the **most relations** with other distinctions.
4. CES-level escalation: pick z*_c/e that maximizes Φ.

**PyPhi current**: PARTIAL — `pyphi/models/distinction.py:199-225`
`resolve_congruence` does same-purview congruence via first-congruent
filter (correct for §2). Does NOT do cross-purview "most relations"
(missing). Replace per Phase C.3.

**Design decision (Q2)**: cross-purview ties resolve by the heuristic
"larger purview" by default (S1: "typically favors larger purviews");
two-pass joint search over CES Φ is opt-in via config
`distinction_cross_purview_tie_resolution: ["LARGER_PURVIEW"]`
(default) or `["MOST_RELATIONS_JOINT", "LARGER_PURVIEW"]` (paper-
literal, expensive).

**Cascade encoding**:
```
DISTINCTION_STATE_CASCADE = [
    Level("Information",  argmax, key=ii),
    Level("Integration",  argmax, key=phi_d),
    # Same purview branch:
    Level("Exclusion",    filter, key=congruent_with_system_state),
    # Cross-purview branch:
    Level("Composition",  argmax, key=larger_purview_or_most_relations),
]
```

The "Exclusion" naming reflects that congruence-with-system-state is
itself an instance of the exclusion principle ("definite" cause-
effect state at the system level).

### 8. 2026 ii(s) clamp interaction

**Status**: not a paper rule; an implementation interaction with the
`|·|+` operator (Eqs 19-20).

**Design decision (Q3)**: keep clamped-tie-resolved-by-lex (current
behavior). The clamp signals "doesn't exist at this partition";
beyond that, signed_phi is preserved as metadata on `.signed_phi`
but does not drive tie-break. No new strategy added.

**PyPhi current**: `pyphi/formalism/iit4/__init__.py:142-172`
`__post_init__` clamps. Unchanged.

### 9. Actual causation: find_mip (per-purview alpha)

**Paper rule** (Albantakis et al. 2019):

> "any occurrence can, thus, have, at most, one actual cause (or
> effect) within a transition — the minimal occurrence with α_max."
>
> "cases of true mechanistic over-determination, due to symmetries
> in the causal network, are resolved by leaving the actual cause
> (effect) indetermined between all x* (y_t) with α_max."

For the **per-purview MIP** (innermost step of α computation): the
partition with minimum |α| wins; the paper does not specify a
tie-break for ties within MIP search. Pragmatic rule: lex-canonical
partition (matches IIT 4.0's `PARTITION_LEX` convention).

**PyPhi current**: `pyphi/actual.py:1086-1125` strict precision
comparison, no tie-set tracking. First-encountered wins on tie.
Deterministic given `mip_partitions` iteration order, but no
canonical key. **Gap** filled by routing through cascade in C.5.

**Cascade encoding**:
```
AC_MIP_CASCADE = [
    Level("Integration",  argmin, key=abs_alpha),
    Level("Determinism",  argmin, key=partition_lex_key),
]
```

### 10. Actual causation: find_causal_link (max RIA across purviews)

**Paper rule** (Albantakis et al. 2019, p. 30):
> "any occurrence can, thus, have, at most, one actual cause (or
> effect) within a transition — the **minimal occurrence with α_max**"

Plus the over-determination clause: ties at α_max with non-comparable
purviews (i.e., not sub/super-sets) → **leave undetermined**, return
all tied minimal candidates.

**Cascade**:
1. Tied at max α → keep all.
2. Among tied: filter to **minimal occurrences** (no strict
   supersets in the purview lattice).
3. If a single minimal occurrence: it's the actual cause/effect.
4. If multiple minimal occurrences (genuine indeterminism): return
   all as the tied set; the result is "undetermined".

**PyPhi current**: `pyphi/actual.py:1217-1228` approximately
correct:
- `max(valid_ria)` via `order_by = [alpha, len(mechanism),
  -len(purview)]` picks a representative.
- Collects all purviews tied at max alpha.
- Filters out strict supersets (`is_not_superset`).
- Returns `CausalLink(max_ria, extended_purview)` with all minimal
  purviews surfaced.

**Two issues with current**:
1. `len(mechanism)` not negated in `order_by` → larger mechanism wins
   on tie. The paper's minimality says **smaller** mechanism wins.
   This is a minor bug: comparing AC RIAs across different
   mechanisms should favor smaller. Fix: negate `len(mechanism)`.
2. The representative `max_ria` doesn't carry the tie-set as a
   first-class value; consumers see `extended_purview` only.
   Cascade should surface the tied set explicitly via `set_ties`.

**Cascade encoding**:
```
AC_CAUSAL_LINK_CASCADE = [
    Level("Information",  argmax, key=alpha),
    Level("Exclusion",    filter, key=minimal_occurrence),  # no strict supersets
    # Remaining ties → undetermined; return tied set.
]
```

### 11. Composition / CES ties (IIT 3.0)

IIT 3.0 emits `ResolvedDistinctions` directly; no distinction-level
state ties. No cascade needed at the CES level for IIT 3.0.

### 12. Tie serialization

**Design decision**: JSON round-trips preserve the canonical
representative AND the tied set. `cascade_level` (the postulate at
which the tie resolved) is surfaced as a diagnostic field.

**PyPhi current**: `pyphi/warnings.py:warn_about_tie_serialization`
drops `.ties` on `to_json`. Replace per Phase C.7.

### 13. Test surface

See companion architectural spec; not a tie-resolution rule.

## Resolved design decisions

| Q | Topic | Decision |
|---|---|---|
| Q1 | Per-direction vs Cartesian product | Per-direction (S1 confirms). |
| Q2 | Cross-purview distinction tie | Heuristic (larger purview) default; two-pass opt-in via config. |
| Q3 | Clamp policy | Keep current; signed_phi diagnostic-only. |
| Q4 | Eq 25 strict reading | Confirmed by S1; implement skip-to-next-unique. |
| Q5 | Intrinsically-identical CESes | Implement canonical-form check (see below). |
| Q6 | Default config | Paper-faithful defaults; legacy strategies remain configurable. |
| Q7 | Cascade primitive | Generator + ResolutionContext (see companion spec). |

## Intrinsically-identical CESes (Q5)

S1's escape clause for the system-state Φ-tie case:
> "(unless the cause-effect structures are actually identical from
> the intrinsic perspective, in which case the tie would be
> extrinsic and not a violation of the information postulate)"

**Definition**: two CESes specified by the same substrate at tied
specified states are **intrinsically identical** when there exists
a substrate-internal symmetry (e.g., a permutation of node labels)
that maps one CES onto the other.

**Canonical-form algorithm** (sketch):
1. For each CES, compute a canonical fingerprint that is invariant
   under substrate automorphisms.
2. Substrate automorphisms are the permutations π of node indices
   that satisfy `T(π(s) | π(s')) = T(s | s')` for all states s, s'
   (where T is the TPM). They form the automorphism group Aut(T) of
   the TPM.
3. The fingerprint enumerates distinctions by their canonical
   mechanism index (under Aut(T) orbits) and records (cause purview
   orbit, effect purview orbit, φ_d, repertoire shape). Two CESes
   match iff their multisets of fingerprints match.
4. Computing Aut(T) is generally hard, but for typical IIT
   substrates (small, sparse) it's tractable. Use a generic graph-
   isomorphism backend (e.g., `pyphi.connectivity` plus a small
   permutation-finding helper).

**Failure mode**: if Aut(T) computation is too expensive or
unimplemented for some substrate class, fall back to:
- A warning ("cascade reached Composition; CES intrinsic equality
  not verified; treating as not-a-complex").
- A user-controllable config knob
  `intrinsic_ces_equality_check: ["AUTOMORPHISM", "DISABLED"]`.

**PyPhi current**: NOT IMPLEMENTED. New code in Phase C.1 (called
from the system-state cascade's final stage).

## Failed-cascade semantics

When a cascade reaches its final postulate level with multiple
candidates still tied and no escape clause applies:

- **System-state cascade level 4 (Composition tie)**: substrate
  fails the information postulate. The system function returns a
  `NullSystemIrreducibilityAnalysis` with `reasons=[INFORMATION_TIE]`.
  The CES function returns an empty CES.
- **Substrate-exclusion cascade**: the substrate does not qualify as
  a complex. `complexes()` skips it and continues with next-best
  unique candidate.
- **Mechanism MIP / distinction state**: lex-canonical fallback (
  `PARTITION_LEX`, lex on state tuple). These ties don't violate any
  postulate per the paper; they're labeling ties.

The cascade primitive exposes the failure as a structured
`CascadeOutcome`:
```
CascadeOutcome = {
    "resolved": single_winner | None,
    "tied_set": tuple_of_candidates,
    "cascade_level": "Information" | "Integration" | ...,
    "outcome": "RESOLVED" | "UNRESOLVED_WITHIN_BUDGET" | "POSTULATE_FAILURE",
    "failure_reason": ... | None,
}
```

## Cross-references

| Tie level | Spec § | Current code | Phase |
|---|---|---|---|
| Specified state per purview | §2 | `_find_mip_iit4` iit4/formalism.py:207-267 | OK |
| Mechanism MIP | §3 | `_find_mip_single_state` queries.py:85-139 | OK |
| System state | §4 | `integration_value` iit4/__init__.py:420-458 | C.1 |
| System MIP | §5 | `sia()` iit4/__init__.py:715-719 | OK |
| Substrate exclusion | §6 | `complexes` substrate.py:513-541 | C.2 |
| Distinction state | §7 | `resolve_congruence` distinction.py:199-225 | C.3 |
| Clamp interaction | §8 | `__post_init__` iit4/__init__.py:142-172 | OK |
| AC find_mip | §9 | actual.py:1086-1125 | C.5 |
| AC find_causal_link | §10 | actual.py:1217-1228 | C.5 |
| Tie serialization | §12 | warn_about_tie_serialization warnings.py | C.7 |

## Paper references

- Albantakis et al. 2023 main text:
  `papers/2023__albantakis-et-al__iit-4.0.pdf`
- S1 — Resolving ties:
  `papers/2023__albantakis-et-al__iit-4.0__S1-resolving-ties.pdf`
- S2 — Comparison to 1.0-3.0:
  `papers/2023__albantakis-et-al__iit-4.0__S2-comparison-to-1.0-3.0.pdf`
- IIT 4.0 algorithm S4:
  `papers/2023__albantakis-et-al__iit-4.0__S4-iit-algorithm.pdf`
- 2026 ii cap:
  `papers/2026__mayner-et-al__intrinsic-cause-effect-power.pdf`
- 2019 actual causation:
  `papers/2019__albantakis-et-al__what-caused-what.pdf` (Phase A.6)
- IIT 3.0:
  `papers/2014__oizumi-et-al__iit-3.0.pdf`
