# B8 — `result.explain()`: Design

**Status:** draft (awaiting user review)
**Date:** 2026-06-16
**Predecessor:** B21 (unified object display — the `pyphi.display` description/backend model `.explain()` plugs into)
**Successors:** B15 (`result.diff()` — reuses the `Explanation` structured object), P14d (`to_pandas` — shares the labeled-field extraction), P15 (surface freeze — `.explain()` and `NullResultReason` are public surface frozen there)

---

## 1. Motivation

When a PyPhi computation returns φ = 0 (or any value), the *reason* is computed and
then thrown away. The IIT 4.0 SIA path captures a `reasons` list of
`ShortCircuitConditions` enum members at each short-circuit site; the IIT 3.0 SIA path
and the actual-causation (AC) path capture nothing — they call `_null_sia(system)` /
`_null_ac_sia(transition, direction)` with no reason. For φ > 0, the winning partition
and the φ-tied peers are retained on the result, but the *runner-up* partition (the
next-best distinct-φ candidate), the φ-gap to it, and the binding direction are
discarded after the minimum-over-partitions reduction selects a winner.

This is a recurring epistemic gap in IIT research: a researcher gets a number and
cannot see *why* it came out that way without re-deriving it by hand. The data exists
transiently during computation — the entire candidate list is materialized before tie
resolution — but is not retained in a structured, inspectable form.

Two concrete problems compound this:

1. **Two divergent `ShortCircuitConditions` enums with the same name.** One lives at
   the mechanism level (`pyphi/models/ria.py:60`: `NO_PURVIEWS`, `NO_PARTITIONS`,
   `EMPTY_PURVIEW`, `UNREACHABLE_STATE`); one at the system level
   (`pyphi/formalism/iit4/__init__.py:608`: `NO_VALID_PARTITIONS`, `NO_CAUSE`,
   `NO_EFFECT`, `NO_SYSTEM`, `NO_STRONG_CONNECTIVITY`, `MONAD_WITH_NO_SELFLOOP`,
   `MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI`). They are disjoint, identically named,
   and live in different modules — a latent collision that should be resolved before the
   P15 freeze.

2. **No stable handle for asserting *why*, not just the value.** Tests can assert
   `sia.phi == 0` but cannot assert *that it was zero because the system was not
   strongly connected* without reaching into internals. A typed explanation is a stable
   test handle.

B8 unifies the enums and adds `.explain()` returning a typed account of why a quantity
came out as it did.

## 2. Goals

- One unified `NullResultReason` enum (flat, with a `.level` property) replacing both
  `ShortCircuitConditions` definitions. Clean rename, no back-compat alias.
- Short-circuit reasons captured on **every** null result across IIT 4.0, IIT 3.0, and
  AC (system *and* mechanism level).
- `.explain()` on every top-level result type, returning a typed, frozen `Explanation`
  (a title + ordered `Finding`s) that is `Displayable` (B21) and `to_pandas`-able (P14d).
- For φ > 0 results: the runner-up partition + φ-gap + binding direction + driving
  mechanism/purview are retained at compute time (lightweight records) and surfaced.
- **No change to any φ value.** B8 is purely additive: reason plumbing + lightweight
  runner-up retention + a read-only `.explain()` surface.

## 3. Non-goals

- **`result.diff()` (B15).** `Explanation` is designed to be reused by B15, but B15 is a
  separate item. We build the shared object; we do not build diff here.
- **Recomputation.** `.explain()` never re-runs a partition search. It is a pure read
  over retained fields (Approach A; see §6).
- **New approximation or pruning.** Retaining the runner-up does not change selection;
  it only stops discarding the second-best candidate.
- **Macro-level `complexes()` explanation.** Out of scope, mirroring B16's boundary.

## 4. The unified reason type — `NullResultReason`

A single flat `enum.Enum`, `unique`, with a `.level` property returning one of
`"system"` / `"mechanism"` / `"actual_causation"`. It replaces both
`ShortCircuitConditions` enums (every reference updated; no alias retained, per the
project's no-back-compat-shims convention for unpushed dev work).

Members, grouped by level (the grouping is documentation; the enum is flat):

**System level** (shared across IIT 3.0 / 4.0 / AC, confirmed by reading all three null
paths):
- `NO_SYSTEM` — empty system / empty transition
  (`iit4:711`, `iit3:375`, `actual_causation/compute.py:507`)
- `NO_STRONG_CONNECTIVITY` — system CM not strongly connected
  (`iit4:715`, `iit3:379`, AC `:512`; AC tests weak-or-strong — see Risks §10)
- `MONAD_WITH_NO_SELFLOOP` (`iit4:724`, `iit3:391`)
- `MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI` (`iit4:727`, `iit3:399`)
- `NO_VALID_PARTITIONS` — no candidate cuts / no tie set survived
  (`iit4:766`, `iit3:337`/`:340`)
- `NO_CAUSE` — cause intrinsic information ≤ 0 (4.0 only; `iit4:623`)
- `NO_EFFECT` — effect intrinsic information ≤ 0 (4.0 only; `iit4:623`)
- `EMPTY_CAUSE_EFFECT_STRUCTURE` — empty unpartitioned CES (3.0; `iit3:411`) /
  empty unpartitioned account (AC; `actual_causation/compute.py:527`)

**Mechanism level** (RIA / MICE + AC RIA):
- `NO_PURVIEWS` (`ria` enum; `queries.py:261`)
- `NO_PARTITIONS` (`ria` enum)
- `EMPTY_PURVIEW` (`queries.py:161`, AC `compute.py:209`)
- `UNREACHABLE_STATE` (`queries.py:166`)
- `REDUCIBLE_OVER_PARTITION` — AC mechanism reducible against the tested partition
  (`actual_causation/compute.py:231`)

`.level` is implemented with a frozen `frozenset` membership map inside the enum module
(not stored per-member), keeping the enum a plain value type.

**Home:** `pyphi/models/explanation.py` (new). A shared module avoids the circular-import
hazard of defining the enum in any one formalism module while `ria.py`, `iit3`, `iit4`,
and `actual_causation` all reference it.

## 5. The `Explanation` and `Finding` types

Two new frozen dataclasses in `pyphi/models/explanation.py`:

```
@dataclass(frozen=True)
class Finding:
    kind: str            # stable machine key, e.g. "null_result", "winning_partition",
                         # "runner_up", "gap", "binding_direction", "driver"
    label: str           # human-readable summary
    value: Any           # the quantity (a NullResultReason, a partition, a float, …)
    detail: tuple[tuple[str, Any], ...] = ()   # optional supporting fields
    tone: str | None = None                    # "cause"/"effect"/severity accent for HTML

@dataclass(frozen=True)
class Explanation(Displayable):
    subject: str                 # what is being explained, e.g. "Φ_s = 0.0"
    level: str                   # "system" / "mechanism" / "actual_causation"
    findings: tuple[Finding, ...]
```

`Explanation` is `Displayable` (implements `_describe()` → `Description`) and has
`to_pandas()`. It is *the* structured object; `.explain()` constructs and returns it.

**Finding sets by result state:**

- **Short-circuited result** (φ = 0 via a null path): one `Finding(kind="null_result")`
  per fired `NullResultReason`, each carrying the offending quantity in `detail`
  (e.g. the empty purview, the all-zero repertoire, the ≤ 0 intrinsic-information value,
  the non-strongly-connected CM indices).
- **φ > 0 result:** findings for
  - `winning_partition` — the MIP (value = the partition; detail = `num_connections_cut`)
  - `runner_up` + `gap` — the next-distinct-φ candidate and `runner_up.phi − mip.phi`
    (omitted, with an explicit `Finding(kind="gap", value=None, label="unique MIP")`,
    when no distinct runner-up exists)
  - `binding_direction` — `argmin(φ_cause, φ_effect)` with both values in detail
  - `driver` — the mechanism/purview that the binding direction is specified over
    (system level: the system-state spec for the binding direction; mechanism level: the
    MICE purview)

Each result type implements a `_findings()` hook returning the level-appropriate
`tuple[Finding, ...]`; `.explain()` wraps them in an `Explanation`. Keeping `_findings()`
separate from `.explain()` lets subtypes (e.g. `NullSystemIrreducibilityAnalysis`)
override only the finding construction.

## 6. Data sourcing (Approach A — retain at compute, never recompute)

The guiding rule: **`.explain()` only reads fields already present on the result object;
it never re-runs a search.** So every finding must trace to data that is on the result by
the time `.explain()` is called. Three categories:

**(a) Already on the result today — read directly.**
The 4.0 SIA already stores its short-circuit reasons and its φ-tied peers (`.ties`); both
SIAs store the winning partition and the cause/effect sub-analyses (whose φ values give
the binding direction = whichever is smaller); the 4.0 SIA stores the `system_state`
specification; the RIA stores its reasons and tie sets. `.explain()` reads these as-is.

**(b) One new piece of saved data: the runner-up.**
The minimum-over-partitions search builds the *full* list of candidate partitions, each
with its φ, then picks the minimum as the MIP and discards the rest. We change it to also
keep the single **runner-up** — the candidate with the smallest φ that is *strictly
larger* than the MIP's φ (compared with `EQUALITY_TOLERANCE`, so a partition that merely
ties the MIP is a tied peer, not the runner-up). It is stored as a tiny frozen
`(partition, phi)` record, not a full SIA (mirroring B16's lightweight
`ExcludedCandidate`, which avoids retaining heavy distinction/relation graphs).

*Concretely:* if five partitions yield φ ∈ {0.0, 0.0, 0.4, 0.7, 1.1}, the MIP is 0.0
(with one tied peer, already on `.ties`) and the runner-up is the 0.4 partition; the
**φ-gap** finding is 0.4 − 0.0 = 0.4 — a measure of how decisively the MIP is the
minimum. This is added at the three search sites (4.0 `_find_mip_for_fixed_state`, 3.0
`_sia_map_reduce`, AC account selection); each reads a list that is *already fully built*
before tie resolution, so the only cost is keeping one extra small record. The record is
a new optional field on the SIA types (`runner_up`, defaulting to `None`).

**(c) Reason plumbing where it is currently dropped.**
The 4.0 path already records *why* it returned a null result. The 3.0 and AC paths build
their null results without recording a reason — `_null_sia(system)` and
`_null_ac_sia(transition, direction)` take no reason argument. We add a `reasons=`
parameter to those constructors and pass the matching member at each null site, so a 3.0
or AC null result carries the same "why" the 4.0 one does.

**Net:** φ values are untouched; we add one small field (the runner-up) and stop dropping
a value (the reason) that is already computed.

## 7. The `.explain()` API surface

A `.explain() -> Explanation` method on each top-level result type:
`SystemIrreducibilityAnalysis` (4.0), `IIT3SystemIrreducibilityAnalysis`,
`RepertoireIrreducibilityAnalysis`, `MaximallyIrreducibleCauseOrEffect` (MICE),
`Distinction`, `AcSystemIrreducibilityAnalysis`, `AcRepertoireIrreducibilityAnalysis`,
`Account`.

- MICE delegates to its `.ria`. `Distinction` composes the explanations of its binding
  direction (the MICE with the smaller φ), surfacing which of cause/effect drives its φ.
- `Account` explains its irreducibility at the system level and lists per-link reasons.
- The method is pure (no recompute) and total (always returns a populated `Explanation`,
  even for a fully-specified φ > 0 result).

## 8. Display & pandas integration

- `Explanation._describe()` returns a `Description` whose title is `subject`, with one
  `Section` per finding-group. Short-circuit findings are toned by severity; cause/effect
  findings reuse B21's `"cause"`/`"effect"` tones (HTML colors `#D55C00`/`#009E73`). The
  ASCII backend reproduces the same content untoned.
- `Explanation.to_pandas()` returns a tidy long-format `DataFrame`: one row per finding
  (`level`, `kind`, `label`, `value`, plus exploded `detail`), consistent with P14d's
  state-spec long-format convention. This is the shared labeled-field extraction P14d and
  B15 reuse.

## 9. Testing

- **Enum migration is value-neutral:** full golden suite (fast + slow) green after the
  rename; no φ value drifts.
- **`.explain()` coverage invariant:** every top-level result type returns a populated
  `Explanation` (parallel to B21's `Displayable` coverage invariant), in
  `test/test_explanation.py`.
- **Per-level numeric assertions on canonical examples:**
  - a non-strongly-connected system → `explain()` finding `short_circuit /
    NO_STRONG_CONNECTIVITY`;
  - a monad without self-loop → `MONAD_WITH_NO_SELFLOOP`;
  - a 3.0 empty-CES system → `EMPTY_CAUSE_EFFECT_STRUCTURE`;
  - an AC reducible link → `REDUCIBLE_OVER_PARTITION`;
  - a φ > 0 system (e.g. `basic_system`) → `runner_up`/`phi_gap`/`binding_direction`
    present, with the gap equal to an independently computed second-best minus MIP φ.
- **"Why, not just value" handle:** assert the *reason* on a known network, demonstrating
  the stable test handle.
- **`.level` correctness:** every `NullResultReason` member maps to exactly one level.
- **Doctest sweep:** `uv run pytest` with **no path argument** (reprs/HTML change; the
  `pyphi/` doctest collection must be exercised).

## 10. Migration & risks

- **Enum rename churn.** Both `ShortCircuitConditions` symbols and every reference
  (imports in `queries.py`, `ria.py`, `iit4`, plus `jsonify` round-trips and any test
  asserting on the old names) are renamed to `NullResultReason`. Risk: a missed
  reference. Mitigation: grep-sweep + the full suite; serialization round-trip test.
- **AC connectivity wording.** AC's system short-circuit logs "not strongly/weakly
  connected" (`compute.py:512`). Confirm which predicate AC actually applies before
  assigning `NO_STRONG_CONNECTIVITY` vs introducing a distinct AC member; do not assume.
  (Confirmation experiment, per the project's don't-defer-confirmation rule.)
- **3.0 SIA `reasons` field.** The 3.0 SIA currently has no `reasons` field. Adding one
  (default empty) is additive; verify it round-trips through `jsonify` and does not
  perturb the 3.0 SIA golden reprs (the B21 display goldens must be regenerated
  deliberately if the SIA card gains a reasons row, and reviewed as an intended surface
  change).
- **Runner-up field on frozen results.** The SIA result dataclasses must accept the new
  optional `runner_up` field with a default so existing constructors and deserialization
  keep working.
- **No φ drift is the invariant that gates the whole change.** Any golden φ movement is a
  bug in the additive plumbing, not an intended effect.

## 11. Build order (for the implementation plan)

1. `pyphi/models/explanation.py`: `NullResultReason` (+ `.level`), `Finding`,
   `Explanation`, `RunnerUp`. Unit-test the enum/level map in isolation.
2. Migrate both old `ShortCircuitConditions` → `NullResultReason`; full suite green
   (value-neutral checkpoint).
3. Plumb `reasons=` into 3.0 `_null_sia` and AC `_null_ac_sia`/`_null_ac_ria`; add the
   3.0 SIA `reasons` field.
4. Retain `runner_up` at the three MIP-selection sites.
5. `_findings()` + `.explain()` per result type (system 4.0 → 3.0 → mechanism → AC).
6. `Explanation._describe()` + `to_pandas()`.
7. Tests (coverage invariant, per-level numeric, doctest sweep), changelog fragment,
   ROADMAP dashboard row (B8 → ✅), AC connectivity confirmation experiment.
