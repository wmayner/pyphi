# B15 — `result.diff()`: Design

**Status:** draft (awaiting user review)
**Date:** 2026-06-16
**Predecessor:** B8 (`result.explain()` — the `Displayable` + `to_pandas` patterns and the per-type hook convention B15 reuses), B21 (the `pyphi.display` model), `ConfigSnapshot.diff` (landed)
**Successors:** P15 (surface freeze — `.diff()` and `ResultDiff` are public surface frozen there)

---

## 1. Motivation

Comparison is the core epistemic operation in IIT research: a researcher runs an
analysis under two conditions — a different state, a config change, a perturbed
substrate — and needs to see *what changed and why*. Today that means eyeballing two
reprs or hand-writing comparison code. The data to compare is fully present on the result
objects (φ, partition with `lex_key`, distinctions keyed by mechanism, relations, and a
`ConfigSnapshot`), but there is no structured delta.

`result.diff(other)` returns a typed `ResultDiff`: the Δφ, whether the MIP genuinely
changed (not merely a tie-reshuffle), which distinctions/relations were gained, lost, or
changed, and — composing the landed `ConfigSnapshot.diff` — which config differences could
explain the change. It pairs with B8 (`result.explain()`): explain answers *why one
result is what it is*; diff answers *what changed between two*.

## 2. Goals

- A frozen `ResultDiff` (named fields + uniform `Change` records) returned by
  `a.diff(b)` on every top-level result type (CES, 4.0/3.0 SIA, RIA/MICE/Distinction,
  AcSIA/Account).
- A real MIP change is distinguished from a co-optimal tie-reshuffle using `lex_key` +
  the tie set.
- `ResultDiff` is `Displayable` (B21) and `to_pandas`-able (P14d), reusing B8's patterns.
- Config-diff attribution by composing `ConfigSnapshot.diff` — no reimplementation.
- Pure comparison: `.diff()` never recomputes and never changes a result.

## 3. Non-goals

- **Reconciling different substrates structurally.** Cross-substrate diffs are allowed
  (a valid research comparison) and keyed by mechanism label, with a surfaced
  substrate-mismatch note; B15 does not attempt to align differing topologies.
- **Macro `complexes()` diff.** Out of scope, mirroring B8/B16.
- **A merge/patch operation.** `ResultDiff` describes a delta; it does not apply one.
- **Relation "changed" detection.** Relations are diffed as gained/lost by key only;
  per-relation change is deferred (low value, and relation identity already encodes φ).

## 4. The types — `pyphi/models/diff.py` (new)

A sibling to `explanation.py`, reusing `Displayable`, the display vocabulary, and the
`records_to_frame` pandas helper.

```
@dataclass(frozen=True)
class Change:
    kind: str          # "distinction_gained" | "distinction_lost" | "distinction_changed"
                       # | "relation_gained" | "relation_lost"
                       # | "link_gained" | "link_lost" | "link_changed"
    key: Any           # mechanism (distinctions/links), relata identity (relations)
    a_value: Any = None
    b_value: Any = None
    tone: str | None = None

@dataclass(frozen=True)
class ResultDiff(Displayable):
    subject: str                              # e.g. "ΔΦ_s = +0.12"
    level: str                                # "system" | "mechanism"
    delta_phi: Any                            # b.phi - a.phi (signed; Δα for AC)
    mip_changed: bool
    binding_direction_changed: bool | None    # None where the concept doesn't apply
    changes: tuple[Change, ...]
    config_diff: dict[str, tuple[Any, Any]]   # from ConfigSnapshot.diff
    substrate_note: str | None = None         # set when the two substrates differ
```

`ResultDiff._describe()` returns a `Description` with sections: **Summary** (Δφ,
MIP-changed, binding-direction-changed, substrate note if any), **Changes** (the
`Change` records as a table, grouped/toned), **Config differences** (the `config_diff`
map as a table). `to_pandas()` returns one tidy long-format `DataFrame`
(`category, key, a, b`) — scalar deltas and config differences as rows alongside the
element changes — built with `records_to_frame`.

## 5. The `.diff()` API

`a.diff(b) -> ResultDiff` on each top-level result type, built via a per-type
`_changes()` hook (mirroring B8's `_findings()`), so subtypes contribute only their
level-appropriate deltas. The named scalar fields (`delta_phi`, `mip_changed`, …) are
computed in `diff()`; `_changes()` returns the element-level `Change` tuple.

Content by type:

- **`CauseEffectStructure`** (richest): Δφ_s, MIP-change, **distinctions** gained / lost /
  changed (keyed by mechanism), **relations** gained / lost (keyed by relata), config-diff.
- **4.0 / 3.0 SIA**: Δφ, MIP-change, binding-direction-change, config-diff.
- **RIA / MICE / Distinction**: Δφ, purview-change, MIP-change, config-diff (Distinction
  reports its binding direction's change; MICE delegates to its RIA).
- **AcSIA**: Δα, MIP-change, config-diff. **Account**: links gained / lost / changed
  (key = direction + mechanism + purview), Δ(Σα).

## 6. MIP-change vs tie-reshuffle (the correctness point)

`mip_changed` is `True` **iff** `b.partition.lex_key()` is *not* among `a`'s tied MIPs:

```
a_tie_keys = {t.partition.lex_key() for t in a.ties}
mip_changed = b.partition.lex_key() not in a_tie_keys
```

A partition `a` could equally have selected (a co-optimal tie) is a reshuffle, not a real
change. This uses `lex_key` plus the tie set, which already encodes `EQUALITY_TOLERANCE`
from tie resolution. For results without a `.ties` set (none of the in-scope SIA types
lack it; RIA carries `partition_ties`), fall back to a direct `lex_key` inequality
combined with `utils.eq` on φ.

## 7. "Changed" semantics for shared elements

- **Distinction** present in both (same mechanism): *changed* if `not utils.eq(a.phi,
  b.phi)` **or** the purview differs **or** the specified state differs. The `Change`
  carries both distinctions so magnitude and direction are inspectable.
- **Account link** present in both (same direction+mechanism+purview): *changed* if
  `not utils.eq(a.alpha, b.alpha)`.
- **Relation**: gained / lost by key only (no *changed*; see §3).

## 8. Comparability

- **Type mismatch** (e.g. SIA vs CES): `a.diff(b)` raises `TypeError` naming both types.
- **Same type, different substrate** (different node labels / indices): allowed. Element
  deltas key by mechanism *label*; `substrate_note` records the mismatch so the reader
  knows gained/lost may reflect topology, not dynamics. `config_diff` still applies.
- **Direction of the delta**: `a.diff(b)` reads "from `a` to `b`" — `delta_phi = b − a`,
  gained = in `b` not `a`, lost = in `a` not `b`.

## 9. Display & pandas integration

- `_describe()` → B21 `Description`; the ASCII backend reproduces content untoned, HTML
  reuses B8's cause/effect tones and renders the change/config tables natively.
- `to_pandas()` → tidy `DataFrame` (`category, key, a, b`) via `records_to_frame`,
  consistent with B8's export and P14d's long-format convention.

## 10. Testing

- **`ResultDiff` unit tests**: construction, `_describe` renders (ASCII + HTML), `to_pandas`
  shape; `Change` records round-trip.
- **MIP-reshuffle invariant**: two analyses that differ only by a co-optimal tie pick
  yield `mip_changed == False`; a genuinely different MIP yields `True`. Pinned on a
  network with a known MIP tie.
- **Per-type numeric assertions** on canonical examples: a state/config change produces the
  expected `delta_phi` sign, the expected distinctions gained/lost/changed, and the
  expected `config_diff` keys.
- **`.diff()` coverage invariant**: every top-level result type returns a valid
  `ResultDiff` (parallel to B8's coverage invariant), in `test/test_result_diff.py`.
- **Comparability**: `TypeError` on type mismatch; cross-substrate diff sets
  `substrate_note`.
- **No value change**: `.diff()` is pure; goldens unaffected. `uv run pytest` (no path
  argument) for the doctest sweep, since `ResultDiff` adds public repr/HTML surface.

## 11. Build order (for the implementation plan)

1. `pyphi/models/diff.py`: `Change`, `ResultDiff` (+ `_describe`, `to_pandas`); unit-test
   in isolation. Re-export from `pyphi/models/__init__.py`.
2. Shared helpers: the MIP-reshuffle test and the `config_diff` extraction (a small
   `_diff_common(a, b)` returning `delta_phi`, `mip_changed`, `config_diff`,
   `substrate_note`).
3. `_changes()` + `diff()` per type: SIA (4.0 → 3.0) → CES → mechanism (RIA/MICE/
   Distinction) → AC (AcSIA/Account).
4. Display polish + `to_pandas`.
5. Tests (coverage invariant, MIP-reshuffle, per-type numeric, doctest sweep), changelog
   fragment, ROADMAP dashboard row (B15 → ✅).
