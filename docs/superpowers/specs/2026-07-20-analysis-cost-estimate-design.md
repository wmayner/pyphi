# Analysis cost estimate (pre-flight for analyze/sia/ces)

**Date:** 2026-07-20
**Status:** Approved design, pending implementation plan
**Relates to:** the grain-search cost pre-flight (`2026-07-10-grain-cost-preflight-design.md`), whose fidelity ruling — structured counts and weights, never predicted seconds — this design inherits.

## 1. Motivation

A single-system analysis has a knowable workload before any φ is computed:
the number of system partitions the SIA sweeps, the number of candidate
mechanisms, the connectivity-pruned purview sets, and the mechanism
partitions evaluated per (mechanism, purview) pair. Today none of this is
visible to users: the only pre-flight is the MCP server's hardcoded node
limits (`_CES_NODE_LIMIT = 7`, `_SIA_NODE_LIMIT = 9`), which are blind to
the partition scheme, the connectivity, and the alphabet — a sparse 8-unit
system refuses even when its real workload is small, and an expensive
sub-limit configuration passes silently.

The grain search already has this surface (`SearchBounds.estimate` →
`SearchEstimate`). This design adds the single-system counterpart and makes
the MCP guard read real counts instead of node counts.

## 2. Scope

- New module `pyphi/estimate.py` with `estimate_analysis(...)` returning an
  `AnalysisEstimate`; exported as `pyphi.estimate_analysis`.
- Counting is **config-generic**: the work axes are enumerated under
  whatever formalism configuration is active (IIT 3.0 and both IIT 4.0
  presets all work). The IIT 4.0-specific structure-size context from
  `formalism/iit4/bounds.py` (possible distinctions and relations) is
  included only when a 4.0 formalism is active and all units are binary.
- The MCP `analyze` guard delegates to the estimate: count thresholds
  replace the node limits, and a new `estimate_cost` MCP tool exposes the
  estimate directly.
- The shared counting utilities (the work-budget counter and the
  system-partition count memo) move to `pyphi/estimate.py`;
  `pyphi/macro/estimate.py` imports them with no behavior change.

Out of scope: wall-time prediction (per the standing fidelity ruling);
changes to the grain-search estimate beyond the import; estimating memory
in bytes; estimating actual-causation analyses.

## 3. API

```python
def estimate_analysis(
    substrate: Substrate,
    subset: tuple[int, ...] | None = None,
    compute: str | None = None,
    limit: int = 1_000_000,
) -> AnalysisEstimate
```

- `substrate`, `subset`, `compute` mirror `pyphi.analyze`: `subset=None`
  uses the whole substrate; `compute=None` estimates the full analysis,
  `"sia"` only the system-partition axis, `"ces"` only the distinction
  axis. Any other `compute` value raises `ValueError`.
- There is no state parameter: every counted quantity is
  state-independent.
- `limit` is the work budget for the estimate itself (counted items plus
  partition-enumeration steps, as in `estimate_search`). When the budget is
  exhausted, counting stops immediately and the result is marked capped.

### `AnalysisEstimate`

Frozen dataclass in `pyphi/estimate.py`, `Displayable` + `ToPandasMixin`
(card rendering and `to_pandas()` record, matching `SearchEstimate`).

Fields (`None` means "not in this estimate's scope" — the axis was excluded
by `compute`, or the context is not applicable under the active config):

- `n_units: int` — candidate system size m.
- `state_space_size: int` — product of the candidate units' alphabet
  sizes. The per-evaluation cost weight; reported, never multiplied into a
  synthetic total.
- `compute: str` — `"full"`, `"sia"`, or `"ces"`.
- `system_partitions: int | None` — count under the active system
  partition scheme (the SIA axis).
- `mechanisms: int | None` — 2ᵐ − 1.
- `purview_evaluations: int | None` — Σ over mechanisms and both
  directions of the connectivity-pruned purview count (the
  repertoire-computation axis).
- `mechanism_partition_sweeps: int | None` — Σ over (mechanism, purview)
  pairs of the mechanism-partition count for that size pair (the dominant
  CES axis).
- `relations_closed_form: bool | None` — `True` when the active relation
  backend computes relations in closed form (ANALYTICAL), `False` when it
  enumerates (CONCRETE); `None` when relations are not part of the
  estimate (`compute="sia"`, or a formalism without relations).
- `possible_distinctions: int | None`, `possible_relations: int | None` —
  structure-size context from `formalism/iit4/bounds.py`; present only
  under a 4.0 formalism with binary units. `possible_relations` doubles as
  the enumeration worst case when `relations_closed_form` is `False`.
- `capped: bool` — the work budget (or a per-pair enumeration cap) was
  hit; counts are lower bounds and the card renders them with a `≥`
  qualifier.

## 4. Counting method

Counts are produced by driving the real enumeration machinery, never by
closed-form formulas that could drift from it:

- **System partitions**: enumerate `pyphi.partition.system_partitions`
  over `range(m)` under the active scheme, memoized on (scheme name, m).
  This memo currently lives in `pyphi/macro/estimate.py`
  (`_partition_counts`); it moves to `pyphi/estimate.py` and the macro
  module imports it.
- **Purviews**: the estimate constructs the candidate `System` internally
  in an all-zeros reference state and calls its `potential_purviews` for
  each mechanism and direction — the exact code path the CES walk uses.
  Purview sets depend only on connectivity, so the reference state is
  immaterial; `Substrate.potential_purviews` is cached on the
  connectivity-matrix fingerprint, so repeated estimates over the same
  topology are cheap.
- **Mechanism partitions**: `pyphi.partition.mechanism_partitions` counts
  depend only on the size pair (|mechanism|, |purview|), so each
  (scheme name, |M|, |P|) is enumerated once and memoized. Enumeration
  steps charge the work budget, so a pathologically large pair trips the
  limit mid-enumeration rather than hanging the estimate.
- **Budget**: a single `_Counter` (moved from `pyphi/macro/estimate.py`)
  is charged per counted item and per enumeration step. Hitting `limit`
  raises the internal stop signal; the estimate returns immediately with
  `capped=True`.

The IIT 4.0 context numbers call
`bounds.number_of_possible_distinctions` / `number_of_possible_relations`
directly (cheap closed forms; these are counting identities, not workload
formulas, so the no-closed-form rule does not apply to them).

## 5. MCP delegation

In `pyphi/mcp/server.py`:

- The node-limit constants are replaced by count thresholds:
  `_SIA_PARTITION_LIMIT` (system-partition count) and `_CES_SWEEP_LIMIT`
  (mechanism-partition sweep count). Their default values are computed at
  implementation time as the counts at today's refusal boundary — the
  fully connected 9-unit binary system for the SIA threshold and the fully
  connected 7-unit binary system for the CES threshold, under the default
  configuration — and recorded as constants with a comment stating the
  derivation. **Contingency:** if the 7-unit sweep count turns out to
  exceed ~10⁸ (making the guard's one-time first-call enumeration cost
  itself unreasonable), the CES threshold is instead set to 10⁸ with a
  comment noting the deliberate tightening.
- The `analyze` tool runs `estimate_analysis(substrate, subset=None,
  compute=..., limit=threshold + 1)` before computing. The gating count
  matches today's limit selection: `compute="sia"` gates on
  `system_partitions` against `_SIA_PARTITION_LIMIT`; `compute="ces"` and
  `"full"` gate on `mechanism_partition_sweeps` against
  `_CES_SWEEP_LIMIT`. If the gating count exceeds its threshold — or the
  estimate capped out — the tool refuses with the actual counts in the
  message, unless `confirm_large=true`. `confirm_large` semantics are unchanged. When the
  caller passes a `formalism` override, the estimate runs under that same
  preset (inside the same config override the analysis will use), so the
  guard sees the workload the analysis will actually have.
- Because the size-pair and scheme memos persist for the server process,
  the guard's cost after the first large call is proportional to the
  number of size pairs, not the partition counts.
- New MCP tool `estimate_cost(handle, compute="full", formalism=None)`:
  returns the estimate's card plus the counts as a dict, so agents can
  pre-flight explicitly before committing to an analysis.

## 6. Testing

New `test/test_estimate.py`, formalism pinned with complete presets:

- Exact counts on a fully connected 3-unit binary system, checked against
  direct enumeration of the same registries (e.g. `system_partitions`
  yields 22 under `DIRECTED_SET_PARTITION`; `mechanism_partition_sweeps`
  equals the hand-summed per-pair counts).
- A sparse-connectivity case where purview pruning visibly reduces
  `purview_evaluations` below the fully connected count.
- A scheme-change case: overriding the system partition scheme changes
  `system_partitions` (config sensitivity).
- An IIT 3.0 preset case: the estimate runs and reports the 3.0 schemes'
  counts (formalism genericity); `possible_distinctions` /
  `possible_relations` are `None`.
- `compute="sia"` / `"ces"` scoping: excluded axes are `None`.
- `limit` early exit: a small budget yields `capped=True` and lower-bound
  counts.
- A non-binary (k-ary) case: work axes are counted, 4.0 context fields are
  `None`.
- `subset` restriction: counts over a proper subset match the subsystem's
  own enumeration.
- Pandas record and card rendering smoke tests.
- Grain-search regression: the existing `estimate_search` tests keep
  passing after the shared-helper move (no new tests needed).

MCP tests (in the existing MCP test module):

- The guard refuses above threshold with counts in the message, passes at
  the calibrated boundary, and `confirm_large=true` bypasses it.
- A sparse system above the old node limit but under the count threshold
  passes without confirmation (the delegation's point).
- `estimate_cost` returns the card and counts for a registered substrate.

## 7. Documentation and bookkeeping

- A short passage in `docs/theory/computational-complexity.md` introducing
  `estimate_analysis` where that page discusses workload growth.
- The MCP `performance.md` content topic gains a pre-flight paragraph
  pointing at the `estimate_cost` tool.
- Changelog fragment `changelog.d/analysis-cost-estimate.feature.md`.
- ROADMAP row updated in the same change.
- `pyphi/__init__.py` exports `estimate_analysis`; the API reference picks
  up the new module through the existing autosummary setup.
