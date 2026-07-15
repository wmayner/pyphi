# Distinction-level reducibility shortcircuit

**Date**: 2026-07-15
**Status**: Approved design, pending implementation plan

## Problem

`pyphi/formalism/queries.py::distinction()` always evaluates the cause
MICE before the effect MICE. When one direction is reducible — most
visibly when a mechanism has an empty candidate purview set in one
direction (e.g. no common effect purview) — the distinction's φ is 0
regardless of the other direction, yet the full, potentially
multi-million-partition search still runs on that other direction. The
same shape exists in IIT 3.0's `concept()`
(`pyphi/formalism/iit3/__init__.py`).

`find_mice` already returns instantly when *its own* direction has no
candidate purviews (the `NO_PURVIEWS` null MICE); the waste is entirely
cross-direction.

## Design

### Config option

`formalism.iit.shortcircuit_distinctions: bool = True`, defined in
`IITConfig` (`pyphi/conf/formalism.py`) beside `shortcircuit_sia`:

- Added to the bool-validation list in `__post_init__`.
- Docstring section mirroring the `shortcircuit_sia` contract: when
  `True`, distinction evaluation stops early on detected reducibility
  and the skipped direction's selection margins and ties are absent;
  when `False`, both directions are always fully evaluated (exact
  margins and complete ties everywhere). φ values of surviving
  (φ > 0) distinctions are identical either way.
- Default `True`: fast by default, exhaustive as the opt-out, matching
  `shortcircuit_sia`. The default never changes any surviving
  distinction's contents — only what zero-φ distinctions carry.
- Presets are complete/absolute; verify the new field flows into the
  IIT 3.0 / IIT 4.0 presets with default `True` (no formalism wants a
  different value).

### New `NullResultReason` member

`OTHER_DIRECTION_REDUCIBLE` (mechanism level; added to the level
mapping in `pyphi/models/explanation.py`): this direction was not
evaluated because the distinction is already reducible via the other
direction. The skipped MICE is a null RIA carrying this reason:

- φ reads 0 as a placeholder. The direction's true φ_max is unknown —
  only the *distinction's* φ (the min across directions) is guaranteed
  to be 0. The reason makes this explicit; `explain()` reports it.
- Purview is empty; margins (`purview_margin`, `state_margin`,
  `partition_margin`) and ties are absent, consistent with the
  documented early-stop contract.

### Shortcircuit logic

One shared helper in `queries.py`, used by both `distinction()` and
IIT 3.0's `concept()` (which threads its `cause_purviews` /
`effect_purviews` kwargs into the checks). Cause-first order is
preserved so results are deterministic. When
`shortcircuit_distinctions` is `True`:

1. **Pre-flight**: if the *effect* direction's candidate purview set is
   empty (`potential_purviews` — cheap, and cached under
   `cache_potential_purviews`), return immediately. The effect side
   gets the canonical `NO_PURVIEWS` null from `find_mice` (instant in
   this case); the cause side gets an `OTHER_DIRECTION_REDUCIBLE`
   null. No cause-side pre-flight is needed: `find_mice(CAUSE)` with an
   empty purview set returns instantly and the next guard fires.
2. **Post-cause**: run the cause search; if the winning MICE has φ = 0
   (trivially or genuinely reducible), skip the effect search and
   return with an `OTHER_DIRECTION_REDUCIBLE` null effect.
3. Otherwise run the effect search as today.

`queries.phi()` gets the same early return (if the cause MIP φ is 0,
skip the effect MIP), unconditionally: it returns a bare float with no
ties/margins surface to truncate, and φ values are non-negative, so the
result is unchanged.

### Machinery interactions (audited)

- **Tie resolution / congruence / relations / CES**: φ = 0 distinctions
  are dropped by `filter(None, …)` in `all_distinctions` before
  reaching `UnresolvedDistinctions`, congruence resolution, relations,
  or any CES; no tie machinery reads a skipped MICE. IIT 3.0
  constellations filter by positive φ likewise.
- **Margins**: the `shortcircuit_sia` docstring already defines the
  contract — early stops leave selection margins undefined, and the
  flag's `False` setting restores exact margins. The skipped direction
  fits that contract; display code handles `None` margins.
  `pyphi.landscape` operates on already-computed structures.
- **Caveat for direct callers**: anyone iterating the MICEs of
  reducible distinctions (e.g. IIT 3.0's `only_positive_phi=False`
  path) sees an unevaluated placeholder in the skipped direction
  unless they set the option to `False`. Documented on the option.

### Verification point (settled during implementation, not deferred)

Confirm the IIT 3.0 partitioned-constellation path does not consume
repertoires from φ = 0 partitioned concepts (the expectation is that
vanished concepts are measured against the null concept using their
unpartitioned repertoires). If it does consume them, that path threads
an explicit internal override rather than shipping the assumption.

## Tests

1. A mechanism with no effect purviews: distinction φ = 0, effect
   reason `NO_PURVIEWS`, cause reason `OTHER_DIRECTION_REDUCIBLE`, and
   the cause search provably never ran (call counter via monkeypatch).
2. A genuinely reducible cause (computed φ = 0): effect search skipped,
   effect carries the new reason.
3. Flag off: both directions fully evaluated; margins present; current
   behavior preserved.
4. Regression: the surviving CES of an example network is identical
   with the flag on and off.

All φ-asserting tests pin their formalism via the preset context
managers (`IIT_3_CONFIG` / `IIT_4_CONFIG`).

## Documentation and surfaces

- Changelog fragment (`.config.md`) in `changelog.d/`.
- Option line in `pyphi/conf/CLAUDE.md`.
- MCP server content check: surface the new option in
  `pyphi/mcp/content/configuration.md` and/or `performance.md` where
  `shortcircuit_sia` appears.
