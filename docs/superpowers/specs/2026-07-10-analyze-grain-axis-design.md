# `analyze()` grain axis — design

**Date:** 2026-07-10
**Status:** Approved design, pending implementation plan
**Depends on:** the bounded intrinsic-unit search (`pyphi.macro.complexes`), complex unification (macro winners are `Complex` objects)
**Relates to:** the grain-search cost pre-flight (`SearchBounds.estimate`)

## Goal

Make the grain search reachable from the front door: a `grains=` keyword on
:func:`pyphi.analyze.analyze` that dispatches to `pyphi.macro.complexes`
under the same formalism handling and returns the `ComplexesResult`.

## Background

`analyze(substrate, state, *, subset, formalism, compute)` is the one
high-level entry point: it builds the candidate system, runs the analysis
under the active (or a named) formalism, and returns an `Analysis` bundle
or a raw result via `compute`. The grain search — "which units, at which
grain, are intrinsic?" — lives in `pyphi.macro.complexes(substrate,
micro_history, bounds, parallel_kwargs=...)` and is not discoverable from
the front door. `ComplexesResult` already carries everything a caller
needs (`complexes`, `maximal_complex`, `records`, `ties`, a display card),
so no new result type is warranted.

## Design

### Signature

```python
def analyze(
    substrate: Any,
    state: tuple[int, ...],
    *,
    subset: Any = None,
    formalism: str | None = None,
    compute: Any = None,
    grains: Any = None,
    parallel_kwargs: dict | None = None,
) -> Analysis | Any:
```

- `grains` — `None` (default): the existing single-system path, unchanged.
  A `SearchBounds` instance: run the grain search with those bounds.
  `True`: run with default bounds (`SearchBounds()`), so opting in is one
  keystroke: `analyze(substrate, state, grains=True)`. Any other value
  raises `ValueError`.
- `parallel_kwargs` — forwarded to `pyphi.macro.complexes`; only
  meaningful with `grains`.

### Dispatch semantics

- With `grains` set, `analyze` calls
  `pyphi.macro.complexes(substrate, state, bounds,
  parallel_kwargs=parallel_kwargs)` inside the same formalism-override
  context the single-system path uses, and returns the `ComplexesResult`.
- `formalism="IIT_3_0"` (or an active IIT 3.0 config) composes naturally:
  the macro driver raises its existing `ValueError` — `analyze` adds no
  duplicate guard.
- **`state` doubles as micro history.** The macro driver's
  `_normalized_history` already accepts a bare state when the bounds
  require one micro step (`max_micro_grain == 1`) and a sequence of states
  (oldest first) otherwise, with error messages that state the required
  length. `analyze` passes `state` through untouched.
- The deferred import of `pyphi.macro` happens inside the `grains` branch,
  so the single-system path pays nothing.

### Errors (all `ValueError`)

- `grains` together with `subset`: a grain search assembles systems from
  unit pools over the whole universe (Eq. 18); a fixed subset contradicts
  it.
- `grains` together with `compute`: `compute` selects single-system
  results.
- `parallel_kwargs` without `grains`: it has no meaning on the
  single-system path.
- `grains` that is neither `None`, `True`, nor a `SearchBounds` instance
  (note: `grains=False` is treated as an error, not as `None` — passing it
  is a sign of confusion, and `isinstance(True, int)` pitfalls are avoided
  by checking `grains is True` explicitly).

### Documentation

- `analyze`'s docstring gains `grains` and `parallel_kwargs` parameter
  entries and a `Returns` addition (`ComplexesResult` when `grains` is
  set), plus a `state` note (a micro-history sequence is required when the
  bounds admit update grains above 1).
- The module docstring widens from "a single system's IIT analysis" to
  cover the grain-search dispatch in one sentence.
- Changelog feature fragment.

## Testing

`test/test_analyze.py` additions:

- `grains=True` on the min-substrate under `presets.iit4_2023` returns a
  `ComplexesResult` equal in content to a direct
  `pyphi.macro.complexes(substrate, state, SearchBounds())` run (same
  winner units and φₛ, same record count).
- `grains=SearchBounds(max_depth=0)` returns the depth-0 result (micro
  complexes as `Complex` objects).
- `formalism="IIT_3_0"` with `grains=True` raises `ValueError` matching
  `"IIT_3_0"`.
- `grains=True, subset=(0,)` raises; `grains=True, compute="sia"` raises;
  `parallel_kwargs={}` without `grains` raises; `grains=0.5` raises.
- The existing single-system tests pass unchanged.

## Out of scope

- Exposing the cost pre-flight through `analyze`
  (`SearchBounds.estimate(substrate)` is that door).
- Any change to the `Analysis` bundle or to `ComplexesResult`.
- Temporal-history convenience beyond pass-through (e.g. simulating a
  history automatically) — the caller supplies the observed micro history.
