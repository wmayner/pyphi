# Precision-aware structural equality comparator

**Date:** 2026-05-26
**Status:** Design

## Problem

`pyphi/models/cmp.py::general_eq` is the entry point for `__eq__` on result objects across the codebase: `SystemIrreducibilityAnalysis` (IIT 3.0 and 4.0), `CauseEffectStructure`, `AcSystemIrreducibilityAnalysis`, `RepertoireIrreducibilityAnalysis`, `StateSpecification`, distinctions. It already special-cases ``phi`` and ``alpha`` for precision-aware comparison (via ``utils.eq``), but every other attribute falls through to ``numpy_aware_eq`` which uses strict comparison: ``np.array_equal`` for ndarrays and ``==`` for scalars.

Strict comparison cannot distinguish two kinds of difference:

- **Op-order drift** (~1e-15-1e-16): float64 arithmetic produces different ULP-level results when the same mathematical operation is evaluated via different code paths. This is *noise*, not a regression.
- **Math regression** (1e-3 and larger): a real change in the algorithm produces a different answer.

Conflating these has cost the project five fixture regenerations during the 2.0 development cycle (commits `593adef4`, `f89e60f2`, `fd135387`, `cb571f16`, and Task 15's `c1b32729`). Each regeneration forces the developer into the amendment's "extraordinary evidence" escape hatch — even though the change was provably equivalent math via a different evaluation path.

The fix is to make the structural-equality comparator absorb op-order noise while still detecting real math regressions. `test_golden_regression.py` already does this for its golden fixtures (using `np.allclose(rtol=1e-12, atol=1e-12)`); the model-level `__eq__` does not.

## Approach

Replace the strict comparison primitives in `numpy_aware_eq` with precision-aware equivalents, using a fixed module-level tolerance constant that both production `__eq__` and the test-fixture comparator consume.

### Tolerance constant

```python
# pyphi/models/cmp.py
EQUALITY_TOLERANCE = 1e-13
"""Tolerance for structural equality on IIT quantities. Absorbs op-order
drift in float64 arithmetic on IIT measures while distinguishing real
math regressions. Used by `numpy_aware_eq` (model `__eq__`) and by
golden-fixture comparisons in the test suite. Independent of
`config.numerics.precision`, which governs user-configurable phi
comparison via `utils.eq`."""
```

Selection of `1e-13`:

- Op-order noise on IIT 4.0 quantities (intrinsic information, repertoires, structure-integrated phi) caps at ~1e-13 after compounding through several arithmetic steps. The five fixture regenerations across 2.0 all reflected drift at this scale or smaller.
- For Φ values in the thousands (the IIT 4.0 paper's Fig 6D shows Φ=11451), `rtol=1e-13` gives an absolute tolerance of ~1e-10 — still ~10⁹ tighter than the smallest plausible real math regression (~1e-3 on bounded quantities, larger on unbounded).
- Coincidentally matches `config.numerics.precision`'s default value (`13`) — but they remain logically independent. See the *Three thresholds* section below.

### Updated `numpy_aware_eq`

```python
def numpy_aware_eq(a: Any, b: Any) -> bool:
    """Return whether two objects are equal via recursion, with float
    leaves compared up to ``EQUALITY_TOLERANCE``.

    Arrays compare via ``np.allclose``; float scalars via ``math.isclose``;
    other types via ``==``.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.allclose(
                a, b,
                rtol=EQUALITY_TOLERANCE,
                atol=EQUALITY_TOLERANCE,
            )
        except (ValueError, TypeError):
            return False
    if (
        (isinstance(a, Iterable) and isinstance(b, Iterable))
        and not isinstance(a, str)
        and not isinstance(b, str)
    ):
        if len(a) != len(b):
            return False
        return all(numpy_aware_eq(x, y) for x, y in zip(a, b, strict=False))
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return math.isclose(
            float(a), float(b),
            rel_tol=EQUALITY_TOLERANCE,
            abs_tol=EQUALITY_TOLERANCE,
        )
    return a == b
```

Notable changes from the current implementation:

- Arrays compare via ``np.allclose`` rather than ``np.array_equal``. ``equal_nan`` is left at its default (``False``): NaN ≠ NaN, preserving the current behavior so we don't smuggle a second semantic change into a precision-aware refactor.
- Float scalars compare via ``math.isclose`` with both ``rel_tol`` and ``abs_tol``, supporting unbounded Φ magnitudes correctly.
- Shape mismatch on arrays → ``ValueError`` from ``np.allclose`` → ``False`` (no exception leak). Same defensive guard for ``TypeError`` from non-numeric arrays.
- Iterable recursion and `==` fallback unchanged.

### Updated `test_golden_regression.py`

```python
from pyphi.models.cmp import EQUALITY_TOLERANCE

# replace the hardcoded RTOL/ATOL definitions
RTOL = EQUALITY_TOLERANCE
ATOL = EQUALITY_TOLERANCE
```

The fixture-comparison code in `_compare` (already using ``np.allclose`` and ``math.isclose``) is otherwise unchanged. The hardcoded `1e-12` becomes `1e-13` — a tightening of one decimal. All 25 current goldens pass byte-identical, so the tightening cannot break anything that currently passes. Future fixture comparisons gain a slightly tighter regression net.

### `general_eq` unchanged in shape

The pattern of special-casing `phi`/`alpha` (precision-aware via `utils.eq`) and `mechanism`/`purview` (set equality on integer indices) and falling through to `numpy_aware_eq` is preserved. The fall-through is now precision-aware, which is the load-bearing change.

`mechanism` and `purview` are tuples of integer indices; set equality is the right operation and is unrelated to floating-point tolerance.

## Two thresholds in the codebase

After this change, the codebase has two distinct precision-related thresholds. Each answers a different question:

| Location | Value | Configurable | Question answered |
|---|---|---|---|
| `utils.eq` | `10**(-config.numerics.precision)`, default `1e-13` | yes (`config.numerics.precision`) | User-facing φ comparison ("are these two φ values equal under my tolerance setting?") |
| `cmp.EQUALITY_TOLERANCE` | `1e-13` | no | Op-order-noise threshold on IIT quantities. Used by `numpy_aware_eq` (production `__eq__` on result objects) AND by `test_golden_regression.py` (fixture comparison) — both ask the same underlying question, so they share the constant by import. |

The values happen to be equal at default configuration (`config.numerics.precision = 13` gives `10**(-13) = 1e-13`). They remain logically independent: changing the user's config affects only `utils.eq`; `EQUALITY_TOLERANCE` is fixed in code. Changing `EQUALITY_TOLERANCE` requires a deliberate edit in `cmp.py` and propagates to both consumers atomically.

`config.numerics.precision` is a holdover from the IIT 3.0 / EMD era. The `pyemd` C library produced numerical noise in low bits, and `13` was tuned to absorb it. The configurability existed so users could tighten or loosen based on their `pyemd` build. For IIT 4.0 (intrinsic-difference measures implemented in pure NumPy / Python), the "right tolerance for op-order drift" is a property of float64 arithmetic on IIT quantities, not a user preference.

ROADMAP follow-up entry (under "Informal notes — pre-release housekeeping"):

> **Retire `config.numerics.precision` when IIT 3.0 is dropped.** The
> setting is a holdover from the EMD-era `pyemd` C-library noise. Once
> 3.0 support is gone, `utils.eq` should migrate to use a fixed
> module-level constant (or import `cmp.EQUALITY_TOLERANCE` directly),
> and the config field should be removed. Independent of P12b.

## Out of scope

- **Replacing `general_eq` with per-class `__eq__` definitions.** The attribute-list pattern (`general_eq(self, other, ['phi', 'alpha', 'mechanism', ...])`) is an EMD-era convention and could be replaced with type-introspecting per-class `__eq__` methods that know which fields are float-bearing. That refactor would be its own project after the post-P12b architecture stabilizes (post AC k-ary cutover, post macro rewrite, post `pyphi/tpm.py` consolidation).
- **Removing `config.numerics.precision`.** Required for IIT 3.0 / `pyemd` support; the ROADMAP entry above captures the post-3.0 follow-up.
- **Tightening or loosening individual attribute tolerances.** All float-bearing comparisons through `numpy_aware_eq` use the same `EQUALITY_TOLERANCE`. If future evidence shows a specific attribute needs a different tolerance, special-case it in `general_eq` (parallel to the existing `phi`/`alpha` pattern).
- **Reworking `set` equality for `mechanism`/`purview`.** Those compare integer index tuples; tolerance is meaningless. No change.
- **Changes to `numpy_aware_eq`'s behavior on non-numeric types.** Strings, ints, None, etc. continue to compare via `==`. Tolerance only affects float-bearing leaves.

## Acceptance criteria

After the change lands:

1. **`numpy_aware_eq` precision-aware.** Arrays use `np.allclose`; float scalars use `math.isclose`; both with `rtol=atol=EQUALITY_TOLERANCE`. Iterables and non-numeric types unchanged.
2. **`test_golden_regression.py`** imports `EQUALITY_TOLERANCE` from `cmp.py`; `RTOL`/`ATOL` removed as separate hardcoded values.
3. **Goldens 25/25** byte-identical (the tightening from `1e-12` to `1e-13` cannot break anything that currently passes byte-identical).
4. **Full `uv run pytest`** (no path; includes doctests) passes.
5. **Pyright + ruff clean.**
6. **`test_models.py` numpy_aware_eq tests** still pass. The 4 existing unit tests use differences ≥1, far beyond any tolerance threshold; no behavioral regression.
7. **ROADMAP** has the post-3.0 `config.numerics.precision` retirement entry.
8. **`changelog.d/`** has a `feature.md` fragment describing the user-visible effect (structural equality on result objects now precision-aware up to `1e-13`).

## Risks

- **A previously-passing test might silently start passing when it should fail.** The new comparator accepts up to `1e-13` drift; any test that relied on detecting drift smaller than this would no longer fire. Mitigation: the existing unit tests for `numpy_aware_eq` use differences ≥1, so they are safe. Risk is on tests we haven't audited. Mitigation: full suite run before and after; review any newly-skipping tests.
- **A real math regression at the threshold might be masked.** Threshold of `1e-13` is conservative (10⁹ tighter than the smallest plausible real regression at `1e-3`), but a regression that happens to land between `1e-15` and `1e-13` would be absorbed as if it were noise. Mitigation: this is the same tradeoff `test_golden_regression.py` already accepts at `1e-12`; tightening to `1e-13` is a slight risk reduction. If a future regression at this scale appears, the threshold becomes the natural place to adjust.
- **NaN handling preserved.** `np.allclose` defaults to `equal_nan=False`, matching the current `np.array_equal` behavior (NaN ≠ NaN). If a future audit shows NaN-bearing fixtures intend "no information here" semantics, switching to `equal_nan=True` is a separate one-line change with its own consideration.

## Implementation note

Single commit, small surface area (~30 lines net across `pyphi/models/cmp.py`, `test/test_golden_regression.py`, plus the ROADMAP entry and changelog fragment). No structural refactor; just comparator-primitive replacement and a new constant. Goldens-stable by construction (tightening from `1e-12` to `1e-13` cannot break byte-identical passes).
